from enum import Enum
from dataclasses import dataclass
from typing import Optional, Union, Dict, Tuple, List
import os
import psutil
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Queue
from tqdm import tqdm
from threading import Lock
import threading
from x_filter.logging_setup import get_logger

log = get_logger()


class ArrayScale(Enum):
    """Unified array size categories."""

    TINY = "tiny"  # < 1M elements
    SMALL = "small"  # 1M - 100M elements
    MEDIUM = "medium"  # 100M - 1B elements
    LARGE = "large"  # 1B - 10B elements
    HUGE = "huge"  # > 10B elements


@dataclass
class ArrayInfo:
    """Information about array characteristics."""

    scale: ArrayScale
    size_gb: float
    elements: int
    element_size: int
    memory_needed_gb: float
    optimal_threads: int


@dataclass
class ChunkingStrategy:
    """Processing strategy parameters."""

    chunk_size: int
    num_chunks: int
    memory_per_chunk_gb: float
    threads_per_chunk: int
    total_threads: int
    cache_friendly: bool


class ThreadPoolManager:
    """Manages thread pools and their resources."""

    def __init__(self, resource_manager, threads_needed: int):
        self.resource_manager = resource_manager
        self.threads_needed = threads_needed
        self.pool = None

    def __enter__(self):
        """Acquire thread pool."""
        with self.resource_manager.thread_lock:
            available = (
                self.resource_manager.max_threads
                - self.resource_manager.active_threads.value
            )
            threads_to_use = min(self.threads_needed, available)
            if threads_to_use > 0:
                self.resource_manager.active_threads.value += threads_to_use
                self.pool = ThreadPoolExecutor(
                    max_workers=threads_to_use,
                    thread_name_prefix="resource_managed_thread",
                )
                return self.pool
            raise RuntimeError(
                f"No threads available. Active: {self.resource_manager.active_threads.value}"
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release thread pool."""
        if self.pool:
            self.pool.shutdown()
            with self.resource_manager.thread_lock:
                self.resource_manager.active_threads.value = max(
                    0, self.resource_manager.active_threads.value - self.threads_needed
                )


class MemoryManager:
    """Manages memory allocation and deallocation."""

    def __init__(self, resource_manager, bytes_needed: int):
        self.resource_manager = resource_manager
        self.bytes_needed = bytes_needed

    def __enter__(self):
        """Acquire memory."""
        with self.resource_manager.memory_lock:
            if (
                self.resource_manager.memory_allocated.value + self.bytes_needed
                <= self.resource_manager.max_memory
            ):
                self.resource_manager.memory_allocated.value += self.bytes_needed
                return self.bytes_needed
            raise RuntimeError(
                f"Not enough memory. Requested: {self.bytes_needed}, Available: {self.resource_manager.available_memory}"
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release memory."""
        with self.resource_manager.memory_lock:
            self.resource_manager.memory_allocated.value = max(
                0, self.resource_manager.memory_allocated.value - self.bytes_needed
            )


class ResourceManager:
    """Manage system resources for processing."""

    def __init__(
        self,
        max_memory: Optional[Union[str, int, float]] = None,
        max_threads: Optional[int] = None,
    ):
        manager = Manager()
        self.memory_lock = manager.Lock()
        self.thread_lock = manager.Lock()
        self.memory_allocated = manager.Value("i", 0)
        self.active_threads = manager.Value("i", 0)

        if max_memory is None:
            vm = psutil.virtual_memory()
            max_memory = int(vm.available * 0.75)
        elif isinstance(max_memory, str):
            max_memory = self.parse_memory_limit(max_memory)

        if max_threads is None:
            max_threads = os.cpu_count() or 1

        self.max_memory = max_memory
        self.max_threads = max_threads

        # Placeholder for detected cache sizes
        self._L3_CACHE_SIZE = self._detect_cache_size()  # Default L3 size
        self._CACHE_LINE_SIZE = 64  # Common default

    def _detect_cache_size(self, level=3) -> int:
        """Attempt to detect L3 cache size (basic placeholder)."""
        try:
            pass
        except Exception:
            pass
        vm = psutil.virtual_memory()
        estimated_cache = min(8 * 1024 * 1024, max(1024 * 1024, int(vm.total / (1024**3)) * 1024 * 1024))
        log.debug(f"Using fallback L3 cache size estimate: {estimated_cache / (1024*1024):.1f} MB")
        return estimated_cache

    def parse_memory_limit(self, memory_limit: str) -> int:
        """Parse memory limit string to bytes."""
        memory_limit = memory_limit.upper().strip()
        units = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}

        number = (
            float(memory_limit[:-1])
            if memory_limit[-1] in units
            else float(memory_limit)
        )
        unit = memory_limit[-1] if memory_limit[-1] in units else "B"

        return int(number * units[unit])

    def get_array_scale(self, num_elements: int) -> ArrayScale:
        """Determine array scale category based on number of elements."""
        if num_elements < 1_000_000:
            return ArrayScale.TINY
        elif num_elements < 100_000_000:
            return ArrayScale.SMALL
        elif num_elements < 1_000_000_000:
            return ArrayScale.MEDIUM
        elif num_elements < 10_000_000_000:
            return ArrayScale.LARGE
        return ArrayScale.HUGE

    def analyze_array(self, arr: np.ndarray) -> ArrayInfo:
        """Analyze array characteristics and determine optimal processing strategy."""
        num_elements = len(arr)
        element_size = arr.dtype.itemsize
        size_gb = arr.nbytes / (1024**3)

        scale = self.get_array_scale(num_elements)

        scale_factors = {
            ArrayScale.TINY: {"memory": 3.0, "threads": 1},
            ArrayScale.SMALL: {"memory": 2.5, "threads": min(2, self.max_threads)},
            ArrayScale.MEDIUM: {"memory": 2.0, "threads": min(4, self.max_threads)},
            ArrayScale.LARGE: {"memory": 1.5, "threads": min(8, self.max_threads)},
            ArrayScale.HUGE: {"memory": 1.2, "threads": self.max_threads},
        }

        factor = scale_factors[scale]
        memory_needed_gb = min(size_gb * factor["memory"], self.max_memory / (1024**3))

        return ArrayInfo(
            scale=scale,
            size_gb=size_gb,
            elements=num_elements,
            element_size=element_size,
            memory_needed_gb=memory_needed_gb,
            optimal_threads=factor["threads"],
        )

    def calculate_chunk_size(
        self,
        arr_info: ArrayInfo,
        num_concurrent_chunks: int,
        cache_optimization: bool = True,
        overhead_factor: float = 0.7,
        min_chunks_target: int = 10,
        max_chunk_elements: int = 100_000_000,
        min_chunk_elements: int = 100_000,
    ) -> ChunkingStrategy:
        """Calculate optimal chunk size and processing strategy based on available resources."""
        limiting_factor = "Initialization"

        if arr_info.elements == 0:
            return ChunkingStrategy(1, 1, 0, 1, 1, False)

        if arr_info.element_size <= 0:
            log.warning("Array element size is zero or negative, cannot calculate chunk size.")
            return ChunkingStrategy(1, arr_info.elements, 0, 1, 1, False)

        if num_concurrent_chunks <= 0:
            log.warning("num_concurrent_chunks must be positive, defaulting to 1.")
            num_concurrent_chunks = 1

        current_available_memory = self.available_memory
        usable_memory_for_chunks = int(current_available_memory * overhead_factor)

        if usable_memory_for_chunks <= 0:
            log.warning("No usable memory available for chunks based on overhead factor, using minimum chunk size.")
            chunk_elements = max(1, min(min_chunk_elements, arr_info.elements))
            num_chunks_actual = (arr_info.elements + chunk_elements - 1) // chunk_elements
            limiting_factor = "No Usable Memory"
            return ChunkingStrategy(chunk_elements, num_chunks_actual, 0, 1, 1, False)

        memory_based_elements = usable_memory_for_chunks // (arr_info.element_size * num_concurrent_chunks)
        memory_based_elements = max(1, memory_based_elements)
        chunk_elements = memory_based_elements
        limiting_factor = f"Available Memory ({current_available_memory/(1024**3):.2f} GB * {overhead_factor:.2f} / {num_concurrent_chunks} concurrent)"

        current_min_chunks = min_chunks_target
        if arr_info.scale == ArrayScale.HUGE:
            current_min_chunks = max(min_chunks_target, 100)
        elif arr_info.scale == ArrayScale.LARGE:
            current_min_chunks = max(min_chunks_target, 50)

        max_elements_for_total_size_target = arr_info.elements // current_min_chunks
        max_elements_reasonable_min = arr_info.elements // 1000
        max_elements_for_total_size = max(1, max_elements_for_total_size_target, max_elements_reasonable_min)

        if chunk_elements > max_elements_for_total_size:
            chunk_elements = max_elements_for_total_size
            limiting_factor = f"Min Chunks Target ({current_min_chunks})"

        if chunk_elements > max_chunk_elements:
            chunk_elements = max_chunk_elements
            limiting_factor = f"Max Chunk Elements ({max_chunk_elements:,})"

        effective_min_elements = min(min_chunk_elements, arr_info.elements)
        if chunk_elements < effective_min_elements:
            chunk_elements = effective_min_elements
            if limiting_factor != f"Min Chunk Elements ({effective_min_elements:,})":
                pass
            limiting_factor = f"Min Chunk Elements ({effective_min_elements:,})"

        if chunk_elements > arr_info.elements:
            chunk_elements = arr_info.elements
            limiting_factor = "Array Total Elements"

        original_chunk_elements = chunk_elements
        if cache_optimization:
            chunk_elements = self._optimize_for_cache(
                chunk_elements, arr_info.element_size, self._L3_CACHE_SIZE, self._CACHE_LINE_SIZE
            )
            chunk_elements = max(min(chunk_elements, max_chunk_elements), min(min_chunk_elements, arr_info.elements))
            chunk_elements = min(chunk_elements, arr_info.elements)
            chunk_elements = max(1, chunk_elements)
            if chunk_elements != original_chunk_elements and not limiting_factor.startswith("Cache"):
                limiting_factor += " + Cache Alignment"

        chunk_elements = max(1, chunk_elements)
        num_chunks_actual = (arr_info.elements + chunk_elements - 1) // chunk_elements
        memory_per_chunk_gb = (chunk_elements * arr_info.element_size) / (1024**3)
        estimated_total_chunk_mem_gb = memory_per_chunk_gb * num_concurrent_chunks

        total_threads = min(arr_info.optimal_threads, self.available_threads)
        threads_per_chunk = max(1, total_threads // num_chunks_actual) if num_chunks_actual > 0 else 1

        log.info(f"Calculated chunk size: {chunk_elements:,} elements. Limiting Factor: {limiting_factor}")
        log.info(f"  - Based on: Available Mem: {current_available_memory/(1024**3):.2f} GB, "
                 f"Concurrent Chunks: {num_concurrent_chunks}, Overhead Factor: {overhead_factor}")
        log.info(f"  - Estimated memory per chunk: {memory_per_chunk_gb:.3f} GB")
        log.info(f"  - Estimated total concurrent chunk memory: {estimated_total_chunk_mem_gb:.3f} GB")
        log.info(f"  - Array Size: {arr_info.elements:,} elements -> Num Chunks: {num_chunks_actual}")
        log.info(f"  - Threading: Total: {total_threads}, Per Chunk (estimated): {threads_per_chunk}")

        return ChunkingStrategy(
            chunk_size=chunk_elements,
            num_chunks=num_chunks_actual,
            memory_per_chunk_gb=memory_per_chunk_gb,
            threads_per_chunk=threads_per_chunk,
            total_threads=total_threads,
            cache_friendly=cache_optimization,
        )

    def _optimize_for_cache(
        self,
        current_chunk_elements: int,
        element_size: int,
        L3_CACHE_SIZE: int,
        CACHE_LINE_SIZE: int,
    ) -> int:
        """Refine chunk size to align better with cache lines."""
        if element_size <= 0:
            return current_chunk_elements

        elements_per_line = max(1, CACHE_LINE_SIZE // element_size)
        aligned_chunk_elements = (current_chunk_elements // elements_per_line) * elements_per_line
        aligned_chunk_elements = max(min(elements_per_line, current_chunk_elements), aligned_chunk_elements)

        final_chunk_elements = aligned_chunk_elements
        final_chunk_elements = max(1, final_chunk_elements)

        return final_chunk_elements

    def slice_chunk(self, arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Slice a chunk of an array with error handling."""
        try:
            return arr[indices]
        except Exception as e:
            log.error(f"Error in slice_chunk: {e}")
            raise

    def slice_arrays(
        self, arrays: Dict[str, np.ndarray], indices: np.ndarray, mmap_folder: str
    ) -> Dict[str, np.ndarray]:
        """Slice multiple arrays using parallel processing."""
        total_elements = len(indices)
        chunk_size = max(1024, total_elements // (self.max_threads * 2))
        num_chunks = (total_elements + chunk_size - 1) // chunk_size
        max_workers_per_array = max(1, self.max_threads // len(arrays))

        total_chunks = num_chunks * len(arrays)

        manager = Manager()
        progress_queue = manager.Queue()
        arrays_list = list(arrays.keys())
        processed_results = {}

        with tqdm(total=total_chunks, desc="Processing arrays", unit="chunks") as pbar:
            def update_progress():
                while True:
                    progress = progress_queue.get()
                    if progress is None:
                        break
                    pbar.update(progress)

            progress_thread = threading.Thread(target=update_progress)
            progress_thread.start()

            try:
                with ProcessPoolExecutor(
                    max_workers=min(len(arrays), self.max_threads)
                ) as executor:
                    futures = []

                    for name in arrays_list:
                        mmap_path = os.path.join(mmap_folder, f"{name}_slice.mmap")
                        future = executor.submit(
                            self.process_array_chunks,
                            name,
                            arrays[name],
                            indices,
                            chunk_size,
                            num_chunks,
                            mmap_path,
                            max_workers_per_array,
                            progress_queue,
                        )
                        futures.append((future, name))

                    for future, name in futures:
                        try:
                            result = future.result()
                            if result is not None:
                                processed_results[name] = result
                        except Exception as e:
                            log.error(f"Error processing array {name}: {e}")

            finally:
                progress_queue.put(None)
                progress_thread.join()

        return processed_results

    def process_array_chunks(
        self,
        name: str,
        arr: np.ndarray,
        indices: np.ndarray,
        chunk_size: int,
        num_chunks: int,
        mmap_path: str,
        max_workers: int,
        progress_queue: Queue,
    ) -> Optional[np.ndarray]:
        """Process chunks of a single array."""
        try:
            mmap_out = np.memmap(
                mmap_path, dtype=arr.dtype, mode="w+", shape=(len(indices),)
            )

            with ProcessPoolExecutor(max_workers=max_workers) as chunk_executor:
                futures = []

                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(indices))
                    chunk_indices = indices[start_idx:end_idx]

                    future = chunk_executor.submit(self.slice_chunk, arr, chunk_indices)
                    futures.append((future, start_idx, end_idx))

                for future, start_idx, end_idx in futures:
                    try:
                        chunk_data = future.result()
                        mmap_out[start_idx:end_idx] = chunk_data
                        progress_queue.put(1)
                    except Exception as e:
                        log.error(f"Error processing chunk {start_idx}-{end_idx}: {e}")

            return mmap_out

        except Exception as e:
            log.error(f"Error processing array {name}: {e}")
            return None

    @property
    def available_memory(self) -> int:
        """Get available memory in bytes."""
        with self.memory_lock:
            return self.max_memory - self.memory_allocated.value

    @property
    def available_threads(self) -> int:
        """Get number of available threads."""
        with self.thread_lock:
            return self.max_threads - self.active_threads.value
