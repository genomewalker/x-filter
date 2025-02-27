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
        cache_optimization: bool = True,
        max_chunks: int = 100,
    ) -> ChunkingStrategy:
        """Calculate optimal chunk size and processing strategy."""
        L3_CACHE_SIZE = 8 * 1024 * 1024  # 8MB typical L3 cache
        CACHE_LINE_SIZE = 64  # 64 bytes typical cache line

        available_memory_gb = min(
            self.available_memory / (1024**3), arr_info.memory_needed_gb
        )

        scale_strategies = {
            ArrayScale.TINY: {
                "chunks": 1,
                "memory_fraction": 1.0,
                "thread_fraction": 1.0,
            },
            ArrayScale.SMALL: {
                "chunks": min(4, max_chunks),
                "memory_fraction": 0.8,
                "thread_fraction": 0.5,
            },
            ArrayScale.MEDIUM: {
                "chunks": min(8, max_chunks),
                "memory_fraction": 0.6,
                "thread_fraction": 0.25,
            },
            ArrayScale.LARGE: {
                "chunks": min(16, max_chunks),
                "memory_fraction": 0.4,
                "thread_fraction": 0.125,
            },
            ArrayScale.HUGE: {
                "chunks": min(32, max_chunks),
                "memory_fraction": 0.2,
                "thread_fraction": 0.0625,
            },
        }

        strategy = scale_strategies[arr_info.scale]
        memory_per_chunk_gb = (
            available_memory_gb * strategy["memory_fraction"]
        ) / strategy["chunks"]
        total_threads = min(arr_info.optimal_threads, self.available_threads)
        threads_per_chunk = max(1, int(total_threads * strategy["thread_fraction"]))

        return ChunkingStrategy(
            chunk_size=self._calculate_chunk_size(
                arr_info,
                memory_per_chunk_gb,
                cache_optimization,
                L3_CACHE_SIZE,
                CACHE_LINE_SIZE,
                threads_per_chunk,
            ),
            num_chunks=strategy["chunks"],
            memory_per_chunk_gb=memory_per_chunk_gb,
            threads_per_chunk=threads_per_chunk,
            total_threads=total_threads,
            cache_friendly=cache_optimization,
        )

    def _calculate_chunk_size(
        self,
        arr_info: ArrayInfo,
        memory_per_chunk_gb: float,
        cache_optimization: bool,
        L3_CACHE_SIZE: int,
        CACHE_LINE_SIZE: int,
        threads_per_chunk: int,
    ) -> int:
        """Calculate optimal chunk size based on memory and cache constraints."""
        bytes_per_chunk = int(memory_per_chunk_gb * 1024**3)
        elements_per_chunk = bytes_per_chunk // arr_info.element_size

        if not cache_optimization:
            return max(1024, min(elements_per_chunk, arr_info.elements))

        # Calculate cache-based chunk size
        cache_elements = L3_CACHE_SIZE // arr_info.element_size
        elements_per_line = max(1, CACHE_LINE_SIZE // arr_info.element_size)
        cache_based_chunk_size = min(
            elements_per_chunk, cache_elements * threads_per_chunk
        )
        cache_based_chunk_size = (
            cache_based_chunk_size // elements_per_line
        ) * elements_per_line

        return min(cache_based_chunk_size, elements_per_chunk)

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

        # Calculate total chunks
        total_chunks = num_chunks * len(arrays)

        # Create progress queue
        manager = Manager()
        progress_queue = manager.Queue()
        arrays_list = list(arrays.keys())
        processed_results = {}

        with tqdm(total=total_chunks, desc="Processing arrays", unit="chunks") as pbar:
            # Progress tracking thread
            def update_progress():
                while True:
                    progress = progress_queue.get()
                    if progress is None:
                        break
                    pbar.update(progress)

            progress_thread = threading.Thread(target=update_progress)
            progress_thread.start()

            try:
                # Process arrays in parallel
                with ProcessPoolExecutor(
                    max_workers=min(len(arrays), self.max_threads)
                ) as executor:
                    futures = []

                    # Submit all arrays
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

                    # Process results
                    for future, name in futures:
                        try:
                            result = future.result()
                            if result is not None:
                                processed_results[name] = result
                        except Exception as e:
                            log.error(f"Error processing array {name}: {e}")

            finally:
                # Clean up progress tracking
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
            # Create memory map for output
            mmap_out = np.memmap(
                mmap_path, dtype=arr.dtype, mode="w+", shape=(len(indices),)
            )

            # Process chunks
            with ProcessPoolExecutor(max_workers=max_workers) as chunk_executor:
                futures = []

                # Submit chunks
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(indices))
                    chunk_indices = indices[start_idx:end_idx]

                    future = chunk_executor.submit(self.slice_chunk, arr, chunk_indices)
                    futures.append((future, start_idx, end_idx))

                # Process results
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
