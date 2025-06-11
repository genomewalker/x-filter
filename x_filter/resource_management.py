from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Union, Dict, Tuple, List, Any, Set
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Queue
from tqdm import tqdm
from threading import Lock
import os
import psutil
import time
import numpy as np
import logging
from x_filter.logging_setup import get_logger  # Fixed import path

log = get_logger()


class ArrayScale(Enum):
    """Enumeration for array scale categories."""
    SMALL = "small"      # < 1GB
    MEDIUM = "medium"    # 1GB - 10GB  
    LARGE = "large"      # 10GB - 100GB
    HUGE = "huge"        # > 100GB


@dataclass
class ArrayInfo:
    """Information about a memory-mapped array."""
    name: str
    path: str
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int
    scale: ArrayScale
    creation_time: float
    last_access: float
    temp: bool = True


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory: int
    available_memory: int
    used_memory: int
    cached_memory: int
    swap_total: int
    swap_used: int
    process_memory: int


@dataclass
class ResourceLimits:
    """Resource limits and thresholds."""
    max_memory: int
    max_threads: int
    max_arrays: int
    cleanup_threshold: float
    warning_threshold: float


@dataclass
class ChunkingStrategy:
    """Strategy for chunking large arrays."""
    chunk_size: int
    num_chunks: int
    cache_friendly: bool
    memory_efficient: bool
    suggested_threads: int
    total_threads: int = 1  # Add missing attribute with default value


class ThreadPoolManager:
    """Enhanced thread pool manager with better resource control."""
    
    def __init__(self, max_workers: int = None, thread_name_prefix: str = "ResourceWorker"):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.thread_name_prefix = thread_name_prefix
        self._executor = None
        self._lock = Lock()
        
    def __enter__(self):
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix=self.thread_name_prefix
                )
        return self._executor
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None


class MemoryManager:
    """Manages memory allocation and deallocation."""

    def __init__(self, resource_manager, bytes_needed: int):
        self.resource_manager = resource_manager
        self.bytes_needed = bytes_needed
        self.acquired = False

    def __enter__(self):
        if self.resource_manager.reserve_memory(self.bytes_needed):
            self.acquired = True
        else:
            raise RuntimeError(f"Cannot allocate {self.bytes_needed} bytes")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            try:
                self.resource_manager.release_memory(self.bytes_needed)
                self.acquired = False
            except Exception:
                # Ignore errors during cleanup to prevent segfaults
                pass


class ResourceManager:
    """
    Centralized resource management for memory allocation, threading, and temporary files.
    Handles billion-element datasets efficiently with adaptive memory management.
    """
    
    def __init__(
        self,
        max_memory: Optional[Union[str, int]] = None,
        max_threads: Optional[int] = None,
        temp_base_dir: Optional[str] = None,
        memory_safety_factor: float = 0.9,
        enable_monitoring: bool = True,
        min_chunk_size: int = 1_000_000,
        max_chunk_size: Optional[int] = None,
        mmap_folder: Optional[str] = None  # Add this parameter
    ):
        """
        Initialize ResourceManager with system resource detection and limits.
        
        Args:
            max_memory: Maximum memory to use (string like "250G" or bytes as int)
            max_threads: Maximum threads to use
            temp_base_dir: Base directory for temporary files
            memory_safety_factor: Safety factor for memory allocation (0.0-1.0)
            enable_monitoring: Whether to enable resource monitoring
            mmap_folder: Directory for memory-mapped files (alias for temp_base_dir)
        """
        # Initialize logger first
        self.log = get_logger()
        
        self.memory_safety_factor = memory_safety_factor
        self.enable_monitoring = enable_monitoring
        
        # Handle mmap_folder parameter - it's an alias for temp_base_dir
        if mmap_folder is not None:
            self.temp_base_dir = mmap_folder
        elif temp_base_dir is not None:
            self.temp_base_dir = temp_base_dir
        else:
            self.temp_base_dir = "/tmp"
        
        # Parse memory limit
        if max_memory is not None:
            if isinstance(max_memory, str):
                self.max_memory_bytes = self.parse_memory_limit(max_memory)
            else:
                self.max_memory_bytes = int(max_memory)
        else:
            # Auto-detect system memory
            self.max_memory_bytes = psutil.virtual_memory().available
        
        # Apply safety factor
        self.usable_memory_bytes = int(self.max_memory_bytes * memory_safety_factor)
        
        # Set thread limits
        if max_threads is not None:
            self.max_threads = max_threads
        else:
            self.max_threads = min(os.cpu_count() or 1, 16)  # Reasonable default
        
        # Initialize tracking variables
        self.allocated_memory = 0
        self.allocated_threads = 0
        self.arrays = {}
        
        log.info(f"ResourceManager initialized:")
        log.info(f"  - Max memory: {self.max_memory_bytes // (1024**3)}GB")
        log.info(f"  - Usable memory: {self.usable_memory_bytes // (1024**3)}GB")
        log.info(f"  - Max threads: {self.max_threads}")

    @property
    def max_memory(self) -> int:
        return self.max_memory_bytes

    @property
    def mmap_folder(self) -> Optional[str]:
        return self.temp_base_dir

    @mmap_folder.setter
    def mmap_folder(self, value: Optional[str]):
        self.temp_base_dir = value

    def parse_memory_limit(self, memory_limit: str) -> int:
        """Parse memory limit string like '10GB', '512MB', etc."""
        memory_limit = memory_limit.upper().strip()

        if memory_limit.endswith("GB") or memory_limit.endswith("G"):
            return int(float(memory_limit[:-1]) * 1024**3)
        elif memory_limit.endswith("MB") or memory_limit.endswith("M"):
            return int(float(memory_limit[:-1]) * 1024**2)
        elif memory_limit.endswith("KB") or memory_limit.endswith("K"):
            return int(float(memory_limit[:-1]) * 1024)
        elif memory_limit.endswith("B"):
            return int(float(memory_limit[:-1]))
        else:
            return int(float(memory_limit))

    def get_array_scale(self, num_elements: int) -> ArrayScale:
        """Determine array scale based on number of elements."""
        if num_elements < 1_000_000:
            return ArrayScale.SMALL
        elif num_elements < 100_000_000:
            return ArrayScale.MEDIUM
        elif num_elements < 1_000_000_000:
            return ArrayScale.LARGE
        elif num_elements < 10_000_000_000:
            return ArrayScale.HUGE
        else:
            return ArrayScale.HUGE

    def analyze_array(self, arr: np.ndarray) -> ArrayInfo:
        """Analyze array characteristics."""
        return ArrayInfo(
            name="",
            path="",
            shape=arr.shape,
            dtype=str(arr.dtype),
            size_bytes=arr.nbytes,
            scale=self.get_array_scale(arr.size),
            creation_time=time.time(),
            last_access=time.time()
        )

    def calculate_optimal_chunk_size(
        self,
        total_elements: int,
        element_size: int,
        operation_overhead: float = 2.0,
        min_chunk_size: int = 1_000_000,
        max_chunk_size: Optional[int] = None,
    ) -> int:
        """Calculate optimal chunk size based on available memory and dataset size."""
        # Calculate memory needed per element (including overhead)
        memory_per_element = element_size * operation_overhead
        
        # Calculate maximum chunk size based on available memory
        max_memory_chunk_size = int(self.usable_memory_bytes * 0.8 / memory_per_element)
        
        # Apply constraints
        chunk_size = max(min_chunk_size, min(max_memory_chunk_size, total_elements))
        
        if max_chunk_size is not None:
            chunk_size = min(chunk_size, max_chunk_size)
        
        return chunk_size

    def calculate_chunk_size(
        self,
        arr_info: ArrayInfo,
        cache_optimization: bool = True,
        min_chunk_size: int = 1_000_000,
        max_chunk_size: Optional[int] = None
    ) -> ChunkingStrategy:
        """Calculate optimal chunking strategy for an array."""
        # Simplified implementation
        total_elements = np.prod(arr_info.shape)
        element_size = np.dtype(arr_info.dtype).itemsize
        
        chunk_size = self.calculate_optimal_chunk_size(
            total_elements, element_size, 
            min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size
        )
        
        num_chunks = (total_elements + chunk_size - 1) // chunk_size
        suggested_threads = min(self.max_threads, num_chunks)
        
        return ChunkingStrategy(
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            cache_friendly=cache_optimization,
            memory_efficient=True,
            suggested_threads=suggested_threads,
            total_threads=suggested_threads  # Set total_threads to match suggested_threads
        )

    def slice_chunk(self, arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Slice array using indices."""
        return arr[indices]

    def reserve_threads(self, num_threads: int) -> bool:
        """Reserve threads for processing."""
        if self.allocated_threads + num_threads <= self.max_threads:
            self.allocated_threads += num_threads
            return True
        return False

    def release_threads(self, num_threads: int):
        """Release reserved threads."""
        self.allocated_threads = max(0, self.allocated_threads - num_threads)

    def reserve_memory(self, bytes_needed: int) -> bool:
        """Reserve memory for processing."""
        if self.allocated_memory + bytes_needed <= self.usable_memory_bytes:
            self.allocated_memory += bytes_needed
            return True
        return False

    def release_memory(self, bytes_used: int):
        """Release reserved memory."""
        self.allocated_memory = max(0, self.allocated_memory - bytes_used)

    def create_array(
        self,
        name: str,
        shape: tuple,
        dtype,
        folder: str = None,
        mode: str = "r+",
        temp: bool = False,
        overwrite: bool = True,
    ) -> Optional[np.ndarray]:
        """Create a memory-mapped array."""
        try:
            if folder is None:
                folder = self.temp_base_dir
            
            # Ensure directory exists
            os.makedirs(folder, mode=0o755, exist_ok=True)
            
            # Create file path
            filepath = os.path.join(folder, f"{name}.mmap")
            
            # Handle existing files
            if os.path.exists(filepath) and not overwrite:
                log.warning(f"Array file {filepath} already exists")
                return None
            
            # Create memory-mapped array
            arr = np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)
            
            # Track the array
            self.arrays[name] = {
                'array': arr,
                'path': filepath,
                'temp': temp
            }
            
            return arr
            
        except Exception as e:
            log.error(f"Failed to create array {name}: {e}")
            return None

    def delete_array(self, name: str) -> bool:
        """Delete a memory-mapped array."""
        if name in self.arrays:
            try:
                arr_info = self.arrays[name]
                del arr_info['array']  # Release reference
                
                if os.path.exists(arr_info['path']):
                    os.remove(arr_info['path'])
                
                del self.arrays[name]
                return True
                
            except Exception as e:
                log.error(f"Failed to delete array {name}: {e}")
                return False
        return False

    def cleanup(self):
        """Clean up all temporary arrays and resources."""
        for name in list(self.arrays.keys()):
            if self.arrays[name].get('temp', False):
                self.delete_array(name)
        
        self.allocated_memory = 0
        self.allocated_threads = 0
