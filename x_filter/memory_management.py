import os
import gc
import time
import numpy as np
import threading
from typing import Dict, Optional, Any, List, Tuple
from x_filter.logging_setup import get_logger
import psutil

log = get_logger()


class MemmapArrayManager:
    """Manages memory-mapped arrays with robust creation and cleanup capabilities."""

    def __init__(self, mmap_folder: str, name: str = "default", max_memory: Optional[str] = None):
        """Initialize the memory-mapped array manager.

        Args:
            mmap_folder: Base directory for storing memory-mapped files
            name: Name for this array manager instance
            max_memory: Maximum memory limit as string (e.g., "32GB")
        """
        self.mmap_folder = mmap_folder
        self.name = name
        self.arrays: Dict[str, np.memmap] = {}
        self.array_paths: Dict[str, str] = {}
        self.locks: Dict[str, threading.Lock] = {}
        
        # Parse memory limit
        if max_memory:
            self.max_memory_bytes = self._parse_memory_limit(max_memory)
        else:
            # Default to 80% of available memory
            vm = psutil.virtual_memory()
            self.max_memory_bytes = int(vm.available * 0.8)

        # Create folder if it doesn't exist
        os.makedirs(mmap_folder, exist_ok=True)

        log.debug(f"Initialized ArrayManager '{name}' in {mmap_folder} with {self.max_memory_bytes / (1024**3):.1f}GB limit")

    def _parse_memory_limit(self, memory_limit: str) -> int:
        """Parse memory limit string to bytes."""
        memory_limit = memory_limit.upper().strip()
        
        # Support both single and multi-letter units
        unit_multipliers = {
            'B': 1,
            'K': 1024, 'KB': 1024, 'KIB': 1024,
            'M': 1024**2, 'MB': 1024**2, 'MIB': 1024**2,
            'G': 1024**3, 'GB': 1024**3, 'GIB': 1024**3,
            'T': 1024**4, 'TB': 1024**4, 'TIB': 1024**4,
        }
        
        # Find unit suffix - check longer units first to avoid partial matches
        for unit_name in sorted(unit_multipliers.keys(), key=len, reverse=True):
            if memory_limit.endswith(unit_name):
                number_str = memory_limit[:-len(unit_name)].strip()
                try:
                    number = float(number_str)
                    return int(number * unit_multipliers[unit_name])
                except ValueError:
                    break
        
        # Fallback
        vm = psutil.virtual_memory()
        return int(vm.available * 0.8)

    def estimate_memory_usage(self) -> int:
        """Estimate current memory usage of all managed arrays in bytes."""
        total_bytes = 0
        for array in self.arrays.values():
            if hasattr(array, 'nbytes'):
                total_bytes += array.nbytes
        return total_bytes

    def check_memory_availability(self, needed_bytes: int) -> bool:
        """Check if enough memory is available for allocation."""
        current_usage = self.estimate_memory_usage()
        return (current_usage + needed_bytes) <= self.max_memory_bytes

    def create_array(
        self, name: str, shape: Tuple, dtype: np.dtype, filename: Optional[str] = None
    ) -> np.memmap:
        """Create a new memory-mapped array.

        Args:
            name: Logical name of the array
            shape: Shape of the array
            dtype: Data type of the array
            filename: Optional specific filename (default: name.dat)

        Returns:
            Memory-mapped numpy array
        """
        if name in self.arrays:
            self.remove_array(name)

        filepath = filename or os.path.join(self.mmap_folder, f"{name}.dat")
        self.array_paths[name] = filepath
        self.locks[name] = threading.Lock()

        try:
            array = np.memmap(filepath, dtype=dtype, mode="w+", shape=shape)
            self.arrays[name] = array
            log.debug(f"Created array '{name}' with shape {shape} at {filepath}")
            return array
        except Exception as e:
            log.error(f"Error creating array '{name}': {e}")
            raise

    def register_existing(self, name: str, array: np.memmap) -> None:
        """Register an existing memory-mapped array for management.

        Args:
            name: Logical name for the array
            array: Existing numpy memmap array
        """
        if not isinstance(array, np.memmap):
            raise TypeError("Only numpy.memmap objects can be registered")

        self.arrays[name] = array
        self.array_paths[name] = array.filename
        self.locks[name] = threading.Lock()
        log.debug(f"Registered existing array '{name}' from {array.filename}")

    def get(self, name: str) -> Optional[np.memmap]:
        """Get a managed array by name.

        Args:
            name: Name of the array

        Returns:
            The memory-mapped array or None if not found
        """
        return self.arrays.get(name)

    def remove_array(self, name: str) -> None:
        """Remove a specific array and its file.

        Args:
            name: Name of the array to remove
        """
        if name in self.arrays:
            with self.locks[name]:
                try:
                    # Delete the array object first
                    del self.arrays[name]

                    # Force garbage collection
                    gc.collect()

                    # Remove file if it exists
                    filepath = self.array_paths[name]
                    if os.path.exists(filepath):
                        os.unlink(filepath)

                    # Clean up internal state
                    del self.array_paths[name]
                    del self.locks[name]

                    log.debug(f"Removed array '{name}'")
                except Exception as e:
                    log.warning(f"Error removing array '{name}': {e}")

    def cleanup(self, force: bool = False) -> None:
        """Clean up all managed arrays and their files.

        Args:
            force: Whether to use more aggressive cleanup methods
        """
        log.info(f"Cleaning up {len(self.arrays)} arrays in '{self.name}'")

        # First delete all array objects
        array_names = list(self.arrays.keys())
        for name in array_names:
            try:
                del self.arrays[name]
            except Exception as e:
                log.warning(f"Error deleting array object '{name}': {e}")

        # Clear the arrays dictionary
        self.arrays.clear()

        # Multiple garbage collections
        gc.collect()
        gc.collect()

        # Give OS time to release file handles
        time.sleep(0.5)

        # Now delete the files with multiple retries
        remaining_files = list(self.array_paths.values())
        max_retries = 3

        for attempt in range(max_retries):
            if not remaining_files:
                break

            still_remaining = []
            for filepath in remaining_files:
                try:
                    if os.path.exists(filepath):
                        try:
                            # Test if file is accessible
                            with open(filepath, "a"):
                                pass
                            # Delete the file
                            os.unlink(filepath)
                        except PermissionError:
                            # File might be locked, try changing permissions
                            os.chmod(filepath, 0o666)
                            os.unlink(filepath)
                except Exception as e:
                    log.warning(
                        f"Failed to delete {filepath} (attempt {attempt+1}): {e}"
                    )
                    still_remaining.append(filepath)

            # Update remaining files for next iteration
            remaining_files = still_remaining

            if still_remaining and attempt < max_retries - 1:
                # Wait longer between retries
                time.sleep(1)
                gc.collect()

        # Last resort - use system command if force=True
        if force and remaining_files:
            log.warning(f"Using force deletion for {len(remaining_files)} files")
            for filepath in remaining_files:
                try:
                    # Last attempt with system command
                    os.system(f"rm -f {filepath}")
                except Exception as e:
                    log.warning(f"Force deletion failed for {filepath}: {e}")

        # Clean up empty subdirectories
        self._cleanup_empty_dirs()

        # Reset internal state
        self.array_paths.clear()
        self.locks.clear()

    def _cleanup_empty_dirs(self) -> None:
        """Clean up empty directories within the mmap folder."""
        # Walk directories bottom-up
        for root, dirs, files in os.walk(self.mmap_folder, topdown=False):
            if root != self.mmap_folder:  # Don't remove the base folder
                try:
                    os.rmdir(root)
                except OSError:
                    # Directory not empty, just continue
                    pass

    def __del__(self) -> None:
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.cleanup(force=False)
        except:
            # Ignore errors in destructor
            pass

    def __enter__(self) -> "MemmapArrayManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.cleanup(force=True)
