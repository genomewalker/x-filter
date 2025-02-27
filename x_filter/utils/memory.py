# x_filter/utils/memory.py

from typing import Optional, Callable, Any, Dict, List
import os
import psutil
import time
import gc
from functools import wraps
from dataclasses import dataclass
from datetime import datetime
import threading
import multiprocessing

from x_filter.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class MemoryStats:
    """Container for memory statistics at a single point in time."""

    timestamp: datetime
    rss: float  # Resident Set Size in MB
    vms: float  # Virtual Memory Size in MB
    shared: float  # Shared Memory in MB
    cpu_percent: float
    num_threads: int
    delta_mb: float  # Change in RSS since last measurement
    event: str


class MemoryTracker:
    """Singleton class to track memory usage across the application."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.initialize()
            return cls._instance

    def initialize(self):
        """Initialize the memory tracker."""
        self.process = psutil.Process(os.getpid())
        self.peak_rss = 0.0
        self.last_rss = 0.0
        self.history = {}
        self.start_times = {}

    def _get_memory_stats(self, event: str) -> MemoryStats:
        """Get current memory statistics."""
        mem = self.process.memory_info()
        rss_mb = mem.rss / (1024 * 1024)
        vms_mb = mem.vms / (1024 * 1024)
        shared_mb = getattr(mem, "shared", 0) / (1024 * 1024)

        delta_mb = rss_mb - self.last_rss
        self.last_rss = rss_mb
        self.peak_rss = max(self.peak_rss, rss_mb)

        return MemoryStats(
            timestamp=datetime.now(),
            rss=rss_mb,
            vms=vms_mb,
            shared=shared_mb,
            cpu_percent=self.process.cpu_percent(),
            num_threads=self.process.num_threads(),
            delta_mb=delta_mb,
            event=event,
        )

    def start_tracking(self, name: str):
        """Start tracking memory for a named section."""
        with self._lock:
            self.history[name] = []
            self.start_times[name] = time.time()
            stats = self._get_memory_stats("start")
            self.history[name].append(stats)
            return stats

    def record_event(self, name: str, event: str):
        """Record a memory usage event."""
        with self._lock:
            if name not in self.history:
                self.start_tracking(name)

            stats = self._get_memory_stats(event)
            self.history[name].append(stats)
            return stats

    def stop_tracking(self, name: str) -> List[MemoryStats]:
        """Stop tracking memory for a named section and return statistics."""
        with self._lock:
            stats = self._get_memory_stats("end")
            self.history[name].append(stats)

            execution_time = time.time() - self.start_times[name]
            initial_mem = self.history[name][0].rss
            final_mem = stats.rss
            mem_change = final_mem - initial_mem

            log.info(
                f"Memory tracking for {name}:\n"
                f"  Execution time: {execution_time:.2f}s\n"
                f"  Initial memory: {initial_mem:.1f}MB\n"
                f"  Final memory: {final_mem:.1f}MB\n"
                f"  Memory change: {mem_change:+.1f}MB\n"
                f"  Peak memory: {self.peak_rss:.1f}MB"
            )

            return self.history[name]

    def get_current_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        return self.process.memory_info().rss

    def get_available_memory(self) -> float:
        """Get available system memory in bytes."""
        return psutil.virtual_memory().available

    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        return self.peak_rss


def track_memory(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    detailed: bool = False,
    log_level: str = "INFO",
) -> Callable:
    """
    Decorator to track memory usage of a function.

    Args:
        func: The function to decorate
        name: Optional custom name for tracking (defaults to function name)
        detailed: If True, logs memory stats for every garbage collection
        log_level: Logging level for memory statistics

    Usage:
        # Basic usage
        @track_memory
        def my_function():
            pass

        # With custom name and detailed tracking
        @track_memory(name="data_processing", detailed=True)
        def process_data():
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            tracking_name = name or func.__name__
            tracker = MemoryTracker()

            # Start tracking
            tracker.start_tracking(tracking_name)

            # Run garbage collection before execution for clean measurement
            gc.collect()

            try:
                # Run the function
                result = func(*args, **kwargs)

                # Run garbage collection after execution
                gc.collect()

                # Stop tracking and get stats
                tracker.stop_tracking(tracking_name)

                return result

            except Exception as e:
                # Record error state
                tracker.record_event(tracking_name, f"error: {str(e)}")
                raise

        return wrapper

    # Handle both @track_memory and @track_memory() syntax
    if func is None:
        return decorator
    return decorator(func)


class MemoryMonitor:
    """
    Class for monitoring memory usage in a separate thread.

    Usage:
        with MemoryMonitor("my_operation", interval=1.0):
            # Do memory-intensive operation
    """

    def __init__(self, name: str, interval: float = 0.5):
        """
        Initialize memory monitor.

        Args:
            name: Name for the monitoring session
            interval: Monitoring interval in seconds
        """
        self.name = name
        self.interval = interval
        self.tracker = MemoryTracker()
        self.running = False
        self.thread = None

    def _monitor_memory(self):
        """Background thread function to monitor memory."""
        event_count = 0

        while self.running:
            try:
                self.tracker.record_event(self.name, f"monitor_{event_count}")
                event_count += 1
                time.sleep(self.interval)
            except Exception as e:
                log.warning(f"Error in memory monitor: {e}")

    def __enter__(self):
        """Start memory monitoring."""
        self.tracker.start_tracking(self.name)
        self.running = True
        self.thread = threading.Thread(
            target=self._monitor_memory, daemon=True, name=f"memory_monitor_{self.name}"
        )
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop memory monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.tracker.stop_tracking(self.name)


def optimize_memory_usage():
    """Try to optimize memory usage of the current process."""
    # Force garbage collection
    gc.collect()

    # Dump unreachable objects info if in debug mode
    if log.isEnabledFor(logging.DEBUG):
        gc.set_debug(gc.DEBUG_LEAK)
        gc.collect()
        gc.set_debug(0)

    # Reduce memory fragmentation (Linux)
    if hasattr(gc, "malloc_trim"):
        gc.malloc_trim()

    # Report memory usage
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    log.debug(
        f"Memory usage after optimization: "
        f"{mem_info.rss / (1024**2):.1f}MB resident, "
        f"{mem_info.vms / (1024**2):.1f}MB virtual"
    )


def get_memory_info() -> Dict[str, float]:
    """Get memory usage information."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    system_mem = psutil.virtual_memory()

    return {
        "rss_mb": mem_info.rss / (1024**2),
        "vms_mb": mem_info.vms / (1024**2),
        "available_mb": system_mem.available / (1024**2),
        "used_percent": system_mem.percent,
        "cpu_percent": process.cpu_percent(),
        "num_threads": process.num_threads(),
    }


def format_size(size_bytes: float) -> str:
    """Format a size in bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes:.2f}B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f}KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f}MB"
    else:
        return f"{size_bytes / (1024**3):.2f}GB"
