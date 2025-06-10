from typing import Optional, Callable, Any, Dict, List
import os
import psutil
import logging
import time
from functools import wraps
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager

log = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Container for memory statistics at a single point in time"""

    timestamp: datetime
    rss: float  # Resident Set Size in MB
    vms: float  # Virtual Memory Size in MB
    shared: float  # Shared Memory in MB
    cpu_percent: float
    num_threads: int
    delta_mb: float  # Change in RSS since last measurement
    event: str


class MemoryTracker:
    """Singleton class to track memory usage across the application"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """Initialize the memory tracker"""
        self.process = psutil.Process(os.getpid())
        self.peak_rss = 0.0
        self.last_rss = 0.0
        self.history: Dict[str, List[MemoryStats]] = {}
        self.start_times: Dict[str, float] = {}

    def _get_memory_stats(self, event: str) -> MemoryStats:
        """Get current memory statistics"""
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
        """Start tracking memory for a named section"""
        self.history[name] = []
        self.start_times[name] = time.time()
        stats = self._get_memory_stats("start")
        self.history[name].append(stats)

    def record_event(self, name: str, event: str):
        """Record a memory usage event"""
        if name not in self.history:
            self.start_tracking(name)

        stats = self._get_memory_stats(event)
        self.history[name].append(stats)
        return stats

    def stop_tracking(self, name: str) -> List[MemoryStats]:
        """Stop tracking memory for a named section and return statistics"""
        stats = self._get_memory_stats("end")
        self.history[name].append(stats)

        execution_time = time.time() - self.start_times[name]
        initial_mem = self.history[name][0].rss
        final_mem = stats.rss
        mem_change = final_mem - initial_mem

        log.debug(
            f"Memory tracking for {name}:\n"
            f"  Execution time: {execution_time:.2f}s\n"
            f"  Initial memory: {initial_mem:.1f}MB\n"
            f"  Final memory: {final_mem:.1f}MB\n"
            f"  Memory change: {mem_change:+.1f}MB\n"
            f"  Peak memory: {self.peak_rss:.1f}MB"
        )

        return self.history[name]

    def get_system_memory_info(self) -> Dict[str, float]:
        """Get comprehensive system memory information in GB."""
        vm = psutil.virtual_memory()
        return {
            'total_gb': vm.total / (1024**3),
            'available_gb': vm.available / (1024**3),
            'used_gb': vm.used / (1024**3),
            'percent_used': vm.percent,
            'process_rss_gb': self.process.memory_info().rss / (1024**3),
            'process_vms_gb': self.process.memory_info().vms / (1024**3),
        }

    def check_memory_pressure(self, threshold_percent: float = 85.0) -> bool:
        """Check if system memory usage is above threshold."""
        vm = psutil.virtual_memory()
        return vm.percent > threshold_percent

    def recommend_memory_limit(self, safety_factor: float = 0.8) -> str:
        """Recommend memory limit as string with units."""
        vm = psutil.virtual_memory()
        recommended_gb = int((vm.available / (1024**3)) * safety_factor)
        return f"{max(1, recommended_gb)}GB"


def track_memory(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    detailed: bool = False,
    log_level: int = logging.INFO,
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

            try:
                # Run the function
                result = func(*args, **kwargs)

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


@contextmanager
def track_memory_block(name: str, detailed: bool = False):
    """
    Context manager for tracking memory usage in a block of code.

    Usage:
        with track_memory_block("data_loading"):
            data = load_large_dataset()
            process_data(data)
    """
    tracker = MemoryTracker()
    tracker.start_tracking(name)

    try:
        yield tracker
    finally:
        tracker.stop_tracking(name)
