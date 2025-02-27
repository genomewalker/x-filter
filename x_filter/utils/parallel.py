# x_filter/utils/parallel.py

import os
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import (
    List,
    Dict,
    Any,
    Callable,
    TypeVar,
    Generic,
    Iterable,
    Iterator,
    Optional,
)
from dataclasses import dataclass
from enum import Enum

from x_filter.utils.logging import get_logger

log = get_logger(__name__)

# Type variables for generic typing
T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type


class ExecutionMode(Enum):
    """Mode for parallel execution."""

    THREAD = "thread"
    PROCESS = "process"


@dataclass
class ParallelTask(Generic[T, R]):
    """Represent a parallel task with input and result."""

    task_id: int
    input: T
    result: Optional[R] = None
    error: Optional[Exception] = None
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        """Get task duration in seconds."""
        if self.start_time > 0 and self.end_time > 0:
            return self.end_time - self.start_time
        return 0.0

    @property
    def completed(self) -> bool:
        """Check if task is completed."""
        return self.end_time > 0


class ParallelExecutor:
    """
    Generic parallel executor that handles threads or processes efficiently.

    This class provides a unified interface for parallel execution with:
    - Automatic task distribution
    - Progress reporting
    - Error handling
    - Resource management
    """

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.THREAD,
        max_workers: Optional[int] = None,
        task_timeout: Optional[float] = None,
        name: str = "ParallelExecutor",
    ):
        """
        Initialize the parallel executor.

        Args:
            mode: Execution mode (THREAD or PROCESS)
            max_workers: Maximum number of workers (default: auto)
            task_timeout: Timeout for each task in seconds
            name: Name for this executor
        """
        self.mode = mode
        self.name = name
        self.task_timeout = task_timeout

        # Determine worker count
        if max_workers is None:
            if mode == ExecutionMode.PROCESS:
                max_workers = max(1, os.cpu_count() or 4)
            else:
                max_workers = max(4, (os.cpu_count() or 4) * 2)

        self.max_workers = max_workers
        self._executor = None
        self._tasks: Dict[int, ParallelTask] = {}
        self._task_counter = 0
        self._lock = threading.RLock()

    def __enter__(self):
        """Initialize executor when entering context."""
        if self.mode == ExecutionMode.PROCESS:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix=f"{self.name}_worker"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def submit(self, func: Callable[[T], R], input_data: T) -> ParallelTask[T, R]:
        """
        Submit a task for execution.

        Args:
            func: Function to execute
            input_data: Input data for the function

        Returns:
            Task object representing the submitted task
        """
        if not self._executor:
            raise RuntimeError("Executor not initialized - use with statement")

        with self._lock:
            task_id = self._task_counter
            self._task_counter += 1

            task = ParallelTask(task_id=task_id, input=input_data)
            self._tasks[task_id] = task

        # Define wrapper to update task with result
        def task_wrapper(task_input, task_id):
            start_time = time.time()
            try:
                result = func(task_input)
                return result, None, task_id, start_time, time.time()
            except Exception as e:
                return None, e, task_id, start_time, time.time()

        # Submit to executor
        future = self._executor.submit(task_wrapper, input_data, task_id)

        # Add callback to update task
        def update_task(fut):
            try:
                result, error, tid, start, end = fut.result(timeout=self.task_timeout)
                with self._lock:
                    if tid in self._tasks:
                        self._tasks[tid].result = result
                        self._tasks[tid].error = error
                        self._tasks[tid].start_time = start
                        self._tasks[tid].end_time = end
            except Exception as e:
                with self._lock:
                    if task_id in self._tasks:
                        self._tasks[task_id].error = e
                        self._tasks[task_id].end_time = time.time()

        future.add_done_callback(update_task)
        return task

    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        show_progress: bool = True,
        progress_desc: Optional[str] = None,
    ) -> List[ParallelTask[T, R]]:
        """
        Execute a function on multiple items in parallel.

        Args:
            func: Function to execute
            items: Items to process
            show_progress: Whether to show progress bar
            progress_desc: Description for progress bar

        Returns:
            List of task objects
        """
        tasks = []

        # Submit all tasks
        for item in items:
            task = self.submit(func, item)
            tasks.append(task)

        # Wait for completion with optional progress bar
        if show_progress:
            try:
                from tqdm import tqdm

                if progress_desc is None:
                    progress_desc = f"{self.name} progress"

                with tqdm(total=len(tasks), desc=progress_desc) as pbar:
                    completed = set()

                    while len(completed) < len(tasks):
                        pending = len(tasks) - len(completed)
                        new_completed = 0

                        for i, task in enumerate(tasks):
                            if i not in completed and task.completed:
                                completed.add(i)
                                new_completed += 1

                        if new_completed > 0:
                            pbar.update(new_completed)

                        if pending > 0:
                            time.sleep(0.1)

            except ImportError:
                # No tqdm available, use simple logging
                log.info(f"Processing {len(tasks)} tasks...")

                while True:
                    completed = sum(1 for task in tasks if task.completed)
                    if completed == len(tasks):
                        break

                    log.info(f"Completed {completed}/{len(tasks)} tasks")
                    time.sleep(1.0)

                log.info(f"All {len(tasks)} tasks completed")

        return tasks

    def get_results(self, tasks: List[ParallelTask[T, R]]) -> List[R]:
        """
        Get results from a list of tasks, raising first error if any.

        Args:
            tasks: List of tasks

        Returns:
            List of results

        Raises:
            Exception: If any task failed
        """
        # Check for errors
        for task in tasks:
            if task.error:
                raise task.error

        # Return results
        return [task.result for task in tasks]

    def get_task_stats(self) -> Dict[str, Any]:
        """Get statistics about task execution."""
        with self._lock:
            total_tasks = len(self._tasks)
            completed_tasks = sum(1 for task in self._tasks.values() if task.completed)
            failed_tasks = sum(
                1 for task in self._tasks.values() if task.error is not None
            )

            durations = [
                task.duration for task in self._tasks.values() if task.completed
            ]

            avg_duration = sum(durations) / len(durations) if durations else 0
            max_duration = max(durations) if durations else 0
            min_duration = min(durations) if durations else 0

            return {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "avg_duration": avg_duration,
                "max_duration": max_duration,
                "min_duration": min_duration,
            }


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    mode: ExecutionMode = ExecutionMode.THREAD,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
    task_timeout: Optional[float] = None,
) -> List[R]:
    """
    Convenience function for parallel mapping of items.

    Args:
        func: Function to execute
        items: Items to process
        mode: Execution mode
        max_workers: Maximum number of workers
        show_progress: Whether to show progress bar
        progress_desc: Description for progress bar
        task_timeout: Timeout for each task

    Returns:
        List of results
    """
    with ParallelExecutor(
        mode=mode,
        max_workers=max_workers,
        task_timeout=task_timeout,
        name=progress_desc or "parallel_map",
    ) as executor:
        tasks = executor.map(func, items, show_progress, progress_desc)
        return executor.get_results(tasks)


def get_optimal_chunk_size(
    total_size: int, num_workers: int, min_chunk: int = 1000
) -> int:
    """
    Calculate optimal chunk size for parallel processing.

    Args:
        total_size: Total size of data to process
        num_workers: Number of workers
        min_chunk: Minimum chunk size

    Returns:
        Optimal chunk size
    """
    if total_size <= 0:
        return min_chunk

    # Start with simple division
    chunk_size = max(min_chunk, total_size // (num_workers * 4))

    # Round to a nice number for better cache behavior
    if chunk_size > 1_000_000:
        # Round to nearest 100,000
        chunk_size = (chunk_size // 100_000) * 100_000
    elif chunk_size > 100_000:
        # Round to nearest 10,000
        chunk_size = (chunk_size // 10_000) * 10_000
    elif chunk_size > 10_000:
        # Round to nearest 1,000
        chunk_size = (chunk_size // 1_000) * 1_000

    return max(min_chunk, chunk_size)
