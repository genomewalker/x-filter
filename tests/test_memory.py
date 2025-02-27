# tests/test_memory.py

import unittest
import numpy as np
import tempfile
import os
import time
from unittest import mock

from x_filter.utils.memory import MemoryTracker, track_memory, MemoryMonitor


class TestMemoryTracker(unittest.TestCase):
    """Test the MemoryTracker class."""

    def setUp(self):
        # Reset the singleton instance for testing
        MemoryTracker._instance = None

    def test_singleton_pattern(self):
        """Test that MemoryTracker is a singleton."""
        tracker1 = MemoryTracker()
        tracker2 = MemoryTracker()
        self.assertIs(tracker1, tracker2)

    def test_start_stop_tracking(self):
        """Test start and stop tracking."""
        tracker = MemoryTracker()
        tracker.start_tracking("test_operation")
        stats = tracker.stop_tracking("test_operation")

        self.assertGreater(len(stats), 1)
        self.assertEqual(stats[0].event, "start")
        self.assertEqual(stats[-1].event, "end")

    def test_record_event(self):
        """Test recording events."""
        tracker = MemoryTracker()
        tracker.start_tracking("test_operation")
        tracker.record_event("test_operation", "milestone_1")
        stats = tracker.stop_tracking("test_operation")

        self.assertGreater(len(stats), 2)
        events = [stat.event for stat in stats]
        self.assertIn("milestone_1", events)


class TestMemoryDecorator(unittest.TestCase):
    """Test the track_memory decorator."""

    @track_memory
    def sample_function(self, size_mb=10):
        """Sample function that allocates memory."""
        # Allocate an array of specified size
        arr = np.ones((size_mb * 1024 * 1024 // 8), dtype=np.float64)
        return arr

    @track_memory(name="custom_name", detailed=True)
    def another_function(self):
        """Another sample function."""
        return 42

    def test_decorator_basic(self):
        """Test basic decorator functionality."""
        result = self.sample_function(size_mb=1)
        self.assertIsNotNone(result)

    def test_decorator_with_args(self):
        """Test decorator with arguments."""
        result = self.another_function()
        self.assertEqual(result, 42)


class TestMemoryMonitor(unittest.TestCase):
    """Test the MemoryMonitor class."""

    def test_context_manager(self):
        """Test memory monitor as context manager."""
        with MemoryMonitor("test_monitoring", interval=0.1) as monitor:
            # Perform some memory-intensive operation
            arrays = [np.ones((1024, 1024)) for _ in range(3)]
            time.sleep(0.3)  # Allow monitor to run a few cycles

        # Verify that tracking occurred
        tracker = MemoryTracker()
        history = tracker.history.get("test_monitoring", [])
        self.assertGreater(len(history), 2)


if __name__ == "__main__":
    unittest.main()
