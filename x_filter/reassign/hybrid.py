from typing import Tuple, Optional
import numpy as np
import logging

from x_filter.reassign.anderson import FastAndersonAccelerator
from x_filter.reassign.quasi_newton import FastLBFGSAccelerator

log = logging.getLogger("my_logger")

class HybridAccelerator:
    """
    Enhanced hybrid accelerator using memory-mapped arrays exclusively.
    """
    
    def __init__(self, dimension: int, anderson_memory: int = 10, lbfgs_memory: int = 10,
                 n_queries: int = None, n_subjects: int = None,
                 mmap_dir: str = None, resource_manager=None):  # Remove random_seed parameter
        self.dimension = dimension
        self.anderson_memory = anderson_memory
        self.lbfgs_memory = lbfgs_memory
        self.resource_manager = resource_manager
        
        # Set default random seed
        random_seed = 42
        np.random.seed(random_seed)
        
        # Initialize both accelerators without seed (they have defaults)
        self.anderson_accelerator = FastAndersonAccelerator(
            dimension=dimension,
            memory_depth=anderson_memory,
            resource_manager=resource_manager
        )
        
        self.lbfgs_accelerator = FastLBFGSAccelerator(
            dimension=dimension,
            memory_depth=lbfgs_memory,
            resource_manager=resource_manager
        )
        
        # Performance tracking
        self.anderson_successes = 0
        self.lbfgs_successes = 0
        self.anderson_failures = 0
        self.lbfgs_failures = 0
        self.current_method = "anderson"

    def step(self, current_iterate: np.ndarray, fixed_point_map: callable,
             em_data: np.ndarray = None) -> np.ndarray:
        """CORRECTED Hybrid step using standard Anderson formulation."""
        try:
            # Try Anderson first (now with correct formulation)
            result = self.anderson_accelerator.step(current_iterate, fixed_point_map)
            if result is not None:
                self.anderson_successes += 1
                return result
            else:
                self.anderson_failures += 1
        except Exception as e:
            self.anderson_failures += 1
            log.debug(f"Anderson failed: {e}")
        
        try:
            # Fallback to L-BFGS
            result = self.lbfgs_accelerator.step(current_iterate, fixed_point_map)
            if result is not None:
                self.lbfgs_successes += 1
                return result
            else:
                self.lbfgs_failures += 1
        except Exception as e:
            self.lbfgs_failures += 1
            log.debug(f"L-BFGS failed: {e}")
        
        # Return basic EM step
        return fixed_point_map(current_iterate)

    def _try_acceleration_method_fast(self, method: str, current_iterate, fixed_point_map):
        """Try a specific acceleration method."""
        if method == "anderson":
            return self.anderson_accelerator.step(current_iterate, fixed_point_map)
        elif method == "lbfgs":
            return self.lbfgs_accelerator.step(current_iterate, fixed_point_map)
        else:
            return None
    
    def cleanup(self):
        """Cleanup both accelerators."""
        try:
            self.anderson_accelerator.cleanup()
            self.lbfgs_accelerator.cleanup()
        except Exception:
            pass
    
    def get_performance_stats(self):
        """Get performance statistics."""
        return {
            'anderson_successes': self.anderson_successes,
            'anderson_failures': self.anderson_failures,
            'lbfgs_successes': self.lbfgs_successes,
            'lbfgs_failures': self.lbfgs_failures
        }
