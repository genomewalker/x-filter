import logging
import numpy as np
from typing import Tuple, Optional

from x_filter.reassign.anderson import FastAndersonAccelerator
from x_filter.reassign.quasi_newton import FastLBFGSAccelerator

log = logging.getLogger("my_logger")

class HybridAccelerator:
    """
    Enhanced hybrid accelerator using memory-mapped arrays exclusively.
    """
    
    def __init__(self, dimension: int, anderson_memory: int = 10, lbfgs_memory: int = 10,
                 n_queries: int = None, n_subjects: int = None,
                 mmap_dir: str = None, resource_manager=None):
        self.dimension = dimension
        self.anderson_memory = anderson_memory
        self.lbfgs_memory = lbfgs_memory
        self.resource_manager = resource_manager
        
        # Initialize both accelerators
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
        """
        Perform hybrid acceleration step.
        """
        try:
            if self.current_method == "anderson":
                result = self._try_acceleration_method_fast("anderson", current_iterate, fixed_point_map)
                if result is not None:
                    self.anderson_successes += 1
                    return result
                else:
                    self.anderson_failures += 1
                    # Switch to L-BFGS
                    self.current_method = "lbfgs"
                    
            if self.current_method == "lbfgs":
                result = self._try_acceleration_method_fast("lbfgs", current_iterate, fixed_point_map)
                if result is not None:
                    self.lbfgs_successes += 1
                    return result
                else:
                    self.lbfgs_failures += 1
                    # Switch back to Anderson
                    self.current_method = "anderson"
            
            # If all methods fail, return basic EM step
            return fixed_point_map(current_iterate)
            
        except Exception as e:
            log.warning(f"Hybrid acceleration failed: {e}")
            return fixed_point_map(current_iterate)

    def _try_acceleration_method_fast(self, method: str, current_iterate, fixed_point_map):
        """
        Try acceleration method with error handling.
        """
        try:
            if method == "anderson":
                return self.anderson_accelerator.step(current_iterate, fixed_point_map)
            elif method == "lbfgs":
                return self.lbfgs_accelerator.step(current_iterate, fixed_point_map)
            else:
                return None
        except Exception as e:
            log.debug(f"{method} acceleration failed: {e}")
            return None
    
    def cleanup(self):
        """Clean up accelerator resources."""
        if hasattr(self.anderson_accelerator, 'cleanup'):
            self.anderson_accelerator.cleanup()
        if hasattr(self.lbfgs_accelerator, 'cleanup'):
            self.lbfgs_accelerator.cleanup()
    
    def get_performance_stats(self):
        """Get performance statistics."""
        total_anderson = self.anderson_successes + self.anderson_failures
        total_lbfgs = self.lbfgs_successes + self.lbfgs_failures
        
        anderson_rate = self.anderson_successes / max(total_anderson, 1) * 100
        lbfgs_rate = self.lbfgs_successes / max(total_lbfgs, 1) * 100
        
        return {
            'anderson_success_rate': anderson_rate,
            'lbfgs_success_rate': lbfgs_rate,
            'anderson_attempts': total_anderson,
            'lbfgs_attempts': total_lbfgs,
            'current_method': self.current_method
        }
