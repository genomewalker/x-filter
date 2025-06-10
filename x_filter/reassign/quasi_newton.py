"""
L-BFGS acceleration for EM algorithm using memory-mapped arrays.
"""
import logging
import numpy as np
from typing import Optional

log = logging.getLogger("my_logger")

class FastLBFGSAccelerator:
    """
    L-BFGS accelerator using memory-mapped arrays for large-scale problems.
    """
    
    def __init__(self, dimension: int, memory_depth: int = 10, resource_manager=None):
        self.dimension = dimension
        self.memory_depth = memory_depth
        self.resource_manager = resource_manager
        self.iteration = 0
        
        # Initialize memory-mapped storage for L-BFGS history
        self.s_history = []  # Position differences
        self.y_history = []  # Gradient differences
        self.rho_history = []  # 1/(y^T s) values
        
        self.initialized = False
        
    def step(self, current_iterate: np.ndarray, fixed_point_map: callable) -> Optional[np.ndarray]:
        """
        Perform L-BFGS acceleration step.
        """
        try:
            # Compute next EM step
            next_iterate = fixed_point_map(current_iterate)
            
            if not self.initialized:
                self.initialized = True
                self.prev_iterate = current_iterate.copy()
                return next_iterate
            
            # Compute differences
            s_k = current_iterate - self.prev_iterate  # Position difference
            y_k = next_iterate - current_iterate       # "Gradient" difference
            
            # Check curvature condition
            sy_dot = np.dot(s_k, y_k)
            if sy_dot <= 1e-8:
                # Skip this update due to poor curvature
                self.prev_iterate = current_iterate.copy()
                return next_iterate
            
            # Update history
            rho_k = 1.0 / sy_dot
            
            if len(self.s_history) >= self.memory_depth:
                # Remove oldest entries
                self.s_history.pop(0)
                self.y_history.pop(0)
                self.rho_history.pop(0)
            
            # Add new entries
            self.s_history.append(s_k.copy())
            self.y_history.append(y_k.copy())
            self.rho_history.append(rho_k)
            
            # Compute L-BFGS direction
            direction = self._compute_lbfgs_direction(next_iterate - current_iterate)
            
            # Line search and update
            accelerated_iterate = current_iterate + direction
            
            # Ensure positivity constraints (for probability distributions)
            accelerated_iterate = np.maximum(accelerated_iterate, 1e-15)
            
            # Normalize if needed (for probability distributions)
            if np.any(accelerated_iterate > 1.0):
                accelerated_iterate = accelerated_iterate / np.sum(accelerated_iterate)
            
            self.prev_iterate = current_iterate.copy()
            self.iteration += 1
            
            return accelerated_iterate
            
        except Exception as e:
            log.debug(f"L-BFGS step failed: {e}")
            return None
    
    def _compute_lbfgs_direction(self, gradient: np.ndarray) -> np.ndarray:
        """
        Compute L-BFGS search direction using two-loop recursion.
        """
        if not self.s_history:
            return gradient
        
        q = gradient.copy()
        alpha = np.zeros(len(self.s_history))
        
        # First loop (backward)
        for i in range(len(self.s_history) - 1, -1, -1):
            alpha[i] = self.rho_history[i] * np.dot(self.s_history[i], q)
            q -= alpha[i] * self.y_history[i]
        
        # Initial Hessian approximation (scaling)
        if len(self.y_history) > 0:
            gamma = np.dot(self.s_history[-1], self.y_history[-1]) / np.dot(self.y_history[-1], self.y_history[-1])
            r = gamma * q
        else:
            r = q
        
        # Second loop (forward)
        for i in range(len(self.s_history)):
            beta = self.rho_history[i] * np.dot(self.y_history[i], r)
            r += (alpha[i] - beta) * self.s_history[i]
        
        return r
    
    def cleanup(self):
        """Clean up accelerator resources."""
        self.s_history.clear()
        self.y_history.clear()
        self.rho_history.clear()
        self.initialized = False
