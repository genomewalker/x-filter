"""
L-BFGS acceleration for EM algorithm using memory-mapped arrays.
"""
from typing import Optional
import numpy as np
import logging
import os

log = logging.getLogger("my_logger")

class FastLBFGSAccelerator:
    """Fast L-BFGS accelerator using memory-mapped arrays."""
    
    def __init__(self, dimension: int, memory_depth: int = 10, resource_manager=None):
        self.dimension = dimension
        self.memory_depth = memory_depth
        self.resource_manager = resource_manager
        
        # L-BFGS state
        self.iteration_count = 0
        self.current_position = 0  # Circular buffer position
        self.stored_pairs = 0  # Number of stored s_k, y_k pairs
        
        # Initialize memory-mapped storage for L-BFGS vectors
        self._initialize_storage()
        
        # Store previous iteration values
        self.prev_x = None
        self.prev_grad = None
        
        # Convergence parameters
        self.min_step_size = 1e-12
        self.max_step_size = 1e3
        self.gradient_tolerance = 1e-8
        
        log.info(f"L-BFGS accelerator initialized with dimension {dimension}, memory depth {memory_depth}")
    
    def _initialize_storage(self):
        """Initialize memory-mapped arrays for L-BFGS storage."""
        if self.resource_manager is not None:
            # Use resource manager for memory-mapped arrays
            self.s_vectors = self.resource_manager.create_array(
                name="lbfgs_s_vectors", 
                shape=(self.memory_depth, self.dimension), 
                dtype=np.float64, 
                temp=True
            )
            self.y_vectors = self.resource_manager.create_array(
                name="lbfgs_y_vectors", 
                shape=(self.memory_depth, self.dimension), 
                dtype=np.float64, 
                temp=True
            )
            self.rho_values = self.resource_manager.create_array(
                name="lbfgs_rho_values", 
                shape=(self.memory_depth,), 
                dtype=np.float64, 
                temp=True
            )
            self.alpha_values = self.resource_manager.create_array(
                name="lbfgs_alpha_values", 
                shape=(self.memory_depth,), 
                dtype=np.float64, 
                temp=True
            )
            
            # Working arrays for L-BFGS computation
            self.work_vector = self.resource_manager.create_array(
                name="lbfgs_work_vector", 
                shape=(self.dimension,), 
                dtype=np.float64, 
                temp=True
            )
            self.gradient_vector = self.resource_manager.create_array(
                name="lbfgs_gradient_vector", 
                shape=(self.dimension,), 
                dtype=np.float64, 
                temp=True
            )
            
            log.debug(f"L-BFGS memory-mapped storage initialized")
        else:
            # Fallback to regular numpy arrays
            self.s_vectors = np.zeros((self.memory_depth, self.dimension), dtype=np.float64)
            self.y_vectors = np.zeros((self.memory_depth, self.dimension), dtype=np.float64)
            self.rho_values = np.zeros(self.memory_depth, dtype=np.float64)
            self.alpha_values = np.zeros(self.memory_depth, dtype=np.float64)
            self.work_vector = np.zeros(self.dimension, dtype=np.float64)
            self.gradient_vector = np.zeros(self.dimension, dtype=np.float64)
            
            log.debug(f"L-BFGS regular array storage initialized")
    
    def _approximate_gradient(self, current_iterate: np.ndarray, fixed_point_map: callable) -> np.ndarray:
        """
        Approximate the gradient of the EM fixed-point residual.
        For EM: F(x) = M(x) - x, where M is the EM map
        Gradient approximation: ∇F(x) ≈ (F(x + εe_i) - F(x)) / ε for each component
        """
        try:
            # Current residual: F(x) = M(x) - x
            current_em_result = fixed_point_map(current_iterate)
            if current_em_result is None:
                return None
                
            current_residual = current_em_result - current_iterate
            
            # Use finite differences for gradient approximation
            epsilon = max(1e-8, np.linalg.norm(current_iterate) * 1e-8)
            gradient = np.zeros_like(current_iterate)
            
            # Coordinate-wise finite differences (more efficient than full Jacobian)
            for i in range(min(self.dimension, 100)):  # Limit to first 100 components for efficiency
                # Perturb coordinate i
                x_plus = current_iterate.copy()
                x_plus[i] += epsilon
                
                # Compute perturbed residual
                em_result_plus = fixed_point_map(x_plus)
                if em_result_plus is None:
                    continue
                    
                residual_plus = em_result_plus - x_plus
                
                # Finite difference approximation
                gradient[i] = (residual_plus[i] - current_residual[i]) / epsilon
            
            # For remaining components, use a simpler approximation
            if self.dimension > 100:
                # Use current residual as gradient approximation for efficiency
                gradient[100:] = current_residual[100:]
            
            return gradient
            
        except Exception as e:
            log.debug(f"Gradient approximation failed: {e}")
            return None
    
    def _compute_lbfgs_direction(self, gradient: np.ndarray) -> np.ndarray:
        """
        Compute L-BFGS search direction using two-loop recursion.
        """
        try:
            if self.stored_pairs == 0:
                # No history available, return negative gradient (steepest descent)
                return -gradient
            
            # Copy gradient to work vector
            self.work_vector[:] = gradient
            
            # First loop: compute alpha values and update work vector
            for i in range(self.stored_pairs):
                # Get circular buffer index (most recent first)
                idx = (self.current_position - 1 - i) % self.memory_depth
                
                # Compute α_i = ρ_i * s_i^T * q
                dot_product = np.dot(self.s_vectors[idx], self.work_vector)
                self.alpha_values[idx] = self.rho_values[idx] * dot_product
                
                # Update q = q - α_i * y_i
                self.work_vector -= self.alpha_values[idx] * self.y_vectors[idx]
            
            # Apply initial Hessian approximation H_0 = γI
            # Use γ = (s^T y) / (y^T y) from most recent pair
            if self.stored_pairs > 0:
                recent_idx = (self.current_position - 1) % self.memory_depth
                s_recent = self.s_vectors[recent_idx]
                y_recent = self.y_vectors[recent_idx]
                
                y_dot_y = np.dot(y_recent, y_recent)
                if y_dot_y > 1e-12:
                    s_dot_y = np.dot(s_recent, y_recent)
                    gamma = max(0.1, min(10.0, s_dot_y / y_dot_y))  # Clamp gamma
                else:
                    gamma = 1.0
                
                self.work_vector *= gamma
            
            # Second loop: correct the direction
            for i in range(self.stored_pairs - 1, -1, -1):
                # Get circular buffer index (oldest first in second loop)
                idx = (self.current_position - 1 - i) % self.memory_depth
                
                # Compute β = ρ_i * y_i^T * r
                dot_product = np.dot(self.y_vectors[idx], self.work_vector)
                beta = self.rho_values[idx] * dot_product
                
                # Update r = r + (α_i - β) * s_i
                self.work_vector += (self.alpha_values[idx] - beta) * self.s_vectors[idx]
            
            # Return negative direction (for minimization)
            return -self.work_vector
            
        except Exception as e:
            log.debug(f"L-BFGS direction computation failed: {e}")
            return -gradient  # Fallback to steepest descent
    
    def _update_history(self, s_k: np.ndarray, y_k: np.ndarray):
        """Update L-BFGS history with new s_k and y_k vectors."""
        try:
            # Compute ρ_k = 1 / (y_k^T s_k)
            y_dot_s = np.dot(y_k, s_k)
            
            if abs(y_dot_s) < 1e-12:
                log.debug("Skipping L-BFGS update: y^T s too small")
                return False
            
            rho_k = 1.0 / y_dot_s
            
            # Store in circular buffer
            self.s_vectors[self.current_position] = s_k
            self.y_vectors[self.current_position] = y_k
            self.rho_values[self.current_position] = rho_k
            
            # Update circular buffer position
            self.current_position = (self.current_position + 1) % self.memory_depth
            self.stored_pairs = min(self.stored_pairs + 1, self.memory_depth)
            
            log.debug(f"L-BFGS history updated: {self.stored_pairs} pairs stored, ρ = {rho_k:.2e}")
            return True
            
        except Exception as e:
            log.debug(f"L-BFGS history update failed: {e}")
            return False
    
    def _line_search(self, current_iterate: np.ndarray, direction: np.ndarray, 
                     fixed_point_map: callable, current_residual: np.ndarray) -> float:
        """
        Simple backtracking line search for step size.
        """
        try:
            # Initial step size
            alpha = 1.0
            c1 = 1e-4  # Armijo condition parameter
            rho = 0.5  # Backtracking parameter
            max_backtracks = 10
            
            # Current function value (residual norm squared)
            current_f = 0.5 * np.dot(current_residual, current_residual)
            
            # Gradient dot direction (should be negative for descent)
            grad_dot_dir = np.dot(current_residual, direction)
            
            if grad_dot_dir >= 0:
                log.debug("L-BFGS direction is not a descent direction")
                return 0.0
            
            for i in range(max_backtracks):
                # Test point
                x_new = current_iterate + alpha * direction
                
                # Ensure weights remain positive and normalized
                x_new = np.maximum(x_new, 1e-15)
                x_new = x_new / np.sum(x_new)
                
                # Compute new residual
                try:
                    em_result = fixed_point_map(x_new)
                    if em_result is None:
                        alpha *= rho
                        continue
                        
                    new_residual = em_result - x_new
                    new_f = 0.5 * np.dot(new_residual, new_residual)
                    
                    # Armijo condition
                    if new_f <= current_f + c1 * alpha * grad_dot_dir:
                        return alpha
                        
                except Exception:
                    pass
                
                alpha *= rho
            
            return 0.0  # Line search failed
            
        except Exception as e:
            log.debug(f"Line search failed: {e}")
            return 0.0
    
    def step(self, current_iterate: np.ndarray, fixed_point_map: callable) -> Optional[np.ndarray]:
        """L-BFGS step for EM acceleration."""
        try:
            self.iteration_count += 1
            
            # Compute current EM result
            current_em_result = fixed_point_map(current_iterate)
            if current_em_result is None:
                log.debug("EM step failed in L-BFGS")
                return None
            
            # Ensure proper normalization
            if np.sum(current_em_result) > 1e-15:
                current_em_result = current_em_result / np.sum(current_em_result)
            else:
                log.debug("L-BFGS: EM result has zero sum")
                return None
            
            # For first iteration, just return EM result
            if self.prev_x is None:
                self.prev_x = current_iterate.copy()
                self.prev_grad = self._approximate_gradient(current_iterate, fixed_point_map)
                return current_em_result
            
            # Compute current gradient (approximate)
            current_grad = self._approximate_gradient(current_iterate, fixed_point_map)
            if current_grad is None:
                log.debug("Gradient computation failed")
                return current_em_result
            
            # Compute s_k = x_k - x_{k-1} and y_k = ∇f_k - ∇f_{k-1}
            s_k = current_iterate - self.prev_x
            y_k = current_grad - self.prev_grad if self.prev_grad is not None else current_grad
            
            # Update L-BFGS history
            if np.linalg.norm(s_k) > 1e-12 and np.linalg.norm(y_k) > 1e-12:
                self._update_history(s_k, y_k)
            
            # Compute current residual for L-BFGS
            current_residual = current_em_result - current_iterate
            residual_norm = np.linalg.norm(current_residual)
            
            # If residual is very small, return EM result
            if residual_norm < self.gradient_tolerance:
                log.debug(f"L-BFGS: residual very small ({residual_norm:.2e}), using EM result")
                self.prev_x = current_iterate.copy()
                self.prev_grad = current_grad
                return current_em_result
            
            # Compute L-BFGS search direction
            direction = self._compute_lbfgs_direction(current_residual)
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm < 1e-12:
                log.debug("L-BFGS direction too small")
                self.prev_x = current_iterate.copy()
                self.prev_grad = current_grad
                return current_em_result
            
            # Normalize direction
            direction = direction / direction_norm
            
            # Line search for step size
            step_size = self._line_search(current_iterate, direction, fixed_point_map, current_residual)
            
            if step_size < self.min_step_size:
                log.debug(f"L-BFGS step size too small: {step_size:.2e}")
                self.prev_x = current_iterate.copy()
                self.prev_grad = current_grad
                return current_em_result
            
            # Compute L-BFGS step
            lbfgs_result = current_iterate + step_size * direction
            
            # Ensure positivity and normalization
            lbfgs_result = np.maximum(lbfgs_result, 1e-15)
            total_weight = np.sum(lbfgs_result)
            if total_weight > 1e-15:
                lbfgs_result = lbfgs_result / total_weight
            else:
                log.debug("L-BFGS result has zero weight sum")
                self.prev_x = current_iterate.copy()
                self.prev_grad = current_grad
                return current_em_result
            
            # Validate L-BFGS result
            if not np.all(np.isfinite(lbfgs_result)):
                log.debug("L-BFGS result contains invalid values")
                self.prev_x = current_iterate.copy()
                self.prev_grad = current_grad
                return current_em_result
            
            # Check if L-BFGS step provides improvement
            lbfgs_step_size = np.linalg.norm(lbfgs_result - current_iterate)
            em_step_size = np.linalg.norm(current_em_result - current_iterate)
            
            # Use L-BFGS result if it provides a meaningful improvement
            if lbfgs_step_size > em_step_size * 0.1:  # At least 10% of EM step size
                log.debug(f"L-BFGS step accepted: step size {lbfgs_step_size:.2e} vs EM {em_step_size:.2e}")
                result = lbfgs_result
            else:
                log.debug(f"L-BFGS step too small, using EM result")
                result = current_em_result
            
            # Update for next iteration
            self.prev_x = current_iterate.copy()
            self.prev_grad = current_grad
            
            return result
            
        except Exception as e:
            log.warning(f"L-BFGS step failed: {e}")
            # Update state even on failure
            try:
                self.prev_x = current_iterate.copy()
                if 'current_grad' in locals():
                    self.prev_grad = current_grad
            except:
                pass
            return None
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            # Clear arrays to free memory
            if hasattr(self, 's_vectors'):
                del self.s_vectors
            if hasattr(self, 'y_vectors'):
                del self.y_vectors
            if hasattr(self, 'rho_values'):
                del self.rho_values
            if hasattr(self, 'alpha_values'):
                del self.alpha_values
            if hasattr(self, 'work_vector'):
                del self.work_vector
            if hasattr(self, 'gradient_vector'):
                del self.gradient_vector
            if hasattr(self, 'prev_x'):
                del self.prev_x
            if hasattr(self, 'prev_grad'):
                del self.prev_grad
                
            log.debug("L-BFGS accelerator cleanup completed")
        except Exception as e:
            log.debug(f"L-BFGS cleanup error: {e}")
