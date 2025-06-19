"""
L-BFGS acceleration for EM algorithm using ResourceManager memory-mapped arrays.
"""
from typing import Optional
import numpy as np
import logging

log = logging.getLogger("my_logger")

class FastLBFGSAccelerator:
    """Fast L-BFGS accelerator using ResourceManager memory-mapped arrays."""
    
    def __init__(self, dimension: int, memory_depth: int = 10, resource_manager=None, random_seed: int = 42):
        self.dimension = dimension
        self.memory_depth = min(memory_depth, 8)  # Conservative for stability
        self.resource_manager = resource_manager
        self.random_seed = random_seed
        
        if resource_manager is None:
            raise ValueError("ResourceManager is required for memory-mapped L-BFGS accelerator")
        
        # Set numpy random seed for deterministic behavior
        np.random.seed(random_seed)
        
        # Pre-allocate ALL arrays using ResourceManager
        try:
            self.s_vectors = self.resource_manager.create_array(
                name=f"lbfgs_s_vectors_{id(self)}",
                shape=(self.memory_depth, dimension),
                dtype=np.float64,
                temp=True
            )
            self.y_vectors = self.resource_manager.create_array(
                name=f"lbfgs_y_vectors_{id(self)}",
                shape=(self.memory_depth, dimension),
                dtype=np.float64,
                temp=True
            )
            self.rho_values = self.resource_manager.create_array(
                name=f"lbfgs_rho_values_{id(self)}",
                shape=(self.memory_depth,),
                dtype=np.float64,
                temp=True
            )
            self.alpha_values = self.resource_manager.create_array(
                name=f"lbfgs_alpha_values_{id(self)}",
                shape=(self.memory_depth,),
                dtype=np.float64,
                temp=True
            )
            self.work_vector = self.resource_manager.create_array(
                name=f"lbfgs_work_vector_{id(self)}",
                shape=(dimension,),
                dtype=np.float64,
                temp=True
            )
            self.gradient_vector = self.resource_manager.create_array(
                name=f"lbfgs_gradient_vector_{id(self)}",
                shape=(dimension,),
                dtype=np.float64,
                temp=True
            )
            
        except Exception as e:
            log.error(f"Failed to create L-BFGS memory-mapped arrays: {e}")
            raise
        
        # L-BFGS state
        self.iteration_count = 0
        self.current_position = 0
        self.stored_pairs = 0
        
        # Store previous iteration values
        self.prev_x = None
        self.prev_grad = None
        
        # Convergence parameters
        self.min_step_size = 1e-12
        self.max_step_size = 1e3
        self.gradient_tolerance = 1e-8
        
        log.debug(f"L-BFGS accelerator initialized with memory-mapped arrays: dim={dimension}, memory={memory_depth}")
    
    def _approximate_gradient_vectorized(self, current_iterate: np.ndarray, fixed_point_map: callable) -> Optional[np.ndarray]:
        """Vectorized gradient approximation using memory-mapped arrays."""
        try:
            # Current residual: F(x) = M(x) - x
            current_em_result = fixed_point_map(current_iterate)
            if current_em_result is None:
                return None
                
            # VECTORIZED: Compute residual
            current_residual = current_em_result - current_iterate
            
            # Use finite differences for gradient approximation
            epsilon = max(1e-8, np.linalg.norm(current_iterate) * 1e-8)
            
            # VECTORIZED: Initialize gradient
            self.gradient_vector[:] = 0.0
            
            # Efficient coordinate-wise finite differences
            num_coords = min(self.dimension, 50)  # Limit for efficiency
            for i in range(num_coords):
                # VECTORIZED: Create perturbed iterate
                self.work_vector[:] = current_iterate
                self.work_vector[i] += epsilon
                
                # Compute perturbed residual
                em_result_plus = fixed_point_map(self.work_vector)
                if em_result_plus is None:
                    continue
                    
                # VECTORIZED: Finite difference
                residual_plus_i = em_result_plus[i] - self.work_vector[i]
                self.gradient_vector[i] = (residual_plus_i - current_residual[i]) / epsilon
            
            # For remaining components, use residual approximation
            if self.dimension > num_coords:
                self.gradient_vector[num_coords:] = current_residual[num_coords:]
            
            return self.gradient_vector.copy()
            
        except Exception as e:
            log.debug(f"Vectorized gradient approximation failed: {e}")
            return None
    
    def _compute_lbfgs_direction_vectorized(self, gradient: np.ndarray) -> np.ndarray:
        """Vectorized L-BFGS direction computation using memory-mapped arrays."""
        try:
            if self.stored_pairs == 0:
                # VECTORIZED: Return negative gradient
                return -gradient
            
            # VECTORIZED: Copy gradient to work vector
            self.work_vector[:] = gradient
            
            # First loop: compute alpha values and update work vector
            for i in range(self.stored_pairs):
                idx = (self.current_position - 1 - i) % self.memory_depth
                
                # VECTORIZED: Compute α_i = ρ_i * s_i^T * q
                dot_product = np.dot(self.s_vectors[idx, :], self.work_vector)
                self.alpha_values[idx] = self.rho_values[idx] * dot_product
                
                # VECTORIZED: Update q = q - α_i * y_i
                self.work_vector[:] -= self.alpha_values[idx] * self.y_vectors[idx, :]
            
            # Apply initial Hessian approximation H_0 = γI
            if self.stored_pairs > 0:
                recent_idx = (self.current_position - 1) % self.memory_depth
                
                # VECTORIZED: Compute gamma
                y_dot_y = np.dot(self.y_vectors[recent_idx, :], self.y_vectors[recent_idx, :])
                if y_dot_y > 1e-12:
                    s_dot_y = np.dot(self.s_vectors[recent_idx, :], self.y_vectors[recent_idx, :])
                    gamma = max(0.1, min(10.0, s_dot_y / y_dot_y))
                else:
                    gamma = 1.0
                
                # VECTORIZED: Apply scaling
                self.work_vector[:] *= gamma
            
            # Second loop: correct the direction
            for i in range(self.stored_pairs - 1, -1, -1):
                idx = (self.current_position - 1 - i) % self.memory_depth
                
                # VECTORIZED: Compute β = ρ_i * y_i^T * r
                beta = self.rho_values[idx] * np.dot(self.y_vectors[idx, :], self.work_vector)
                
                # VECTORIZED: Update r = r + (α_i - β) * s_i
                self.work_vector[:] += (self.alpha_values[idx] - beta) * self.s_vectors[idx, :]
            
            # Return negative direction (for minimization)
            return -self.work_vector.copy()
            
        except Exception as e:
            log.debug(f"Vectorized L-BFGS direction computation failed: {e}")
            return -gradient
    
    def _update_history_vectorized(self, s_k: np.ndarray, y_k: np.ndarray) -> bool:
        """Update L-BFGS history using vectorized operations with memory-mapped arrays."""
        try:
            # VECTORIZED: Compute ρ_k = 1 / (y_k^T s_k)
            y_dot_s = np.dot(y_k, s_k)
            
            if abs(y_dot_s) < 1e-12:
                log.debug("Skipping L-BFGS update: y^T s too small")
                return False
            
            rho_k = 1.0 / y_dot_s
            
            # VECTORIZED: Store in memory-mapped circular buffer
            self.s_vectors[self.current_position, :] = s_k
            self.y_vectors[self.current_position, :] = y_k
            self.rho_values[self.current_position] = rho_k
            
            # Update circular buffer position
            self.current_position = (self.current_position + 1) % self.memory_depth
            self.stored_pairs = min(self.stored_pairs + 1, self.memory_depth)
            
            log.debug(f"L-BFGS history updated: {self.stored_pairs} pairs stored, ρ = {rho_k:.2e}")
            return True
            
        except Exception as e:
            log.debug(f"Vectorized L-BFGS history update failed: {e}")
            return False
    
    def step(self, current_iterate: np.ndarray, fixed_point_map: callable) -> Optional[np.ndarray]:
        """L-BFGS step using vectorized operations with memory-mapped arrays."""
        try:
            self.iteration_count += 1
            
            # Compute current EM result
            current_em_result = fixed_point_map(current_iterate)
            if current_em_result is None:
                log.debug("EM step failed in L-BFGS")
                return None
            
            # VECTORIZED: Normalize
            em_sum = np.sum(current_em_result)
            if em_sum > 1e-15:
                current_em_result = current_em_result / em_sum
            else:
                log.debug("L-BFGS: EM result has zero sum")
                return None
            
            # For first iteration, just return EM result
            if self.prev_x is None:
                self.prev_x = current_iterate.copy()
                self.prev_grad = self._approximate_gradient_vectorized(current_iterate, fixed_point_map)
                return current_em_result
            
            # Compute current gradient
            current_grad = self._approximate_gradient_vectorized(current_iterate, fixed_point_map)
            if current_grad is None:
                log.debug("Gradient computation failed")
                return current_em_result
            
            # VECTORIZED: Compute s_k and y_k
            s_k = current_iterate - self.prev_x
            y_k = current_grad - self.prev_grad if self.prev_grad is not None else current_grad
            
            # Update L-BFGS history
            if np.linalg.norm(s_k) > 1e-12 and np.linalg.norm(y_k) > 1e-12:
                self._update_history_vectorized(s_k, y_k)
            
            # VECTORIZED: Compute current residual
            current_residual = current_em_result - current_iterate
            residual_norm = np.linalg.norm(current_residual)
            
            # If residual is very small, return EM result
            if residual_norm < self.gradient_tolerance:
                log.debug(f"L-BFGS: residual very small ({residual_norm:.2e}), using EM result")
                self.prev_x = current_iterate.copy()
                self.prev_grad = current_grad
                return current_em_result
            
            # Compute L-BFGS search direction
            direction = self._compute_lbfgs_direction_vectorized(current_residual)
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm < 1e-12:
                log.debug("L-BFGS direction too small")
                self.prev_x = current_iterate.copy()
                self.prev_grad = current_grad
                return current_em_result
            
            # VECTORIZED: Normalize direction
            direction = direction / direction_norm
            
            # Simple step size (avoiding complex line search for speed)
            step_size = min(0.1, 1.0 / max(1.0, residual_norm))
            
            # VECTORIZED: Compute L-BFGS step
            lbfgs_result = current_iterate + step_size * direction
            
            # VECTORIZED: Ensure positivity and normalization
            np.maximum(lbfgs_result, 1e-15, out=lbfgs_result)
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
            
            # Use L-BFGS result if it provides meaningful improvement
            if lbfgs_step_size > em_step_size * 0.1:
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
        """Clean up memory-mapped arrays through ResourceManager."""
        try:
            if hasattr(self, 'resource_manager') and self.resource_manager is not None:
                array_names = [
                    f"lbfgs_s_vectors_{id(self)}",
                    f"lbfgs_y_vectors_{id(self)}",
                    f"lbfgs_rho_values_{id(self)}",
                    f"lbfgs_alpha_values_{id(self)}",
                    f"lbfgs_work_vector_{id(self)}",
                    f"lbfgs_gradient_vector_{id(self)}"
                ]
                
                for name in array_names:
                    try:
                        self.resource_manager.delete_array(name)
                    except Exception:
                        pass
                
                # Clear instance variables
                for attr in ['s_vectors', 'y_vectors', 'rho_values', 'alpha_values', 
                           'work_vector', 'gradient_vector', 'prev_x', 'prev_grad']:
                    if hasattr(self, attr):
                        setattr(self, attr, None)
                
                log.debug("L-BFGS accelerator cleanup completed")
        except Exception as e:
            log.debug(f"L-BFGS cleanup error (non-fatal): {e}")
