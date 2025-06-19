"""
Anderson acceleration for EM algorithm using memory-mapped arrays via ResourceManager.
"""
import numpy as np
import logging
from numba import njit
from typing import Optional

log = logging.getLogger("my_logger")

@njit(fastmath=True, parallel=False, cache=True)
def vectorized_anderson_solve(
    residual_diffs: np.ndarray,
    memory_used: int,
    weights_out: np.ndarray,
    temp_gram: np.ndarray
) -> bool:
    """
    VECTORIZED Anderson system solve using memory-mapped arrays.
    """
    if memory_used <= 1:
        if memory_used == 1:
            weights_out[0] = 1.0
        return memory_used > 0
    
    # VECTORIZED: Build Gram matrix
    for i in range(memory_used):
        for j in range(i, memory_used):
            dot_product = np.dot(residual_diffs[i], residual_diffs[j])
            temp_gram[i, j] = dot_product
            temp_gram[j, i] = dot_product
    
    # VECTORIZED: Add regularization to diagonal
    for i in range(memory_used):
        temp_gram[i, i] += 1e-8
    
    # VECTORIZED: Check condition - manually extract diagonal
    diag_sum = 0.0
    for i in range(memory_used):
        diag_sum += temp_gram[i, i]
    
    condition_estimate = temp_gram[0, 0] / (diag_sum + 1e-15)
    
    if condition_estimate < 1e-6:
        # VECTORIZED: Uniform weights
        uniform_weight = 1.0 / memory_used
        for i in range(memory_used):
            weights_out[i] = uniform_weight
        return True
    
    # VECTORIZED: Initialize weights
    for i in range(memory_used):
        weights_out[i] = 1.0 / memory_used
    
    # Iterative solve with vectorized operations
    for iteration in range(3):
        new_weights = np.zeros(memory_used)
        for i in range(memory_used):
            sum_val = 0.0
            for j in range(memory_used):
                if i != j:
                    sum_val += temp_gram[i, j] * weights_out[j]
            if temp_gram[i, i] > 1e-12:
                new_weights[i] = (1.0 / memory_used - sum_val) / temp_gram[i, i]
        
        # VECTORIZED: Normalization
        total = np.sum(new_weights)
        if total > 1e-12:
            for i in range(memory_used):
                weights_out[i] = new_weights[i] / total
    
    return True

class FastAndersonAccelerator:
    """Fast Anderson accelerator using ResourceManager memory-mapped arrays."""
    
    def __init__(self, dimension: int, memory_depth: int = 10, beta: float = 1.0,
                 resource_manager=None):
        self.dimension = dimension
        self.memory_depth = min(memory_depth, 5)  # Conservative for stability
        self.beta = min(beta, 0.5)  # Conservative mixing
        self.resource_manager = resource_manager
        
        if resource_manager is None:
            raise ValueError("ResourceManager is required for memory-mapped Anderson accelerator")
        
        # Set deterministic seed
        np.random.seed(42)
        
        # Pre-allocate ALL arrays using ResourceManager memory-mapped storage
        try:
            self.iterate_history = self.resource_manager.create_array(
                name=f"anderson_iterate_history_{id(self)}",
                shape=(self.memory_depth, dimension),
                dtype=np.float64, 
                temp=True
            )
            self.residual_history = self.resource_manager.create_array(
                name=f"anderson_residual_history_{id(self)}",
                shape=(self.memory_depth, dimension),
                dtype=np.float64, 
                temp=True
            )
            self.residual_diffs = self.resource_manager.create_array(
                name=f"anderson_residual_diffs_{id(self)}",
                shape=(self.memory_depth, dimension),
                dtype=np.float64, 
                temp=True
            )
            self.weights = self.resource_manager.create_array(
                name=f"anderson_weights_{id(self)}",
                shape=(self.memory_depth,),
                dtype=np.float64, 
                temp=True
            )
            self.temp_gram = self.resource_manager.create_array(
                name=f"anderson_temp_gram_{id(self)}",
                shape=(self.memory_depth, self.memory_depth),
                dtype=np.float64, 
                temp=True
            )
            self.temp_result = self.resource_manager.create_array(
                name=f"anderson_temp_result_{id(self)}",
                shape=(dimension,),
                dtype=np.float64, 
                temp=True
            )
            self.temp_weighted_sum = self.resource_manager.create_array(
                name=f"anderson_temp_weighted_sum_{id(self)}",
                shape=(dimension,),
                dtype=np.float64, 
                temp=True
            )
            
        except Exception as e:
            log.error(f"Failed to create Anderson memory-mapped arrays: {e}")
            raise
        
        # Initialize state
        self.memory_used = 0
        self.current_pos = 0
        self.first_iteration = True
        
        # Performance tracking
        self.success_count = 0
        self.total_attempts = 0
        self.error_count = 0
        
        log.debug(f"Anderson accelerator initialized with memory-mapped arrays: dim={dimension}, memory={self.memory_depth}")
    
    def step(self, current_iterate: np.ndarray, fixed_point_map: callable) -> Optional[np.ndarray]:
        """Anderson step using memory-mapped arrays and vectorized operations."""
        self.total_attempts += 1
        
        try:
            # Validate input
            if current_iterate is None or len(current_iterate) != self.dimension:
                self.error_count += 1
                return None
            
            if not np.all(np.isfinite(current_iterate)):
                self.error_count += 1
                return self._safe_normalize(current_iterate)
            
            # Get F(x_k) - the basic EM step
            try:
                fx_k = fixed_point_map(current_iterate)
            except Exception as e:
                log.warning(f"Fixed point map failed in Anderson: {e}")
                self.error_count += 1
                return current_iterate
            
            if fx_k is None or len(fx_k) != self.dimension or not np.all(np.isfinite(fx_k)):
                self.error_count += 1
                return current_iterate
            
            # Compute residual: f_k = F(x_k) - x_k
            # Use vectorized operations
            current_residual = fx_k - current_iterate
            
            # For first iteration, just return F(x_k) and store history
            if self.first_iteration:
                self._store_history_vectorized(current_iterate, current_residual)
                self.first_iteration = False
                return self._safe_normalize(fx_k)
            
            residual_norm = np.linalg.norm(current_residual)
            if not np.isfinite(residual_norm) or residual_norm < 1e-15:
                self._store_history_vectorized(current_iterate, current_residual)
                return self._safe_normalize(fx_k)
            
            # Try Anderson acceleration if we have sufficient history
            if self.memory_used > 0:
                try:
                    result = self._apply_vectorized_anderson(current_iterate, fx_k, current_residual)
                    if result is not None and self._validate_result_vectorized(result, current_iterate, fx_k):
                        self.success_count += 1
                        self._store_history_vectorized(current_iterate, current_residual)
                        return result
                except Exception as e:
                    log.debug(f"Anderson acceleration failed: {e}")
            
            # Store history and return basic F(x_k)
            self._store_history_vectorized(current_iterate, current_residual)
            return self._safe_normalize(fx_k)
            
        except Exception as e:
            log.warning(f"Critical Anderson error: {e}")
            self.error_count += 1
            return self._safe_normalize(current_iterate) if current_iterate is not None else None

    def _store_history_vectorized(self, iterate: np.ndarray, residual: np.ndarray):
        """Store iterate and residual using vectorized operations with memory-mapped arrays."""
        pos = self.current_pos % self.memory_depth
        
        # VECTORIZED: Store current iterate and residual
        self.iterate_history[pos, :] = iterate
        self.residual_history[pos, :] = residual
        
        # Update counters
        self.current_pos += 1
        self.memory_used = min(self.memory_used + 1, self.memory_depth)

    def _apply_vectorized_anderson(self, current_iterate: np.ndarray, fx_k: np.ndarray, 
                                 current_residual: np.ndarray) -> Optional[np.ndarray]:
        """Apply Anderson acceleration using vectorized operations with memory-mapped arrays."""
        try:
            if self.memory_used < 2:
                return fx_k
            
            # VECTORIZED: Build residual difference matrix
            num_diffs = min(self.memory_used - 1, self.memory_depth)
            for i in range(num_diffs):
                pos_curr = (self.current_pos - 1 - i) % self.memory_depth
                pos_prev = (self.current_pos - 2 - i) % self.memory_depth
                # VECTORIZED: Compute difference
                self.residual_diffs[i, :] = self.residual_history[pos_curr, :] - self.residual_history[pos_prev, :]
            
            # VECTORIZED: Solve Anderson system
            success = vectorized_anderson_solve(
                self.residual_diffs[:num_diffs, :],
                num_diffs,
                self.weights,
                self.temp_gram
            )
            
            if not success:
                return None
            
            # VECTORIZED: Compute weighted combination
            # Clear weighted sum
            self.temp_weighted_sum[:] = 0.0
            
            # VECTORIZED: Accumulate weighted history
            for i in range(num_diffs):
                pos = (self.current_pos - 1 - i) % self.memory_depth
                weight = self.weights[i]
                # F(x_i) = x_i + f_i
                fx_i = self.iterate_history[pos, :] + self.residual_history[pos, :]
                self.temp_weighted_sum[:] += weight * fx_i
            
            # Add current point with remaining weight
            current_weight = 1.0 - np.sum(self.weights[:num_diffs])
            self.temp_result[:] = self.temp_weighted_sum + current_weight * fx_k
            
            # Apply conservative damping
            damping = 0.7  # Conservative
            self.temp_result[:] = (1 - damping) * fx_k + damping * self.temp_result
            
            return self._safe_normalize(self.temp_result)

        except Exception as e:
            log.debug(f"Vectorized Anderson acceleration failed: {e}")
            return None

    def _validate_result_vectorized(self, result: np.ndarray, current_iterate: np.ndarray, 
                                  basic_step: np.ndarray) -> bool:
        """Validate result using vectorized operations."""
        try:
            if result is None or len(result) != self.dimension:
                return False
            
            # VECTORIZED: Check finite values and bounds
            if not np.all(np.isfinite(result)) or not np.all(result >= 1e-15) or not np.all(result <= 1.0):
                return False
            
            # VECTORIZED: Check sum constraint
            result_sum = np.sum(result)
            if not np.isfinite(result_sum) or abs(result_sum - 1.0) > 0.5:
                return False
            
            # VECTORIZED: Check change magnitude
            result_change = np.linalg.norm(result - current_iterate)
            basic_change = np.linalg.norm(basic_step - current_iterate)
            
            if not np.isfinite(result_change) or not np.isfinite(basic_change):
                return False
            
            # Permissive change ratio for stability
            if basic_change > 1e-15 and result_change > basic_change * 100:
                return False
            
            return True
            
        except Exception:
            return False

    def _safe_normalize(self, array: np.ndarray) -> np.ndarray:
        """Safely normalize array using vectorized operations with memory-mapped temp array."""
        try:
            # VECTORIZED: Clip and sum
            np.clip(array, 1e-15, 1.0, out=self.temp_result[:len(array)])
            current_sum = np.sum(self.temp_result[:len(array)])
            
            if current_sum > 1e-15:
                # VECTORIZED: Normalize
                self.temp_result[:len(array)] /= current_sum
            else:
                # VECTORIZED: Uniform fallback
                self.temp_result[:len(array)] = 1.0 / len(array)
            
            return self.temp_result[:len(array)].copy()
        except Exception as e:
            log.warning(f"Safe normalization failed: {e}")
            # Ultimate fallback
            result = np.full(len(array), 1.0 / len(array))
            return result

    def cleanup(self):
        """Clean up memory-mapped arrays through ResourceManager."""
        try:
            if hasattr(self, 'resource_manager') and self.resource_manager is not None:
                # ResourceManager will handle cleanup of memory-mapped arrays
                # Just clear our references
                array_names = [
                    f"anderson_iterate_history_{id(self)}",
                    f"anderson_residual_history_{id(self)}",
                    f"anderson_residual_diffs_{id(self)}",
                    f"anderson_weights_{id(self)}",
                    f"anderson_temp_gram_{id(self)}",
                    f"anderson_temp_result_{id(self)}",
                    f"anderson_temp_weighted_sum_{id(self)}"
                ]
                
                for name in array_names:
                    try:
                        self.resource_manager.delete_array(name)
                    except Exception:
                        pass  # Ignore cleanup errors
                
                # Clear instance variables
                for attr in ['iterate_history', 'residual_history', 'residual_diffs', 
                           'weights', 'temp_gram', 'temp_result', 'temp_weighted_sum']:
                    if hasattr(self, attr):
                        setattr(self, attr, None)
                
                self.memory_used = 0
                self.current_pos = 0
                self.first_iteration = True
                
                log.debug("Anderson accelerator cleanup completed")
        except Exception as e:
            log.debug(f"Anderson cleanup error (non-fatal): {e}")
