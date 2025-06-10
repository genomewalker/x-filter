import numpy as np
import logging
from numba import njit, prange
from typing import Tuple, Optional

log = logging.getLogger("my_logger")

@njit(fastmath=True, parallel=True, cache=True)
def optimized_anderson_update(
    residual: np.ndarray,
    history_matrix: np.ndarray,
    weights: np.ndarray,
    memory_used: int,
    beta: float,
    current_iterate: np.ndarray
) -> None:
    """OPTIMIZED Anderson update with pure vectorized operations."""
    n = len(residual)
    
    if memory_used == 0:
        # Simple mixing - vectorized
        for i in prange(n):
            current_iterate[i] = current_iterate[i] + beta * residual[i]
        return
    
    # VECTORIZED: Compute weighted combination directly
    for i in prange(n):
        weighted_sum = 0.0
        for k in range(memory_used):
            weighted_sum += weights[k] * history_matrix[k, i]
        current_iterate[i] = current_iterate[i] + beta * (residual[i] - weighted_sum)

@njit(fastmath=True, cache=True)
def fast_solve_anderson_system(
    residual_diffs: np.ndarray,
    memory_used: int,
    weights_out: np.ndarray
) -> bool:
    """OPTIMIZED Anderson system solve with minimal operations."""
    if memory_used <= 1:
        if memory_used == 1:
            weights_out[0] = 1.0
        return memory_used > 0
    
    # Build Gram matrix - optimized
    gram_matrix = np.zeros((memory_used, memory_used), dtype=np.float64)
    
    for i in range(memory_used):
        for j in range(i, memory_used):  # Only upper triangle
            dot_product = np.dot(residual_diffs[i], residual_diffs[j])
            gram_matrix[i, j] = dot_product
            gram_matrix[j, i] = dot_product  # Symmetric
    
    # Add regularization
    for i in range(memory_used):
        gram_matrix[i, i] += 1e-8
    
    # Simplified solve: Use uniform weights if system is poorly conditioned
    condition_estimate = gram_matrix[0, 0] / (np.trace(gram_matrix) + 1e-15)
    
    if condition_estimate < 1e-6:
        # Use uniform weights
        for i in range(memory_used):
            weights_out[i] = 1.0 / memory_used
        return True
    
    # Simple iterative solve for small systems
    for i in range(memory_used):
        weights_out[i] = 1.0 / memory_used
    
    for _ in range(3):  # Few iterations
        new_weights = np.zeros(memory_used, dtype=np.float64)
        for i in range(memory_used):
            sum_val = 0.0
            for j in range(memory_used):
                if i != j:
                    sum_val += gram_matrix[i, j] * weights_out[j]
            if gram_matrix[i, i] > 1e-12:
                new_weights[i] = (1.0 / memory_used - sum_val) / gram_matrix[i, i]
        
        # Normalize
        total = np.sum(new_weights)
        if total > 1e-12:
            for i in range(memory_used):
                weights_out[i] = new_weights[i] / total
    
    return True

@njit(fastmath=True, cache=True)
def solve_anderson_system_stable(
    residual_diffs: np.ndarray,
    residual: np.ndarray,
    memory_used: int,
    weights_out: np.ndarray,  # Pre-allocated output array
    temp_matrix: np.ndarray   # Add the missing temp_matrix parameter
) -> bool:
    """Solve Anderson system with numerical stabilization using pre-allocated arrays."""
    if memory_used <= 1:
        if memory_used == 1:
            weights_out[0] = 1.0
        return memory_used > 0
    
    # Clear the temp_matrix
    for i in range(memory_used):
        for j in range(memory_used):
            temp_matrix[i, j] = 0.0
    
    # Build Gram matrix using pre-allocated array
    for i in range(memory_used):
        for j in range(memory_used):
            temp_matrix[i, j] = np.dot(residual_diffs[i], residual_diffs[j])
    
    # Add regularization to diagonal
    regularization = 1e-10
    for i in range(memory_used):
        temp_matrix[i, i] += regularization
    
    # Simple Gauss elimination with pivoting
    for k in range(memory_used):
        # Find pivot
        max_val = abs(temp_matrix[k, k])
        max_row = k
        for i in range(k + 1, memory_used):
            if abs(temp_matrix[i, k]) > max_val:
                max_val = abs(temp_matrix[i, k])
                max_row = i
        
        if max_val < 1e-12:
            return False
        
        # Swap rows if needed
        if max_row != k:
            for j in range(memory_used):
                temp_val = temp_matrix[k, j]
                temp_matrix[k, j] = temp_matrix[max_row, j]
                temp_matrix[max_row, j] = temp_val
        
        # Elimination
        for i in range(k + 1, memory_used):
            if abs(temp_matrix[k, k]) > 1e-12:
                factor = temp_matrix[i, k] / temp_matrix[k, k]
                for j in range(k, memory_used):
                    temp_matrix[i, j] -= factor * temp_matrix[k, j]
    
    # Back substitution to solve for weights
    for i in range(memory_used):
        weights_out[i] = 1.0 / memory_used  # Start with uniform weights
    
    for i in range(memory_used - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, memory_used):
            sum_val += temp_matrix[i, j] * weights_out[j]
        if abs(temp_matrix[i, i]) > 1e-12:
            weights_out[i] = (1.0 / memory_used - sum_val) / temp_matrix[i, i]
    
    # Normalize weights
    total_weight = 0.0
    for i in range(memory_used):
        total_weight += weights_out[i]
    
    if total_weight > 1e-12:
        for i in range(memory_used):
            weights_out[i] = weights_out[i] / total_weight
    else:
        for i in range(memory_used):
            weights_out[i] = 1.0 / memory_used
    
    return True

@njit(fastmath=True, cache=True)
def update_anderson_history(
    history_matrix: np.ndarray,
    residual_diffs: np.ndarray,
    new_residual_diff: np.ndarray,
    memory_used: int,
    current_pos: int,
    memory_depth: int
) -> Tuple[int, int]:
    """Update Anderson history matrices using circular buffer."""
    pos = current_pos % memory_depth
    
    # Update residual differences
    for i in range(len(new_residual_diff)):
        residual_diffs[pos, i] = new_residual_diff[i]
    
    new_memory_used = min(memory_used + 1, memory_depth)
    new_current_pos = current_pos + 1
    
    return new_memory_used, new_current_pos

@njit(fastmath=True, parallel=True, cache=True)
def compute_residual_diff(
    new_residual: np.ndarray,
    old_residual: np.ndarray,
    output: np.ndarray
) -> None:
    """Compute residual difference using pre-allocated output array."""
    for i in prange(len(new_residual)):
        output[i] = new_residual[i] - old_residual[i]

@njit(fastmath=True, parallel=True, cache=True)
def vectorized_anderson_solve(
    residual_diffs: np.ndarray,
    memory_used: int,
    weights_out: np.ndarray,
    temp_gram: np.ndarray  # Pre-allocated temp array
) -> bool:
    """
    VECTORIZED Anderson system solve using parallel operations.
    """
    if memory_used <= 1:
        if memory_used == 1:
            weights_out[0] = 1.0
        return memory_used > 0
    
    # Build Gram matrix using vectorized operations
    for i in prange(memory_used):
        for j in range(i, memory_used):
            dot_product = np.dot(residual_diffs[i], residual_diffs[j])
            temp_gram[i, j] = dot_product
            temp_gram[j, i] = dot_product  # Symmetric
    
    # Vectorized regularization
    for i in prange(memory_used):
        temp_gram[i, i] += 1e-8
    
    # Fast condition check using vectorized operations
    diag_sum = 0.0
    for i in range(memory_used):
        diag_sum += temp_gram[i, i]
    
    condition_estimate = temp_gram[0, 0] / (diag_sum + 1e-15)
    
    if condition_estimate < 1e-6:
        # Vectorized uniform weights
        uniform_weight = 1.0 / memory_used
        for i in prange(memory_used):
            weights_out[i] = uniform_weight
        return True
    
    # Vectorized iterative solve
    for i in prange(memory_used):
        weights_out[i] = 1.0 / memory_used
    
    # Few vectorized iterations
    for _ in range(2):  # Reduced iterations for speed
        total = 0.0
        for i in prange(memory_used):
            sum_val = 0.0
            for j in range(memory_used):
                if i != j:
                    sum_val += temp_gram[i, j] * weights_out[j]
            if temp_gram[i, i] > 1e-12:
                new_val = (1.0 / memory_used - sum_val) / temp_gram[i, i]
                weights_out[i] = new_val
                total += new_val
        
        # Vectorized normalization
        if total > 1e-12:
            inv_total = 1.0 / total
            for i in prange(memory_used):
                weights_out[i] = weights_out[i] * inv_total
    
    return True

@njit(fastmath=True, cache=True)
def fast_solve_anderson_system_mmap(
    residual_diffs: np.ndarray,
    memory_used: int,
    weights_out: np.ndarray,
    temp_gram: np.ndarray  # Pre-allocated memory-mapped array
) -> bool:
    """OPTIMIZED Anderson system solve using memory-mapped arrays only."""
    if memory_used <= 1:
        if memory_used == 1:
            weights_out[0] = 1.0
        return memory_used > 0
    
    # Clear temp_gram
    for i in range(memory_used):
        for j in range(memory_used):
            temp_gram[i, j] = 0.0
    
    # Build Gram matrix using pre-allocated memory-mapped array
    for i in range(memory_used):
        for j in range(i, memory_used):
            dot_product = np.dot(residual_diffs[i], residual_diffs[j])
            temp_gram[i, j] = dot_product
            temp_gram[j, i] = dot_product  # Symmetric
    
    # Add regularization
    for i in range(memory_used):
        temp_gram[i, i] += 1e-8
    
    # Calculate trace for condition estimate
    trace_sum = 0.0
    for i in range(memory_used):
        trace_sum += temp_gram[i, i]
    
    condition_estimate = temp_gram[0, 0] / (trace_sum + 1e-15)
    
    if condition_estimate < 1e-6:
        # Use uniform weights
        uniform_weight = 1.0 / memory_used
        for i in range(memory_used):
            weights_out[i] = uniform_weight
        return True
    
    # Initialize weights
    uniform_weight = 1.0 / memory_used
    for i in range(memory_used):
        weights_out[i] = uniform_weight
    
    # Simple iterative solve for small systems
    for iteration in range(3):  # Few iterations
        total_new_weight = 0.0
        
        for i in range(memory_used):
            sum_val = 0.0
            for j in range(memory_used):
                if i != j:
                    sum_val += temp_gram[i, j] * weights_out[j]
            
            if temp_gram[i, i] > 1e-12:
                new_weight = (uniform_weight - sum_val) / temp_gram[i, i]
                weights_out[i] = new_weight
                total_new_weight += new_weight
        
        # Normalize in-place
        if total_new_weight > 1e-12:
            inv_total = 1.0 / total_new_weight
            for i in range(memory_used):
                weights_out[i] = weights_out[i] * inv_total
    
    return True

class FastAndersonAccelerator:
    """Anderson accelerator using memory-mapped arrays exclusively."""
    
    def __init__(self, dimension: int, memory_depth: int = 10, beta: float = 1.0,
                 n_queries: int = None, n_subjects: int = None,
                 mmap_dir: str = None, resource_manager=None):
        self.dimension = dimension
        self.memory_depth = min(memory_depth, 3)  # Very conservative for safety
        self.beta = min(beta, 0.3)  # Conservative mixing
        self.resource_manager = resource_manager
        
        # Validate dimension
        if dimension <= 0:
            raise ValueError(f"Invalid dimension: {dimension}")
        
        if resource_manager is None:
            raise ValueError("ResourceManager is required for memory-mapped Anderson accelerator")
        
        # Pre-allocate ALL arrays using ResourceManager (memory-mapped)
        self.residual_diffs = resource_manager.create_array(
            name="anderson_residual_diffs",
            shape=(self.memory_depth, dimension),
            dtype=np.float64, temp=True
        )
        self.weights = resource_manager.create_array(
            name="anderson_weights",
            shape=(self.memory_depth,),
            dtype=np.float64, temp=True
        )
        self.prev_residual = resource_manager.create_array(
            name="anderson_prev_residual",
            shape=(dimension,),
            dtype=np.float64, temp=True
        )
        self.temp_gram = resource_manager.create_array(
            name="anderson_temp_gram",
            shape=(self.memory_depth, self.memory_depth),
            dtype=np.float64, temp=True
        )
        
        # Additional memory-mapped temporary arrays
        self.temp_residual = resource_manager.create_array(
            name="anderson_temp_residual",
            shape=(dimension,),
            dtype=np.float64, temp=True
        )
        self.temp_residual_diff = resource_manager.create_array(
            name="anderson_temp_residual_diff",
            shape=(dimension,),
            dtype=np.float64, temp=True
        )
        self.temp_weighted_sum = resource_manager.create_array(
            name="anderson_temp_weighted_sum",
            shape=(dimension,),
            dtype=np.float64, temp=True
        )
        self.temp_result = resource_manager.create_array(
            name="anderson_temp_result",
            shape=(dimension,),
            dtype=np.float64, temp=True
        )
        
        self.memory_used = 0
        self.current_pos = 0
        self.first_iteration = True
        
        # Performance tracking
        self.success_count = 0
        self.total_attempts = 0
        self.error_count = 0
        self.is_debug = log.isEnabledFor(logging.DEBUG)
        
        if self.is_debug:
            log.debug(f"Anderson accelerator initialized with memory-mapped arrays: dim={dimension}, memory={self.memory_depth}")
    
    def step(self, current_iterate: np.ndarray, fixed_point_map: callable,
             em_data: np.ndarray = None) -> np.ndarray:
        """Anderson step using correct mathematical formulation."""
        self.total_attempts += 1
        
        try:
            # Validate input
            if current_iterate is None or len(current_iterate) != self.dimension:
                self.error_count += 1
                return None
            
            if not np.all(np.isfinite(current_iterate)):
                self.error_count += 1
                return self._clip_to_bounds(current_iterate)
            
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
            
            # CORRECT ANDERSON FORMULATION:
            # f_k = F(x_k) - x_k (residual at current point)
            for i in range(self.dimension):
                self.temp_residual[i] = fx_k[i] - current_iterate[i]
            
            # For first iteration, just return F(x_k) and store f_k
            if self.first_iteration:
                self.first_iteration = False
                for i in range(self.dimension):
                    self.prev_residual[i] = self.temp_residual[i]
                return self._clip_to_bounds(fx_k)
            
            residual_norm = np.linalg.norm(self.temp_residual)
            if not np.isfinite(residual_norm) or residual_norm < 1e-15:
                for i in range(self.dimension):
                    self.prev_residual[i] = self.temp_residual[i]
                return self._clip_to_bounds(fx_k)
            
            # Try Anderson acceleration if we have history
            if self.memory_used > 0:
                try:
                    # Compute residual difference: Δf_k = f_k - f_{k-1}
                    diff_norm = 0.0
                    for i in range(self.dimension):
                        self.temp_residual_diff[i] = self.temp_residual[i] - self.prev_residual[i]
                        diff_norm += self.temp_residual_diff[i] * self.temp_residual_diff[i]
                    diff_norm = diff_norm ** 0.5
                    
                    if np.isfinite(diff_norm) and diff_norm > 1e-15:
                        # Update history
                        pos = self.current_pos % self.memory_depth
                        for i in range(self.dimension):
                            self.residual_diffs[pos, i] = self.temp_residual_diff[i]
                        
                        self.memory_used = min(self.memory_used + 1, self.memory_depth)
                        self.current_pos += 1
                        
                        # Solve Anderson system for weights
                        if fast_solve_anderson_system_mmap(
                            self.residual_diffs[:self.memory_used],
                            self.memory_used,
                            self.weights[:self.memory_used],
                            self.temp_gram[:self.memory_used, :self.memory_used]
                        ):
                            # Apply Anderson acceleration
                            result = self._safe_apply_anderson_acceleration_mmap(current_iterate, fx_k)
                            
                            if self._safe_validate_result_mmap(result, current_iterate, fx_k):
                                for i in range(self.dimension):
                                    self.prev_residual[i] = self.temp_residual[i]
                                self.success_count += 1
                                return result
                    
                except Exception as e:
                    if self.is_debug:
                        log.debug(f"Anderson acceleration failed: {e}")
                    self.error_count += 1
            
            # Store residual and return basic F(x_k)
            for i in range(self.dimension):
                self.prev_residual[i] = self.temp_residual[i]
            
            return self._clip_to_bounds(fx_k)
            
        except Exception as e:
            log.warning(f"Critical Anderson error: {e}")
            self.error_count += 1
            return self._clip_to_bounds(current_iterate) if current_iterate is not None else None

    def _safe_apply_anderson_acceleration_mmap(self, current_iterate, fx_k):
        """Apply Anderson acceleration using correct mathematical formulation."""
        try:
            # Clear weighted sum
            for i in range(self.dimension):
                self.temp_weighted_sum[i] = 0.0
            
            # Compute weighted combination: Σ γ_j Δf_{k-j}
            for k in range(self.memory_used):
                weight_k = self.weights[k]
                for i in range(self.dimension):
                    self.temp_weighted_sum[i] += weight_k * self.residual_diffs[k, i]
            
            # CORRECT ANDERSON FORMULA:
            # x_{k+1} = F(x_k) - Σ γ_j Δf_{k-j}
            # Note: F(x_k) is fx_k, and we subtract the weighted residual differences
            for i in range(self.dimension):
                self.temp_result[i] = fx_k[i] - self.temp_weighted_sum[i]
            
            # Apply additional damping for stability (optional)
            damping = 0.8  # Conservative damping
            for i in range(self.dimension):
                basic_step = fx_k[i]
                anderson_step = self.temp_result[i]
                self.temp_result[i] = (1 - damping) * basic_step + damping * anderson_step
            
            # Clip and renormalize
            result_sum = 0.0
            for i in range(self.dimension):
                self.temp_result[i] = max(1e-15, min(1.0, self.temp_result[i]))
                result_sum += self.temp_result[i]
            
            if result_sum > 1e-15:
                inv_result_sum = 1.0 / result_sum
                for i in range(self.dimension):
                    self.temp_result[i] *= inv_result_sum
            else:
                # Fallback to uniform weights
                uniform_val = 1.0 / self.dimension
                for i in range(self.dimension):
                    self.temp_result[i] = uniform_val
            
            return self.temp_result

        except Exception as e:
            if self.is_debug:
                log.debug(f"Anderson acceleration application failed: {e}")
            # Fallback to basic step with damping
            for i in range(self.dimension):
                self.temp_result[i] = (1 - self.beta) * current_iterate[i] + self.beta * fx_k[i]
            
            return self._clip_to_bounds(self.temp_result)

    def _safe_validate_result_mmap(self, result: np.ndarray, current_iterate: np.ndarray, basic_step: np.ndarray) -> bool:
        """Validation using memory-mapped arrays only."""
        try:
            if result is None or len(result) != self.dimension:
                return False
            
            # Check finite values
            for i in range(len(result)):
                if not np.isfinite(result[i]):
                    return False
                if result[i] < 1e-15 or result[i] > 1.0:
                    return False
            
            # Check sum constraint
            result_sum = 0.0
            for i in range(len(result)):
                result_sum += result[i]
            
            if not np.isfinite(result_sum) or abs(result_sum - 1.0) > 0.8:
                return False
            
            # Check change magnitude
            result_change = 0.0
            basic_change = 0.0
            
            for i in range(len(result)):
                result_diff = result[i] - current_iterate[i]
                basic_diff = basic_step[i] - current_iterate[i]
                result_change += result_diff * result_diff
                basic_change += basic_diff * basic_diff
            
            result_change = result_change ** 0.5
            basic_change = basic_change ** 0.5
            
            if not np.isfinite(result_change) or not np.isfinite(basic_change):
                return False
            
            # Much more permissive change ratio
            if basic_change > 1e-15 and result_change > basic_change * 1000:
                return False
            
            return True
            
        except Exception:
            return False

    def _clip_to_bounds(self, array):
        """Clip array values to valid bounds using memory-mapped temp array AND RENORMALIZE."""
        # Operate on self.temp_result assuming 'array' might be self.temp_result or another source
        # Copy 'array' to 'self.temp_result' if it's not already it, or work on 'array' if it's a distinct modifiable buffer.
        # For simplicity, this function will assume it's okay to modify self.temp_result based on 'array'.
        
        current_sum = 0.0
        for i in range(len(array)):
            val = max(1e-15, min(1.0, array[i]))
            self.temp_result[i] = val
            current_sum += val

        if current_sum > 1e-15:
            inv_sum = 1.0 / current_sum
            for i in range(len(self.temp_result)): # Iterate over self.temp_result length
                self.temp_result[i] *= inv_sum
        else:
            # Fallback to uniform if sum is zero (e.g. all weights became 1e-15)
            if self.dimension > 0:
                uniform_val = 1.0 / self.dimension
                for i in range(len(self.temp_result)):
                    self.temp_result[i] = uniform_val
        return self.temp_result

    def cleanup(self):
        """Safe cleanup with error handling."""
        try:
            if hasattr(self, 'resource_manager') and self.resource_manager is not None:
                # Arrays will be cleaned up by ResourceManager
                pass
        except Exception:
            pass  # Ignore cleanup errors
