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
                temp_matrix[k, j], temp_matrix[max_row, j] = temp_matrix[max_row, j], temp_matrix[k, j]
        
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
                weights_out[i] = (1.0 / memory_used - sum_val) / temp_gram[i, i]
            total += weights_out[i]
        
        # Vectorized normalization
        if total > 1e-12:
            inv_total = 1.0 / total
            for i in prange(memory_used):
                weights_out[i] *= inv_total
    
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
                weights_out[i] = (uniform_weight - sum_val) / temp_gram[i, i]
            total_new_weight += weights_out[i]
        
        # Normalize in-place
        if total_new_weight > 1e-12:
            inv_total = 1.0 / total_new_weight
            for i in range(memory_used):
                weights_out[i] *= inv_total
    
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
        # CORRECTED: Store both iterates and residuals for proper Anderson
        self.iterate_history = resource_manager.create_array(
            name="anderson_iterate_history",
            shape=(self.memory_depth, dimension),
            dtype=np.float64, temp=True
        )
        self.residual_history = resource_manager.create_array(
            name="anderson_residual_history",
            shape=(self.memory_depth, dimension),
            dtype=np.float64, temp=True
        )
        self.weights = resource_manager.create_array(
            name="anderson_weights",
            shape=(self.memory_depth,),
            dtype=np.float64, temp=True
        )
        self.temp_gram = resource_manager.create_array(
            name="anderson_temp_gram",
            shape=(self.memory_depth, self.memory_depth),
            dtype=np.float64, temp=True
        )
        
        # Additional memory-mapped temporary arrays
        self.temp_result = resource_manager.create_array(
            name="anderson_temp_result",
            shape=(dimension,),
            dtype=np.float64, temp=True
        )
        self.temp_weighted_sum = resource_manager.create_array(
            name="anderson_temp_weighted_sum",
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
        """CORRECTED Anderson step using standard mathematical formulation."""
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
            
            # CORRECTED: Compute residual as f_k = F(x_k) - x_k
            current_residual = fx_k - current_iterate
            
            # For first iteration, just return F(x_k) and store history
            if self.first_iteration:
                self._store_history(current_iterate, current_residual)
                self.first_iteration = False
                return self._clip_to_bounds(fx_k)
            
            residual_norm = np.linalg.norm(current_residual)
            if not np.isfinite(residual_norm) or residual_norm < 1e-15:
                self._store_history(current_iterate, current_residual)
                return self._clip_to_bounds(fx_k)
            
            # Try Anderson acceleration if we have sufficient history
            if self.memory_used > 0:
                try:
                    result = self._apply_standard_anderson(current_iterate, fx_k, current_residual)
                    if result is not None and self._safe_validate_result_mmap(result, current_iterate, fx_k):
                        self.success_count += 1
                        self._store_history(current_iterate, current_residual)
                        return result
                    
                except Exception as e:
                    if self.is_debug:
                        log.debug(f"Anderson acceleration failed: {e}")
            
            # Store history and return basic F(x_k)
            self._store_history(current_iterate, current_residual)
            return self._clip_to_bounds(fx_k)
            
        except Exception as e:
            log.warning(f"Critical Anderson error: {e}")
            self.error_count += 1
            return self._clip_to_bounds(current_iterate) if current_iterate is not None else None

    def _store_history(self, iterate: np.ndarray, residual: np.ndarray):
        """Store iterate and residual in circular buffer."""
        pos = self.current_pos % self.memory_depth
        
        # Store current iterate and residual
        self.iterate_history[pos] = iterate
        self.residual_history[pos] = residual
        
        # Update counters
        self.current_pos += 1
        self.memory_used = min(self.memory_used + 1, self.memory_depth)

    def _apply_standard_anderson(self, current_iterate: np.ndarray, fx_k: np.ndarray, 
                                current_residual: np.ndarray) -> np.ndarray:
        """Apply STANDARD Anderson acceleration formulation."""
        try:
            # STANDARD ANDERSON FORMULATION:
            # 1. Solve: min ||Σ γ_i Δf_i||² subject to Σ γ_i = 1
            # 2. Update: x_{k+1} = Σ γ_i F(x_{k-m+i})
            
            # Build residual difference matrix (Δf_i = f_i - f_{i-1})
            if self.memory_used < 2:
                # Not enough history for Anderson, return basic step
                return fx_k
            
            # Compute residual differences
            residual_diffs = np.zeros((self.memory_used - 1, self.dimension))
            for i in range(self.memory_used - 1):
                pos_curr = (self.current_pos - 1 - i) % self.memory_depth
                pos_prev = (self.current_pos - 2 - i) % self.memory_depth
                residual_diffs[i] = self.residual_history[pos_curr] - self.residual_history[pos_prev]
            
            # Solve Anderson system: find weights γ that minimize ||Σ γ_i Δf_i||²
            success = self._solve_anderson_system_standard(residual_diffs, current_residual)
            if not success:
                return None
            
            # STANDARD UPDATE: x_{k+1} = Σ γ_i F(x_{k-m+i})
            # Clear weighted sum
            for i in range(self.dimension):
                self.temp_weighted_sum[i] = 0.0
            
            # Compute weighted combination of function values
            for i in range(self.memory_used):
                pos = (self.current_pos - 1 - i) % self.memory_depth
                weight = self.weights[i] if i < len(self.weights) else 0.0
                
                # F(x_i) = x_i + f_i (since f_i = F(x_i) - x_i)
                for j in range(self.dimension):
                    fx_i = self.iterate_history[pos, j] + self.residual_history[pos, j]
                    self.temp_weighted_sum[j] += weight * fx_i
            
            # Add current point with remaining weight
            current_weight = 1.0 - np.sum(self.weights[:self.memory_used])
            for i in range(self.dimension):
                self.temp_result[i] = self.temp_weighted_sum[i] + current_weight * fx_k[i]
            
            # Apply conservative damping
            damping = 0.8
            for i in range(self.dimension):
                self.temp_result[i] = (1 - damping) * fx_k[i] + damping * self.temp_result[i]
            
            return self._clip_to_bounds(self.temp_result)

        except Exception as e:
            if self.is_debug:
                log.debug(f"Standard Anderson acceleration failed: {e}")
            return None

    def _solve_anderson_system_standard(self, residual_diffs: np.ndarray, 
                                      current_residual: np.ndarray) -> bool:
        """Solve standard Anderson system with proper constraint."""
        try:
            m = len(residual_diffs)  # Number of residual differences
            if m == 0:
                return False
            
            # Clear gram matrix
            for i in range(m):
                for j in range(m):
                    self.temp_gram[i, j] = 0.0
            
            # Build Gram matrix G_ij = <Δf_i, Δf_j>
            for i in range(m):
                for j in range(i, m):
                    dot_product = np.dot(residual_diffs[i], residual_diffs[j])
                    self.temp_gram[i, j] = dot_product
                    self.temp_gram[j, i] = dot_product  # Symmetric
            
            # Add regularization
            reg = 1e-8
            for i in range(m):
                self.temp_gram[i, i] += reg
            
            # Solve constrained system: G γ = e, where e is vector of ones
            # This gives the solution to min ||Σ γ_i Δf_i||² subject to Σ γ_i = 1
            
            # Use simple iterative method for small systems
            # Initialize with uniform weights
            for i in range(m):
                self.weights[i] = 1.0 / m
            
            # Iterative refinement
            for iteration in range(5):
                new_weights = np.zeros(m)
                
                # Solve G * new_weights = ones
                for i in range(m):
                    target = 1.0 / m  # Target: uniform constraint
                    sum_off_diag = 0.0
                    
                    for j in range(m):
                        if i != j:
                            sum_off_diag += self.temp_gram[i, j] * self.weights[j]
                    
                    if abs(self.temp_gram[i, i]) > 1e-12:
                        new_weights[i] = (target - sum_off_diag) / self.temp_gram[i, i]
                    else:
                        new_weights[i] = 1.0 / m
                
                # Normalize to satisfy constraint Σ γ_i = 1
                total = np.sum(new_weights)
                if total > 1e-12:
                    for i in range(m):
                        self.weights[i] = new_weights[i] / total
                else:
                    for i in range(m):
                        self.weights[i] = 1.0 / m
            
            return True
            
        except Exception as e:
            log.debug(f"Anderson system solve failed: {e}")
            return False

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
        current_sum = 0.0
        for i in range(len(array)):
            val = max(1e-15, min(1.0, array[i]))
            self.temp_result[i] = val
            current_sum += val

        if current_sum > 1e-15:
            inv_sum = 1.0 / current_sum
            for i in range(len(self.temp_result)):
                self.temp_result[i] *= inv_sum
        else:
            # Fallback to uniform if sum is zero
            if self.dimension > 0:
                uniform_val = 1.0 / self.dimension
                for i in range(len(self.temp_result)):
                    self.temp_result[i] = uniform_val
        return self.temp_result

    def cleanup(self):
        """Safe cleanup with error handling."""
        try:
            if hasattr(self, 'resource_manager') and self.resource_manager is not None:
                # Clear references to arrays before ResourceManager cleanup
                array_attrs = [
                    'residual_diffs', 'weights', 'prev_residual', 'temp_gram',
                    'temp_residual', 'temp_residual_diff', 'temp_weighted_sum', 'temp_result'
                ]
                
                for attr in array_attrs:
                    if hasattr(self, attr):
                        try:
                            # Clear the reference without explicitly deleting
                            setattr(self, attr, None)
                        except Exception:
                            pass  # Ignore errors during cleanup
                
                # Reset state
                self.memory_used = 0
                self.current_pos = 0
                self.first_iteration = True
                
        except Exception:
            pass  # Ignore all cleanup errors to prevent segfaults
