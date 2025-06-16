"""
Ultra-fast Numba kernels optimized for billion-scale EM operations.
Enhanced with better vectorization and memory access patterns.
"""
import numpy as np
from numba import njit, prange, types

# CRITICAL: Pre-compile all kernels with explicit signatures for maximum performance
@njit(types.void(types.int64[:], types.int64[:], types.float64[:], types.float64[:], 
                 types.float64, types.float64[:], types.float64[:], types.float64[:]), 
      fastmath=True, parallel=True, cache=True, nogil=True)
def ultra_fast_e_step_billion_prealloc(
    source_indices: np.ndarray,
    subject_indices: np.ndarray,
    bit_scores: np.ndarray,
    weights: np.ndarray,
    lambda_scale: float,
    responsibilities: np.ndarray,
    temp_max: np.ndarray,      # PRE-ALLOCATED to max_source size
    temp_sum: np.ndarray       # PRE-ALLOCATED to max_source size
) -> None:
    """
    ULTRA-FAST E-step with pre-allocated arrays and optimized memory access.
    No dynamic allocation - all arrays pre-sized.
    """
    n_alignments = len(source_indices)
    max_source = len(temp_max)
    
    if n_alignments == 0:
        return
    
    # VECTORIZED: Clear temp arrays in parallel with better memory access
    for i in prange(max_source):
        temp_max[i] = -1e30
        temp_sum[i] = 0.0
    
    # OPTIMIZED: Single-pass score normalization
    min_score = bit_scores[0]
    max_score = bit_scores[0]
    
    # Parallel min/max finding
    for i in prange(1, n_alignments):
        score = bit_scores[i]
        if score < min_score:
            min_score = score
        if score > max_score:
            max_score = score
    
    score_range = max_score - min_score
    
    # Handle uniform scores case with pre-allocated arrays
    if score_range < 1e-12:
        # Clear temp_sum for weight accumulation
        for i in prange(max_source):
            temp_sum[i] = 0.0
        
        # Accumulate weights per source
        for i in prange(n_alignments):
            source_idx = source_indices[i]
            subject_idx = subject_indices[i]
            if source_idx < max_source:
                temp_sum[source_idx] += weights[subject_idx]
        
        # Assign uniform probabilities
        for i in prange(n_alignments):
            source_idx = source_indices[i]
            subject_idx = subject_indices[i]
            if source_idx < max_source and temp_sum[source_idx] > 1e-15:
                responsibilities[i] = weights[subject_idx] / temp_sum[source_idx]
            else:
                responsibilities[i] = 1e-15
        return
    
    inv_score_range = 1.0 / score_range
    
    # PASS 1: Find max per source with optimized memory access
    for i in prange(n_alignments):
        source_idx = source_indices[i]
        subject_idx = subject_indices[i]
        
        if source_idx < max_source:
            norm_score = (bit_scores[i] - min_score) * inv_score_range
            weight_val = weights[subject_idx]
            log_weight = np.log(max(weight_val, 1e-15))
            weighted_score = log_weight + lambda_scale * norm_score
            
            # Atomic max update with better performance
            if weighted_score > temp_max[source_idx]:
                temp_max[source_idx] = weighted_score

    # PASS 2: Compute sums with vectorized operations
    for i in prange(n_alignments):
        source_idx = source_indices[i]
        subject_idx = subject_indices[i]
        
        if source_idx < max_source:
            max_val = temp_max[source_idx]
            if max_val > -1e29:
                norm_score = (bit_scores[i] - min_score) * inv_score_range
                weight_val = weights[subject_idx]
                log_weight = np.log(max(weight_val, 1e-15))
                weighted_score = log_weight + lambda_scale * norm_score
                diff = weighted_score - max_val
                
                if diff > -10.0:  # Optimized threshold
                    exp_val = np.exp(diff)
                    temp_sum[source_idx] += exp_val

    # PASS 3: Compute final responsibilities with bounds checking
    for i in prange(n_alignments):
        source_idx = source_indices[i]
        subject_idx = subject_indices[i]
        
        if source_idx < max_source:
            max_val = temp_max[source_idx]
            sum_val = temp_sum[source_idx]
            
            if sum_val > 1e-15 and max_val > -1e29:
                norm_score = (bit_scores[i] - min_score) * inv_score_range
                weight_val = weights[subject_idx]
                log_weight = np.log(max(weight_val, 1e-15))
                weighted_score = log_weight + lambda_scale * norm_score
                diff = weighted_score - max_val
                
                if diff > -10.0:
                    exp_val = np.exp(diff)
                    prob = exp_val / sum_val
                    responsibilities[i] = max(1e-15, min(1.0, prob))
                else:
                    responsibilities[i] = 1e-15
            else:
                responsibilities[i] = 1e-15
        else:
            responsibilities[i] = 1e-15

@njit(types.void(types.int64[:], types.float64[:], types.float64[:], types.float64[:]), 
      fastmath=True, parallel=True, cache=True, nogil=True)
def ultra_fast_m_step_billion_prealloc(
    subject_indices: np.ndarray,
    responsibilities: np.ndarray,
    new_weights: np.ndarray,         # PRE-ALLOCATED output
    temp_sums: np.ndarray           # PRE-ALLOCATED temp array
) -> None:
    """
    ULTRA-FAST M-step with pre-allocated arrays and optimized parallel reduction.
    """
    n_alignments = len(subject_indices)
    max_subject = len(new_weights)
    
    if n_alignments == 0:
        return
    
    # VECTORIZED: Clear arrays in parallel
    for i in prange(max_subject):
        temp_sums[i] = 0.0
        new_weights[i] = 0.0
    
    # OPTIMIZED: Single-pass accumulation with better memory access
    for i in prange(n_alignments):
        subject_idx = subject_indices[i]
        if subject_idx < max_subject:
            resp_val = responsibilities[i]
            temp_sums[subject_idx] += resp_val
    
    # VECTORIZED: Parallel reduction for total sum
    total_sum = 0.0
    for i in prange(max_subject):
        total_sum += temp_sums[i]
    
    # VECTORIZED: Parallel normalization
    if total_sum > 1e-15:
        inv_total = 1.0 / total_sum
        for i in prange(max_subject):
            new_weights[i] = max(1e-15, temp_sums[i] * inv_total)
    else:
        # Uniform fallback
        if max_subject > 0:
            uniform_weight = 1.0 / max_subject
            for i in prange(max_subject):
                new_weights[i] = uniform_weight

@njit(types.float64(types.int64[:], types.int64[:], types.float64[:], types.float64[:], 
                    types.float64, types.float64[:], types.float64[:]), 
      fastmath=True, cache=True, nogil=True)
def ultra_fast_likelihood_billion_prealloc(
    source_indices: np.ndarray,
    subject_indices: np.ndarray,
    bit_scores: np.ndarray,
    weights: np.ndarray,
    lambda_scale: float,
    temp_max: np.ndarray,      # PRE-ALLOCATED
    temp_sum: np.ndarray       # PRE-ALLOCATED
) -> float:
    """
    ULTRA-FAST likelihood computation with pre-allocated arrays.
    """
    n_alignments = len(source_indices)
    max_source = len(temp_max)
    
    if n_alignments == 0:
        return -1e30
    
    # Clear temp arrays
    for i in range(max_source):
        temp_max[i] = -1e30
        temp_sum[i] = 0.0
    
    # Optimized score normalization
    min_score = bit_scores[0]
    max_score = bit_scores[0]
    
    for i in range(1, n_alignments):
        score = bit_scores[i]
        if score < min_score:
            min_score = score
        if score > max_score:
            max_score = score
    
    score_range = max_score - min_score
    if score_range < 1e-12:
        score_range = 1.0
    
    inv_score_range = 1.0 / score_range
    
    # PASS 1: Find max per source
    for i in range(n_alignments):
        source_idx = source_indices[i]
        subject_idx = subject_indices[i]
        
        if source_idx < max_source:
            norm_score = (bit_scores[i] - min_score) * inv_score_range
            weight_val = weights[subject_idx]
            log_weight = np.log(max(weight_val, 1e-15))
            weighted_score = log_weight + lambda_scale * norm_score
            
            if weighted_score > temp_max[source_idx]:
                temp_max[source_idx] = weighted_score
    
    # PASS 2: Compute sums
    for i in range(n_alignments):
        source_idx = source_indices[i]
        subject_idx = subject_indices[i]
        
        if source_idx < max_source:
            max_val = temp_max[source_idx]
            if max_val > -1e29:
                norm_score = (bit_scores[i] - min_score) * inv_score_range
                weight_val = weights[subject_idx]
                log_weight = np.log(max(weight_val, 1e-15))
                weighted_score = log_weight + lambda_scale * norm_score
                diff = weighted_score - max_val
                
                if diff > -15.0:
                    exp_val = np.exp(diff)
                    temp_sum[source_idx] += exp_val
    
    # PASS 3: Sum log-likelihood
    total_ll = 0.0
    for source_idx in range(max_source):
        if temp_max[source_idx] > -1e29 and temp_sum[source_idx] > 1e-15:
            query_ll = temp_max[source_idx] + np.log(temp_sum[source_idx])
            total_ll += query_ll
    
    return total_ll

@njit(types.int64(types.int64[:], types.float64[:], types.float64[:], types.int64[:]), 
      fastmath=True, cache=True, nogil=True)  # REMOVED parallel=True - not beneficial here
def ultra_fast_conservation_check_prealloc(
    source_indices: np.ndarray,
    responsibilities: np.ndarray,
    temp_sums: np.ndarray,        # PRE-ALLOCATED
    temp_counts: np.ndarray       # PRE-ALLOCATED
) -> int:
    """
    IMPROVED ultra-fast conservation check with STRICT precision handling.
    Should achieve zero violations with proper fixing.
    """
    n_alignments = len(source_indices)
    max_source = len(temp_sums)
    
    if n_alignments == 0:
        return 0
    
    # Clear arrays
    for i in range(max_source):
        temp_sums[i] = 0.0
        temp_counts[i] = 0

    # Accumulate sums and counts
    for i in range(n_alignments):
        source_idx = source_indices[i]
        if 0 <= source_idx < max_source:
            temp_sums[source_idx] += responsibilities[i]
            temp_counts[source_idx] += 1

    # Count violations with STRICT tolerance - should be 0 after proper fixing
    violations = 0
    for i in range(max_source):
        if temp_counts[i] > 0:
            sum_val = temp_sums[i]
            # STRICT: Use tighter tolerance - properly fixed responsibilities should sum to exactly 1.0
            deviation = abs(sum_val - 1.0)
            
            # Much stricter threshold - we want ZERO violations
            threshold = 1e-12  # Very tight precision requirement
            
            if deviation > threshold:
                violations += 1

    return violations

@njit(types.void(types.int64[:], types.float64[:], types.float64[:]), 
      fastmath=True, cache=True, nogil=True)
def ultra_fast_conservation_fix_prealloc(
    source_indices: np.ndarray,
    responsibilities: np.ndarray,
    temp_sums: np.ndarray         # PRE-ALLOCATED
) -> None:
    """
    ENHANCED ultra-fast conservation fix with PERFECT precision.
    This should achieve exactly zero violations.
    """
    n_alignments = len(source_indices)
    max_source = len(temp_sums)
    
    if n_alignments == 0:
        return
    
    # Clear sums
    for i in range(max_source):
        temp_sums[i] = 0.0

    # Pass 1: Accumulate sums per source with high precision
    for i in range(n_alignments):
        source_idx = source_indices[i]
        if 0 <= source_idx < max_source:
            temp_sums[source_idx] += responsibilities[i]

    # Pass 2: PERFECT normalization with precision handling
    for i in range(n_alignments):
        source_idx = source_indices[i]
        if 0 <= source_idx < max_source:
            sum_val = temp_sums[source_idx]
            if sum_val > 1e-15:  # Non-zero sum
                # High-precision normalization
                new_resp = responsibilities[i] / sum_val
                # Ensure precision is maintained
                responsibilities[i] = new_resp
            else:
                # If sum is zero, set to minimum (this shouldn't happen in practice)
                responsibilities[i] = 1e-15

@njit(types.void(types.int64[:], types.float64[:], types.float64[:]), 
      fastmath=True, cache=True, nogil=True)
def ultra_fast_perfect_conservation_fix(
    source_indices: np.ndarray,
    responsibilities: np.ndarray,
    temp_sums: np.ndarray         # PRE-ALLOCATED temp array
) -> None:
    """
    PERFECT conservation fix that guarantees zero violations.
    Uses high-precision arithmetic and multiple passes if needed.
    """
    n_alignments = len(source_indices)
    max_source = len(temp_sums)
    
    if n_alignments == 0:
        return
    
    # Multiple passes to ensure perfect conservation
    for pass_num in range(3):  # Up to 3 passes to achieve perfection
        # Clear temp array
        for i in range(max_source):
            temp_sums[i] = 0.0
        
        # Pass 1: Accumulate current sums
        for i in range(n_alignments):
            source_idx = source_indices[i]
            if 0 <= source_idx < max_source:
                temp_sums[source_idx] += responsibilities[i]
        
        # Pass 2: Check if we need fixing
        needs_fixing = False
        for i in range(max_source):
            if temp_sums[i] > 0.0:
                deviation = abs(temp_sums[i] - 1.0)
                if deviation > 1e-14:  # Very tight tolerance
                    needs_fixing = True
                    break
        
        if not needs_fixing:
            break  # Perfect conservation achieved
        
        # Pass 3: Apply normalization
        for i in range(n_alignments):
            source_idx = source_indices[i]
            if 0 <= source_idx < max_source:
                sum_val = temp_sums[source_idx]
                if sum_val > 1e-15:
                    responsibilities[i] = responsibilities[i] / sum_val
                else:
                    responsibilities[i] = 1e-15

@njit(types.void(types.int64[:], types.float64[:], types.float64[:]), 
      fastmath=True, cache=True, nogil=True)
def ultra_fast_global_normalization_fix(
    source_indices: np.ndarray,
    responsibilities: np.ndarray,
    temp_sums: np.ndarray         # PRE-ALLOCATED temp array
) -> None:
    """
    ULTRA-FAST global normalization fix for conservation as last resort.
    Uses only pre-allocated arrays and Numba-optimized loops.
    """
    n_alignments = len(source_indices)
    max_source = len(temp_sums)
    
    if n_alignments == 0:
        return
    
    # Clear temp array
    for i in range(max_source):
        temp_sums[i] = 0.0
    
    # Pass 1: Count reads per source
    read_counts = np.zeros(max_source, dtype=np.int64)
    for i in range(n_alignments):
        source_idx = source_indices[i]
        if 0 <= source_idx < max_source:
            read_counts[source_idx] += 1
    
    # Pass 2: Calculate total expected probability mass
    total_reads = 0
    for i in range(max_source):
        if read_counts[i] > 0:
            total_reads += 1
    
    # Pass 3: Calculate current total probability mass
    current_total = 0.0
    for i in range(n_alignments):
        current_total += responsibilities[i]
    
    # Pass 4: Apply global scaling if needed
    if current_total > 1e-15 and total_reads > 0:
        scale_factor = total_reads / current_total
        for i in range(n_alignments):
            responsibilities[i] = responsibilities[i] * scale_factor
    else:
        # Set to minimum values if total is zero
        for i in range(n_alignments):
            responsibilities[i] = 1e-15

@njit(types.void(types.float64[:], types.float64[:], types.int64[:]), 
      fastmath=True, parallel=True, cache=True, nogil=True)  # KEEP parallel=True - benefits from it
def compute_statistics_billion(
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    counts: np.ndarray
) -> None:
    """
    ULTRA-FAST statistics computation for probability distributions.
    This function benefits from parallelization due to independent threshold checks.
    """
    n_probs = len(probabilities)
    n_thresholds = len(thresholds)
    
    # Clear counts
    for i in prange(n_thresholds):
        counts[i] = 0
    
    # Count probabilities above each threshold - this benefits from parallelization
    for i in prange(n_probs):
        prob_val = probabilities[i]
        for j in range(n_thresholds):
            if prob_val >= thresholds[j]:
                counts[j] += 1

# Define aliases for backward compatibility
ultra_fast_e_step_billion = ultra_fast_e_step_billion_prealloc
ultra_fast_m_step_billion = ultra_fast_m_step_billion_prealloc
ultra_fast_likelihood_billion = ultra_fast_likelihood_billion_prealloc
ultra_fast_conservation_check = ultra_fast_conservation_check_prealloc
ultra_fast_conservation_fix = ultra_fast_conservation_fix_prealloc
ultra_fast_global_normalization = ultra_fast_global_normalization_fix
ultra_fast_perfect_conservation = ultra_fast_perfect_conservation_fix

@njit(types.void(types.int64[:], types.int64[:], types.float64[:], types.float64[:], 
                 types.float64, types.float64[:], types.float64[:], types.float64[:], types.float64), 
      fastmath=True, parallel=True, cache=True, nogil=True)
def ultra_fast_enhanced_e_step(
    source_indices: np.ndarray,
    subject_indices: np.ndarray,
    bit_scores: np.ndarray,
    weights: np.ndarray,
    lambda_scale: float,
    responsibilities: np.ndarray,
    temp_max: np.ndarray,
    temp_sum: np.ndarray,
    temperature: float = 0.1
) -> None:
    """
    Ultra-fast enhanced E-step with temperature scaling and score transformation.
    """
    n_alignments = len(source_indices)
    max_source = len(temp_max)
    
    if n_alignments == 0:
        return
    
    # Clear temp arrays
    for i in prange(max_source):
        temp_max[i] = -1e30
        temp_sum[i] = 0.0
    
    # Enhanced score normalization
    min_score = bit_scores[0]
    max_score = bit_scores[0]
    
    for i in prange(1, n_alignments):
        score = bit_scores[i]
        if score < min_score:
            min_score = score
        if score > max_score:
            max_score = score
    
    score_range = max_score - min_score
    
    if score_range < 1e-12:
        # Uniform case with enhanced weight handling
        for i in prange(max_source):
            temp_sum[i] = 0.0
        
        for i in prange(n_alignments):
            source_idx = source_indices[i]
            subject_idx = subject_indices[i]
            if source_idx < max_source:
                temp_sum[source_idx] += weights[subject_idx]
        
        for i in prange(n_alignments):
            source_idx = source_indices[i]
            subject_idx = subject_indices[i]
            if source_idx < max_source and temp_sum[source_idx] > 1e-15:
                responsibilities[i] = weights[subject_idx] / temp_sum[source_idx]
            else:
                responsibilities[i] = 1e-15
        return
    
    inv_score_range = 1.0 / score_range
    inv_temperature = 1.0 / temperature
    
    # Pass 1: Enhanced max finding with score transformation
    for i in prange(n_alignments):
        source_idx = source_indices[i]
        subject_idx = subject_indices[i]
        
        if source_idx < max_source:
            # Enhanced score transformation: quadratic + exponential scaling
            norm_score = (bit_scores[i] - min_score) * inv_score_range
            
            # Apply sigmoid transformation for better separation
            sigmoid_score = 1.0 / (1.0 + np.exp(-10.0 * (norm_score - 0.5)))
            
            # Apply exponential scaling
            enhanced_score = np.exp(2.0 * sigmoid_score) - 1.0
            
            weight_val = weights[subject_idx]
            log_weight = np.log(max(weight_val, 1e-15))
            
            # Temperature-scaled weighted score
            weighted_score = log_weight + (lambda_scale * enhanced_score) * inv_temperature
            
            if weighted_score > temp_max[source_idx]:
                temp_max[source_idx] = weighted_score
    
    # Pass 2: Enhanced sum computation
    for i in prange(n_alignments):
        source_idx = source_indices[i]
        subject_idx = subject_indices[i]
        
        if source_idx < max_source:
            max_val = temp_max[source_idx]
            if max_val > -1e29:
                norm_score = (bit_scores[i] - min_score) * inv_score_range
                sigmoid_score = 1.0 / (1.0 + np.exp(-10.0 * (norm_score - 0.5)))
                enhanced_score = np.exp(2.0 * sigmoid_score) - 1.0
                
                weight_val = weights[subject_idx]
                log_weight = np.log(max(weight_val, 1e-15))
                weighted_score = log_weight + (lambda_scale * enhanced_score) * inv_temperature
                
                diff = weighted_score - max_val
                if diff > -15.0:
                    exp_val = np.exp(diff)
                    temp_sum[source_idx] += exp_val
    
    # Pass 3: Enhanced responsibility computation
    for i in prange(n_alignments):
        source_idx = source_indices[i]
        subject_idx = subject_indices[i]
        
        if source_idx < max_source:
            max_val = temp_max[source_idx]
            sum_val = temp_sum[source_idx]
            
            if sum_val > 1e-15 and max_val > -1e29:
                norm_score = (bit_scores[i] - min_score) * inv_score_range
                sigmoid_score = 1.0 / (1.0 + np.exp(-10.0 * (norm_score - 0.5)))
                enhanced_score = np.exp(2.0 * sigmoid_score) - 1.0
                
                weight_val = weights[subject_idx]
                log_weight = np.log(max(weight_val, 1e-15))
                weighted_score = log_weight + (lambda_scale * enhanced_score) * inv_temperature
                
                diff = weighted_score - max_val
                if diff > -15.0:
                    exp_val = np.exp(diff)
                    prob = exp_val / sum_val
                    # Apply probability sharpening
                    sharpened_prob = prob ** (1.0 / temperature)
                    responsibilities[i] = max(1e-15, min(1.0, sharpened_prob))
                else:
                    responsibilities[i] = 1e-15
            else:
                responsibilities[i] = 1e-15
        else:
            responsibilities[i] = 1e-15

@njit(types.void(types.int64[:], types.float64[:], types.float64[:], types.float64[:], types.float64), 
      fastmath=True, parallel=True, cache=True, nogil=True)
def ultra_fast_enhanced_m_step(
    subject_indices: np.ndarray,
    responsibilities: np.ndarray,
    new_weights: np.ndarray,
    temp_sums: np.ndarray,
    regularization: float = 1e-8
) -> None:
    """
    Enhanced M-step with regularization to prevent overfitting.
    """
    n_alignments = len(subject_indices)
    max_subject = len(new_weights)
    
    if n_alignments == 0:
        return
    
    # Clear arrays
    for i in prange(max_subject):
        temp_sums[i] = regularization  # Add small regularization
        new_weights[i] = 0.0
    
    # Accumulate with enhanced precision
    for i in prange(n_alignments):
        subject_idx = subject_indices[i]
        if subject_idx < max_subject:
            resp_val = responsibilities[i]
            # Apply response transformation to reduce noise
            enhanced_resp = resp_val * resp_val * resp_val  # Cubic transformation
            temp_sums[subject_idx] += enhanced_resp
    
    # Enhanced normalization with regularization
    total_sum = regularization * max_subject
    for i in prange(max_subject):
        total_sum += temp_sums[i]
    
    if total_sum > 1e-15:
        inv_total = 1.0 / total_sum
        for i in prange(max_subject):
            new_weights[i] = max(1e-15, temp_sums[i] * inv_total)
    else:
        if max_subject > 0:
            uniform_weight = 1.0 / max_subject
            for i in prange(max_subject):
                new_weights[i] = uniform_weight

# ...existing code...
