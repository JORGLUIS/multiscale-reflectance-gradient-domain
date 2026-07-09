import numpy as np

def ssq_error(correct: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute the sum-squared-error for a patch/image, where the estimate is
    multiplied by a scalar which minimizes the error. Sums over all pixels
    where mask is True.
    
    Args:
        correct (np.ndarray): Ground truth image patch (2D).
        estimate (np.ndarray): Estimated image patch (2D).
        mask (np.ndarray): Binary mask patch (2D).
        
    Returns:
        float: Scaled sum-squared-error.
    """
    assert correct.ndim == 2, "Inputs must be 2D arrays"
    assert estimate.ndim == 2, "Inputs must be 2D arrays"
    assert mask.ndim == 2, "Mask must be a 2D array"
    
    # Calculate scale factor alpha that minimizes ||alpha * estimate - correct||^2 on mask
    denom = np.sum(estimate**2 * mask)
    if denom > 1e-5:
        alpha = np.sum(correct * estimate * mask) / denom
    else:
        alpha = 0.0
        
    return float(np.sum(mask * (correct - alpha * estimate) ** 2))

def local_error(correct: np.ndarray, estimate: np.ndarray, mask: np.ndarray, 
                window_size: int, window_shift: int) -> float:
    """
    Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error.
    
    Args:
        correct (np.ndarray): Ground truth image (2D).
        estimate (np.ndarray): Estimated image (2D).
        mask (np.ndarray): Binary mask (2D).
        window_size (int): Size of the local window.
        window_shift (int): Shift step between overlapping windows.
        
    Returns:
        float: Local Mean Squared Error.
    """
    M, N = correct.shape[:2]
    ssq = 0.0
    total = 0.0
    
    for i in range(0, M - window_size + 1, window_shift):
        for j in range(0, N - window_size + 1, window_shift):
            correct_curr = correct[i:i+window_size, j:j+window_size]
            estimate_curr = estimate[i:i+window_size, j:j+window_size]
            mask_curr = mask[i:i+window_size, j:j+window_size]
            
            ssq += ssq_error(correct_curr, estimate_curr, mask_curr)
            total += np.sum(mask_curr * (correct_curr ** 2))
            
    if total > 1e-5:
        return float(ssq / total)
    return 0.0

def score_image(true_shading: np.ndarray, true_refl: np.ndarray, 
                estimate_shading: np.ndarray, estimate_refl: np.ndarray, 
                mask: np.ndarray, window_size: int = 20) -> float:
    """
    Computes the average local error (LMSE) for reflectance and shading components.
    
    Args:
        true_shading (np.ndarray): Ground truth shading (2D).
        true_refl (np.ndarray): Ground truth reflectance (2D).
        estimate_shading (np.ndarray): Estimated shading (2D).
        estimate_refl (np.ndarray): Estimated reflectance (2D).
        mask (np.ndarray): Binary mask (2D).
        window_size (int): Size of the local window. Default is 20 (approx 10% of MIT image size).
        
    Returns:
        float: Combined LMSE score (average of shading LMSE and reflectance LMSE).
    """
    # Convert input parameters to float64 for numerical precision
    true_shading = true_shading.astype(np.float64)
    true_refl = true_refl.astype(np.float64)
    estimate_shading = estimate_shading.astype(np.float64)
    estimate_refl = estimate_refl.astype(np.float64)
    mask = mask.astype(bool)
    
    window_shift = window_size // 2
    
    err_shading = local_error(true_shading, estimate_shading, mask, window_size, window_shift)
    err_refl = local_error(true_refl, estimate_refl, mask, window_size, window_shift)
    
    return float(0.5 * err_shading + 0.5 * err_refl)
