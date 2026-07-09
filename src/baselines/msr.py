import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, List, Optional
from src.baselines.homomorphic import rgb_to_luminance

def msr_decomposition(image: np.ndarray, mask: np.ndarray, 
                      sigmas: List[float] = [15.0, 80.0, 250.0], 
                      weights: Optional[List[float]] = None, 
                      eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Multi-Scale Retinex (MSR) decomposition.
    Estimates shading S as the weighted geometric mean of linear blurred versions at multiple scales.
    Computes reflectance R = I / S.
    Supports grayscale (2D) or color (3D) images.
    
    Args:
        image (np.ndarray): Input image, shape [H, W] or [H, W, 3], in range [0, 1].
        mask (np.ndarray): Boolean mask of shape [H, W].
        sigmas (List[float]): List of sigmas for the multi-scale Gaussian filters.
        weights (List[float], optional): Weights for each scale. If None, uses equal weights.
        eps (float): Small constant to avoid log(0) and division by zero.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Estimated shading S [H, W].
            - Estimated reflectance R [H, W] or [H, W, 3].
    """
    is_color = (image.ndim == 3)
    num_scales = len(sigmas)
    
    if weights is None:
        weights = [1.0 / num_scales] * num_scales
    else:
        # Normalize weights
        sum_w = sum(weights)
        weights = [w / sum_w for w in weights]
        
    # 1. Get luminance
    if is_color:
        L = rgb_to_luminance(image)
    else:
        L = image.copy()
        
    # 2. Fill background of mask with mean foreground value
    if not np.all(mask):
        fg_mean = np.mean(L[mask]) if np.any(mask) else 0.5
        L_filled = np.where(mask, L, fg_mean)
    else:
        L_filled = L
        
    # 3. Compute log-shading as weighted sum of log-blurred scales
    log_S = np.zeros_like(L)
    for sigma, weight in zip(sigmas, weights):
        S_n = gaussian_filter(L_filled, sigma=sigma, mode='reflect')
        S_n_clipped = np.clip(S_n, eps, 1.0)
        log_S += weight * np.log(S_n_clipped)
        
    S = np.exp(log_S)
    
    # 4. Compute reflectance R = I / S
    if is_color:
        R = image / S[..., np.newaxis]
    else:
        R = image / S
        
    # Apply mask
    S = S * mask
    if is_color:
        R = R * mask[..., np.newaxis]
    else:
        R = R * mask
        
    return S, R
