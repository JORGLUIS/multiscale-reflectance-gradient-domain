import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple
from src.baselines.homomorphic import rgb_to_luminance

def ssr_decomposition(image: np.ndarray, mask: np.ndarray, 
                      sigma: float = 15.0, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Single-Scale Retinex (SSR) decomposition.
    Estimates shading S in linear domain using Gaussian blur,
    and computes reflectance R = I / S.
    Supports grayscale (2D) or color (3D) images.
    
    Args:
        image (np.ndarray): Input image, shape [H, W] or [H, W, 3], in range [0, 1].
        mask (np.ndarray): Boolean mask of shape [H, W].
        sigma (float): Standard deviation of the Gaussian filter.
        eps (float): Small constant to avoid division by zero.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Estimated shading S [H, W].
            - Estimated reflectance R [H, W] or [H, W, 3].
    """
    is_color = (image.ndim == 3)
    
    # 1. Get luminance
    if is_color:
        L = rgb_to_luminance(image)
    else:
        L = image.copy()
        
    # 2. Fill background of mask with mean foreground value to prevent leakage
    if not np.all(mask):
        fg_mean = np.mean(L[mask]) if np.any(mask) else 0.5
        L_filled = np.where(mask, L, fg_mean)
    else:
        L_filled = L
        
    # 3. Estimate shading as the blurred luminance in linear domain
    S = gaussian_filter(L_filled, sigma=sigma, mode='reflect')
    S = np.clip(S, eps, 1.0)
    
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
