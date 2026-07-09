import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple

def rgb_to_luminance(rgb_image: np.ndarray) -> np.ndarray:
    """
    Computes luminance from an RGB image using standard weights.
    
    Args:
        rgb_image (np.ndarray): Color image of shape [H, W, 3].
        
    Returns:
        np.ndarray: Grayscale luminance image of shape [H, W].
    """
    return 0.2989 * rgb_image[..., 0] + 0.5870 * rgb_image[..., 1] + 0.1140 * rgb_image[..., 2]

def homomorphic_decomposition(image: np.ndarray, mask: np.ndarray, 
                              sigma: float = 15.0, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs homomorphic decomposition (log -> Gaussian low-pass -> exp).
    Supports grayscale (2D) or color (3D) images.
    
    Args:
        image (np.ndarray): Input image, shape [H, W] or [H, W, 3], in range [0, 1].
        mask (np.ndarray): Boolean mask of shape [H, W].
        sigma (float): Standard deviation of the Gaussian filter for low-pass shading extraction.
        eps (float): Small constant to avoid log(0) and division by zero.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Estimated shading S [H, W].
            - Estimated reflectance R [H, W] or [H, W, 3].
    """
    is_color = (image.ndim == 3)
    
    # 1. Get luminance representation
    if is_color:
        L = rgb_to_luminance(image)
    else:
        L = image.copy()
        
    # 2. Go to log-luminance
    # Apply mask before log to avoid issues, though clipping/eps handles general cases
    L_masked = np.clip(L, eps, 1.0)
    log_L = np.log(L_masked)
    
    # 3. Apply Gaussian low-pass filter to log-luminance to get log-shading
    # The mask tells us which region is valid. 
    # For a simple baseline, we can run gaussian_filter on the log-luminance.
    # Note: to avoid border leakage from background (which is 0), we can fill background
    # with mean foreground log-luminance or use normalized convolution.
    # A simple but effective method is to fill masked-out areas with the mean foreground value.
    if not np.all(mask):
        fg_mean = np.mean(log_L[mask]) if np.any(mask) else 0.0
        log_L_filled = np.where(mask, log_L, fg_mean)
    else:
        log_L_filled = log_L
        
    log_S = gaussian_filter(log_L_filled, sigma=sigma, mode='reflect')
    
    # 4. Reflectance in log-domain: r = l - s
    log_R = log_L - log_S
    
    # 5. Convert back to linear domain
    S = np.exp(log_S)
    R_L = np.exp(log_R)
    
    # Recompose color if input was color
    if is_color:
        # Chromaticity: C = image / (L + eps)
        # Recomposed color reflectance: R = C * R_L
        C = image / (L[..., np.newaxis] + eps)
        R = C * R_L[..., np.newaxis]
    else:
        R = R_L
        
    # Apply mask to output
    S = S * mask
    if is_color:
        R = R * mask[..., np.newaxis]
    else:
        R = R * mask
        
    return S, R
