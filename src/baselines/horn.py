import numpy as np
from typing import Tuple
from src.utils.poisson import get_gradients, solve, solve_L1
from src.baselines.homomorphic import rgb_to_luminance

def horn_decomposition(image: np.ndarray, mask: np.ndarray, 
                       threshold: float = 0.1, L1: bool = False, 
                       clip_val: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Retinex-Horn decomposition (Poisson equation solver on thresholded gradients).
    Supports grayscale (2D) or color (3D) images.
    
    Args:
        image (np.ndarray): Input image, shape [H, W] or [H, W, 3], in range [0, 1].
        mask (np.ndarray): Boolean mask of shape [H, W].
        threshold (float): Threshold value for detail gradient selection.
        L1 (bool): If True, uses L1 sparse penalty solver. If False, uses least squares.
        clip_val (float): Minimum value to clip image intensities before taking log.
        
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
        
    # 2. Go to log-luminance with clipping to avoid log(0)
    L_clipped = np.clip(L, clip_val, np.inf)
    log_L = np.where(mask, np.log(L_clipped), 0.0)
    
    # 3. Compute gradients of log-luminance
    i_y, i_x = get_gradients(log_L)
    
    # 4. Threshold gradients: gradients larger than threshold are kept as reflectance
    # (Horn assumes large gradients are reflectance changes, small gradients are shading)
    r_y = np.where(np.abs(i_y) > threshold, i_y, 0.0)
    r_x = np.where(np.abs(i_x) > threshold, i_x, 0.0)
    
    # 5. Solve Poisson equation to reconstruct log-reflectance
    if L1:
        log_R_L = solve_L1(r_y, r_x, mask)
    else:
        log_R_L = solve(r_y, r_x, mask)
        
    R_L = mask * np.exp(log_R_L)
    R_L = np.clip(R_L, 0.0, 1.0)
    
    # 6. Shading is estimated as linear division: S = L / R
    # (or S = image / R, and mask out)
    # We clip R_L to avoid division by zero
    S = np.where(mask, L / (R_L + 1e-5), 0.0)
    S = np.clip(S, 0.0, 1.0)
    
    # Recompose color if color input
    if is_color:
        C = image / (L[..., np.newaxis] + 1e-5)
        R = C * R_L[..., np.newaxis]
    else:
        R = R_L
        
    # Apply mask
    S = S * mask
    if is_color:
        R = R * mask[..., np.newaxis]
    else:
        R = R * mask
        
    return S, R
