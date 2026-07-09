import numpy as np
from scipy.ndimage import convolve1d
from typing import List, Tuple

def get_starlet_filter(scale_idx: int) -> np.ndarray:
    """
    Constructs the 1D B3-spline à trous filter for a given scale index.
    The filter has non-zero elements separated by 2^(scale_idx) - 1 zeros.
    
    Args:
        scale_idx (int): The scale level (0-indexed).
        
    Returns:
        np.ndarray: The 1D filter weights.
    """
    step = 2 ** scale_idx
    filter_len = 4 * step + 1
    h = np.zeros(filter_len, dtype=np.float64)
    h[0] = 0.0625       # 1/16
    h[step] = 0.25      # 1/4
    h[2 * step] = 0.375 # 3/8
    h[3 * step] = 0.25  # 1/4
    h[4 * step] = 0.0625# 1/16
    return h

def starlet_decompose(image: np.ndarray, levels: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Performs 2D Starlet (à trous) decomposition on an image.
    Supports 2D (grayscale) or 3D (color) arrays.
    
    Args:
        image (np.ndarray): Input image, shape [H, W] or [H, W, C].
        levels (int): Number of decomposition levels L.
        
    Returns:
        Tuple[List[np.ndarray], np.ndarray]:
            - List of L detail coefficient arrays, each of shape matching input.
            - Final approximation residual array, shape matching input.
    """
    coeffs = []
    c_j = image.astype(np.float64)
    
    for j in range(levels):
        h = get_starlet_filter(j)
        
        # Convolve separably along axis 0 and axis 1
        # If image is 3D, convolve1d handles axis correctly when iterating or specifying axis
        c_next = convolve1d(c_j, h, axis=0, mode='reflect')
        c_next = convolve1d(c_next, h, axis=1, mode='reflect')
        
        # Detail coefficients w_{j+1} = c_j - c_{j+1}
        w_j = c_j - c_next
        coeffs.append(w_j)
        
        c_j = c_next
        
    return coeffs, c_j

def starlet_reconstruct(coeffs: List[np.ndarray], residual: np.ndarray) -> np.ndarray:
    """
    Reconstructs an image from its Starlet decomposition coefficients.
    
    Args:
        coeffs (List[np.ndarray]): List of L detail coefficient arrays.
        residual (np.ndarray): Final approximation residual array.
        
    Returns:
        np.ndarray: Reconstructed image of shape matching the coefficients.
    """
    reconstructed = residual.copy()
    for w_j in coeffs:
        reconstructed += w_j
    return reconstructed
