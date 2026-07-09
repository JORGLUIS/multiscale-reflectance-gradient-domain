import numpy as np
from typing import List, Tuple

def atrous_median_filter2d(image: np.ndarray, step: int, kernel_size: int = 3) -> np.ndarray:
    """
    Applies a 2D à trous median filter to an image with a given spacing step.
    
    Args:
        image (np.ndarray): 2D input array of shape [H, W].
        step (int): Spacing between kernel samples (2^j for scale j).
        kernel_size (int): Size of the median filter window (e.g. 3 or 5).
        
    Returns:
        np.ndarray: Filtered image of shape [H, W].
    """
    H, W = image.shape
    r = kernel_size // 2
    pad_width = r * step
    
    # Reflect padding to handle boundary conditions
    padded = np.pad(image, pad_width, mode='reflect')
    
    # Collect shifted versions of the image corresponding to the kernel window
    shifted_slices = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            y_start = pad_width + dy * step
            x_start = pad_width + dx * step
            shifted_slices.append(padded[y_start:y_start + H, x_start:x_start + W])
            
    # Stack slices and compute median along the stack axis
    stacked = np.stack(shifted_slices, axis=0)
    return np.median(stacked, axis=0)

def mmt_decompose(image: np.ndarray, levels: int, kernel_size: int = 3) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Performs a 2D Multiscale Median Transform (MMT) decomposition on an image.
    Supports both 2D (grayscale) and 3D (color) arrays.
    
    Args:
        image (np.ndarray): Input image of shape [H, W] or [H, W, C].
        levels (int): Number of decomposition levels L.
        kernel_size (int): Median filter kernel window size (usually 3 or 5).
        
    Returns:
        Tuple[List[np.ndarray], np.ndarray]:
            - List of L detail coefficient arrays, each of shape matching input.
            - Final approximation residual array, shape matching input.
    """
    coeffs = []
    c_j = image.astype(np.float64)
    
    for j in range(levels):
        step = 2 ** j
        if c_j.ndim == 2:
            c_next = atrous_median_filter2d(c_j, step, kernel_size)
        else:
            # Multi-channel image: process each channel separately
            c_next = np.empty_like(c_j)
            for c in range(c_j.shape[2]):
                c_next[..., c] = atrous_median_filter2d(c_j[..., c], step, kernel_size)
                
        w_j = c_j - c_next
        coeffs.append(w_j)
        c_j = c_next
        
    return coeffs, c_j

def mmt_reconstruct(coeffs: List[np.ndarray], residual: np.ndarray) -> np.ndarray:
    """
    Reconstructs an image from its Multiscale Median Transform (MMT) coefficients.
    
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
