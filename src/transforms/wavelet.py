import numpy as np
import pywt
from typing import Tuple, List, Union

def wavelet_decompose(image: np.ndarray, levels: int, wavelet_name: str = "db2") -> Tuple[list, Tuple[int, int]]:
    """
    Performs a 2D Stationary Wavelet Transform (SWT) decomposition on a grayscale image.
    Applies reflect padding so that image dimensions are multiples of 2^levels.
    
    Args:
        image (np.ndarray): 2D input image of shape [H, W].
        levels (int): Number of decomposition levels L.
        wavelet_name (str): Wavelet family name (e.g., "db2", "haar", "bior1.3").
        
    Returns:
        Tuple[list, Tuple[int, int]]:
            - List of coefficients: [(cA_L, (cH_L, cV_L, cD_L)), ..., (cH_1, cV_1, cD_1)].
            - Original shape of the image: (H, W).
    """
    H, W = image.shape
    factor = 2 ** levels
    pad_H = ((H + factor - 1) // factor) * factor
    pad_W = ((W + factor - 1) // factor) * factor
    
    pad_y = pad_H - H
    pad_x = pad_W - W
    
    # Pad using reflect mode to handle boundary conditions smoothly
    padded = np.pad(image, ((0, pad_y), (0, pad_x)), mode='reflect')
    
    coeffs = pywt.swt2(padded, wavelet_name, level=levels)
    return coeffs, (H, W)

def wavelet_reconstruct(coeffs: list, wavelet_name: str = "db2", original_shape: Tuple[int, int] = None) -> np.ndarray:
    """
    Reconstructs a 2D image from its SWT coefficients using inverse SWT.
    Crops the output to the original shape if provided.
    
    Args:
        coeffs (list): Coeffs list: [(cA_L, (cH_L, cV_L, cD_L)), ..., (cH_1, cV_1, cD_1)].
        wavelet_name (str): Wavelet family name.
        original_shape (Tuple[int, int]): If provided, crops output to shape (H, W).
        
    Returns:
        np.ndarray: Reconstructed 2D image.
    """
    padded_reconstructed = pywt.iswt2(coeffs, wavelet_name)
    if original_shape is not None:
        H, W = original_shape
        return padded_reconstructed[:H, :W]
    return padded_reconstructed
