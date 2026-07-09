import numpy as np
from typing import Tuple, List
from src.transforms.starlet import starlet_decompose
from src.baselines.homomorphic import rgb_to_luminance
from src.transforms.wavelet import wavelet_decompose, wavelet_reconstruct

def median_absolute_deviation(arr: np.ndarray, mask: np.ndarray) -> float:
    """
    Computes the Median Absolute Deviation (MAD) of an array within a mask.
    
    Args:
        arr (np.ndarray): Input array.
        mask (np.ndarray): Boolean mask.
        
    Returns:
        float: The MAD value.
    """
    valid_vals = arr[mask]
    if len(valid_vals) == 0:
        return 0.0
    med = np.median(valid_vals)
    return float(np.median(np.abs(valid_vals - med)))

def multiscale_decomposition(image: np.ndarray, mask: np.ndarray, 
                            levels: int = 3, threshold_factor: float = 2.0, 
                            transform_type: str = "starlet",
                            scale_coherence: bool = False,
                            color_coherence: bool = False,
                            color_threshold: float = 0.05,
                            color_beta: float = 0.5,
                            eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs multiscale intrinsic image decomposition using Starlet, MMT, or Wavelet transform
    with advanced magnitude, scale-coherence, and color-coherence attribution rules.
    Supports grayscale (2D) or color (3D) images.
    
    Args:
        image (np.ndarray): Input image, shape [H, W] or [H, W, 3], in range [0, 1].
        mask (np.ndarray): Boolean mask of shape [H, W].
        levels (int): Number of decomposition levels L.
        threshold_factor (float): Multiplier for the MAD threshold.
        transform_type (str): Type of transform ("starlet", "mmt", or "wavelet").
        scale_coherence (bool): If True, applies inter-scale coherence constraint.
        color_coherence (bool): If True, uses chromaticity gradient to modulate threshold.
        color_threshold (float): Multiplier for chromaticity gradient MAD to find edges.
        color_beta (float): Fraction to reduce threshold in color edge regions.
        eps (float): Small constant to avoid log(0) and division by zero.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Estimated shading S [H, W].
            - Estimated reflectance R [H, W] or [H, W, 3].
    """
    is_color = (image.ndim == 3)
    H_orig, W_orig = image.shape[:2]
    
    # 1. Get luminance
    if is_color:
        L = rgb_to_luminance(image)
    else:
        L = image.copy()
        
    # 2. Go to log-luminance
    L_masked = np.clip(L, eps, 1.0)
    log_L = np.log(L_masked)
    
    # Fill background with mean foreground log-luminance to prevent border leakage
    if not np.all(mask):
        fg_mean = np.mean(log_L[mask]) if np.any(mask) else 0.0
        log_L_filled = np.where(mask, log_L, fg_mean)
    else:
        log_L_filled = log_L
        
    # 3. Perform decomposition on the filled log-luminance
    if transform_type == "starlet":
        coeffs, residual = starlet_decompose(log_L_filled, levels=levels)
    elif transform_type == "mmt":
        from src.transforms.mmt import mmt_decompose
        coeffs, residual = mmt_decompose(log_L_filled, levels=levels)
    elif transform_type == "wavelet":
        coeffs_pywt, orig_shape = wavelet_decompose(log_L_filled, levels=levels, wavelet_name="db2")
    else:
        raise ValueError(f"Unknown transform_type '{transform_type}'")
        
    # 4. Compute chromaticity edges if color_coherence is enabled
    if is_color and color_coherence:
        C = image / (L[..., np.newaxis] + eps)
        grad_C_x = np.zeros_like(C)
        grad_C_y = np.zeros_like(C)
        grad_C_x[:, :-1, :] = C[:, 1:, :] - C[:, :-1, :]
        grad_C_y[:-1, :, :] = C[1:, :, :] - C[:-1, :, :]
        # L2 norm over color channels
        grad_C_mag = np.sqrt(np.sum(grad_C_x**2 + grad_C_y**2, axis=-1))
        
        # Compute threshold based on MAD of chromaticity gradient
        grad_C_mad = median_absolute_deviation(grad_C_mag, mask)
        T_color = color_threshold * grad_C_mad if grad_C_mad > 1e-6 else color_threshold
        is_color_edge = (grad_C_mag > T_color)
    else:
        is_color_edge = None
        
    # 5. Apply advanced attribution rules per level
    if transform_type in ["starlet", "mmt"]:
        log_R = np.zeros_like(log_L)
        log_S = residual.copy()
        
        # Precompute standard noise thresholds (MAD) per scale
        mads = [median_absolute_deviation(w_j, mask) for w_j in coeffs]
        T = [max(threshold_factor * mad, 1e-6) for mad in mads]
        
        for j in range(levels):
            w_j = coeffs[j]
            T_j = T[j]
            
            # Color coherence: modulate threshold dynamically
            if is_color_edge is not None:
                T_j_local = np.where(is_color_edge, T_j * color_beta, T_j)
            else:
                T_j_local = T_j
                
            # Basic magnitude-based edge check
            is_reflectance_edge = (np.abs(w_j) > T_j_local)
            
            # Scale coherence check: verify detail persistence across scales
            if scale_coherence and levels > 1:
                if j < levels - 1:
                    T_next = T[j+1]
                    T_next_local = np.where(is_color_edge, T_next * color_beta, T_next) if is_color_edge is not None else T_next
                    # Check if the next scale also has details above 0.5 * threshold
                    is_reflectance_edge = is_reflectance_edge & (np.abs(coeffs[j+1]) > 0.5 * T_next_local)
                else:
                    T_prev = T[j-1]
                    T_prev_local = np.where(is_color_edge, T_prev * color_beta, T_prev) if is_color_edge is not None else T_prev
                    # Check if the previous scale has details above 0.5 * threshold
                    is_reflectance_edge = is_reflectance_edge & (np.abs(coeffs[j-1]) > 0.5 * T_prev_local)
                    
            # Attribute coefficients
            log_R += np.where(is_reflectance_edge, w_j, 0.0)
            log_S += np.where(is_reflectance_edge, 0.0, w_j)
            
        # Convert back to linear domain
        S = np.exp(log_S)
        R_L = np.exp(log_R)
        
    else:  # transform_type == "wavelet"
        # coeffs_pywt is list: [(cA_L, (cH_L, cV_L, cD_L)), ..., (cA_1, (cH_1, cV_1, cD_1))]
        # Extract H, V, D tuples in order from L to 1
        detail_tuples = [item[1] for item in coeffs_pywt]
            
        # Reverse to order from level 1 (finest) to level L (coarsest)
        detail_tuples.reverse()
        
        # Calculate local magnitude for each level
        w_mags = [np.sqrt(w[0]**2 + w[1]**2 + w[2]**2) for w in detail_tuples]
        
        # Compute thresholds based on MAD of cropped magnitudes
        mads = [median_absolute_deviation(w_j[:H_orig, :W_orig], mask) for w_j in w_mags]
        T = [max(threshold_factor * mad, 1e-6) for mad in mads]
        
        # We need to map is_color_edge to padded space if color coherence is enabled
        is_color_edge_padded = None
        if is_color_edge is not None:
            pad_H, pad_W = w_mags[0].shape
            is_color_edge_padded = np.zeros((pad_H, pad_W), dtype=bool)
            is_color_edge_padded[:H_orig, :W_orig] = is_color_edge
            
        refl_details = []
        shading_details = []
        
        for j in range(levels):
            w_j = w_mags[j]
            T_j = T[j]
            w_H, w_V, w_D = detail_tuples[j]
            
            # Color coherence: modulate threshold dynamically
            if is_color_edge_padded is not None:
                T_j_local = np.where(is_color_edge_padded, T_j * color_beta, T_j)
            else:
                T_j_local = T_j
                
            # Basic magnitude-based edge check
            is_reflectance_edge = (w_j > T_j_local)
            
            # Scale coherence check: verify detail persistence across scales
            if scale_coherence and levels > 1:
                if j < levels - 1:
                    T_next = T[j+1]
                    T_next_local = np.where(is_color_edge_padded, T_next * color_beta, T_next) if is_color_edge_padded is not None else T_next
                    is_reflectance_edge = is_reflectance_edge & (w_mags[j+1] > 0.5 * T_next_local)
                else:
                    T_prev = T[j-1]
                    T_prev_local = np.where(is_color_edge_padded, T_prev * color_beta, T_prev) if is_color_edge_padded is not None else T_prev
                    is_reflectance_edge = is_reflectance_edge & (w_mags[j-1] > 0.5 * T_prev_local)
                    
            # Separate details into reflectance and shading
            refl_H = np.where(is_reflectance_edge, w_H, 0.0)
            refl_V = np.where(is_reflectance_edge, w_V, 0.0)
            refl_D = np.where(is_reflectance_edge, w_D, 0.0)
            
            shading_H = np.where(is_reflectance_edge, 0.0, w_H)
            shading_V = np.where(is_reflectance_edge, 0.0, w_V)
            shading_D = np.where(is_reflectance_edge, 0.0, w_D)
            
            refl_details.append((refl_H, refl_V, refl_D))
            shading_details.append((shading_H, shading_V, shading_D))
            
        # Reverse details back to match PyWavelets swt2 structure (L to 1)
        refl_details.reverse()
        shading_details.reverse()
        
        # Build coefficient structures for reconstruction
        refl_coeffs = []
        shading_coeffs = []
        
        for k in range(levels):
            refl_coeffs.append((np.zeros_like(coeffs_pywt[k][0]), refl_details[k]))
            shading_coeffs.append((coeffs_pywt[k][0], shading_details[k]))
            
        # Reconstruct padded representations
        log_R_padded = wavelet_reconstruct(refl_coeffs, wavelet_name="db2")
        log_S_padded = wavelet_reconstruct(shading_coeffs, wavelet_name="db2")
        
        # Crop back to original shape and convert to linear domain
        R_L = np.exp(log_R_padded[:H_orig, :W_orig])
        S = np.exp(log_S_padded[:H_orig, :W_orig])
        
    # Recompose color if color input
    if is_color:
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
