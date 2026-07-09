import numpy as np
from skimage.metrics import structural_similarity


def scale_to_reference(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Ajusta la escala de una estimación al ground-truth dentro de la máscara.
    """
    denom = np.sum(estimate ** 2 * mask)
    if denom <= 1e-5:
        return np.zeros_like(estimate)

    alpha = np.sum(reference * estimate * mask) / denom
    return alpha * estimate


def masked_psnr(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    """
    Calcula PSNR sobre la zona válida después de alinear escala.
    """
    aligned = scale_to_reference(reference, estimate, mask)
    mse = np.sum(mask * (reference - aligned) ** 2) / np.sum(mask)
    if mse <= 1e-12:
        return 99.0

    return float(10.0 * np.log10(1.0 / mse))


def masked_ssim(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    """
    Calcula SSIM rellenando el fondo con el ground-truth para no penalizar la máscara.
    """
    aligned = scale_to_reference(reference, estimate, mask)
    reference_filled = np.where(mask, reference, 0.0)
    estimate_filled = np.where(mask, aligned, reference_filled)
    return float(structural_similarity(reference_filled, estimate_filled, data_range=1.0))


def gradient_sparsity(image: np.ndarray, mask: np.ndarray, threshold: float = 0.02) -> float:
    """
    Fracción de píxeles válidos con gradiente pequeño en la reflectancia.
    """
    grad_y = np.zeros_like(image)
    grad_x = np.zeros_like(image)
    grad_y[:-1, :] = image[1:, :] - image[:-1, :]
    grad_x[:, :-1] = image[:, 1:] - image[:, :-1]

    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    valid = mask.astype(bool)
    if not np.any(valid):
        return 0.0

    return float(np.mean(grad_mag[valid] <= threshold))
