"""
src/eval/diagnostico.py
Diagnostico de separabilidad de la regla de atribucion por magnitud.

Mide si la magnitud de los coeficientes multiescala discrimina reflectancia de shading
usando el ground truth de MIT y Sintel, en vez de asumirlo por herencia del denoising wavelet.
"""

import numpy as np
from scipy.stats import rankdata
from typing import Dict, List, Tuple, Optional

from src.baselines.homomorphic import rgb_to_luminance
from src.transforms.starlet import starlet_decompose
from src.transforms.mmt import mmt_decompose


def log_component(component: np.ndarray, mask: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Convierte una componente (reflectancia, shading o imagen observada) a log-luminancia,
    con el mismo relleno de fondo (media del foreground) que usa multiscale_decomposition.

    Args:
        component (np.ndarray): Imagen [H, W] o [H, W, 3] en rango [0, 1].
        mask (np.ndarray): Mascara booleana [H, W] del foreground.
        eps (float): Constante para evitar log(0).

    Returns:
        np.ndarray: Log-luminancia [H, W], con el fondo relleno con la media del foreground.
    """
    L = rgb_to_luminance(component) if component.ndim == 3 else component.copy()
    L = np.clip(L, eps, 1.0)
    log_L = np.log(L)
    if not np.all(mask):
        fg_mean = np.mean(log_L[mask]) if np.any(mask) else 0.0
        log_L = np.where(mask, log_L, fg_mean)
    return log_L


def decompose_log_component(log_component_filled: np.ndarray, transform_type: str,
                             levels: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Aplica la transformada multiescala (starlet o MMT) a una log-luminancia ya rellenada.

    Args:
        log_component_filled (np.ndarray): Log-luminancia [H, W], fondo ya relleno.
        transform_type (str): "starlet" o "mmt".
        levels (int): Numero de niveles L.

    Returns:
        Tuple[List[np.ndarray], np.ndarray]: Coeficientes de detalle por escala (fino a grueso)
            y el residuo de aproximacion final.
    """
    if transform_type == "starlet":
        return starlet_decompose(log_component_filled, levels=levels)
    elif transform_type == "mmt":
        return mmt_decompose(log_component_filled, levels=levels)
    else:
        raise ValueError(f"transform_type '{transform_type}' no soportado en el diagnostico")


def decompose_gt_components(R_gt: np.ndarray, S_gt: np.ndarray, mask: np.ndarray,
                             transform_type: str, levels: int,
                             eps: float = 1e-5) -> Dict[str, object]:
    """
    Descompone por separado el ground truth de reflectancia y de shading en coeficientes
    multiescala, para poder etiquetar cada coeficiente de la imagen observada por dominancia.

    Nota: para MMT la aditividad w_I = w_R + w_S no es exacta porque la mediana no es lineal
    (usar additivity_error_per_scale con los coeficientes de la imagen observada para cuantificarlo).

    Args:
        R_gt (np.ndarray): Reflectancia ground truth [H, W] o [H, W, 3], rango [0, 1].
        S_gt (np.ndarray): Shading ground truth [H, W], rango [0, 1].
        mask (np.ndarray): Mascara booleana [H, W] del foreground.
        transform_type (str): "starlet" o "mmt".
        levels (int): Numero de niveles L.
        eps (float): Constante para evitar log(0).

    Returns:
        Dict[str, object]: coeffs_R, residual_R, coeffs_S, residual_S.
    """
    log_R = log_component(R_gt, mask, eps)
    log_S = log_component(S_gt, mask, eps)
    coeffs_R, residual_R = decompose_log_component(log_R, transform_type, levels)
    coeffs_S, residual_S = decompose_log_component(log_S, transform_type, levels)
    return {
        "coeffs_R": coeffs_R,
        "residual_R": residual_R,
        "coeffs_S": coeffs_S,
        "residual_S": residual_S,
    }


def additivity_error_per_scale(coeffs_I: List[np.ndarray], coeffs_R: List[np.ndarray],
                                coeffs_S: List[np.ndarray], mask: np.ndarray) -> List[float]:
    """
    Error relativo de aditividad w_I ~ w_R + w_S por escala. Exacto (cercano a cero) para
    starlet, aproximado para MMT porque la mediana no es un operador lineal.

    Args:
        coeffs_I (List[np.ndarray]): Coeficientes de la imagen observada, por escala.
        coeffs_R (List[np.ndarray]): Coeficientes del ground truth de reflectancia, por escala.
        coeffs_S (List[np.ndarray]): Coeficientes del ground truth de shading, por escala.
        mask (np.ndarray): Mascara booleana [H, W] del foreground.

    Returns:
        List[float]: Norma relativa de (w_I - w_R - w_S) dentro de la mascara, por escala.
    """
    errors = []
    for w_I, w_R, w_S in zip(coeffs_I, coeffs_R, coeffs_S):
        diff = w_I[mask] - w_R[mask] - w_S[mask]
        denom = np.linalg.norm(w_I[mask]) + 1e-12
        errors.append(float(np.linalg.norm(diff) / denom))
    return errors


def label_dominance(w_R: np.ndarray, w_S: np.ndarray) -> np.ndarray:
    """
    Etiqueta binaria por coeficiente segun cual componente del ground truth domina.

    Args:
        w_R (np.ndarray): Coeficientes de detalle del ground truth de reflectancia, una escala.
        w_S (np.ndarray): Coeficientes de detalle del ground truth de shading, misma escala.

    Returns:
        np.ndarray: Mascara booleana, True si domina reflectancia (|w_R| >= |w_S|).
    """
    return np.abs(w_R) >= np.abs(w_S)


def _auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    AUC de un clasificador escalar via el estadistico de Mann-Whitney (sin sklearn).

    Args:
        scores (np.ndarray): Puntajes continuos (ya restringidos a la mascara).
        labels (np.ndarray): Etiquetas booleanas (ya restringidas a la mascara).

    Returns:
        float: AUC en [0, 1], o NaN si una de las clases esta vacia.
    """
    n_pos = int(np.sum(labels))
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    sum_ranks_pos = np.sum(ranks[labels])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def magnitude_auc_per_scale(w_I: List[np.ndarray], labels: List[np.ndarray],
                             mask: np.ndarray) -> List[float]:
    """
    AUC de |w_I| como clasificador de la etiqueta de dominancia, por escala.

    Args:
        w_I (List[np.ndarray]): Coeficientes de la imagen observada, por escala.
        labels (List[np.ndarray]): Etiquetas de dominancia (label_dominance), por escala.
        mask (np.ndarray): Mascara booleana [H, W] del foreground.

    Returns:
        List[float]: AUC por escala.
    """
    aucs = []
    for w_j, label_j in zip(w_I, labels):
        scores = np.abs(w_j[mask])
        labs = label_j[mask]
        aucs.append(_auc_from_scores(scores, labs))
    return aucs


def coefficient_histograms(w_R: np.ndarray, w_S: np.ndarray, mask: np.ndarray,
                            bins: int = 50, eps: float = 1e-8) -> Dict[str, np.ndarray]:
    """
    Histogramas de log10|w| por componente, para una escala, dentro de la mascara.

    Args:
        w_R (np.ndarray): Coeficientes del ground truth de reflectancia, una escala.
        w_S (np.ndarray): Coeficientes del ground truth de shading, misma escala.
        mask (np.ndarray): Mascara booleana [H, W] del foreground.
        bins (int): Numero de bins del histograma.
        eps (float): Constante para evitar log10(0).

    Returns:
        Dict[str, np.ndarray]: edges, hist_R, hist_S.
    """
    log_abs_R = np.log10(np.abs(w_R[mask]) + eps)
    log_abs_S = np.log10(np.abs(w_S[mask]) + eps)
    combined_min = float(min(log_abs_R.min(), log_abs_S.min()))
    combined_max = float(max(log_abs_R.max(), log_abs_S.max()))
    edges = np.linspace(combined_min, combined_max, bins + 1)
    hist_R, _ = np.histogram(log_abs_R, bins=edges)
    hist_S, _ = np.histogram(log_abs_S, bins=edges)
    return {"edges": edges, "hist_R": hist_R, "hist_S": hist_S}


def interscale_decay_stats(coeffs: List[np.ndarray], mask: np.ndarray) -> Dict[str, List[float]]:
    """
    Razon |w_(j+1)| / |w_j| por pixel entre escalas consecutivas (j=0 la mas fina), con
    estadisticas resumen por componente. Anticipa la variante D2b-2: la iluminacion suave
    deberia decaer distinto que los bordes de reflectancia a traves de las escalas.

    Args:
        coeffs (List[np.ndarray]): Coeficientes de detalle de una sola componente (R o S), por escala.
        mask (np.ndarray): Mascara booleana [H, W] del foreground.

    Returns:
        Dict[str, List[float]]: "mean" y "median" de la razon, una entrada por par de escalas consecutivas.
    """
    ratios_mean = []
    ratios_median = []
    for j in range(len(coeffs) - 1):
        num = np.abs(coeffs[j + 1][mask])
        den = np.abs(coeffs[j][mask]) + 1e-12
        ratio = num / den
        ratios_mean.append(float(np.mean(ratio)))
        ratios_median.append(float(np.median(ratio)))
    return {"mean": ratios_mean, "median": ratios_median}


def chroma_alignment_auc(image: np.ndarray, labels: np.ndarray, mask: np.ndarray,
                          eps: float = 1e-5) -> float:
    """
    AUC del gradiente de cromaticidad |grad C| como clasificador de la etiqueta de dominancia,
    con el mismo calculo que usa multiscale_decomposition para color_coherence. Anticipa la
    variante D2b-3 (el shading es acromatico).

    Args:
        image (np.ndarray): Imagen observada [H, W, 3] en rango [0, 1]. Si es 2D (sin color),
            no hay cromaticidad y se retorna NaN.
        labels (np.ndarray): Etiqueta de dominancia (label_dominance) de referencia, una escala.
        mask (np.ndarray): Mascara booleana [H, W] del foreground.
        eps (float): Constante para evitar division por cero.

    Returns:
        float: AUC en [0, 1], o NaN si la imagen es gris o una clase esta vacia.
    """
    if image.ndim != 3:
        return float("nan")
    L = rgb_to_luminance(image)
    C = image / (L[..., np.newaxis] + eps)
    grad_C_x = np.zeros_like(C)
    grad_C_y = np.zeros_like(C)
    grad_C_x[:, :-1, :] = C[:, 1:, :] - C[:, :-1, :]
    grad_C_y[:-1, :, :] = C[1:, :, :] - C[:-1, :, :]
    grad_C_mag = np.sqrt(np.sum(grad_C_x ** 2 + grad_C_y ** 2, axis=-1))
    scores = grad_C_mag[mask]
    labs = labels[mask]
    return _auc_from_scores(scores, labs)


def oracle_attribution(image: np.ndarray, mask: np.ndarray, w_I_coeffs: List[np.ndarray],
                        residual: np.ndarray, labels_per_scale: List[np.ndarray],
                        eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Atribuye cada coeficiente de la imagen observada segun la etiqueta de dominancia del
    ground truth (1 = reflectancia, 0 = shading), y reconstruye S y R con exactamente la
    misma logica de reconstruccion y recomposicion de color que multiscale_decomposition
    (el residuo siempre se asigna a S). Es el techo de LMSE de cualquier regla de atribucion
    binaria por coeficiente: comparar contra Horn indica si el problema es la calibracion
    del umbral o la familia de reglas completa.

    Args:
        image (np.ndarray): Imagen observada [H, W] o [H, W, 3], rango [0, 1].
        mask (np.ndarray): Mascara booleana [H, W] del foreground.
        w_I_coeffs (List[np.ndarray]): Coeficientes de detalle de la imagen observada, por escala.
        residual (np.ndarray): Residuo de aproximacion final de la imagen observada.
        labels_per_scale (List[np.ndarray]): Etiquetas de dominancia (label_dominance), por escala.
        eps (float): Constante para evitar division por cero al recomponer color.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Shading oracle S y reflectancia oracle R.
    """
    is_color = image.ndim == 3
    log_R = np.zeros_like(residual)
    log_S = residual.copy()
    for w_j, label_j in zip(w_I_coeffs, labels_per_scale):
        log_R += np.where(label_j, w_j, 0.0)
        log_S += np.where(label_j, 0.0, w_j)

    S = np.exp(log_S)
    R_L = np.exp(log_R)

    if is_color:
        L = rgb_to_luminance(image)
        C = image / (L[..., np.newaxis] + eps)
        R = C * R_L[..., np.newaxis]
    else:
        R = R_L

    S = S * mask
    if is_color:
        R = R * mask[..., np.newaxis]
    else:
        R = R * mask

    return S, R
