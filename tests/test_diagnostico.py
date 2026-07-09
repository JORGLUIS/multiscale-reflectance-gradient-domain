import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.eval.diagnostico import (
    log_component,
    decompose_log_component,
    decompose_gt_components,
    additivity_error_per_scale,
    label_dominance,
    magnitude_auc_per_scale,
    oracle_attribution,
)
from src.metrics.lmse import local_error


def _stepped_reflectance(H: int, W: int) -> np.ndarray:
    """R con un unico borde nitido vertical al centro (dos niveles constantes)."""
    R = np.full((H, W), 0.3)
    R[:, W // 2:] = 0.8
    return R


def _smooth_shading(H: int, W: int) -> np.ndarray:
    """S de variacion suave y de baja frecuencia (sin bordes nitidos)."""
    x = np.linspace(0.0, 1.0, W)
    y = np.linspace(0.0, 1.0, H)
    xx, yy = np.meshgrid(x, y)
    return 0.6 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)


def test_starlet_additivity_is_exact():
    """El starlet es una transformada lineal: w_I debe igualar w_R + w_S por escala."""
    H, W = 64, 64
    mask = np.ones((H, W), dtype=bool)
    R = _stepped_reflectance(H, W)
    S = _smooth_shading(H, W)
    I = R * S

    levels = 3
    gt = decompose_gt_components(R, S, mask, transform_type="starlet", levels=levels)
    log_I = log_component(I, mask)
    coeffs_I, _ = decompose_log_component(log_I, "starlet", levels)

    errors = additivity_error_per_scale(coeffs_I, gt["coeffs_R"], gt["coeffs_S"], mask)
    for err in errors:
        assert err < 1e-8, f"Error de aditividad starlet demasiado alto: {err}"


def test_magnitude_auc_high_at_fine_scale_and_oracle_recovers_gt():
    """R con borde nitido y S suave: la magnitud de w_I debe discriminar bien en escala fina
    y el oracle (atribucion con la etiqueta GT) debe acercarse al piso de error de Horn."""
    H, W = 64, 64
    mask = np.ones((H, W), dtype=bool)
    R = _stepped_reflectance(H, W)
    S = _smooth_shading(H, W)
    I = R * S

    levels = 3
    gt = decompose_gt_components(R, S, mask, transform_type="starlet", levels=levels)
    log_I = log_component(I, mask)
    coeffs_I, residual_I = decompose_log_component(log_I, "starlet", levels)

    labels_per_scale = [
        label_dominance(gt["coeffs_R"][j], gt["coeffs_S"][j]) for j in range(levels)
    ]
    aucs = magnitude_auc_per_scale(coeffs_I, labels_per_scale, mask)

    assert aucs[0] > 0.9, f"AUC en la escala mas fina deberia ser alta, fue {aucs[0]}"

    S_oracle, R_oracle = oracle_attribution(I, mask, coeffs_I, residual_I, labels_per_scale)
    lmse_r = local_error(R, R_oracle, mask, window_size=20, window_shift=10)
    lmse_s = local_error(S, S_oracle, mask, window_size=20, window_shift=10)
    combined = 0.5 * lmse_r + 0.5 * lmse_s
    assert combined < 0.03, f"LMSE del oracle deberia ser cercano a 0, fue {combined}"


def test_identity_constant_shading_all_reflectance_dominated():
    """Si S es constante, todos los coeficientes deben quedar dominados por reflectancia."""
    H, W = 64, 64
    mask = np.ones((H, W), dtype=bool)
    R = _stepped_reflectance(H, W)
    S = np.full((H, W), 0.6)
    I = R * S

    levels = 3
    gt = decompose_gt_components(R, S, mask, transform_type="starlet", levels=levels)

    for j in range(levels):
        assert np.all(gt["coeffs_S"][j] == 0.0), "w_S deberia ser exactamente cero (S constante)"
        labels = label_dominance(gt["coeffs_R"][j], gt["coeffs_S"][j])
        assert np.all(labels), "Con S constante todos los coeficientes deben dominarlos R"

    log_I = log_component(I, mask)
    coeffs_I, residual_I = decompose_log_component(log_I, "starlet", levels)
    labels_per_scale = [
        label_dominance(gt["coeffs_R"][j], gt["coeffs_S"][j]) for j in range(levels)
    ]
    S_oracle, R_oracle = oracle_attribution(I, mask, coeffs_I, residual_I, labels_per_scale)

    lmse_r = local_error(R, R_oracle, mask, window_size=20, window_shift=10)
    assert lmse_r < 0.03, f"El oracle deberia recuperar R casi exacto, LMSE fue {lmse_r}"
