import sys
from pathlib import Path
import numpy as np
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.decompose.gradient_domain import (
    reintegrate_from_weight,
    gradient_domain_decomposition,
    beta_values,
    residual_pyramid,
)
from src.utils.poisson import get_gradients


def _synthetic_log_image(H=32, W=32):
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xx, yy = np.meshgrid(x, y)
    image = 0.4 + 0.3 * np.sin(3 * np.pi * xx) * np.cos(2 * np.pi * yy)
    return np.log(np.clip(image, 1e-3, 1.0))


def test_p_equal_one_recovers_full_gradient_as_reflectance():
    """Con p=1, el gradiente de log_R_L debe igualar al de log_L (identidad del Poisson)."""
    log_L = _synthetic_log_image()
    mask = np.ones_like(log_L, dtype=bool)
    p = np.ones_like(log_L)

    log_R_L, log_S_L = reintegrate_from_weight(log_L, mask, p)

    grad_L_y, grad_L_x = get_gradients(log_L)
    grad_R_y, grad_R_x = get_gradients(log_R_L)

    # El solver de Poisson (pyamg) es iterativo: converge a un residuo chico, no exacto.
    assert np.allclose(grad_R_y, grad_L_y, atol=1e-4)
    assert np.allclose(grad_R_x, grad_L_x, atol=1e-4)
    assert np.std(log_S_L) < 1e-4, "Con p=1 el shading deberia quedar constante"


def test_p_equal_zero_shading_matches_luminance_gradient():
    """Con p=0, el gradiente de log_S_L debe igualar al de log_L y R queda constante."""
    log_L = _synthetic_log_image()
    mask = np.ones_like(log_L, dtype=bool)
    p = np.zeros_like(log_L)

    log_R_L, log_S_L = reintegrate_from_weight(log_L, mask, p)

    grad_L_y, grad_L_x = get_gradients(log_L)
    grad_S_y, grad_S_x = get_gradients(log_S_L)

    assert np.allclose(grad_S_y, grad_L_y, atol=1e-4)
    assert np.allclose(grad_S_x, grad_L_x, atol=1e-4)
    assert np.std(log_R_L) < 1e-4, "Con p=0 la reflectancia deberia quedar constante"


def test_beta_profile_linear_and_constant():
    """El perfil lineal decae de beta_fine a beta_coarse, el constante no varia."""
    linear = beta_values(4, beta_fine=1.1, beta_coarse=0.1, beta_profile="linear")
    assert linear[0] == pytest.approx(1.1)
    assert linear[-1] == pytest.approx(0.1)
    assert np.all(np.diff(linear) < 0)

    constant = beta_values(4, beta_fine=1.1, beta_coarse=0.1, beta_profile="constant")
    assert np.allclose(constant, 1.1)


def test_determinism_seed_zero():
    """Dos corridas con la misma entrada deben dar exactamente el mismo resultado."""
    np.random.seed(0)
    H, W = 40, 40
    image = np.random.rand(H, W).astype(np.float64) * 0.6 + 0.2
    mask = np.ones((H, W), dtype=bool)

    S1, R1 = gradient_domain_decomposition(image, mask, levels=2)
    S2, R2 = gradient_domain_decomposition(image, mask, levels=2)

    assert np.array_equal(S1, S2)
    assert np.array_equal(R1, R2)


def test_gradient_domain_decomposition_runs_on_color_image():
    """Sanity end-to-end: corre sobre imagen color con croma y da formas y rangos correctos."""
    np.random.seed(0)
    H, W = 32, 32
    image = np.random.rand(H, W, 3).astype(np.float64) * 0.6 + 0.2
    mask = np.ones((H, W), dtype=bool)

    S, R = gradient_domain_decomposition(image, mask, levels=2, chroma_modulation=True)

    assert S.shape == (H, W)
    assert R.shape == (H, W, 3)
    assert np.all(np.isfinite(S))
    assert np.all(np.isfinite(R))


@pytest.mark.parametrize("transform_type", ["starlet", "mmt", "wavelet"])
def test_residual_pyramid_preserves_amplitude_across_transforms(transform_type):
    """Cada R_k debe quedar en las mismas unidades que la imagen (sin deriva de escala)."""
    log_L = _synthetic_log_image()
    levels = 3

    pyramid = residual_pyramid(log_L, levels, transform_type=transform_type)

    assert len(pyramid) == levels + 1
    assert np.array_equal(pyramid[0], log_L)
    for R_k in pyramid[1:]:
        assert R_k.shape == log_L.shape
        assert np.isfinite(R_k).all()
        # Tolerancia amplia: MMT (mediana, no lineal) no preserva la media tan
        # ajustado como starlet/wavelet (lineales); esto solo debe atrapar una
        # deriva de escala sistematica (p.ej. el bug de normalizacion 2^nivel
        # de pywt.swt2), no el sesgo esperado de un filtro de mediana.
        assert np.mean(R_k) == pytest.approx(np.mean(log_L), abs=0.2)


@pytest.mark.parametrize("transform_type", ["starlet", "mmt", "wavelet"])
def test_gradient_domain_decomposition_runs_with_each_transform(transform_type):
    """Sanity end-to-end de la generalizacion de transformada (starlet/mmt/wavelet)."""
    np.random.seed(0)
    H, W = 32, 32
    image = np.random.rand(H, W).astype(np.float64) * 0.6 + 0.2
    mask = np.ones((H, W), dtype=bool)

    S, R = gradient_domain_decomposition(image, mask, levels=3, transform_type=transform_type)

    assert S.shape == (H, W)
    assert R.shape == (H, W)
    assert np.all(np.isfinite(S))
    assert np.all(np.isfinite(R))
