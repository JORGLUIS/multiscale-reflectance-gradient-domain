import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.metrics.extra import masked_psnr, masked_ssim, gradient_sparsity


def test_extra_metrics_identity_case():
    image = np.ones((32, 32), dtype=np.float64) * 0.4
    mask = np.ones((32, 32), dtype=bool)

    assert masked_psnr(image, image, mask) == 99.0
    assert np.isclose(masked_ssim(image, image, mask), 1.0)


def test_gradient_sparsity_on_flat_image():
    image = np.ones((16, 16), dtype=np.float64)
    mask = np.ones((16, 16), dtype=bool)

    assert gradient_sparsity(image, mask) == 1.0
