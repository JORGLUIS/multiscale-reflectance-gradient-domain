#!/usr/bin/env python3
"""
scripts/plot_pyramid_demo.py
Genera Informe/pyramid_demo.png: la piramide de residuos R_0,...,R_L (Starlet)
de un objeto real de MIT, mas el gradiente de cromaticidad |grad C| y el
gradiente de cada nivel ||grad R_k|| (el ingrediente de Q_k), para mostrar
sobre una imagen concreta lo que hace el metodo final (Figura 1 del informe).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.mit import MITIntrinsicDataset
from src.baselines.homomorphic import rgb_to_luminance
from src.decompose.gradient_domain import residual_pyramid, centered_gradient_magnitude

OBJECT_ID = "cup1"
LEVELS = 3
EPS = 1e-5


def chroma_gradient(diffuse, L, mask):
    """Mismo calculo que gradient_domain_decomposition cuando chroma_modulation=True."""
    C = diffuse / (L[..., np.newaxis] + EPS)
    grad_C_x = np.zeros_like(C)
    grad_C_y = np.zeros_like(C)
    grad_C_x[:, :-1, :] = C[:, 1:, :] - C[:, :-1, :]
    grad_C_y[:-1, :, :] = C[1:, :, :] - C[:-1, :, :]
    grad_C = np.sqrt(np.sum(grad_C_x ** 2 + grad_C_y ** 2, axis=-1))
    return grad_C * mask


def normalize_for_display(field, mask):
    valid = field[mask]
    if valid.size == 0 or valid.max() <= 0:
        return field
    return np.clip(field / valid.max(), 0.0, 1.0)


def main():
    dataset = MITIntrinsicDataset()
    data = dataset.load_object(OBJECT_ID)
    diffuse = data["diffuse"]
    mask = data["mask"]

    L = rgb_to_luminance(diffuse)
    L_masked = np.clip(L, EPS, 1.0)
    log_L = np.log(L_masked)
    fg_mean = np.mean(log_L[mask]) if np.any(mask) else 0.0
    log_L_filled = np.where(mask, log_L, fg_mean)

    pyramid = residual_pyramid(log_L_filled, LEVELS, transform_type="starlet")

    fig, axes = plt.subplots(2, len(pyramid), figsize=(2.6 * len(pyramid), 5.6))

    # Fila 1: R_0,...,R_L (misma pirámide que antes).
    for k, (ax, R_k) in enumerate(zip(axes[0], pyramid)):
        display = np.clip(np.exp(R_k), 0.0, 1.0) * mask
        ax.imshow(display, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"$R_{{{k}}}$", fontsize=14)
        ax.axis("off")

    # Fila 2: bajo R_0 va |grad C| (no tiene nivel propio); bajo R_1,R_2,R_3
    # va ||grad R_k||, el ingrediente real de Q_k = ||grad R_k||^(beta_k-1).
    grad_c = normalize_for_display(chroma_gradient(diffuse, L, mask), mask)
    im = axes[1][0].imshow(grad_c, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1][0].set_title(r"$|\nabla C|$", fontsize=14)
    axes[1][0].axis("off")

    for k in range(1, len(pyramid)):
        grad_mag = centered_gradient_magnitude(pyramid[k])
        grad_mag = normalize_for_display(grad_mag * mask, mask)
        axes[1][k].imshow(grad_mag, cmap="magma", vmin=0.0, vmax=1.0)
        axes[1][k].set_title(rf"$\|\nabla R_{{{k}}}\|$", fontsize=14)
        axes[1][k].axis("off")

    fig.tight_layout()

    colorbar = fig.colorbar(im, ax=list(axes[1]), orientation="horizontal",
                             fraction=0.06, pad=0.08, aspect=40)
    colorbar.set_label("magnitud normalizada (por el máximo de cada panel)", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)

    out_path = project_root / "Informe" / "pyramid_demo.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()
