#!/usr/bin/env python3
"""
scripts/plot_pyramid_demo.py
Genera Informe/pyramid_demo.png: la piramide de residuos R_0,...,R_L (Starlet)
de un objeto real de MIT, para mostrar sobre una imagen concreta lo que hace
el metodo final (Figura 1 del informe).
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
from src.decompose.gradient_domain import residual_pyramid

OBJECT_ID = "cup1"
LEVELS = 3
EPS = 1e-5


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

    fig, axes = plt.subplots(1, len(pyramid), figsize=(2.6 * len(pyramid), 2.9))
    for k, (ax, R_k) in enumerate(zip(axes, pyramid)):
        display = np.clip(np.exp(R_k), 0.0, 1.0) * mask
        ax.imshow(display, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"$R_{{{k}}}$", fontsize=14)
        ax.axis("off")
    fig.tight_layout()

    out_path = project_root / "Informe" / "pyramid_demo.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()
