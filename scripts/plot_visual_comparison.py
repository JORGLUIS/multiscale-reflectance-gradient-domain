#!/usr/bin/env python3
"""
scripts/plot_visual_comparison.py
Genera Informe/mit_multiscale_comparison.png y sintel_multiscale_comparison.png
(Figura 3 del informe): reflectancia estimada y error absoluto del metodo final
(atenuacion continua en dominio de gradiente) en sus tres variantes de
transformada (Starlet, MMT, Wavelet), en tres ejemplos por dataset.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
configs_dir = project_root / "configs"

from src.decompose.multiscale import multiscale_decomposition
from src.decompose.gradient_domain import gradient_domain_decomposition
from src.baselines.homomorphic import homomorphic_decomposition
from src.baselines.ssr import ssr_decomposition
from src.baselines.msr import msr_decomposition
from src.baselines.horn import horn_decomposition
from src.data.mit import MITIntrinsicDataset
from src.data.sintel import MPISintelDataset
from src.metrics.extra import masked_psnr, masked_ssim, scale_to_reference
from src.metrics.lmse import local_error


def load_config(exp_id):
    with open(configs_dir / f"{exp_id}.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_decomposition_from_config(diffuse, mask, exp_id):
    config = load_config(exp_id)
    method = config.get("method")
    params = dict(config.get("params", {}))

    if method == "homomorphic":
        return homomorphic_decomposition(diffuse, mask, **params)
    if method == "ssr":
        return ssr_decomposition(diffuse, mask, **params)
    if method == "msr":
        return msr_decomposition(diffuse, mask, **params)
    if method == "horn":
        return horn_decomposition(diffuse, mask, **params)
    if method in ["starlet", "mmt", "multiscale"]:
        return multiscale_decomposition(diffuse, mask, **params)
    if method == "gradient_domain":
        return gradient_domain_decomposition(diffuse, mask, **params)
    raise ValueError(f"Metodo no soportado: {method}")


def to_gray(image):
    return np.mean(image, axis=2) if image.ndim == 3 else image


def compute_example_metrics(data, est_shading, est_refl):
    mask = data["mask"]
    true_refl_gray = to_gray(data["reflectance"])
    true_shading_gray = to_gray(data["shading"])
    est_refl_gray = to_gray(est_refl)
    est_shading_gray = to_gray(est_shading)

    refl_lmse = local_error(true_refl_gray, est_refl_gray, mask, window_size=20, window_shift=10)
    shading_lmse = local_error(true_shading_gray, est_shading_gray, mask, window_size=20, window_shift=10)
    combined_lmse = 0.5 * refl_lmse + 0.5 * shading_lmse
    refl_psnr = masked_psnr(true_refl_gray, est_refl_gray, mask)
    refl_ssim = masked_ssim(true_refl_gray, est_refl_gray, mask)

    aligned_refl = scale_to_reference(true_refl_gray, est_refl_gray, mask)
    error_map = np.where(mask, np.abs(true_refl_gray - aligned_refl), np.nan)

    metrics = {
        "Refl_LMSE": refl_lmse,
        "Shading_LMSE": shading_lmse,
        "Combined_LMSE": combined_lmse,
        "Refl_PSNR": refl_psnr,
        "Refl_SSIM": refl_ssim,
    }
    return metrics, error_map


def show_rgb(ax, image, title, fontsize=8):
    ax.imshow(np.clip(image, 0.0, 1.0))
    if title:
        ax.set_title(title, fontsize=fontsize)
    ax.axis("off")


def show_multiscale_comparison(dataset_name, items, experiments, output_path):
    """Una fila por ejemplo: Entrada, GT, y por cada variante del metodo final
    su reflectancia estimada y su error absoluto como columnas separadas
    (2 + 2*len(experiments) columnas en total)."""
    nrows = len(items)
    ncols = 2 + 2 * len(experiments)
    cell_w_in = 1.85

    row_heights = []
    item_titles = []
    for item in items:
        if dataset_name == "mit":
            example_id, data = item
            item_titles.append(example_id)
        else:
            scene_id, frame_file, data = item
            item_titles.append(f"{scene_id}/{frame_file.replace('.png', '')}")
        img_h, img_w = data["diffuse"].shape[:2]
        row_heights.append(img_h / img_w)
    fig_height = sum(row_heights) * cell_w_in + 0.55 * nrows + 0.25

    fig = plt.figure(figsize=(cell_w_in * ncols, fig_height))
    grid = fig.add_gridspec(nrows, ncols, height_ratios=row_heights, hspace=0.45, wspace=0.05)

    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad(color="black")
    error_axes = []
    last_error = None
    records = []

    for row, (item, item_title) in enumerate(zip(items, item_titles)):
        if dataset_name == "mit":
            _, data = item
            summary_id = item_title
        else:
            _, _, data = item
            summary_id = item_title

        header = (row == 0)
        input_ax = fig.add_subplot(grid[row, 0])
        gt_ax = fig.add_subplot(grid[row, 1])
        show_rgb(input_ax, data["diffuse"], (f"Entrada\n{item_title}" if header else f"\n{item_title}"))
        show_rgb(gt_ax, data["reflectance"], ("Reflectancia\nGT" if header else "\n"))

        for method_index, method_info in enumerate(experiments):
            method_label, exp_id = method_info
            est_shading, est_refl = run_decomposition_from_config(
                data["diffuse"], data["mask"], exp_id,
            )
            metrics, error_map = compute_example_metrics(data, est_shading, est_refl)
            r_col = 2 + 2 * method_index
            err_col = r_col + 1

            estimate_ax = fig.add_subplot(grid[row, r_col])
            error_ax = fig.add_subplot(grid[row, err_col])
            show_rgb(estimate_ax, est_refl, (f"{method_label}\nR estimada" if header else "\n"))
            last_error = error_ax.imshow(error_map, cmap=error_cmap, vmin=0.0, vmax=0.5)
            error_ax.set_title((f"Error {method_label}\n" if header else "\n") + f"LMSE {metrics['Refl_LMSE']:.3f}",
                                fontsize=8)
            error_ax.axis("off")
            error_axes.append(error_ax)

            records.append({
                "dataset": dataset_name,
                "example": summary_id,
                "experiment_id": exp_id,
                **metrics,
            })

    colorbar = fig.colorbar(last_error, ax=error_axes, fraction=0.015, pad=0.01)
    colorbar.set_label("Error abs.", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(records)


def main():
    mit_visual_experiments = [
        ("Starlet", "d2_graddom_mit"),
        ("MMT", "d2_graddom_mit_mmt"),
        ("Wavelet", "d2_graddom_mit_wavelet"),
    ]
    mit_dataset = MITIntrinsicDataset()
    mit_items = [(object_id, mit_dataset.load_object(object_id))
                 for object_id in ["box", "cup1", "paper1"]]
    mit_summary = show_multiscale_comparison(
        "mit", mit_items, mit_visual_experiments,
        project_root / "Informe" / "mit_multiscale_comparison.png",
    )
    print(mit_summary.round(4).to_string(index=False))

    sintel_visual_experiments = [
        ("Starlet", "d2_graddom_sintel"),
        ("MMT", "d2_graddom_sintel_mmt"),
        ("Wavelet", "d2_graddom_sintel_wavelet"),
    ]
    sintel_dataset = MPISintelDataset()
    sintel_items = [
        (scene_id, frame_file, sintel_dataset.load_frame(scene_id, frame_file))
        for scene_id, frame_file in [
            ("alley_1", "frame_0001.png"),
            ("ambush_2", "frame_0001.png"),
            ("bamboo_1", "frame_0001.png"),
        ]
    ]
    sintel_summary = show_multiscale_comparison(
        "sintel", sintel_items, sintel_visual_experiments,
        project_root / "Informe" / "sintel_multiscale_comparison.png",
    )
    print(sintel_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
