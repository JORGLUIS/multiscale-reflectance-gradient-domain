#!/usr/bin/env python3
"""
scripts/plot_visual_comparison.py
Genera Informe/mit_multiscale_comparison.png y sintel_multiscale_comparison.png
(Figura 3 del informe): compara la regla por magnitud (Starlet, Wavelet, MMT)
contra el metodo final (atenuacion continua en dominio de gradiente) en tres
ejemplos por dataset, con su error absoluto de reflectancia.
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


def show_rgb(ax, image, title):
    ax.imshow(np.clip(image, 0.0, 1.0))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def show_multiscale_comparison(dataset_name, items, experiments, output_path):
    rows = 2 * len(items)
    cols = 2 + len(experiments)
    fig_height = 3.1 * len(items)
    fig_width = 4.5 + 3.0 * len(experiments)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(
        rows,
        cols,
        width_ratios=[0.9, 0.9] + [1.15] * len(experiments),
    )

    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad(color="black")
    error_axes = []
    last_error = None
    records = []

    for item_index, item in enumerate(items):
        image_row = 2 * item_index
        error_row = image_row + 1

        if dataset_name == "mit":
            example_id, data = item
            item_title = example_id
            summary_id = example_id
        else:
            scene_id, frame_file, data = item
            frame_label = frame_file.replace(".png", "")
            item_title = f"{scene_id}\n{frame_label}"
            summary_id = f"{scene_id}/{frame_label}"

        input_ax = fig.add_subplot(grid[image_row:error_row + 1, 0])
        gt_ax = fig.add_subplot(grid[image_row:error_row + 1, 1])
        show_rgb(input_ax, data["diffuse"], f"Entrada\n{item_title}")
        show_rgb(gt_ax, data["reflectance"], "Reflectancia\nGT")

        for method_index, method_info in enumerate(experiments):
            method_label, exp_id = method_info
            est_shading, est_refl = run_decomposition_from_config(
                data["diffuse"], data["mask"], exp_id,
            )
            metrics, error_map = compute_example_metrics(data, est_shading, est_refl)
            col = 2 + method_index

            estimate_ax = fig.add_subplot(grid[image_row, col])
            error_ax = fig.add_subplot(grid[error_row, col])
            show_rgb(estimate_ax, est_refl, f"{method_label}\nR estimada")
            last_error = error_ax.imshow(error_map, cmap=error_cmap, vmin=0.0, vmax=0.5)
            error_ax.set_title(f"Error {method_label}\nLMSE {metrics['Refl_LMSE']:.3f}", fontsize=8)
            error_ax.axis("off")
            error_axes.append(error_ax)

            records.append({
                "dataset": dataset_name,
                "example": summary_id,
                "experiment_id": exp_id,
                **metrics,
            })

    colorbar = fig.colorbar(last_error, ax=error_axes, fraction=0.02, pad=0.015)
    colorbar.set_label("Error abs.", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(records)


def main():
    mit_visual_experiments = [
        ("Starlet", "f3_starlet_color"),
        ("Wavelet", "f3_wavelet_all"),
        ("MMT", "f3_mmt"),
        ("Metodo final", "d2_graddom_mit"),
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
        ("Starlet", "f2_sintel_starlet"),
        ("Wavelet", "f3_sintel_wavelet_all"),
        ("MMT", "f3_sintel_mmt"),
        ("Metodo final", "d2_graddom_sintel"),
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
