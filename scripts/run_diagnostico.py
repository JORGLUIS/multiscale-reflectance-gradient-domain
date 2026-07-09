#!/usr/bin/env python3
"""
scripts/run_diagnostico.py
Mide si la magnitud de los coeficientes multiescala discrimina reflectancia de
shading, usando el ground truth de MIT o Sintel.

Escribe una fila por imagen y escala en results/tables/diag_<id>.csv y una figura
de histogramas |w_R| vs |w_S| pooleados por escala en results/figures/diag_<id>/.
"""

import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.mit import MITIntrinsicDataset, EVAL_OBJECTS
from src.metrics.lmse import local_error
from src.eval.diagnostico import (
    log_component,
    decompose_log_component,
    decompose_gt_components,
    additivity_error_per_scale,
    label_dominance,
    magnitude_auc_per_scale,
    chroma_alignment_auc,
    interscale_decay_stats,
    coefficient_histograms,
    oracle_attribution,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnostico D1 de separabilidad reflectancia/shading")
    parser.add_argument("--config", type=str, required=True, help="Path al YAML de configuracion")
    return parser.parse_args()


def load_items(dataset_name: str):
    """Carga el dataset y devuelve la lista de (tag, data_dict) a procesar."""
    items = []
    if dataset_name == "mit":
        dataset = MITIntrinsicDataset()
        for tag in EVAL_OBJECTS:
            items.append((tag, dataset.load_object(tag)))
    elif dataset_name == "sintel":
        from src.data.sintel import MPISintelDataset
        dataset = MPISintelDataset()
        for scene in dataset.get_scenes():
            frames = dataset.get_frames_in_scene(scene)
            if frames:
                tag = f"{scene}_{frames[0]}"
                items.append((tag, dataset.load_frame(scene, frames[0])))
    else:
        raise ValueError(f"Dataset desconocido '{dataset_name}'")
    return items


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config no encontrado en {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    experiment_id = config.get("experiment_id", config_path.stem)
    dataset_name = config["dataset"]
    transform_type = config["transform_type"]
    params = config.get("params", {})
    levels = params.get("levels", 3)
    eps = params.get("eps", 1e-5)

    print("==================================================")
    print(f"Diagnostico D1: {experiment_id}")
    print(f"Dataset:        {dataset_name}")
    print(f"Transformada:   {transform_type}")
    print(f"Niveles:        {levels}")
    print("==================================================")

    items = load_items(dataset_name)

    rows = []
    pooled_wR = [[] for _ in range(levels)]
    pooled_wS = [[] for _ in range(levels)]

    for idx, (tag, data) in enumerate(items):
        print(f"Procesando {tag} ({idx + 1}/{len(items)})...")
        diffuse = data["diffuse"]
        true_refl = data["reflectance"]
        true_shading = data["shading"]
        mask = data["mask"]

        gt = decompose_gt_components(true_refl, true_shading, mask, transform_type, levels, eps)
        log_I = log_component(diffuse, mask, eps)
        coeffs_I, residual_I = decompose_log_component(log_I, transform_type, levels)

        additivity_errors = additivity_error_per_scale(coeffs_I, gt["coeffs_R"], gt["coeffs_S"], mask)
        labels_per_scale = [label_dominance(gt["coeffs_R"][j], gt["coeffs_S"][j]) for j in range(levels)]
        aucs_magnitude = magnitude_auc_per_scale(coeffs_I, labels_per_scale, mask)
        aucs_chroma = [chroma_alignment_auc(diffuse, labels_per_scale[j], mask, eps) for j in range(levels)]
        decay_R = interscale_decay_stats(gt["coeffs_R"], mask)
        decay_S = interscale_decay_stats(gt["coeffs_S"], mask)

        S_oracle, R_oracle = oracle_attribution(diffuse, mask, coeffs_I, residual_I, labels_per_scale, eps)

        true_refl_gray = np.mean(true_refl, axis=2) if true_refl.ndim == 3 else true_refl
        R_oracle_gray = np.mean(R_oracle, axis=2) if R_oracle.ndim == 3 else R_oracle
        true_shading_gray = np.mean(true_shading, axis=2) if true_shading.ndim == 3 else true_shading
        S_oracle_gray = np.mean(S_oracle, axis=2) if S_oracle.ndim == 3 else S_oracle

        lmse_refl = local_error(true_refl_gray, R_oracle_gray, mask, window_size=20, window_shift=10)
        lmse_shading = local_error(true_shading_gray, S_oracle_gray, mask, window_size=20, window_shift=10)
        lmse_combined = 0.5 * lmse_refl + 0.5 * lmse_shading

        for j in range(levels):
            pooled_wR[j].append(gt["coeffs_R"][j][mask].ravel())
            pooled_wS[j].append(gt["coeffs_S"][j][mask].ravel())

            rows.append({
                "Object": tag,
                "Scale": j,
                "AUC_Magnitude": aucs_magnitude[j],
                "AUC_Chroma": aucs_chroma[j],
                "Additivity_Error": additivity_errors[j],
                "Decay_Mean_R": decay_R["mean"][j] if j < levels - 1 else np.nan,
                "Decay_Mean_S": decay_S["mean"][j] if j < levels - 1 else np.nan,
                "Decay_Median_R": decay_R["median"][j] if j < levels - 1 else np.nan,
                "Decay_Median_S": decay_S["median"][j] if j < levels - 1 else np.nan,
                "Oracle_Refl_LMSE": lmse_refl,
                "Oracle_Shading_LMSE": lmse_shading,
                "Oracle_Combined_LMSE": lmse_combined,
            })

    df = pd.DataFrame(rows)

    tables_dir = project_root / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"diag_{experiment_id}.csv"
    df.to_csv(csv_path, index=False)

    # Resumen por escala (promedio sobre imagenes)
    summary = df.groupby("Scale").agg({
        "AUC_Magnitude": "mean",
        "AUC_Chroma": "mean",
        "Additivity_Error": "mean",
        "Decay_Mean_R": "mean",
        "Decay_Mean_S": "mean",
    })
    oracle_combined_mean = df.drop_duplicates("Object")["Oracle_Combined_LMSE"].mean()
    oracle_refl_mean = df.drop_duplicates("Object")["Oracle_Refl_LMSE"].mean()
    oracle_shading_mean = df.drop_duplicates("Object")["Oracle_Shading_LMSE"].mean()

    print("\n=================== RESUMEN POR ESCALA ===================")
    print(summary.to_string(float_format="{:,.4f}".format))
    print("============================================================")
    print(f"Oracle Refl LMSE (promedio):     {oracle_refl_mean:.4f}")
    print(f"Oracle Shading LMSE (promedio):  {oracle_shading_mean:.4f}")
    print(f"Oracle Combined LMSE (promedio): {oracle_combined_mean:.4f}")
    print(f"CSV guardado en: {csv_path}")

    # Figura de histogramas |w_R| vs |w_S| pooleados, una fila de subplots por escala
    figures_dir = project_root / "results" / "figures" / f"diag_{experiment_id}"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, levels, figsize=(5 * levels, 4), squeeze=False)
    for j in range(levels):
        w_R_all = np.concatenate(pooled_wR[j])
        w_S_all = np.concatenate(pooled_wS[j])
        full_mask = np.ones_like(w_R_all, dtype=bool)
        hist = coefficient_histograms(w_R_all, w_S_all, full_mask, bins=50)
        centers = 0.5 * (hist["edges"][:-1] + hist["edges"][1:])
        ax = axes[0][j]
        ax.bar(centers, hist["hist_R"], width=np.diff(hist["edges"]), alpha=0.5, label="R (reflectancia)")
        ax.bar(centers, hist["hist_S"], width=np.diff(hist["edges"]), alpha=0.5, label="S (shading)")
        ax.set_title(f"Escala {j}")
        ax.set_xlabel("log10|w|")
        ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=8)

    fig.suptitle(f"{experiment_id}: distribucion de |w| por componente")
    fig.tight_layout()
    fig_path = figures_dir / "histogramas.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Figura guardada en: {fig_path}")


if __name__ == "__main__":
    main()
