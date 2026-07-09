#!/usr/bin/env python3
"""
scripts/plot_informe_summary.py
Genera Informe/d1_d2_summary.png (Figura 2 del informe): baselines, mejor
metodo clasico previo, techo oracle de discriminacion por magnitud y el
metodo final (atenuacion continua en dominio de gradiente), en MIT y Sintel.
Lee los promedios directamente de results/tables/*.csv.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

TABLES_DIR = project_root / "results" / "tables"

METHODS_MIT = [
    ("Homomorfico", "f1_homomorphic.csv", "Combined_LMSE"),
    ("SSR", "f2_ssr.csv", "Combined_LMSE"),
    ("MSR", "f2_msr.csv", "Combined_LMSE"),
    ("Horn (Poisson)", "f2_horn.csv", "Combined_LMSE"),
    ("Mejor clasico previo\n(starlet+color, L4)", "f3_starlet_color_L4.csv", "Combined_LMSE"),
    ("Techo oracle D1\n(starlet, L3)", "diag_d1_diag_mit_starlet.csv", "Oracle_Combined_LMSE"),
    ("Metodo final\n(gradiente cont.)", "d2_graddom_mit.csv", "Combined_LMSE"),
]

METHODS_SINTEL = [
    ("Homomorfico", "f2_sintel_homomorphic.csv", "Combined_LMSE"),
    ("SSR", "f2_sintel_ssr.csv", "Combined_LMSE"),
    ("MSR", "f2_sintel_msr.csv", "Combined_LMSE"),
    ("Horn (Poisson)", "f2_sintel_horn.csv", "Combined_LMSE"),
    ("Mejor clasico previo\n(MMT, L2)", "f3_sintel_mmt_L2.csv", "Combined_LMSE"),
    ("Techo oracle D1\n(starlet, L3)", "diag_d1_diag_sintel_starlet.csv", "Oracle_Combined_LMSE"),
    ("Metodo final\n(gradiente cont.)", "d2_graddom_sintel.csv", "Combined_LMSE"),
]


def read_average(csv_name: str, column: str) -> float:
    """Lee el valor promedio de una columna desde un CSV de resultados."""
    df = pd.read_csv(TABLES_DIR / csv_name)
    if "Object" in df.columns and (df["Object"] == "Average").any():
        return float(df.loc[df["Object"] == "Average", column].iloc[0])
    return float(df.drop_duplicates("Object")[column].mean())


def build_summary(methods) -> pd.DataFrame:
    rows = [{"Metodo": label, "Combined_LMSE": read_average(csv_name, column)}
            for label, csv_name, column in methods]
    return pd.DataFrame(rows)


def main():
    summary_mit = build_summary(METHODS_MIT)
    summary_sintel = build_summary(METHODS_SINTEL)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, summary, title in zip(axes, [summary_mit, summary_sintel], ["MIT Intrinsic Images", "MPI-Sintel"]):
        colors = ["#9e9e9e"] * (len(summary) - 3) + ["#5b8def", "#e0a422", "#2ca25f"]
        bars = ax.bar(summary["Metodo"], summary["Combined_LMSE"], color=colors)
        ax.set_ylabel("Combined LMSE")
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        for bar, value in zip(bars, summary["Combined_LMSE"]):
            ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)

    fig.tight_layout()

    out_path = project_root / "Informe" / "d1_d2_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()
