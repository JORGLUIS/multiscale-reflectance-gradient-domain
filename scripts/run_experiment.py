#!/usr/bin/env python3
"""
scripts/run_experiment.py
Runs an intrinsic image decomposition experiment based on a YAML configuration file.
Evaluates on the MIT Intrinsic Images dataset, computes LMSE and MSE,
and outputs results to a CSV table in results/tables/.
"""

import os
import sys
import time
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.mit import MITIntrinsicDataset, EVAL_OBJECTS
from src.metrics.lmse import local_error
from src.metrics.extra import masked_psnr, masked_ssim, gradient_sparsity
from src.baselines.homomorphic import homomorphic_decomposition
from src.baselines.ssr import ssr_decomposition
from src.baselines.msr import msr_decomposition
from src.baselines.horn import horn_decomposition
from src.decompose.multiscale import multiscale_decomposition
from src.decompose.gradient_domain import gradient_domain_decomposition

def parse_args():
    parser = argparse.ArgumentParser(description="Run Intrinsic Decomposition Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    experiment_id = config.get("experiment_id", config_path.stem)
    dataset_name = config.get("dataset", "mit")
    method = config.get("method")
    params = config.get("params", {})
    
    print(f"==================================================")
    print(f"Running Experiment: {experiment_id}")
    print(f"Dataset:            {dataset_name}")
    print(f"Method:             {method}")
    print(f"Parameters:         {params}")
    print(f"==================================================")
    
    if dataset_name == "mit":
        dataset = MITIntrinsicDataset()
        items = [(None, tag) for tag in EVAL_OBJECTS]
    elif dataset_name == "sintel":
        from src.data.sintel import MPISintelDataset
        dataset = MPISintelDataset()
        scenes = dataset.get_scenes()
        items = []
        for scene in scenes:
            frames = dataset.get_frames_in_scene(scene)
            if frames:
                items.append((scene, frames[0]))
    else:
        print(f"Error: Unknown dataset '{dataset_name}'", file=sys.stderr)
        sys.exit(1)
        
    results = []
    
    for idx, item in enumerate(items):
        if dataset_name == "sintel":
            scene, frame = item
            tag = f"{scene}_{frame}"
            print(f"Processing Sintel: {tag} ({idx+1}/{len(items)})...")
            try:
                data = dataset.load_frame(scene, frame)
            except Exception as e:
                print(f"Error loading Sintel frame {scene}/{frame}: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            _, tag = item
            print(f"Processing MIT: {tag} ({idx+1}/{len(items)})...")
            try:
                data = dataset.load_object(tag)
            except Exception as e:
                print(f"Error loading object {tag}: {e}. Make sure download_data.py has run successfully.", file=sys.stderr)
                sys.exit(1)
                
        diffuse = data['diffuse']       # Input color image
        true_refl = data['reflectance'] # GT color reflectance
        true_shading = data['shading']   # GT grayscale shading
        mask = data['mask']             # Binary mask
        
        # Prepare inputs based on method (some methods require grayscale or color)
        t0 = time.time()
        
        if method == "homomorphic":
            est_shading, est_refl = homomorphic_decomposition(
                diffuse, mask, **params
            )
        elif method == "ssr":
            est_shading, est_refl = ssr_decomposition(
                diffuse, mask, **params
            )
        elif method == "msr":
            est_shading, est_refl = msr_decomposition(
                diffuse, mask, **params
            )
        elif method == "horn":
            est_shading, est_refl = horn_decomposition(
                diffuse, mask, **params
            )
        elif method in ["starlet", "mmt", "multiscale"]:
            est_shading, est_refl = multiscale_decomposition(
                diffuse, mask, **params
            )
        elif method == "gradient_domain":
            est_shading, est_refl = gradient_domain_decomposition(
                diffuse, mask, **params
            )
        else:
            print(f"Error: Unknown method '{method}'", file=sys.stderr)
            sys.exit(1)
            
        elapsed_time = time.time() - t0
        
        # Calculate evaluation metrics (always grayscale as per benchmark)
        # Convert reflectance arrays to grayscale
        true_refl_gray = np.mean(true_refl, axis=2) if true_refl.ndim == 3 else true_refl
        est_refl_gray = np.mean(est_refl, axis=2) if est_refl.ndim == 3 else est_refl
        
        # Grayscale shading arrays
        true_shading_gray = np.mean(true_shading, axis=2) if true_shading.ndim == 3 else true_shading
        est_shading_gray = np.mean(est_shading, axis=2) if est_shading.ndim == 3 else est_shading
        
        # Compute LMSE
        lmse_refl = local_error(true_refl_gray, est_refl_gray, mask, window_size=20, window_shift=10)
        lmse_shading = local_error(true_shading_gray, est_shading_gray, mask, window_size=20, window_shift=10)
        lmse_combined = 0.5 * lmse_refl + 0.5 * lmse_shading
        
        # Compute standard scale-invariant MSE (si-MSE) globally for comparison
        # (scale est to match true on mask)
        denom_r = np.sum(est_refl_gray**2 * mask)
        alpha_r = np.sum(true_refl_gray * est_refl_gray * mask) / denom_r if denom_r > 1e-5 else 0.0
        mse_refl = float(np.sum(mask * (true_refl_gray - alpha_r * est_refl_gray) ** 2) / np.sum(mask))
        
        denom_s = np.sum(est_shading_gray**2 * mask)
        alpha_s = np.sum(true_shading_gray * est_shading_gray * mask) / denom_s if denom_s > 1e-5 else 0.0
        mse_shading = float(np.sum(mask * (true_shading_gray - alpha_s * est_shading_gray) ** 2) / np.sum(mask))

        psnr_refl = masked_psnr(true_refl_gray, est_refl_gray, mask)
        ssim_refl = masked_ssim(true_refl_gray, est_refl_gray, mask)
        dssim_refl = 0.5 * (1.0 - ssim_refl)
        sparsity_refl = gradient_sparsity(est_refl_gray, mask)
        
        # Append result
        results.append({
            "Object": tag,
            "Refl_LMSE": lmse_refl,
            "Shading_LMSE": lmse_shading,
            "Combined_LMSE": lmse_combined,
            "Refl_siMSE": mse_refl,
            "Shading_siMSE": mse_shading,
            "Refl_PSNR": psnr_refl,
            "Refl_SSIM": ssim_refl,
            "Refl_DSSIM": dssim_refl,
            "Refl_Sparsity": sparsity_refl,
            "Time_s": elapsed_time
        })
        
    # Create DataFrame and compute averages
    df = pd.DataFrame(results)
    
    # Compute mean across objects
    mean_row = {
        "Object": "Average",
        "Refl_LMSE": df["Refl_LMSE"].mean(),
        "Shading_LMSE": df["Shading_LMSE"].mean(),
        "Combined_LMSE": df["Combined_LMSE"].mean(),
        "Refl_siMSE": df["Refl_siMSE"].mean(),
        "Shading_siMSE": df["Shading_siMSE"].mean(),
        "Refl_PSNR": df["Refl_PSNR"].mean(),
        "Refl_SSIM": df["Refl_SSIM"].mean(),
        "Refl_DSSIM": df["Refl_DSSIM"].mean(),
        "Refl_Sparsity": df["Refl_Sparsity"].mean(),
        "Time_s": df["Time_s"].mean()
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    
    # Save to CSV
    tables_dir = project_root / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path = tables_dir / f"{experiment_id}.csv"
    df.to_csv(csv_output_path, index=False)
    
    print("\n=================== EXPERIMENT SUMMARY ===================")
    print(df.to_string(index=False, formatters={
        "Refl_LMSE": "{:,.4f}".format,
        "Shading_LMSE": "{:,.4f}".format,
        "Combined_LMSE": "{:,.4f}".format,
        "Refl_siMSE": "{:,.4f}".format,
        "Shading_siMSE": "{:,.4f}".format,
        "Refl_PSNR": "{:,.4f}".format,
        "Refl_SSIM": "{:,.4f}".format,
        "Refl_DSSIM": "{:,.4f}".format,
        "Refl_Sparsity": "{:,.4f}".format,
        "Time_s": "{:,.4f}".format
    }))
    print(f"==========================================================")
    print(f"CSV saved to: {csv_output_path}")
    print(f"Combined LMSE Average: {mean_row['Combined_LMSE']:.4f}\n")

if __name__ == "__main__":
    main()
