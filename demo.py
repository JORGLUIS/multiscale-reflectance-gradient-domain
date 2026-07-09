#!/usr/bin/env python3
"""
demo.py
Demonstration script for intrinsic image decomposition.
Loads a sample object from the MIT dataset, decomposes it using the implemented
baselines and advanced multiscale methods, saves the visual outputs, and prints
a comparative table of metrics.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import shutil

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.data.mit import MITIntrinsicDataset
from src.metrics.lmse import local_error
from src.baselines.homomorphic import homomorphic_decomposition
from src.baselines.ssr import ssr_decomposition
from src.baselines.horn import horn_decomposition
from src.decompose.multiscale import multiscale_decomposition

def main():
    print("==================================================")
    # 1. Load data
    print("1. Loading sample object 'box' from MIT dataset...")
    dataset = MITIntrinsicDataset()
    try:
        data = dataset.load_object('box')
    except Exception as e:
        print(f"\nError loading 'box': {e}")
        print("Please make sure you have downloaded the dataset by running:")
        print("  python scripts/download_data.py")
        sys.exit(1)
        
    diffuse = data['diffuse']       # RGB Input
    true_refl = data['reflectance'] # RGB GT Reflectance
    true_shading = data['shading']   # Grayscale GT Shading
    mask = data['mask']             # Binary mask
    
    # Grayscale targets for metric calculation
    true_refl_gray = np.mean(true_refl, axis=2) if true_refl.ndim == 3 else true_refl
    true_shading_gray = np.mean(true_shading, axis=2) if true_shading.ndim == 3 else true_shading
    
    # 2. Setup output folder
    output_dir = project_root / "results" / "demo_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visual results to: {output_dir}\n")
    
    # Save input and GT for reference
    mpimg.imsave(output_dir / "0_input.png", diffuse)
    mpimg.imsave(output_dir / "0_gt_reflectance.png", true_refl)
    mpimg.imsave(output_dir / "0_gt_shading.png", true_shading_gray, cmap='gray')
    
    # 3. Define methods to run
    methods = {
        "Homomorphic": lambda img, msk: homomorphic_decomposition(img, msk, sigma=15.0),
        "SSR (sigma=15)": lambda img, msk: ssr_decomposition(img, msk, sigma=15.0),
        "Retinex-Horn": lambda img, msk: horn_decomposition(img, msk, threshold=0.1),
        "Starlet (base)": lambda img, msk: multiscale_decomposition(
            img, msk, levels=3, threshold_factor=2.0, transform_type="starlet"
        ),
        "Starlet + Color": lambda img, msk: multiscale_decomposition(
            img, msk, levels=3, threshold_factor=2.0, transform_type="starlet",
            color_coherence=True, color_threshold=0.05, color_beta=0.5
        ),
        "MMT (base)": lambda img, msk: multiscale_decomposition(
            img, msk, levels=3, threshold_factor=2.0, transform_type="mmt"
        ),
        "MMT Complete": lambda img, msk: multiscale_decomposition(
            img, msk, levels=3, threshold_factor=2.0, transform_type="mmt",
            scale_coherence=True, color_coherence=True, color_threshold=0.05, color_beta=0.5
        ),
        "Wavelet (db2)": lambda img, msk: multiscale_decomposition(
            img, msk, levels=3, threshold_factor=2.0, transform_type="wavelet"
        ),
        "Wavelet Complete": lambda img, msk: multiscale_decomposition(
            img, msk, levels=3, threshold_factor=2.0, transform_type="wavelet",
            scale_coherence=True, color_coherence=True, color_threshold=0.05, color_beta=0.5
        )
    }
    
    results = []
    plot_data = {}
    print("2. Running descomposition methods...")
    
    for name, func in methods.items():
        print(f"   Executing {name}...")
        t0 = time.time()
        est_shading, est_refl = func(diffuse, mask)
        elapsed = time.time() - t0
        
        # Save output images
        # Clean prefix name for filename
        prefix = name.lower().replace(" (", "_").replace(")", "").replace(" + ", "_").replace("=", "_").replace(" ", "_")
        
        # Normalize and clip for saving
        shading_save = np.clip(est_shading, 0.0, 1.0)
        refl_save = np.clip(est_refl, 0.0, 1.0)
        
        mpimg.imsave(output_dir / f"est_{prefix}_reflectance.png", refl_save)
        mpimg.imsave(output_dir / f"est_{prefix}_shading.png", shading_save, cmap='gray')
        
        if name in ["Starlet (base)", "MMT (base)", "Wavelet (db2)"]:
            plot_data[name] = (refl_save, shading_save)
            
        # Calculate metrics (grayscale)
        est_refl_gray = np.mean(est_refl, axis=2) if est_refl.ndim == 3 else est_refl
        est_shading_gray = np.mean(est_shading, axis=2) if est_shading.ndim == 3 else est_shading
        
        lmse_refl = local_error(true_refl_gray, est_refl_gray, mask, window_size=20, window_shift=10)
        lmse_shading = local_error(true_shading_gray, est_shading_gray, mask, window_size=20, window_shift=10)
        lmse_combined = 0.5 * lmse_refl + 0.5 * lmse_shading
        
        results.append({
            "Method": name,
            "Refl_LMSE": lmse_refl,
            "Shading_LMSE": lmse_shading,
            "Combined_LMSE": lmse_combined,
            "Time_s": elapsed
        })
        
    # 4. Print results table
    print("\n=================== DEMO SUMMARY TABLE ===================")
    print(f"{'Method':<18} | {'Refl LMSE':<10} | {'Shad LMSE':<10} | {'Comb LMSE':<10} | {'Time (s)':<8}")
    print("-" * 64)
    for res in results:
        print(f"{res['Method']:<18} | {res['Refl_LMSE']:<10.4f} | {res['Shading_LMSE']:<10.4f} | {res['Combined_LMSE']:<10.4f} | {res['Time_s']:<8.4f}")
    print("==========================================================")
    print("\nVisual results are successfully saved in results/demo_outputs/.")
    print("You can verify estimated reflectances and shadings there.")
    
    # Generate visual comparison grid
    print("\nGenerating 3x3 visual comparison panel...")
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
    # Row 0: Inputs and Ground Truths
    axes[0, 0].imshow(diffuse)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(true_refl)
    axes[0, 1].set_title("GT Reflectance")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(true_shading_gray, cmap='gray')
    axes[0, 2].set_title("GT Shading")
    axes[0, 2].axis('off')
    
    # Extract plot data
    starlet_refl, starlet_shading = plot_data["Starlet (base)"]
    mmt_refl, mmt_shading = plot_data["MMT (base)"]
    wavelet_refl, wavelet_shading = plot_data["Wavelet (db2)"]
    
    # Row 1: Reflectances
    axes[1, 0].imshow(starlet_refl)
    axes[1, 0].set_title("Starlet Reflectance")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mmt_refl)
    axes[1, 1].set_title("MMT Reflectance")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(wavelet_refl)
    axes[1, 2].set_title("Wavelet (db2) Reflectance")
    axes[1, 2].axis('off')
    
    # Row 2: Shadings
    axes[2, 0].imshow(starlet_shading, cmap='gray')
    axes[2, 0].set_title("Starlet Shading")
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(mmt_shading, cmap='gray')
    axes[2, 1].set_title("MMT Shading")
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(wavelet_shading, cmap='gray')
    axes[2, 2].set_title("Wavelet (db2) Shading")
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    fig_path = output_dir / "visual_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Copy to Informe directory
    informe_dir = project_root / "Informe"
    informe_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fig_path, informe_dir / "visual_comparison.png")
    print(f"Panel saved to {fig_path} and copied to {informe_dir / 'visual_comparison.png'}")
    print("==================================================")

if __name__ == "__main__":
    main()
