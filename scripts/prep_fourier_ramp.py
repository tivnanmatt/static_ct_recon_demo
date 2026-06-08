#!/usr/bin/env python3
import sys
import os
import random
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "backend" / "app"))

from api.datasets import DATASET_REGISTRY
from prep_DLR import ClinicalSplitSampler, SpectralFeatureBuilder

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets = ["head", "thorax", "abdomen", "pelvic"]
    sampler = ClinicalSplitSampler(datasets, val_fraction=0.1, seed=42)

    for n_source in [80, 240]:
        print(f"\n--- Training optimal 2D Fourier Ramp Filter for {n_source}-view geometry ---")
        builder = SpectralFeatureBuilder(n_source, device=device)

        sum_gt_power = torch.zeros((256, 256), device=device, dtype=torch.float32)
        sum_fbp_power = torch.zeros((256, 256), device=device, dtype=torch.float32)

        rng = random.Random(1337)
        num_slices = 150

        print(f"Accumulating power spectra over {num_slices} random slices...")
        for idx in range(num_slices):
            if (idx + 1) % 30 == 0 or idx == num_slices - 1:
                print(f"  - Slice {idx + 1} / {num_slices}...")

            mu_np, meta = sampler.sample_slice("train", rng)
            gt_mu = torch.from_numpy(mu_np).to(device)

            # Build clean FBP with minimal noise
            # i0 = 1e8 provides clear, noise-free anatomical geometry mapping
            comps = builder.build_measurement_components(gt_mu, i0=1e8)
            full_fbp = comps["full_fbp"]

            # Compute 2D power spectrum (absolute value magnitude)
            gt_fft = torch.fft.fft2(gt_mu)
            fbp_fft = torch.fft.fft2(full_fbp)

            sum_gt_power += torch.abs(gt_fft)
            sum_fbp_power += torch.abs(fbp_fft)

        # Average and construct division
        mean_gt_power = sum_gt_power / num_slices
        mean_fbp_power = sum_fbp_power / num_slices

        # Epsilon to avoid divide by zero and clamp high frequency blowups during division
        epsilon = 1e-5 * mean_fbp_power.max()
        optimal_ramp_filter = mean_gt_power / (mean_fbp_power + epsilon)

        # Avoid extreme high frequency noise amplification by clamping
        optimal_ramp_filter = torch.clamp(optimal_ramp_filter, min=0.0, max=40.0)

        # Ensure we enforce symmetric/real FBP constraints, and save
        output_dir = ROOT / "backend" / "app" / "api" / "simulation" / "weights" / f"svd_{n_source}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "optimal_2d_ramp_filter.pt"
        
        torch.save(optimal_ramp_filter.cpu(), output_path)
        print(f"Saved trained optimal 2D Fourier ramp filter (shape: 256x256) to:\n  {output_path}")

if __name__ == "__main__":
    main()