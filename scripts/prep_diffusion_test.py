import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))
sys.path.append(str(ROOT / "backend" / "app"))

from api.datasets import DATASET_REGISTRY  # noqa: E402
from api.deep_learning.model_registry import (  # noqa: E402
    CANONICAL_MODEL_NAMESPACE,
    get_model_dir,
)
from api.simulation.projector import MU_WATER_60KEV  # noqa: E402
from scripts.prep_DLR import (  # noqa: E402
    DLRUNet,
    SpectralFeatureBuilder,
    exposure_mas_to_diffusion_time,
)

def solve_heun_null_space(
    model: nn.Module,
    builder: SpectralFeatureBuilder,
    inputs: torch.Tensor,
    num_steps: int = 20,
    sigma_min: float = 0.002,
    sigma_max: float = 0.5,
    rho: float = 7.0,
    device: Optional[torch.device] = None,
):
    """
    Reverse diffusion in the null space using Heun's method (2nd order ODE solver).
    Yields intermediate states for streaming.
    """
    if device is None:
        device = inputs.device
    batch_size = inputs.shape[0]

    # Time steps following EDM schedule
    step_indices = torch.arange(num_steps, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # Add t=0

    # Start from X0_range + noise_null
    xt_full = inputs[:, 4:5, ...].clone()
    
    for i in range(num_steps):
        t_cur = t_steps[i]
        t_next = t_steps[i+1]
        
        # update inputs with current state
        inputs[:, 3:4, ...] = builder.project_batch_null(xt_full) # current null
        inputs[:, 4:5, ...] = xt_full # current full
        
        # 1. Predict x0 (denoised)
        with torch.no_grad():
            sigma_time = torch.log(torch.tensor([t_cur], device=device).unsqueeze(0)).repeat(batch_size)
            prediction = model(inputs, sigma_time)
            # Apply null-space constraint on the residual
            residual = builder.project_batch_null(prediction - xt_full)
            x0_hat = xt_full + residual
            
        yield {"step": i+1, "sigma": t_cur.item(), "xt": xt_full, "x0": x0_hat}

        # 2. Convert to score/derivative: d_cur = (xt - x0_hat) / t_cur
        d_cur = (xt_full - x0_hat) / t_cur
        
        # 3. Euler step
        x_next = xt_full + (t_next - t_cur) * d_cur
        
        # 4. 2nd order correction (Heun)
        if t_next > 0:
            inputs[:, 3:4, ...] = builder.project_batch_null(x_next)
            inputs[:, 4:5, ...] = x_next
            with torch.no_grad():
                sigma_time_next = torch.log(torch.tensor([t_next], device=device).unsqueeze(0)).repeat(batch_size)
                prediction_next = model(inputs, sigma_time_next)
                residual_next = builder.project_batch_null(prediction_next - x_next)
                x0_hat_next = x_next + residual_next
                
            d_next = (x_next - x0_hat_next) / t_next
            xt_full = xt_full + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_next)
        else:
            xt_full = x_next

def solve_euler_null_space(
    model: nn.Module,
    builder: SpectralFeatureBuilder,
    inputs: torch.Tensor,
    num_steps: int = 20,
    sigma_min: float = 0.002,
    sigma_max: float = 0.5,
    rho: float = 7.0,
    device: Optional[torch.device] = None,
):
    """
    Reverse diffusion in the null space using Euler's method (1st order ODE solver).
    Yields intermediate states for streaming.
    """
    if device is None:
        device = inputs.device
    batch_size = inputs.shape[0]

    # Time steps following EDM schedule
    step_indices = torch.arange(num_steps, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # Add t=0

    xt_full = inputs[:, 4:5, ...].clone()
    
    for i in range(num_steps):
        t_cur = t_steps[i]
        t_next = t_steps[i+1]
        
        inputs[:, 3:4, ...] = builder.project_batch_null(xt_full)
        inputs[:, 4:5, ...] = xt_full
        
        with torch.no_grad():
            sigma_time = torch.log(torch.tensor([t_cur], device=device).unsqueeze(0)).repeat(batch_size)
            prediction = model(inputs, sigma_time)
            residual = builder.project_batch_null(prediction - xt_full)
            x0_hat = xt_full + residual
            
        yield {"step": i+1, "sigma": t_cur.item(), "xt": xt_full, "x0": x0_hat}

        d_cur = (xt_full - x0_hat) / t_cur
        xt_full = xt_full + (t_next - t_cur) * d_cur
            
    return xt_full

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="head")
    parser.add_argument("--n-source", type=int, default=80)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--exposure-mAs", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    builder = SpectralFeatureBuilder(args.n_source, device=device)
    
    # Load model
    model_dir = get_model_dir(CANONICAL_MODEL_NAMESPACE, args.n_source)
    checkpoint_path = model_dir / f"diffusion_{args.n_source}.pt"
    if not checkpoint_path.exists():
        print(f"Error: Model not found at {checkpoint_path}. Run training first.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle the fact that we might have skipped base_channels or different config
    base_channels = checkpoint["config"]["base_channels"]
    model = DLRUNet(in_channels=7, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load a test slice
    ds = DATASET_REGISTRY[args.dataset]
    patient_id = ds.patients[0]
    slice_index = 0
    slice_path = ds.get_patient_slices(patient_id)[slice_index]
    mu_np = ds.get_processed_slice(slice_path).astype(np.float32)
    gt_mu = torch.from_numpy(mu_np).to(device).unsqueeze(0).unsqueeze(0)

    # Prepare inputs
    comps = builder.build_measurement_components(gt_mu[0,0], i0=args.exposure_mAs * 1e5)
    pinv = comps["pinv"].unsqueeze(0).unsqueeze(0)
    meas_null = comps["measurement_null"].unsqueeze(0).unsqueeze(0)
    full_fbp = comps["full_fbp"].unsqueeze(0).unsqueeze(0)
    
    # Start from Range component + Max Noise in Null space
    gt_null = builder.project_batch_null(gt_mu)
    gt_range = gt_mu - gt_null
    
    sigma_max = checkpoint["config"].get("sigma_max", 0.5)
    noise_null = builder.project_batch_null(torch.randn_like(gt_mu)) * sigma_max
    xt_start = gt_range + noise_null
    
    coords_x = builder.coord_x.unsqueeze(0).unsqueeze(0)
    coords_y = builder.coord_y.unsqueeze(0).unsqueeze(0)
    
    inputs = torch.cat([
        pinv,
        meas_null,
        full_fbp,
        builder.project_batch_null(xt_start),
        xt_start,
        coords_x,
        coords_y
    ], dim=1)
    
    exp_cond = exposure_mas_to_diffusion_time(torch.tensor([args.exposure_mAs], device=device))
    
    # Sampling
    recon = solve_heun_null_space(model, builder, inputs, exp_cond, num_steps=args.num_steps, sigma_max=sigma_max)
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    def to_hu(t):
        return (t.squeeze().cpu().numpy() * 1000.0 / MU_WATER_60KEV) - 1000.0

    axes[0].imshow(to_hu(gt_mu), cmap='gray', vmin=-200, vmax=400)
    axes[0].set_title("Ground Truth")
    axes[1].imshow(to_hu(full_fbp), cmap='gray', vmin=-200, vmax=400)
    axes[1].set_title("Full FBP")
    axes[2].imshow(to_hu(xt_start), cmap='gray', vmin=-200, vmax=400)
    axes[2].set_title("Initial (Range + Noise)")
    axes[3].imshow(to_hu(recon), cmap='gray', vmin=-200, vmax=400)
    axes[3].set_title("Diffusion Recon")
    
    for ax in axes: ax.axis('off')
    
    out_dir = Path("outputs/diffusion_tests")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{args.dataset}_{args.n_source}_{int(args.exposure_mAs)}mAs.png")
    print(f"Result saved to {out_dir}")

if __name__ == "__main__":
    main()
