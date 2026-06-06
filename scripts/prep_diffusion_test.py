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

SIGMA_DATA = 100.0 * MU_WATER_60KEV / 1000.0 # Must match training, 100 HU in attenuation units


def set_null_state(inputs: torch.Tensor, builder: SpectralFeatureBuilder, null_state: torch.Tensor) -> torch.Tensor:
    """Update diffusion channels while preserving the measured range-space FBP."""
    range_fbp = inputs[:, 0:1, ...]
    inputs[:, 3:4, ...] = builder.project_batch_null(null_state)
    inputs[:, 4:5, ...] = range_fbp + inputs[:, 3:4, ...]
    return inputs


def edm_preconditioning(sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma_data = torch.as_tensor(SIGMA_DATA, device=sigma.device, dtype=sigma.dtype)
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in = 1.0 / (sigma**2 + sigma_data**2).sqrt()
    c_noise = 0.25 * torch.log(sigma)
    return c_skip, c_out, c_in, c_noise


def estimate_null_signal_and_score(model, inputs, sigma, builder):
    """Return EDM null signal estimate and DSM score, both projected to null space."""
    device = inputs.device
    batch_size = inputs.shape[0]
    sigma = torch.as_tensor(sigma, device=device, dtype=inputs.dtype).reshape(1, 1, 1, 1)
    c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma)

    model_inputs = inputs.clone()
    model_inputs[:, :5, ...] *= c_in

    prediction = model(model_inputs, c_noise.flatten().repeat(batch_size))
    prediction_null = builder.project_batch_null(prediction)
    xt_null = inputs[:, 3:4, ...]
    null_signal_hat = builder.project_batch_null(c_skip * xt_null + c_out * prediction_null)
    score_null = builder.project_batch_null((null_signal_hat - xt_null) / sigma.square().clamp_min(1e-12))
    return null_signal_hat, score_null


def probability_flow_ode_step(null_state: torch.Tensor, score_null: torch.Tensor, sigma_cur: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
    delta_variance = sigma_next.square() - sigma_cur.square()
    return null_state - 0.5 * delta_variance * score_null


def langevin_step(null_state: torch.Tensor, score_null: torch.Tensor, builder: SpectralFeatureBuilder, step_size: torch.Tensor) -> torch.Tensor:
    if float(step_size.item()) <= 0.0:
        return null_state
    noise_null = builder.project_batch_null(torch.randn_like(null_state))
    return builder.project_batch_null(null_state + step_size * score_null + torch.sqrt(2.0 * step_size) * noise_null)

def solve_heun_null_space(
    model: nn.Module,
    builder: SpectralFeatureBuilder,
    inputs: torch.Tensor,
    num_steps: int = 20,
    sigma_min: float = 1e-4,
    sigma_max: float = 10.0,
    rho: float = 7.0,
    temperature: float = 0.0,
    device: Optional[torch.device] = None,
):
    """
    Reverse diffusion in the null space using Heun's method (2nd order ODE solver).
    Yields intermediate states for streaming.
    """
    if device is None:
        device = inputs.device
    # Time steps following EDM schedule
    step_indices = torch.arange(num_steps, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # Add t=0

    null_state = builder.project_batch_null(inputs[:, 3:4, ...].clone())
    
    print(f"DEBUG [Solver]: Starting Heun with {num_steps} steps. t_max={t_steps[0]:.4f}")
    
    for i in range(num_steps):
        t_cur = t_steps[i]
        t_next = t_steps[i+1]
        
        inputs = set_null_state(inputs, builder, null_state)
        
        # 1. Score function estimation in null space only.
        with torch.no_grad():
            null_hat, score_cur = estimate_null_signal_and_score(model, inputs, t_cur, builder)
        recon_hat = inputs[:, 0:1, ...] + null_hat
            
        yield {"step": i+1, "sigma": t_cur.item(), "xt": inputs[:, 4:5, ...], "null_hat": null_hat, "x0": recon_hat}

        # 2. Probability-flow ODE step using dx/d(var) = -0.5 * score.
        x_next = probability_flow_ode_step(null_state, score_cur, t_cur, t_next)
        
        # 3. 2nd order correction (Heun), still in null space only.
        if t_next > 0:
            inputs = set_null_state(inputs, builder, x_next)
            with torch.no_grad():
                _, score_next = estimate_null_signal_and_score(model, inputs, t_next, builder)
            delta_variance = t_next.square() - t_cur.square()
            null_state = builder.project_batch_null(null_state - 0.25 * delta_variance * (score_cur + score_next))
        else:
            null_state = builder.project_batch_null(x_next)

        if temperature > 0.0 and t_next > 0:
            inputs = set_null_state(inputs, builder, null_state)
            with torch.no_grad():
                _, score_next = estimate_null_signal_and_score(model, inputs, t_next, builder)
            base_step = 0.5 * torch.clamp(t_cur.square() - t_next.square(), min=0.0)
            null_state = langevin_step(null_state, score_next, builder, base_step * float(temperature))

def solve_euler_null_space(
    model: nn.Module,
    builder: SpectralFeatureBuilder,
    inputs: torch.Tensor,
    num_steps: int = 20,
    sigma_min: float = 0.002,
    sigma_max: float = 0.5,
    rho: float = 7.0,
    temperature: float = 0.0,
    device: Optional[torch.device] = None,
):
    """
    Reverse diffusion in the null space using Euler's method (1st order ODE solver).
    Yields intermediate states for streaming.
    """
    if device is None:
        device = inputs.device
    # Time steps following EDM schedule
    step_indices = torch.arange(num_steps, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # Add t=0

    null_state = builder.project_batch_null(inputs[:, 3:4, ...].clone())
    
    # Debug info to verify parameter usage
    print(f"DEBUG [Euler Solver]: sigma_max={sigma_max:.4f}, sigma_min={sigma_min:.4f}, steps={num_steps}")
    
    for i in range(num_steps):
        t_cur = t_steps[i]
        t_next = t_steps[i+1]
        
        inputs = set_null_state(inputs, builder, null_state)
        
        with torch.no_grad():
            null_hat, score_cur = estimate_null_signal_and_score(model, inputs, t_cur, builder)
        recon_hat = inputs[:, 0:1, ...] + null_hat
            
        yield {"step": i+1, "sigma": t_cur.item(), "xt": inputs[:, 4:5, ...], "null_hat": null_hat, "x0": recon_hat}

        null_state = probability_flow_ode_step(null_state, score_cur, t_cur, t_next)

        if temperature > 0.0 and t_next > 0:
            inputs = set_null_state(inputs, builder, null_state)
            with torch.no_grad():
                _, score_next = estimate_null_signal_and_score(model, inputs, t_next, builder)
            base_step = 0.5 * torch.clamp(t_cur.square() - t_next.square(), min=0.0)
            null_state = langevin_step(null_state, score_next, builder, base_step * float(temperature))

    return inputs[:, 0:1, ...] + null_state

def solve_langevin_walk_only(
    model: nn.Module,
    builder: SpectralFeatureBuilder,
    inputs: torch.Tensor,
    num_steps: int = 100,
    sigma_min: float = 0.001,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
):
    """
    Performs perpetual Langevin steps at constant noise level sigma_min (no probability flow / ODE steps).
    Yields intermediate states for streaming.
    """
    if device is None:
        device = inputs.device
    
    null_state = builder.project_batch_null(inputs[:, 3:4, ...].clone())
    t_cur = torch.as_tensor(sigma_min, device=device, dtype=inputs.dtype)

    print(f"DEBUG [Langevin Walk]: Starting Langevin walk, steps={num_steps}, sigma_min={sigma_min}, temp={temperature}")

    for i in range(num_steps):
        inputs = set_null_state(inputs, builder, null_state)
        
        with torch.no_grad():
            null_hat, score_cur = estimate_null_signal_and_score(model, inputs, t_cur, builder)
        recon_hat = inputs[:, 0:1, ...] + null_hat
        
        yield {"step": i+1, "sigma": t_cur.item(), "xt": inputs[:, 4:5, ...], "null_hat": null_hat, "x0": recon_hat}

        # Step size is proportional to noise variance (sigma_min^2) and calibrated with temperature.
        step_size = torch.as_tensor(0.05 * (sigma_min**2) * max(0.05, float(temperature)), device=device, dtype=inputs.dtype)
        null_state = langevin_step(null_state, score_cur, builder, step_size)

def solve_combined_diffusion_langevin(
    model: nn.Module,
    builder: SpectralFeatureBuilder,
    inputs: torch.Tensor,
    num_steps_diff: int = 20,
    num_steps_lang: int = 100,
    sigma_max: float = 0.5,
    sigma_min: float = 0.002,
    rho: float = 7.0,
    temperature: float = 0.0,
    solver: str = "heun",
    device: Optional[torch.device] = None,
):
    """
    Combined reverse diffusion sampling in the null space followed by continuous Langevin walk at constant sigma_min.
    """
    if device is None:
        device = inputs.device

    # --- PART 1: DIFFUSION PHASE ---
    if num_steps_lang > 0:
        # If Langevin walk steps follow, the diffusion bridge/phase ends exactly at sigma_min.
        step_indices = torch.arange(num_steps_diff + 1, device=device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / num_steps_diff * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    else:
        # If no Langevin walk, the diffusion phase proceeds all the way to t=0.0.
        step_indices = torch.arange(num_steps_diff, device=device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps_diff - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # Add t=0

    null_state = builder.project_batch_null(inputs[:, 3:4, ...].clone())
    total_steps = num_steps_diff + num_steps_lang

    print(f"DEBUG [Combined Solver]: Starting {solver} with {num_steps_diff} steps + {num_steps_lang} Langevin steps. total_steps={total_steps}")

    for i in range(num_steps_diff):
        t_cur = t_steps[i]
        t_next = t_steps[i+1]

        inputs = set_null_state(inputs, builder, null_state)

        with torch.no_grad():
            null_hat, score_cur = estimate_null_signal_and_score(model, inputs, t_cur, builder)
        recon_hat = inputs[:, 0:1, ...] + null_hat

        yield {
            "step": i + 1,
            "total_steps": total_steps,
            "sigma": t_cur.item(),
            "xt": inputs[:, 4:5, ...].clone(),
            "null_hat": null_hat.clone(),
            "x0": recon_hat.clone()
        }

        # probability-flow ODE step
        x_next = probability_flow_ode_step(null_state, score_cur, t_cur, t_next)

        if solver == "heun":
            if t_next > 0:
                inputs = set_null_state(inputs, builder, x_next)
                with torch.no_grad():
                    _, score_next = estimate_null_signal_and_score(model, inputs, t_next, builder)
                delta_variance = t_next.square() - t_cur.square()
                null_state = builder.project_batch_null(null_state - 0.25 * delta_variance * (score_cur + score_next))
            else:
                null_state = builder.project_batch_null(x_next)
        else: # euler
            null_state = probability_flow_ode_step(null_state, score_cur, t_cur, t_next)

        if temperature > 0.0 and t_next > 0:
            inputs = set_null_state(inputs, builder, null_state)
            with torch.no_grad():
                _, score_next = estimate_null_signal_and_score(model, inputs, t_next, builder)
            base_step = 0.5 * torch.clamp(t_cur.square() - t_next.square(), min=0.0)
            null_state = langevin_step(null_state, score_next, builder, base_step * float(temperature))

    # set final state of diffusion
    inputs = set_null_state(inputs, builder, null_state)

    # --- PART 2: CONTINUOUS LANGEVIN WALK PHASE ---
    if num_steps_lang > 0:
        t_langevin = torch.as_tensor(sigma_min, device=device, dtype=inputs.dtype)
        for i in range(num_steps_lang):
            inputs = set_null_state(inputs, builder, null_state)

            with torch.no_grad():
                null_hat, score_cur = estimate_null_signal_and_score(model, inputs, t_langevin, builder)
            recon_hat = inputs[:, 0:1, ...] + null_hat

            yield {
                "step": num_steps_diff + i + 1,
                "total_steps": total_steps,
                "sigma": t_langevin.item(),
                "xt": inputs[:, 4:5, ...].clone(),
                "null_hat": null_hat.clone(),
                "x0": recon_hat.clone()
            }

            # Step size proportional to noise variance (sigma_min^2) and temperature
            step_size = torch.as_tensor(0.05 * (sigma_min**2) * max(0.05, float(temperature if temperature > 0.0 else 0.5)), device=device, dtype=inputs.dtype)
            null_state = langevin_step(null_state, score_cur, builder, step_size)

def run_single_test(dataset, n_source, num_steps, temperature, exposure_mAs, device, out_dir):
    builder = SpectralFeatureBuilder(n_source, device=device)
    
    # Load model
    model_dir = get_model_dir(CANONICAL_MODEL_NAMESPACE, n_source)
    checkpoint_path = model_dir / f"diffusion_{n_source}.pt"
    if not checkpoint_path.exists():
        print(f"Error: Model not found at {checkpoint_path}. Skipped.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_channels = checkpoint["config"]["base_channels"]
    model = DLRUNet(in_channels=7, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load a test slice
    ds = DATASET_REGISTRY[dataset]
    patient_id = ds.patients[0]
    slice_index = 0
    slice_path = ds.get_patient_slices(patient_id)[slice_index]
    mu_np = ds.get_processed_slice(slice_path).astype(np.float32)
    gt_mu = torch.from_numpy(mu_np).to(device).unsqueeze(0).unsqueeze(0)

    # Prepare inputs
    comps = builder.build_measurement_components(gt_mu[0,0], i0=exposure_mAs * 1e5)
    pinv = comps["pinv"].unsqueeze(0).unsqueeze(0)
    meas_null = comps["measurement_null"].unsqueeze(0).unsqueeze(0)
    full_fbp = comps["full_fbp"].unsqueeze(0).unsqueeze(0)
    
    # Validation-style testing: Instead of reverse generative ODE integration, we directly testing
    # the 1-step EDM denoiser function. We add null-space noise from the forward process at 1000 HU to
    # the GROUND TRUTH null-space component (plus the measurement-null shift if required), and test
    # how well Tweedie's formula predicts the clean target null.
    gt_null = builder.project_batch_null(gt_mu)
    noise_std = 1000.0 * MU_WATER_60KEV / 1000.0 # 1000 HU
    noise_null = builder.project_batch_null(torch.randn_like(gt_mu)) * noise_std
    
    # Forward process corrupted null and total slice
    xt_null = gt_null + noise_null
    xt_start = pinv + xt_null
    
    coords_x = builder.coord_x.unsqueeze(0).unsqueeze(0)
    coords_y = builder.coord_y.unsqueeze(0).unsqueeze(0)
    
    inputs = torch.cat([
        pinv,
        meas_null,
        full_fbp,
        xt_null,
        xt_start,
        coords_x,
        coords_y
    ], dim=1)
    
    # Directly query the EDM 1-step denoiser to estimate/denoise the signal
    with torch.no_grad():
        null_hat, _ = estimate_null_signal_and_score(model, inputs, noise_std, builder)
    
    # Generative Reconstruction = the static range-space FBP (pinv) + one-step estimated clean null_hat
    recon = pinv + null_hat
    
    # Plotting
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    def to_hu(t):
        return (t.squeeze().cpu().numpy() * 1000.0 / MU_WATER_60KEV) - 1000.0

    def component_to_hu(t):
        return t.squeeze().cpu().numpy() * 1000.0 / MU_WATER_60KEV

    axes[0].imshow(to_hu(gt_mu), cmap='gray', vmin=-200, vmax=400)
    axes[0].set_title("Ground Truth")
    axes[1].imshow(to_hu(full_fbp), cmap='gray', vmin=-200, vmax=400)
    axes[1].set_title("Initial FBP (full)")
    axes[2].imshow(component_to_hu(null_hat), cmap='gray', vmin=-100, vmax=100)
    axes[2].set_title("Predicted Null Residual")
    axes[3].imshow(to_hu(inputs[:, 4:5, ...]), cmap='gray', vmin=-200, vmax=400)
    axes[3].set_title("Forward Process (1000 HU)")
    axes[4].imshow(to_hu(recon), cmap='gray', vmin=-200, vmax=400)
    axes[4].set_title("Denoised Reconstruction")
    
    for ax in axes: ax.axis('off')
    
    fig.tight_layout()
    img_name = f"{dataset}_{n_source}views_{int(exposure_mAs)}mAs.png"
    plt.savefig(out_dir / img_name, bbox_inches='tight')
    plt.close(fig)
    print(f"Result saved to {out_dir / img_name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="head")
    parser.add_argument("--n-source", type=int, default=80)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--exposure-mAs", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--all", action="store_true", help="Run on all datasets, views, and exposure levels")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    out_dir = ROOT / "outputs" / "diffusion_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        datasets = ["head", "thorax", "abdomen", "pelvic"]
        views = [80, 240]
        exposures = [1.0, 10.0, 100.0]
        
        print(f"Running full sweep of prep_diffusion_test.py over {len(datasets)} datasets, {len(views)} layout views, and {len(exposures)} exposure levels...")
        for dataset in datasets:
            for n_source in views:
                for exposure in exposures:
                    print(f"\nEvaluating: Dataset={dataset}, Views={n_source}, Exposure={exposure}mAs, Temp={args.temperature}")
                    run_single_test(
                        dataset=dataset,
                        n_source=n_source,
                        num_steps=args.num_steps,
                        temperature=args.temperature,
                        exposure_mAs=exposure,
                        device=device,
                        out_dir=out_dir
                    )
    else:
        run_single_test(
            dataset=args.dataset,
            n_source=args.n_source,
            num_steps=args.num_steps,
            temperature=args.temperature,
            exposure_mAs=args.exposure_mAs,
            device=device,
            out_dir=out_dir
        )


if __name__ == "__main__":
    main()
