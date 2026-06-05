import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "backend" / "app"))

from api.datasets import DATASET_REGISTRY  # noqa: E402
from api.deep_learning.model_registry import (  # noqa: E402
    CANONICAL_MODEL_NAMESPACE,
    get_best_checkpoint_path,
    get_checkpoint_dir,
    get_latest_checkpoint_path,
    get_model_dir,
    get_run_config_path,
    is_canonical_model_id,
)
from api.simulation.projector import MU_WATER_60KEV  # noqa: E402
from prep_DLR import (  # noqa: E402
    DLRUNet,
    SpectralFeatureBuilder,
    ClinicalSplitSampler,
    ensure_run_layout,
    append_history,
    build_lr_scheduler,
    exposure_mas_to_diffusion_time,
)

# Diffusion Constants
SIGMA_MIN = 0.002
SIGMA_MAX = 0.5
P_MEAN = -1.2
P_STD = 1.2

CHANNEL_DESCRIPTIONS = [
    "pinv_fbp_signal_only",
    "measurement_null_component",
    "full_fbp_signal_plus_null",
    "diffusion_null_xt_placeholder",
    "diffusion_total_xt_placeholder",
    "coord_x_position",
    "coord_y_position",
]

@dataclass
class DiffusionTrainingConfig:
    dataset_id: str
    training_dataset_ids: Tuple[str, ...]
    n_source: int
    device: str
    epochs: int
    train_steps_per_epoch: int
    val_steps_per_epoch: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    checkpoint_every: int
    exposures_mas: Tuple[float, ...]
    val_fraction: float
    seed: int
    base_channels: int
    warmup_fraction: float
    min_lr_scale: float
    sigma_min: float = SIGMA_MIN
    sigma_max: float = SIGMA_MAX

def save_diffusion_checkpoint(
    dataset_id: str,
    n_source: int,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    config: DiffusionTrainingConfig,
    metrics: Dict[str, float],
    checkpoint_name: str,
) -> None:
    checkpoint_path = get_model_dir(dataset_id, n_source) / checkpoint_name
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": asdict(config),
        "metrics": metrics,
        "feature_channels": CHANNEL_DESCRIPTIONS,
        "training_target_mode": "diffusion_null_only",
    }
    torch.save(payload, checkpoint_path)

def write_diffusion_run_config(config: DiffusionTrainingConfig) -> None:
    # Use a separate config file or a different subdirectory if needed, but for now we follow the pattern
    config_path = get_run_config_path(config.dataset_id, config.n_source).parent / f"diffusion_config_{config.n_source}.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

def sample_diffusion_batch(
    sampler: ClinicalSplitSampler,
    split: str,
    builder: SpectralFeatureBuilder,
    config: DiffusionTrainingConfig,
    batch_size: int,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        inputs: (B, 7, H, W) where placeholders are filled with current noisy state
        targets: (B, 1, H, W) ground truth mu
        sigmas: (B, 1, 1, 1) sampled noise levels
        exposure_time_embedding: (B, 1, 1, 1) exposure-based time conditioning
    """
    device = builder.device
    batch_mu = []
    exposures = []
    
    for mu_np, sample_meta in sampler.sample_batch_slices(split, batch_size, rng):
        batch_mu.append(torch.from_numpy(mu_np).to(device))
        exposures.append(float(rng.choice(config.exposures_mas)))
    
    gt_mu = torch.stack(batch_mu, dim=0).unsqueeze(1) # (B, 1, H, W)
    exposures_torch = torch.tensor(exposures, device=device)
    
    # 1. Project GT to Range/Null
    gt_null = builder.project_batch_null(gt_mu)
    gt_range = gt_mu - gt_null
    
    # 2. Simulate Noisy measurements & FBP components
    # We do this per-sample because builder doesn't support batch measurement yet
    batch_pinv = []
    batch_meas_null = []
    batch_full_fbp = []
    
    with torch.no_grad():
        for i in range(batch_size):
            comps = builder.build_measurement_components(gt_mu[i, 0], i0=exposures[i] * 1e5)
            batch_pinv.append(comps["pinv"])
            batch_meas_null.append(comps["measurement_null"])
            batch_full_fbp.append(comps["full_fbp"])
            
    pinv = torch.stack(batch_pinv, dim=0).unsqueeze(1)
    meas_null = torch.stack(batch_meas_null, dim=0).unsqueeze(1)
    full_fbp = torch.stack(batch_full_fbp, dim=0).unsqueeze(1)
    
    # 3. Sample Sigma (Log-normal distribution for training)
    # sigma = exp(P_MEAN + P_STD * epsilon)
    rnd_normal = torch.randn((batch_size, 1, 1, 1), device=device)
    sigmas = torch.exp(P_MEAN + P_STD * rnd_normal)
    sigmas = torch.clamp(sigmas, config.sigma_min, config.sigma_max)
    
    # 4. Null-space noise forward process
    noise_null = builder.project_batch_null(torch.randn_like(gt_mu))
    xt_null = gt_null + sigmas * noise_null
    xt_full = gt_range + xt_null
    
    # 5. Construct inputs
    # Channel indices:
    # 0: pinv_fbp_signal_only
    # 1: measurement_null_component
    # 2: full_fbp_signal_plus_null
    # 3: diffusion_null_xt_placeholder -> xt_null
    # 4: diffusion_total_xt_placeholder -> xt_full
    # 5: coord_x
    # 6: coord_y
    
    coords_x = builder.coord_x.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    coords_y = builder.coord_y.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    inputs = torch.cat([
        pinv,
        meas_null,
        full_fbp,
        xt_null,
        xt_full,
        coords_x,
        coords_y
    ], dim=1)
    
    exposure_cond = exposure_mas_to_diffusion_time(exposures_torch)
    
    return inputs, gt_mu, sigmas, exposure_cond

def train_diffusion_epoch(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: Optional[LambdaLR],
    sampler: ClinicalSplitSampler,
    builder: SpectralFeatureBuilder,
    config: DiffusionTrainingConfig,
    split: str,
    epoch: int,
    rng: random.Random,
) -> Dict[str, float]:
    is_train = split == "train"
    model.train(is_train)
    steps = config.train_steps_per_epoch if is_train else config.val_steps_per_epoch
    total_loss = 0.0

    for step_idx in range(1, steps + 1):
        inputs, gt_mu, sigmas, exp_cond = sample_diffusion_batch(sampler, split, builder, config, config.batch_size, rng)
        
        # We need a time embedding for the noise level sigma
        # For simplicity in this first version, we'll combine it with exp_cond or use it as the main diffusion time
        # Let's use log(sigma) as the "time" for the UNet bottlenecks
        sigma_time = torch.log(sigmas.squeeze(-1).squeeze(-1))
        
        # Concatenate exposure time and sigma time? Or just use sigma time and pass exposure as another channel if needed.
        # DLRUNet currently takes (B, 1) time. Let's pass sigma_time.
        
        with torch.set_grad_enabled(is_train):
            # The model predicts the denoised image x0 (or the residual component)
            # In our case, we want the model to predict the ground truth mu
            # But the constraint is always in the null space
            prediction = model(inputs, sigma_time)
            
            # Tweedie-like target or direct x0 prediction?
            # Standard EDM training: minimize weighted MSE between model(xt, sigma) and x0
            # weight = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
            # Here sigma_data ~ 0.1 (typical value for medical images in mu domain)
            sigma_data = 0.1
            weight = (sigmas**2 + sigma_data**2) / (sigmas * sigma_data)**2
            
            # Physics Constraint: The residual update must be in the null space
            # Predicted x0 = xt + P_null(model(xt, sigma) - xt)
            residual = builder.project_batch_null(prediction - inputs[:, 4:5, ...])
            recon_x0 = inputs[:, 4:5, ...] + residual
            
            loss = (weight * (recon_x0 - gt_mu)**2).mean()

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        total_loss += float(loss.item())
        if step_idx % 10 == 0 or steps < 10:
            print(f"[{split}] epoch={epoch} step={step_idx}/{steps} loss={loss.item():.6e} lr={optimizer.param_groups[0]['lr']:.6e}", flush=True)

    return {"loss": total_loss / max(steps, 1)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["head", "thorax", "abdomen", "pelvic"])
    parser.add_argument("--n-sources", nargs="+", type=int, default=[80, 240])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-steps-per-epoch", type=int, default=100)
    parser.add_argument("--val-steps-per-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4) # Slightly higher for diffusion
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--reset-training", action="store_true")
    parser.add_argument("--base-channels", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    for n_source in args.n_sources:
        dataset_id = CANONICAL_MODEL_NAMESPACE if len(args.datasets) > 1 else args.datasets[0]
        config = DiffusionTrainingConfig(
            dataset_id=dataset_id,
            training_dataset_ids=tuple(args.datasets),
            n_source=n_source,
            device=args.device,
            epochs=args.epochs,
            train_steps_per_epoch=args.train_steps_per_epoch,
            val_steps_per_epoch=args.val_steps_per_epoch,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=1e-5,
            checkpoint_every=5,
            exposures_mas=(1.0, 10.0, 100.0),
            val_fraction=0.1,
            seed=42,
            base_channels=args.base_channels, # Bigger model for diffusion
            warmup_fraction=0.05,
            min_lr_scale=0.1
        )

        ensure_run_layout(config.dataset_id, config.n_source)
        write_diffusion_run_config(config)

        sampler = ClinicalSplitSampler(config.training_dataset_ids, val_fraction=config.val_fraction, seed=config.seed)
        builder = SpectralFeatureBuilder(config.n_source, device=device)
        
        # We reuse the DLRUNet architecture but with more channels
        model = DLRUNet(in_channels=7, out_channels=1, base_channels=config.base_channels).to(device)
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        total_steps = config.epochs * config.train_steps_per_epoch
        scheduler = build_lr_scheduler(optimizer, total_steps, config.warmup_fraction, config.min_lr_scale)

        rng_train = random.Random(config.seed + 1)
        rng_val = random.Random(config.seed + 2)

        checkpoint_name = f"diffusion_{n_source}.pt"
        best_val = float('inf')

        for epoch in range(1, config.epochs + 1):
            train_results = train_diffusion_epoch(model, optimizer, scheduler, sampler, builder, config, "train", epoch, rng_train)
            with torch.no_grad():
                val_results = train_diffusion_epoch(model, optimizer, None, sampler, builder, config, "val", epoch, rng_val)
            
            summary = {
                "epoch": epoch,
                "train_loss": train_results["loss"],
                "val_loss": val_results["loss"],
                "lr": optimizer.param_groups[0]["lr"]
            }
            append_history(config.dataset_id, config.n_source, summary)
            
            if val_results["loss"] < best_val:
                best_val = val_results["loss"]
                save_diffusion_checkpoint(config.dataset_id, config.n_source, epoch, model, optimizer, scheduler, config, summary, checkpoint_name)
            
            print(f"[Epoch {epoch}] Train Loss: {train_results['loss']:.6e}, Val Loss: {val_results['loss']:.6e}")

if __name__ == "__main__":
    main()
