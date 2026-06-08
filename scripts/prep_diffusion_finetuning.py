import os
import sys
import argparse
import shutil
import random
import torch
from pathlib import Path

# Fix paths to allow importing from backend and scripts
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
sys.path.append(str(SCRIPTS_DIR))
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "backend" / "app"))

from dataclasses import asdict
from api.deep_learning.model_registry import get_model_dir
from prep_DLR import DLRUNet, SpectralFeatureBuilder, ClinicalSplitSampler, build_lr_scheduler
from prep_diffusion_train import DiffusionTrainingConfig, train_diffusion_epoch, hu_to_atten

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-sources", nargs="+", type=int, default=[80, 240])
    parser.add_argument("--exposures", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0])
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"DEBUG: Using device={device}, epochs={args.epochs}")

    n_sources = args.n_sources
    exposures = args.exposures
    
    for n_source in n_sources:
        model_dir = get_model_dir("main", n_source)
        base_checkpoint_path = model_dir / f"diffusion_{n_source}.pt"
        
        if not base_checkpoint_path.exists():
            print(f"ERROR: Base checkpoint {base_checkpoint_path} not found. Skipping {n_source} sources.")
            continue
            
        # 1. Backup the base checkpoint
        backup_path = model_dir / f"diffusion_{n_source}_original_backup.pt"
        if not backup_path.exists():
            print(f"DEBUG: Backing up {base_checkpoint_path} to {backup_path}")
            shutil.copy(base_checkpoint_path, backup_path)
        else:
            print(f"DEBUG: Backup {backup_path} already exists.")
            
        # 2. Setup sampler and builder
        sampler = ClinicalSplitSampler(["head", "thorax", "abdomen", "pelvic"], val_fraction=0.1, seed=42)
        builder = SpectralFeatureBuilder(n_source, device=device)
        
        # We need to extract the base channels from the checkpoint.
        print(f"DEBUG: Loading base checkpoint {base_checkpoint_path} to extract config.")
        checkpoint = torch.load(base_checkpoint_path, map_location="cpu")
        base_channels = int(checkpoint.get("config", {}).get("base_channels", 32))
        del checkpoint # free memory
        
        # 3. For each exposure level: Copy and fine-tune
        for exposure_mas in exposures:
            exposure_filename = f"diffusion_{n_source}_{exposure_mas}mas.pt"
            exposure_path = model_dir / exposure_filename
            
            # Always (re-)initialize this exposure model from the freshly trained base,
            # then fine-tune. The base changed (new input style), so we do NOT resume the
            # stale exposure weights that were fine-tuned from the old base.
            print(f"DEBUG: Initializing {exposure_path} from base {base_checkpoint_path}.")
            shutil.copy(base_checkpoint_path, exposure_path)
            
            # Setup Training Config for this exposure
            sigma_min = hu_to_atten(1.0)
            sigma_max = hu_to_atten(1000.0)
            
            # Use exact same learning parameters as base diffusion training (2e-4, AdamW, weight-decay 1e-5)
            config = DiffusionTrainingConfig(
                dataset_id="main",
                training_dataset_ids=("head", "thorax", "abdomen", "pelvic"),
                n_source=n_source,
                device=str(device),
                epochs=args.epochs,
                train_steps_per_epoch=100,
                val_steps_per_epoch=10,
                batch_size=8,
                learning_rate=2e-4, # Exact same learning rate as base training
                weight_decay=1e-5,
                checkpoint_every=5,
                exposures_mas=(exposure_mas,), # LOCK to this specific exposure!
                val_fraction=0.1,
                seed=42,
                base_channels=base_channels,
                warmup_fraction=0.05,
                min_lr_scale=0.1,
                sigma_min=sigma_min,
                sigma_max=sigma_max
            )
            
            print(f"--- FINE-TUNING: {n_source} sources | {exposure_mas} mAs ---")
            
            # Load model and load_state_dict from the specific copy
            model = DLRUNet(in_channels=7, out_channels=1, base_channels=base_channels).to(device)
            ckpt = torch.load(exposure_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            
            # Optimizer & Scheduler
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            total_steps = config.epochs * config.train_steps_per_epoch
            scheduler = build_lr_scheduler(optimizer, total_steps, config.warmup_fraction, config.min_lr_scale)
            
            if "optimizer_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception as e:
                    print(f"Warning: Could not restore optimizer state: {e}. Starting fresh.")
                    
            if "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state"])
                except Exception as e:
                    print(f"Warning: Could not restore scheduler state: {e}. Starting fresh.")
            
            # RNG
            rng_train = random.Random(config.seed + 1)
            rng_val = random.Random(config.seed + 2)
            
            # Fine-tune for the requested number of epochs
            for epoch in range(1, args.epochs + 1):
                train_results = train_diffusion_epoch(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    builder=builder,
                    config=config,
                    split="train",
                    epoch=epoch,
                    rng=rng_train
                )
                
                with torch.no_grad():
                    val_results = train_diffusion_epoch(
                        model=model,
                        optimizer=optimizer,
                        scheduler=None,
                        sampler=sampler,
                        builder=builder,
                        config=config,
                        split="val",
                        epoch=epoch,
                        rng=rng_val
                    )
                    
                print(f"EPOCH {epoch}/{args.epochs} - Train Loss: {train_results['loss']:.6e} | Val Loss: {val_results['loss']:.6e}")
                
            # Copy new trained payload and save
            ckpt["epoch"] = ckpt.get("epoch", 0) + args.epochs
            ckpt["model_state"] = model.state_dict()
            ckpt["optimizer_state"] = optimizer.state_dict()
            ckpt["scheduler_state"] = scheduler.state_dict()
            ckpt["config"] = asdict(config)
            ckpt["metrics"] = {"val_loss": val_results["loss"], "train_loss": train_results["loss"]}
            
            torch.save(ckpt, exposure_path)
            print(f"SAVED fine-tuned weights to {exposure_path}")

if __name__ == "__main__":
    main()
