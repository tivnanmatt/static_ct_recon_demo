
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Ensure backend/app is in path
sys.path.append(os.path.dirname(__file__))

# Import the actual classes to ensure we use the exactly implemented logic
from api.datasets import DATASET_REGISTRY
from api.simulation.projector import sim_manager, MU_WATER_60KEV, SVDImageFilter

def debug_svd_comparison(n_source=240):
    print(f"DEBUG: Starting SVD Comparison for N={n_source}...")
    
    # Use a standard dataset slice
    dataset_id = "thorax"
    ds_obj = DATASET_REGISTRY[dataset_id]
    patient_id = ds_obj.get_patient_ids()[0]
    slices = ds_obj.get_patient_slices(patient_id)
    mu_img = ds_obj.get_processed_slice(slices[0])
    
    # 1. Generate Sinogram (Clean)
    sim_manager._warmup_projectors(n_source)
    full_proj = sim_manager.full_projectors[n_source]
    geom = sim_manager.geometry_cache[n_source]
    
    img_torch = torch.from_numpy(mu_img.astype(np.float32)).to(sim_manager.device)
    with torch.no_grad():
        sino_torch = full_proj.forward(img_torch)
        laminogram = full_proj.back_project(sino_torch)

    # 2. Get SVD Weights
    weight_dir = Path(__file__).parent / "api/simulation/weights" / f"svd_{n_source}"
    s = torch.load(weight_dir / "S.pt").to(sim_manager.device)
    v = torch.load(weight_dir / "V.pt").to(sim_manager.device)
    
    # 3. Create P-INV Filter (Null = 0)
    filter_pinv = SVDImageFilter(s, v).to(sim_manager.device)
    # Override logic for exact pseudo-inverse
    with torch.no_grad():
        filter_pinv.null_weight.fill_(0.0)
        filter_pinv.diag_diff.copy_(1.0 / (s**2))
    
    # 4. Create FBP Filter (Null = 1/Smin^2)
    # This uses the current implementation in projector.py
    filter_fbp = SVDImageFilter(s, v).to(sim_manager.device)
    
    # Run Reconstructions
    with torch.no_grad():
        recon_pinv_mu = filter_pinv(laminogram).cpu().numpy()
        recon_fbp_mu = filter_fbp(laminogram).cpu().numpy()
    
    # Convert to HU
    recon_pinv_hu = (recon_pinv_mu * 1000.0 / MU_WATER_60KEV) - 1000.0
    recon_fbp_hu = (recon_fbp_mu * 1000.0 / MU_WATER_60KEV) - 1000.0
    
    # Save comparison image
    output_path = Path(__file__).parent / "static/outputs/svd_debug_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im1 = axes[0].imshow(recon_pinv_hu, cmap='gray', vmin=-200, vmax=400)
    axes[0].set_title("P-INV (Null Space Gain = 0)")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(recon_fbp_hu, cmap='gray', vmin=-200, vmax=400)
    axes[1].set_title(f"FBP (Null Space Gain = 1/Smin^2)")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    plt.suptitle(f"SVD Reconstruction Comparison (N={n_source} Sources)")
    plt.tight_layout()
    plt.savefig(output_path)
    
    print(f"DEBUG: Saved comparison to {output_path}")
    print(f"P-INV: Mean HU={recon_pinv_hu.mean():.2f}")
    print(f"FBP:   Mean HU={recon_fbp_hu.mean():.2f}")

if __name__ == "__main__":
    debug_svd_comparison()
