
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Ensure backend/app is in path
sys.path.append(os.path.dirname(__file__))

from api.datasets import DATASET_REGISTRY
from api.simulation.projector import sim_manager, MU_WATER_60KEV

def debug_svd_comparison(dataset_id="thorax", patient_id="LIDC-IDRI-0028", slice_index=0, n_source=240):
    print(f"DEBUG: Starting SVD Comparison for {dataset_id}/{patient_id}...")
    
    ds_obj = DATASET_REGISTRY[dataset_id]
    slices = ds_obj.get_patient_slices(patient_id)
    mu_img = ds_obj.get_processed_slice(slices[slice_index])
    
    # 1. Generate Sinogram
    print("DEBUG: Generating forward projection...")
    full_sino = sim_manager.project_full(mu_img, n_source, m_as=100.0)
    
    # 2. Setup SVD Filter
    sim_manager._warmup_projectors(n_source)
    weight_dir = Path(__file__).parent / "api/simulation/weights" / f"svd_{n_source}"
    if not weight_dir.exists():
        print(f"ERROR: No SVD weights found for N={n_source}")
        return

    s = torch.load(weight_dir / "S.pt").to(sim_manager.device)
    v = torch.load(weight_dir / "V.pt").to(sim_manager.device)
    
    # 3. P-INV Reconstruction (Zeros in null space)
    print("DEBUG: Running P-INV (Zeros in null space)...")
    beta_pinv = 1e10 # Effectively infinity to zero out
    # Or just set weights manually
    s2 = s**2
    
    # Construct filter for pinv
    # Gain in signal space = 1/s^2, Gain in null space = 0
    # H = V @ diag(1/s^2) @ V^T
    # We can use our SVDImageFilter but override the buffers
    from api.simulation.projector import SVDImageFilter
    
    filter_pinv = SVDImageFilter(s, v).to(sim_manager.device)
    filter_pinv.null_weight.fill_(0.0)
    filter_pinv.diag_diff.copy_(1.0 / s2)
    
    # Get laminogram
    full_proj = sim_manager.full_projectors[n_source]
    geom = sim_manager.geometry_cache[n_source]
    sino_active = []
    for i in range(n_source):
        mask = geom["source_module_mask"][i]
        view_sino = full_sino[i]
        for m_idx, active in enumerate(mask):
            if active:
                sino_active.append(view_sino[m_idx*48 : (m_idx+1)*48])
    sino_active = np.concatenate(sino_active)
    sino_torch = torch.from_numpy(sino_active).float().to(sim_manager.device)
    laminogram = full_proj.back_project(sino_torch)
    
    recon_pinv_mu = filter_pinv(laminogram).cpu().numpy()
    recon_pinv_hu = (recon_pinv_mu * 1000.0 / MU_WATER_60KEV) - 1000.0
    
    # 4. FBP Reconstruction (With Null Space Gain 1/S_min^2)
    print("DEBUG: Running FBP (Null space gain 1/S_min^2)...")
    filter_fbp = SVDImageFilter(s, v).to(sim_manager.device) 
    # This uses the default updated logic: diag_diff = 1/s^2 - 1/S_min^2, null_weight = 1/S_min^2
    
    recon_fbp_mu = filter_fbp(laminogram).cpu().numpy()
    recon_fbp_hu = (recon_fbp_mu * 1000.0 / MU_WATER_60KEV) - 1000.0
    
    # 5. Plotting
    output_path = Path(__file__).parent / "static/debug_fbp_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im1 = axes[0].imshow(recon_pinv_hu, cmap='gray', vmin=-200, vmax=400)
    axes[0].set_title("P-INV (Null Space = 0)")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(recon_fbp_hu, cmap='gray', vmin=-200, vmax=400)
    axes[1].set_title(f"FBP (Null Space Gain 1/S_min^2)")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"DEBUG: Saved comparison plot to {output_path}")
    
    # Print stats
    print(f"P-INV: Mean={recon_pinv_hu.mean():.1f}, Std={recon_pinv_hu.std():.1f}, Max={recon_pinv_hu.max():.1f}")
    print(f"FBP:   Mean={recon_fbp_hu.mean():.1f}, Std={recon_fbp_hu.std():.1f}, Max={recon_fbp_hu.max():.1f}")

if __name__ == "__main__":
    debug_svd_comparison()
