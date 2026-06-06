import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Add backend to path
sys.path.append("/app/backend")
sys.path.append("/app/backend/app")

try:
    from api.datasets import DATASET_REGISTRY
    from ct_laboratory import StaticCTProjector2D, build_uniform_static_2d_geometry
except ImportError:
    # Fallback for local execution if paths differ
    sys.path.append("/home/staticct/matt/workspace/static_ct_recon_demo/backend")
    sys.path.append("/home/staticct/matt/workspace/static_ct_recon_demo/backend/app")
    from api.datasets import DATASET_REGISTRY
    from ct_laboratory import StaticCTProjector2D, build_uniform_static_2d_geometry

# Constants
MU_WATER_60KEV = 0.0183
PRECOMPUTED_WEIGHTS_DIR = Path("/app/backend/app/api/simulation/weights")
DEBUG_OUTPUT_DIR = Path("/app/backend/app/static/outputs/debug_fbp")
DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_debug_fbp(n_source=80):
    print(f"\n===== Debugging FBP: {n_source} Sources =====")
    device = torch.device("cuda:0")
    h, w, sp = 256, 256, 1.6
    
    # 1. Load clinical image (Head CQ500)
    ds_id = "head"
    ds = DATASET_REGISTRY[ds_id]
    patients = ds.patients
    p_id = "CQ500CT0 CQ500CT0" if "CQ500CT0 CQ500CT0" in patients else patients[0]
    slices = ds.get_patient_slices(p_id)
    s_idx = len(slices) // 2 
    
    mu_img = ds.get_processed_slice(slices[s_idx])
    mu_img = np.nan_to_num(mu_img, nan=0.0)
    mu_img = np.maximum(0, mu_img)
    
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    fov_mask_np = (xx**2 + yy**2 <= 0.98)
    mu_img[~fov_mask_np] = 0.0
    
    # 2. Setup Projector
    M = torch.tensor([[0.0, sp], [sp, 0.0]], device=device, dtype=torch.float32)
    b = torch.tensor([-(w-1)*sp/2.0, -(h-1)*sp/2.0], device=device, dtype=torch.float32)
    source_radius, module_radius, n_module, det_n_col, det_spacing = 400.0, 366.17, 48, 48, 1.0
    
    mask_s = torch.zeros((n_source, n_module), dtype=torch.bool, device=device)
    for s in range(n_source):
        mc = int(round(((s/n_source+0.5)%1.0)*n_module))%n_module
        for o in range(-15,16): 
            mask_s[s,(mc+o)%n_module]=True
            
    Mg = torch.eye(2,device=device, dtype=torch.float32).unsqueeze(0).repeat(n_source,1,1)
    bg = torch.zeros((n_source,2),device=device, dtype=torch.float32)
    asrc = torch.eye(n_source,dtype=torch.bool,device=device)
    
    pos, m_c, m_o, _ = build_uniform_static_2d_geometry(n_source, source_radius, n_module, module_radius, det_n_col, det_spacing, Mg, bg, asrc)
    proj = StaticCTProjector2D(h, w, Mg, bg, pos.to(torch.float32), m_c.to(torch.float32), m_o.to(torch.float32), det_n_col, det_spacing, mask_s, asrc, M, b, backend="cuda", device=device)
    
    # 3. Project
    img_torch = torch.from_numpy(mu_img).to(device).to(torch.float32)
    sinogram = proj.forward(img_torch)
    # Noiseless sinogram
    sinogram_noisy = sinogram 
    
    # 4. Phase 1: Unfiltered Back Projection (Rescaled)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        img_ones = torch.ones((h, w), device=device, dtype=torch.float32)
        norm_map = proj.back_project(proj.forward(img_ones)).clamp(min=1e-6)
    raw_bp = proj.back_project(sinogram_noisy)
    unfiltered_recon = raw_bp / norm_map
    torch.cuda.synchronize()
    t_unfiltered = (time.time() - t0) * 1000.0
    print(f"Phase 1 (Unfiltered BP) Time: {t_unfiltered:.2f} ms")
    
    # 5. Phase 2: Sparse Eigen Filter
    weight_dir = PRECOMPUTED_WEIGHTS_DIR / f"svd_{n_source}"
    s = torch.load(weight_dir / "S.pt").to(device)
    v = torch.load(weight_dir / "V.pt").to(device)
    
    torch.cuda.synchronize()
    t1 = time.time()
    s2_min = (s**2).min()
    range_weight = 1.0 / (s**2 + 1e-9)
    null_weight = 1.0 / (s2_min + 1e-9)
    diag_diff = range_weight - null_weight
    
    x_flat = raw_bp.view(-1, 1)
    coeffs = torch.matmul(v.T, x_flat)
    recon_part = torch.matmul(v, (coeffs * diag_diff.view(-1, 1)))
    filtered = (null_weight * x_flat + recon_part).view(h, w)
    filtered = filtered * torch.from_numpy(fov_mask_np).to(device)
    torch.cuda.synchronize()
    t_filtered = (time.time() - t1) * 1000.0
    print(f"Phase 2 (Spectral SVD Filter) Time: {t_filtered:.2f} ms")
    
    # Stats Function
    def print_stats(name, data):
        data_np = data.cpu().numpy() if torch.is_tensor(data) else data
        corner = data_np[:10, :10]
        center = data_np[h//2-5:h//2+5, w//2-5:w//2+5]
        print(f"\nStats for {name}:")
        print(f"  Overall: mean={np.mean(data_np):.4f}, min={np.min(data_np):.4f}, max={np.max(data_np):.4f}")
        print(f"  Center (10x10): mean={np.mean(center):.4f}")

    print_stats("Unfiltered Recon", unfiltered_recon)
    print_stats("Filtered Recon", filtered)

    # Save outputs
    out_dir = DEBUG_OUTPUT_DIR / f"n{n_source}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    def to_hu(mu):
        return np.clip((mu.cpu().numpy() * 1000.0 / MU_WATER_60KEV) - 1000.0, -1000, 1500)
    
    def win(img, ww=120, wl=40, auto=False):
        if auto:
            win_min, win_max = img.min(), img.max()
            ww = win_max - win_min
        else:
            win_min, win_max = wl - ww/2, wl + ww/2
        
        clipped = np.clip(img, win_min, win_max)
        if ww > 0:
            return (clipped - win_min) / ww
        return np.zeros_like(clipped)

    orig_hu = to_hu(img_torch)
    unfiltered_hu = to_hu(unfiltered_recon)
    filtered_hu = to_hu(filtered)

    print(f"\nWINDOWING LOG (N={n_source}):")
    print(f"  Orig HU: min={orig_hu.min():.1f}, max={orig_hu.max():.1f}")
    print(f"  Unfiltered HU: min={unfiltered_hu.min():.1f}, max={unfiltered_hu.max():.1f}, mean={unfiltered_hu.mean():.1f}")
    print(f"  Filtered HU: min={filtered_hu.min():.1f}, max={filtered_hu.max():.1f}, mean={filtered_hu.mean():.1f}")

    plt.imsave(out_dir / "original.png", win(orig_hu), cmap='gray')
    plt.imsave(out_dir / "unfiltered_bp.png", win(unfiltered_hu, auto=True), cmap='gray')
    plt.imsave(out_dir / "filtered_recon.png", win(filtered_hu), cmap='gray')
    print(f"Saved results to {out_dir}")

if __name__ == "__main__":
    for n in [80, 240]:
        run_debug_fbp(n)
