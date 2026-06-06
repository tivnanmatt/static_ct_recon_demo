import torch
import numpy as np
import time
from pathlib import Path
import sys

# Add backend to path
sys.path.append("/app/backend")
from ct_laboratory import StaticCTProjector2D, build_uniform_static_2d_geometry

def prep_fbp_timing(n_source=240):
    print(f"\n===== Timing FBP & Sparse Eigen Filter: {n_source} Sources =====")
    device = torch.device("cuda:0")
    h, w, sp = 256, 256, 1.6
    
    # 1. Setup Projector (High Fidelity)
    M = torch.tensor([[0.0, sp], [sp, 0.0]], device=device)
    b = torch.tensor([-(w-1)*sp/2.0, -(h-1)*sp/2.0], device=device)
    source_radius, module_radius, n_module, det_n_col, det_spacing = 400.0, 366.17, 48, 48, 1.0
    
    mask_s = torch.zeros((n_source, n_module), dtype=torch.bool, device=device)
    for s in range(n_source):
        mc = int(round(((s/n_source+0.5)%1.0)*n_module))%n_module
        for o in range(-15,16): 
            mask_s[s,(mc+o)%n_module]=True
            
    Mg = torch.eye(2,device=device).unsqueeze(0).repeat(n_source,1,1)
    bg = torch.zeros((n_source,2),device=device)
    asrc = torch.eye(n_source,dtype=torch.bool,device=device)
    
    pos, mc, mo, _ = build_uniform_static_2d_geometry(n_source, source_radius, n_module, module_radius, det_n_col, det_spacing, Mg, bg, asrc)
    proj = StaticCTProjector2D(h, w, Mg, bg, pos, mc, mo, det_n_col, det_spacing, mask_s, asrc, M, b, backend="cuda", device=device)
    
    # 2. Load SVD Weights
    weight_dir = Path(f"/app/backend/app/api/simulation/weights/svd_{n_source}")
    if not weight_dir.exists():
        print(f"ERROR: No weights at {weight_dir}")
        return
    
    s = torch.load(weight_dir / "S.pt").to(device)
    v = torch.load(weight_dir / "V.pt").to(device)
    s2 = s**2
    
    # 3. Dummy Sinogram
    n_ray = proj.src.shape[0]
    sinogram = torch.randn(n_ray, device=device)
    
    # --- PHASE 1: BACK PROJECTION ---
    torch.cuda.synchronize()
    start_bp = time.time()
    laminogram = proj.back_project(sinogram)
    torch.cuda.synchronize()
    time_bp = (time.time() - start_bp) * 1000.0
    
    # --- PHASE 2: SPARSE EIGEN FILTER ---
    # Formula: F = V(S^-2 - s_min^-2 I)V^T + s_min^-2 I
    # Modeled as: y = V @ ((S^-2 - s_min^-2) * (V^T @ x)) + s_min^-2 * x
    s_min = s.min()
    s2_min = s_min**2
    
    # Weightings
    range_weight = 1.0 / (s2 + 1e-9)
    null_weight = 1.0 / (s2_min + 1e-9)
    diag_diff = range_weight - null_weight
    
    x_flat = laminogram.view(-1, 1)
    
    torch.cuda.synchronize()
    start_filter = time.time()
    
    # y = null_weight * x + V @ (diag_diff * (V^T @ x))
    coeffs = torch.matmul(v.T, x_flat)
    coeffs_scaled = coeffs * diag_diff.view(-1, 1)
    recon_part = torch.matmul(v, coeffs_scaled)
    filtered = (null_weight * x_flat + recon_part).view(h, w)
    
    torch.cuda.synchronize()
    time_filter = (time.time() - start_filter) * 1000.0
    
    print(f"Results for N={n_source}:")
    print(f"  Back Projection: {time_bp:.3f} ms")
    print(f"  Sparse Eigen Filter: {time_filter:.3f} ms")
    print(f"  Total Recon Time: {time_bp + time_filter:.3f} ms")

if __name__ == "__main__":
    for ns in [80, 240]:
        prep_fbp_timing(ns)
