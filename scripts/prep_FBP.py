import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from scipy.sparse.linalg import eigsh, LinearOperator

sys.path.append(str(Path(__file__).parent.parent / "backend" / "app"))
from api.datasets import DATASET_REGISTRY
from ct_laboratory import StaticCTProjector2D, build_uniform_static_2d_geometry

def get_ata_op(projector, device, mask=None, verbose=True):
    n_row, n_col = projector.n_row, projector.n_col
    N = n_row * n_col
    count = [0]
    start_time = [time.time()]

    def matvec(x):
        x_np = np.asarray(x)
        is_1d = (x_np.ndim == 1)
        x_in = x_np.reshape(-1, 1) if is_1d else x_np
        n_vecs = x_in.shape[1]
        
        count[0] += 1
        if verbose and (count[0] % 10 == 0 or count[0] < 5):
            print(f"  [ATA Op] Iteration {count[0]} | n_vecs: {n_vecs} | Time: {time.time()-start_time[0]:.2f}s")

        x_torch = torch.from_numpy(x_in.T).to(device).view(n_vecs, n_row, n_col).float()
        
        with torch.no_grad():
            if mask is not None:
                x_torch = x_torch * mask
            
            y_torch = projector.forward(x_torch)
            if y_torch.dim() == 1: y_torch = y_torch.unsqueeze(0)
            
            ata_x_torch = projector.back_project(y_torch)
            if ata_x_torch.dim() == 2: ata_x_torch = ata_x_torch.unsqueeze(0)
            
            if mask is not None:
                ata_x_torch = ata_x_torch * mask
        
        res_np = ata_x_torch.cpu().numpy().reshape(n_vecs, -1).T
        return res_np.ravel() if is_1d else res_np

    return LinearOperator((N, N), matvec=matvec, dtype=np.float32)

def run(n_source, k=4096):
    print(f"\n===== Starting SVD Precomputation: {n_source} Sources, k={k} =====")
    device = torch.device("cuda:0")
    h, w, sp = 256, 256, 1.6
    
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
    
    yy, xx = torch.meshgrid(torch.linspace(-1,1,h,device=device), torch.linspace(-1,1,w,device=device), indexing="ij")
    m = (xx**2+yy**2<=1.0).float()
    
    op = get_ata_op(proj, device, mask=m)
    
    out = Path(f"/app/backend/app/api/simulation/weights/svd_{n_source}")
    s_path = out / "S.pt"
    v_path = out / "V.pt"

    if s_path.exists() and v_path.exists():
        print(f"Loading precomputed SVD from {out}...")
        s_torch = torch.load(s_path).to(device)
        v_torch = torch.load(v_path).to(device)
    else:
        print(f"SVD cache not found at {out}. Computing Eigen decomposition for k={k}...")
        w_vals, v_vecs = eigsh(op, k=k, which="LM", tol=1e-3, maxiter=5000)
        
        out.mkdir(parents=True, exist_ok=True)
        idx = np.argsort(w_vals)[::-1]
        s_vals = np.sqrt(np.maximum(w_vals[idx], 0.0))
        v_vecs = v_vecs[:, idx]
        
        s_torch = torch.from_numpy(s_vals).float().to(device)
        v_torch = torch.from_numpy(v_vecs).float().to(device)
        
        torch.save(s_torch.cpu(), s_path)
        torch.save(v_torch.cpu(), v_path)

    # --- CLINICAL RECONSTRUCTION COMPARISON ---
    print(f"DEBUG: Comparing P-INV vs FBP for {n_source} sources (CLINICAL CASE)...")
    
    # Load a real slice (Head CT)
    ds = DATASET_REGISTRY["head"]
    patients = ds.patients
    if not patients:
        print("WARNING: No patients found for 'head' dataset. Falling back to phantom.")
        phantom = torch.zeros((h, w), device=device)
        yy, xx = torch.meshgrid(torch.linspace(-1,1,h,device=device), torch.linspace(-1,1,w,device=device), indexing="ij")
        phantom[xx**2 + yy**2 < 0.5] = 0.02
    else:
        p_id = patients[0]
        slices = ds.get_patient_slices(p_id)
        # Use middle slice
        s_path = slices[len(slices)//2]
        mu_img = ds.get_processed_slice(s_path)
        phantom = torch.from_numpy(mu_img).float().to(device)
    
    # Forward project
    with torch.no_grad():
        y = proj.forward(phantom.unsqueeze(0))
        # Back project to get laminogram
        laminogram = proj.back_project(y).squeeze(0)
        # MUST MASK LAMINOGRAM FOR SVD DECOMPOSITION
        laminogram = laminogram * m
    
    # 1. P-INV (Null space = 0)
    s2 = s_torch**2
    coeffs = torch.matmul(v_torch.T, laminogram.view(-1, 1))
    recon_pinv_flat = torch.matmul(v_torch, coeffs / s2.view(-1, 1))
    recon_pinv = recon_pinv_flat.view(h, w)
    
    # 2. FBP (Null space gain 1/S_min^2)
    s_min2 = s2.min()
    recon_fbp_flat = (1.0 / s_min2) * laminogram.view(-1, 1) + torch.matmul(v_torch, coeffs * (1.0/s2 - 1.0/s_min2).view(-1, 1))
    recon_fbp = recon_fbp_flat.view(h, w)
    
    # Save individual debug images
    import matplotlib.pyplot as plt
    
    # Save clinical files in separate output folder
    results_dir = out / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Use standard windowing for clinical display (HU context if possible, or just mu range)
    vmax = 0.04 # 0.04 mm^-1 is approx dense bone
    
    plt.imsave(results_dir / "clinical_phantom.png", phantom.cpu().numpy(), cmap='gray', vmin=0, vmax=vmax)
    plt.imsave(results_dir / "clinical_laminogram.png", laminogram.cpu().numpy(), cmap='gray', vmin=0, vmax=vmax*n_source)
    plt.imsave(results_dir / "clinical_pinv.png", recon_pinv.cpu().numpy(), cmap='gray', vmin=0, vmax=vmax)
    plt.imsave(results_dir / "clinical_fbp.png", recon_fbp.cpu().numpy(), cmap='gray', vmin=0, vmax=vmax)
    
    # Also save the comparison plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(phantom.cpu().numpy(), cmap='gray', vmin=0, vmax=vmax); axes[0].set_title("Clinical Phantom")
    axes[1].imshow(laminogram.cpu().numpy(), cmap='gray', vmin=0, vmax=vmax*n_source); axes[1].set_title("BP (A^T y)")
    axes[2].imshow(recon_pinv.cpu().numpy(), cmap='gray', vmin=0, vmax=vmax); axes[2].set_title("P-INV (Null=0)")
    axes[3].imshow(recon_fbp.cpu().numpy(), cmap='gray', vmin=0, vmax=vmax); axes[3].set_title("FBP (Null=1/Smin^2)")
    for a in axes: a.axis('off')
    plt.savefig(results_dir / "clinical_comparison.png")
    plt.close()
    
    print(f"Done {n_source}. Results saved to {results_dir}")
    print(f"P-INV Mean: {recon_pinv.mean().item():.6f}")
    print(f"FBP Mean:   {recon_fbp.mean().item():.6f} (Phantom Mean: {phantom.mean().item():.6f})")

if __name__ == "__main__":
    for ns in [80, 240]:
        run(ns)
