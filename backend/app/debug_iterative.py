import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

from api.datasets import DATASET_REGISTRY
from api.simulation.projector import SimulationManager, MU_WATER_60KEV

def debug_iterative():
    print("DEBUG: Initializing SimulationManager...")
    sim = SimulationManager()
    n_source = 240
    
    # 1. Load Data
    print(f"DEBUG: Loading thorax slice...")
    ds = DATASET_REGISTRY['thorax']
    pId = ds.patients[0]
    slices = ds.get_patient_slices(pId)
    mu_img = ds.get_processed_slice(slices[0])
    
    # 2. Simulate Sinogram
    print(f"DEBUG: Simulating sinogram for {n_source} views...")
    full_sino = None
    for i in range(n_source):
        view = sim.project_view(mu_img, n_source, i, m_as=300)
        if full_sino is None:
            full_sino = np.zeros((n_source, len(view)))
        full_sino[i] = view
        
    print(f"DEBUG: Sinogram shape: {full_sino.shape}")
    print(f"DEBUG: Sinogram range: [{full_sino.min():.4f}, {full_sino.max():.4f}], mean: {full_sino.mean():.4f}")
    
    # 3. Run Iterative Recon
    final_results = []
    for lr_to_test in [1.0, 0.1, 0.01, 1e-4]:
        print(f"\n" + "="*50)
        print(f"DEBUG: Testing LR = {lr_to_test} (iters=20, tv=0.0)...")
        results = []
        try:
            failed = False
            for update in sim.run_iterative_recon(n_source, full_sino, num_iters=20, tv_weight=0.0, lr=lr_to_test, use_precond=True):
                it = update["iteration"]
                img = update["image"]
                ll = update["log_likelihood"]
                rmse = update["rmse"]
                
                # Check for NaNs/Infs
                if np.isnan(ll) or np.isinf(ll):
                    print(f"  Iter {it:03d} | FAILED (NaN/Inf)")
                    failed = True
                    break

                # Print extra debug info on first iter
                if it == 1 and hasattr(sim, 'last_precond_s'):
                    s = sim.last_precond_s
                    print(f"    [PRECOND DEBUG] S range: [{s.min().item():.2e}, {s.max().item():.2e}]")
                    
                print(f"  Iter {it:03d} | LL: {ll:10.2e} | RMSE: {rmse:8.4f} | HU mean: {img.mean():6.1f}")
                if it % 5 == 0 or it == 1:
                    results.append((it, img))
                
            if not failed and len(results) > 0:
                print(f"DEBUG: LR {lr_to_test} stable!")
                final_results = results
                break
        except Exception as e:
            print(f"ERROR during iterative recon with LR={lr_to_test}: {e}")
            continue
        
    # 4. Save results (Side-by-side comparison)
    if final_results:
        rows = 2
        cols = (len(final_results) + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        axes = axes.flatten()
        for i, (it, img) in enumerate(final_results):
            im = axes[i].imshow(img, cmap='gray', vmin=-200, vmax=400)
            axes[i].set_title(f"Iteration {it}")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Hide unneeded axes
        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        out_path = Path(__file__).parent / "debug_iter_results.png"
        plt.savefig(out_path)
        print(f"DEBUG: Saved results to {out_path}")
    else:
        print("ERROR: No stable results found!")

if __name__ == "__main__":
    debug_iterative()
