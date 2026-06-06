import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

from api.datasets import DATASET_REGISTRY
from api.simulation.projector import SimulationManager, SparseEigenPreconditioner, MU_WATER_60KEV, PRECOMPUTED_WEIGHTS_DIR
from ct_laboratory import LinearGaussianLogLikelihood, TotalVariancePrior2D, MaximumAPosterioriReconstructor

def debug_pgd():
    print("DEBUG: Initializing SimulationManager...")
    sim = SimulationManager()
    n_source = 240
    device = sim.device
    
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
    
    # Prepare active sinogram data
    geom = sim.geometry_cache[n_source]
    sino_active = []
    for i in range(n_source):
        mask = geom["source_module_mask"][i]
        view_sino = full_sino[i]
        for m_idx, active in enumerate(mask):
            if active:
                sino_active.append(view_sino[m_idx*48 : (m_idx+1)*48])
    sino_active = np.concatenate(sino_active)
    y = torch.as_tensor(sino_active, device=device, dtype=torch.float32)
    
    # 3. Setup Projector and Preconditioner
    projector = sim.full_projectors[n_source]
    
    weight_dir = PRECOMPUTED_WEIGHTS_DIR / f"svd_{n_source}"
    if not (weight_dir / "S.pt").exists():
        print(f"ERROR: Missing SVD weights at {weight_dir}")
        return

    s = torch.load(weight_dir / "S.pt").to(device).to(torch.float32)
    v = torch.load(weight_dir / "V.pt").to(device).to(torch.float32)
    
    # Keep top 1000 singular values for preconditioner
    k = min(1000, s.numel())
    precond = SparseEigenPreconditioner(s[:k], v[:, :k])
    
    # 4. Setup MAP Reconstructor
    likelihood = LinearGaussianLogLikelihood(projector, measurements=y)
    prior = TotalVariancePrior2D(regularization_weight=0.01)
    
    # Initial volume (FBP)
    fbp_hu = sim.reconstruct_step(n_source, full_sino, step='filter')
    fbp_mu = (torch.as_tensor(fbp_hu, device=device, dtype=torch.float32) + 1000.0) * MU_WATER_60KEV / 1000.0
    
    # Sweep LR
    for lr in [1.0, 0.5, 0.1, 0.05]:
        print(f"\nTesting LR = {lr}")
        reconstructor = MaximumAPosterioriReconstructor(
            log_likelihood_fn=likelihood,
            log_prior_fn=prior,
            volume_init=fbp_mu,
            preconditioner=precond,
            inv_preconditioner=precond.inverse,
            lr=lr
        )
        
        for it in range(21):
            ll, lp, lpost, gn_lik, gn_prior, gn_total = reconstructor.map_step()
            
            # Check for divergence
            if np.isnan(lpost) or np.isinf(lpost):
                print(f"  Iter {it:02d} | Diverged!")
                break
                
            if it % 5 == 0:
                print(f"  Iter {it:02d} | LL: {ll:.4e} | LP: {lp:.4e} | Grad Norm: {gn_total:.4e}")

    print("\nPGD Diagnostic Complete.")

if __name__ == "__main__":
    debug_pgd()
