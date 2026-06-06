import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

from api.datasets import DATASET_REGISTRY
from api.simulation.projector import SimulationManager
from ct_laboratory import LinearGaussianLogLikelihood

def debug_gd():
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
    
    # 3. Setup Gradient Descent
    projector = sim.full_projectors[n_source]
    likelihood = LinearGaussianLogLikelihood(projector, measurements=y)
    
    # Initial volume (FBP or Zeros)
    fbp_hu = sim.reconstruct_step(n_source, full_sino, step='filter')
    from api.simulation.projector import MU_WATER_60KEV
    fbp_mu = (torch.as_tensor(fbp_hu, device=device, dtype=torch.float32) + 1000.0) * MU_WATER_60KEV / 1000.0
    
    # Sweep LR
    for lr in [1e-6, 5e-7, 1e-7]:
        print(f"\nTesting LR = {lr}")
        x = fbp_mu.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr=lr)
        
        for it in range(21):
            optimizer.zero_grad()
            ll = likelihood(x)
            loss = -1.0 * ll
            loss.backward()
            
            # Diagnostic stats
            with torch.no_grad():
                gn = x.grad.norm().item()
                # Check for divergence
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Iter {it:02d} | Diverged!")
                    break
                if it % 5 == 0:
                    print(f"  Iter {it:02d} | LL: {ll.item():.4e} | Grad Norm: {gn:.4e}")
            
            optimizer.step()

if __name__ == "__main__":
    debug_gd()
