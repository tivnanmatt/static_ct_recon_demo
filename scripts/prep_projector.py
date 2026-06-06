import os
import sys
import time
import subprocess
from pathlib import Path

# Add the app directory to sys.path so we can import the projector module
sys.path.append(str(Path(__file__).parent / "backend"))

import torch
import numpy as np
from app.api.simulation.projector import SimulationManager
from ct_laboratory import StaticCTProjector2D, UniformStaticCTProjector2D

def run_benchmarks(n_source=80):
    print(f"\n===== Intensive Benchmarking: {n_source} Sources =====")
    manager = SimulationManager()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Grid config
    h, w = 256, 256
    dummy_img = torch.randn((h, w), device=device)
    
    # We need to manually build geometry bits to test different backends/projectors
    # The manager's warmup handles the 'cuda' production case, but we'll do raw tests here
    manager._warmup_projectors(n_source) # Ensures geometry_cache is populated
    geom = manager.geometry_cache[n_source]
    
    source_positions = torch.from_numpy(geom["source_positions"]).to(device)
    module_centers = torch.from_numpy(geom["module_centers"]).to(device)
    module_orientations = torch.from_numpy(geom["module_orientations"]).to(device)
    source_module_mask = torch.from_numpy(geom["source_module_mask"]).to(device)
    det_n_col = geom["det_n_col"]
    det_spacing = geom["det_spacing"]
    
    spacing = 1.6
    M = torch.tensor([[0.0, spacing], [spacing, 0.0]], device=device)
    b = torch.tensor([-(w-1)*spacing/2.0, -(h-1)*spacing/2.0], device=device)

    backends = ['cpu', 'torch', 'cuda']
    
    results = []

    for backend in backends:
        if backend == 'cpu':
            test_device = torch.device('cpu')
        else:
            test_device = device # cuda
            
        print(f"\nTesting Backend: {backend.upper()} (Device: {test_device})")
        
        try:
            # 1. Single View Test (using StaticCTProjector2D)
            mask = torch.zeros((1, n_source), dtype=torch.bool, device=test_device)
            mask[0, 0] = True
            
            p_single = StaticCTProjector2D(
                n_row=h, n_col=w, M=M.to(test_device), b=b.to(test_device),
                source_positions=source_positions.to(test_device), 
                module_centers=module_centers.to(test_device), 
                module_orientations=module_orientations.to(test_device),
                det_n_col=det_n_col, det_spacing=det_spacing, 
                source_module_mask=source_module_mask.to(test_device),
                active_sources=mask, 
                M_gantry=torch.eye(2, device=test_device).unsqueeze(0), 
                b_gantry=torch.zeros((1, 2), device=test_device),
                backend=backend, device=test_device
            )
            
            # Warm / Precompute
            _ = p_single.forward(dummy_img.to(test_device))
            
            # Measure
            start = time.time()
            iters = 10
            for _ in range(iters):
                _ = p_single.forward(dummy_img.to(test_device))
            t_single = ((time.time() - start) / iters) * 1000.0
            print(f"  [Single View] Avg ({backend}): {t_single:.3f} ms")

            # 2. All View Test (using UniformStaticCTProjector2D)
            # Uniform projector builds its own geometry
            p_full = UniformStaticCTProjector2D(
                n_row=h, n_col=w, M=M.to(test_device), b=b.to(test_device),
                n_source=n_source, source_radius=400.0,
                n_module=48, module_radius=366.17,
                det_n_col=48, det_spacing=1.0,
                M_gantry=torch.eye(2, device=test_device).unsqueeze(0).repeat(n_source, 1, 1),
                b_gantry=torch.zeros((n_source, 2), device=test_device),
                active_sources=torch.eye(n_source, dtype=torch.bool, device=test_device),
                backend=backend, device=test_device
            )
            
            # Warm / Precompute
            _ = p_full.forward(dummy_img.to(test_device))
            
            # Measure
            start = time.time()
            iters = 5
            for _ in range(iters):
                _ = p_full.forward(dummy_img.to(test_device))
            t_full = ((time.time() - start) / iters) * 1000.0
            print(f"  [Full Batch ] Avg ({backend}): {t_full:.3f} ms")
            
            results.append({
                'backend': backend,
                'single': t_single,
                'full': t_full
            })

        except Exception as e:
            print(f"  Error with {backend}: {e}")

    return results

def main():
    print("=== Environment Check ===")
    try:
        smi = subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8')
        print(f"GPU Info: {smi.strip()}")
    except:
        print("GPU Info: nvidia-smi failed")
        
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # 1. Standard Precomputation
    source_configs = [80, 240]
    manager = SimulationManager()
    
    for n in source_configs:
        print(f"\nStandard Warmup for {n} sources...")
        manager._warmup_projectors(n)
        
    # 2. Detailed Benchmarks
    run_benchmarks(80)
    run_benchmarks(240)

    print("\nTests complete.")

if __name__ == "__main__":
    main()
