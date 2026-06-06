#!/usr/bin/env python3
import os
import sys
import numpy as np
import io
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.simulation.projector import sim_manager

def precompute_animations():
    static_dir = Path(__file__).resolve().parent / "static"
    output_dir = static_dir / "sim_animations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting precomputation of simulation animation overlays...")
    print(f"Output directory: {output_dir}")

    for n_source in [80, 240]:
        print(f"Precomputing {n_source}-source simulation animation...")
        n_source_dir = output_dir / str(n_source)
        n_source_dir.mkdir(parents=True, exist_ok=True)

        sim_manager._warmup_projectors(n_source)
        geom = sim_manager.geometry_cache[n_source]
        src_pos = geom["source_positions"]
        mod_centers = geom["module_centers"]
        det_radius = geom["module_radius"]

        # Loop through each view/frame
        for view_idx in range(n_source):
            if view_idx % 20 == 0 or view_idx == n_source - 1:
                print(f"  - Frame {view_idx + 1} / {n_source}...")

            mask = geom["source_module_mask"][view_idx]

            fig = Figure(figsize=(8, 8), dpi=100)
            fig.patch.set_alpha(0.0)
            ax = fig.add_subplot(111)
            ax.set_facecolor((0, 0, 0, 0))
            ax.axis('off')
            ax.set_xlim(-450, 450)
            ax.set_ylim(-450, 450)

            # 1. Inactive Sources (color: '#0957BE', smaller size: 5)
            inactive_indices = [i for i in range(n_source) if i != view_idx]
            inactive_srcs = src_pos[inactive_indices]
            ax.scatter(inactive_srcs[:, 0], inactive_srcs[:, 1], c='#0957BE', s=5, zorder=1, alpha=0.9)

            # 2. Active Source (color: '#2FB1E6', larger circular marker: 150)
            ax.scatter([src_pos[view_idx, 0]], [src_pos[view_idx, 1]], c='#2FB1E6', marker='o', s=150, zorder=10)

            # 3. Detectors (colors: inactive '#188478', active '#29E1CB')
            active_segs = []
            inactive_segs = []
            for i in range(len(mod_centers)):
                center = mod_centers[i]
                angle = np.arctan2(center[1], center[0])
                half_angle = (24.0 / det_radius) 
                p1 = [det_radius * np.cos(angle - half_angle), det_radius * np.sin(angle - half_angle)]
                p2 = [det_radius * np.cos(angle + half_angle), det_radius * np.sin(angle + half_angle)]
                if mask[i]:
                    active_segs.append([p1, p2])
                else:
                    inactive_segs.append([p1, p2])

            if inactive_segs:
                lc_inactive = LineCollection(inactive_segs, colors=['#188478']*len(inactive_segs), linewidths=3, zorder=3, capstyle='round', alpha=0.4)
                ax.add_collection(lc_inactive)
                
            if active_segs:
                lc_active = LineCollection(active_segs, colors=['#29E1CB']*len(active_segs), linewidths=4, zorder=11, capstyle='round')
                ax.add_collection(lc_active)

            # 4. Rays (color: '#29E1CB')
            proj = sim_manager.view_projectors[n_source][view_idx]
            s_coords = proj.src.cpu().numpy()
            d_coords = proj.dst.cpu().numpy()
            
            # Use fixed random choices for reproducible output, or standard random choice
            np.random.seed(view_idx)
            indices = np.random.choice(s_coords.shape[0], min(200, s_coords.shape[0]), replace=False)
            ray_segs = np.stack([s_coords[indices], d_coords[indices]], axis=1)
            lc_rays = LineCollection(ray_segs, color='#29E1CB', alpha=0.15, linewidths=0.5, zorder=5)
            ax.add_collection(lc_rays)

            # Save frame
            frame_path = n_source_dir / f"frame_{view_idx}.png"
            fig.savefig(frame_path, format='png', facecolor='none', edgecolor='none', bbox_inches='tight', pad_inches=0.0)

    print("Success! Precomputation completed successfully.")

if __name__ == "__main__":
    precompute_animations()
