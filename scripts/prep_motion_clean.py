import os
import sys
import json
import math
import time
import base64
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Ensure PYTHONPATH includes backend and ct_laboratory
sys.path.append("/workspace/static_ct_recon_demo")
sys.path.append("/workspace/static_ct_recon_demo/backend/app")
sys.path.append("/ct_laboratory")

# Import the manager
from api.motion import MotionScopeManager, mu_to_hu, hu_to_mu, encode_hu_png_bytes

def create_and_save_animation(frame_paths, out_gif_path, out_mp4_path, original_duration_sec=1.0, target_fps=25):
    if not frame_paths or len(frame_paths) == 0:
        print(f"Warning: No frames found for {out_gif_path.name}")
        return
    
    num_original_frames = len(frame_paths)
    num_target_frames = int(original_duration_sec * target_fps)
    
    print(f"Resampling {out_gif_path.name} from {num_original_frames} to {num_target_frames} frames at {target_fps} FPS...")
    
    resampled_paths = []
    for i in range(num_target_frames):
        t = i / float(num_target_frames)
        orig_idx = min(int(t * num_original_frames), num_original_frames - 1)
        resampled_paths.append(frame_paths[orig_idx])
        
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    ims = []
    for fp in resampled_paths:
        img = Image.open(fp)
        im = ax.imshow(img, cmap="gray", animated=True)
        ims.append([im])
        
    ani = animation.ArtistAnimation(fig, ims, interval=1000.0/target_fps, blit=True, repeat_delay=0)
    
    # Save as gif using pillow writer
    ani.save(str(out_gif_path), writer="pillow", fps=target_fps)
    
    # Save as mp4 using ffmpeg writer
    ani.save(str(out_mp4_path), writer="ffmpeg", fps=target_fps)
    
    plt.close(fig)

def main():
    print("Initializing MotionScopeManager...")
    m = MotionScopeManager()
    
    patient_id = "case7_axslowinf"
    slice_idx = 0
    
    ww = 1500.0
    wl = -500.0
    
    out_root = Path("/workspace/static_ct_recon_demo/backend/app/static/precomputed/motion_clean")
    out_root.mkdir(parents=True, exist_ok=True)
    
    out_ani_dir = Path("/workspace/static_ct_recon_demo/outputs/motion_animations")
    out_ani_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Precompute clean Ground Truth frames: t=0 to 980ms in steps of 20ms
    print("Precomputing Ground Truth frames...")
    for clock_val in [True, False]:
        clock_str = "yes" if clock_val else "no"
        dir_name = out_root / "gt" / f"clock_{clock_str}"
        
        gt_gif_path = out_ani_dir / f"gt_clock_{clock_str}.gif"
        gt_mp4_path = out_ani_dir / f"gt_clock_{clock_str}.mp4"
        
        if gt_gif_path.exists() and gt_mp4_path.exists():
            print(f"Ground Truth clock_{clock_str} already exists. Skipping.")
            continue
            
        dir_name.mkdir(parents=True, exist_ok=True)
        for t_ms in range(0, 1000, 20):
            t_sec = t_ms / 1000.0
            # Interpolate mu with clock overlay
            mu_img = m.interpolate_phase_mu(patient_id, slice_idx, t_sec, overlay_clock=clock_val)
            hu_img = mu_to_hu(mu_img)
            
            # Save as PNG
            png_bytes = encode_hu_png_bytes(hu_img, wl - ww / 2.0, wl + ww / 2.0)
            with open(dir_name / f"frame_{t_ms}.png", "wb") as f:
                f.write(png_bytes)
                
        # Generate Ground Truth GIF & MP4 animations
        gt_frames = sorted(list(dir_name.glob("frame_*.png")), key=lambda p: int(p.stem.split("_")[1]))
        # Ground truth has 50 frames in 1.0s cardiac cycle. Resample to 25 FPS for exact 1.0s playback.
        create_and_save_animation(gt_frames, gt_gif_path, gt_mp4_path, original_duration_sec=1.0, target_fps=25)
                
    # 2. Precompute reconstructions:
    # All combinations of views (80, 240), clock (True, False), method (flashfbp, neuralspark), and RPS (4, 20, 80)
    total_recons = 2 * 2 * 2 * (4 + 20 + 80)
    completed = 0
    print(f"Precomputing {total_recons} reconstructions...")
    
    for views in [80, 240]:
        for clock_val in [True, False]:
            clock_str = "yes" if clock_val else "no"
            for method in ["flashfbp", "neuralspark"]:
                method_val = "flash-fbp" if method == "flashfbp" else "neuralspark"
                for rps in [4, 20, 80]:
                    recon_gif_path = out_ani_dir / f"recon_views_{views}_clock_{clock_str}_method_{method}_rps_{rps}.gif"
                    recon_mp4_path = out_ani_dir / f"recon_views_{views}_clock_{clock_str}_method_{method}_rps_{rps}.mp4"
                    
                    if recon_gif_path.exists() and recon_mp4_path.exists():
                        print(f"Reconstruction views_{views}_clock_{clock_str}_method_{method}_rps_{rps} already exists. Skipping simulation.")
                        completed += rps
                        continue
                        
                    dir_name = out_root / f"views_{views}" / f"clock_{clock_str}" / f"method_{method}" / f"rps_{rps}"
                    dir_name.mkdir(parents=True, exist_ok=True)
                    
                    print(f"Simulating Views={views}, Clock={clock_str}, Method={method}, RPS={rps}...")
                    
                    num_rotations = rps
                    total_views = views * num_rotations
                    
                    current_sinogram = np.zeros((views, 2304), dtype=np.float32)
                    
                    for global_view_index in range(total_views):
                        rotation_index = global_view_index // views
                        view_index = global_view_index % views
                        acquisition_time = global_view_index / max(float(views) * float(rps), 1e-6)
                        
                        # Use the manager's interpolate and project_view logic
                        phase_mu = m.interpolate_phase_mu(patient_id, slice_idx, acquisition_time, overlay_clock=clock_val)
                        current_sinogram[view_index] = m.sim_manager.project_view(phase_mu, views, view_index, m_as=0.1)
                        
                        if view_index == views - 1:
                            # Reconstruct this rotation!
                            start_recon = time.time()
                            reconstruct_data = m.run_reconstruction(
                                method_val,
                                views,
                                current_sinogram.copy(),
                                0.1,
                                ww,
                                wl,
                            )
                            # reconstruct_data["image"] has raw base64 data url. We need to save it as a binary PNG file.
                            img_str = reconstruct_data["image"].replace("data:image/png;base64,", "")
                            img_bytes = base64.b64decode(img_str)
                            
                            with open(dir_name / f"rot_{rotation_index}.png", "wb") as f:
                                f.write(img_bytes)
                                
                            elapsed = (time.time() - start_recon) * 1000.0
                            completed += 1
                            print(f"  Rotation {rotation_index + 1}/{num_rotations} done in {elapsed:.1f}ms ({completed}/{total_recons})")
                            
                            # Reset sinogram
                            current_sinogram = np.zeros((views, 2304), dtype=np.float32)

                    # Now generate Reconstruction GIF & MP4 animations from the saved rot_*.png files
                    rot_frames = sorted(list(dir_name.glob("rot_*.png")), key=lambda p: int(p.stem.split("_")[1]))
                    recon_gif_path = out_ani_dir / f"recon_views_{views}_clock_{clock_str}_method_{method}_rps_{rps}.gif"
                    recon_mp4_path = out_ani_dir / f"recon_views_{views}_clock_{clock_str}_method_{method}_rps_{rps}.mp4"
                    # Play back at exact real-time speed (1.0 second duration) using standard 25 FPS resampling
                    create_and_save_animation(rot_frames, recon_gif_path, recon_mp4_path, original_duration_sec=1.0, target_fps=25)

    print("Precomputation and animation rendering finished successfully!")

if __name__ == "__main__":
    main()
