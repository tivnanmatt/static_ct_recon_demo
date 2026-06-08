import sys
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image, ImageSequence

# Physics constants
MU_WATER_60KEV = 0.0183 # mm^-1

def hu_to_mu(hu_img: np.ndarray) -> np.ndarray:
    return np.maximum((hu_img + 1000.0) * MU_WATER_60KEV / 1000.0, 0.0)

def main():
    root_candidates = [
        Path("/data/4d_cardiac_gif"),
        Path("/home/staticct/data/4d_cardiac_gif"),
        Path("/workspace/static_ct_recon_demo/data/4d_cardiac_gif"),
    ]
    root_dir = None
    for candidate in root_candidates:
        if candidate.exists():
            root_dir = candidate
            break
            
    if root_dir is None:
        print("ERROR: Could not find 4d_cardiac_gif directory.")
        sys.exit(1)
        
    print(f"Using search root: {root_dir}")
    gif_files = sorted(list(root_dir.glob("*.gif")))
    if not gif_files:
        print("ERROR: No gif files found in directory.")
        sys.exit(1)
        
    output_dict = {}
    
    for gif_path in gif_files:
        patient_id = gif_path.stem
        print(f"Precomputing {patient_id}...")
        
        img = Image.open(str(gif_path))
        mu_frames = []
        durations = []
        
        for frame in ImageSequence.Iterator(img):
            w, h = frame.size
            ymin, ymax = int(0.05 * h), int(0.95 * h)
            xmin, xmax = int(0.05 * w), int(0.95 * w)
            
            cropped = frame.crop((xmin, ymin, xmax, ymax))
            cropped_w, cropped_h = cropped.size
            zoom_factor = (350.0 / cropped_w) / 1.6
            new_w = int(round(cropped_w * zoom_factor))
            new_h = int(round(cropped_h * zoom_factor))
            
            resized = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)
            arr = np.array(resized.convert('L'), dtype=np.float32)
            hu_resized = -1250.0 + (arr / 255.0) * 1500.0
            
            # Place in center of 256x256 grid padded with -1000.0 (air)
            hu_256 = np.full((256, 256), -1000.0, dtype=np.float32)
            src_h, src_w = hu_resized.shape
            dst_h_start = (256 - src_h) // 2
            dst_w_start = (256 - src_w) // 2
            
            h_crop = min(src_h, 256)
            w_crop = min(src_w, 256)
            src_h_start = max(0, (src_h - 256) // 2)
            src_w_start = max(0, (src_w - 256) // 2)
            
            hu_256[
                dst_h_start : dst_h_start + h_crop,
                dst_w_start : dst_w_start + w_crop
            ] = hu_resized[
                src_h_start : src_h_start + h_crop,
                src_w_start : src_w_start + w_crop
            ]
            hu = np.maximum(hu_256, -1000.0)
            
            mu_frames.append(hu_to_mu(hu))
            durations.append(frame.info.get('duration', 100) / 1000.0)
            
        N = len(mu_frames)
        phases = [float((i / N) * 100.0) for i in range(N)]
        
        cum_times = [0.0]
        curr = 0.0
        for d in durations:
            curr += d
            cum_times.append(curr)
            
        total_duration = sum(durations)
        
        output_dict[patient_id] = {
            "patient_id": patient_id,
            "phases": phases,
            "slice_count": 1,
            "default_wl": -500.0,
            "default_ww": 1500.0,
            "mu_frames": [f.tolist() for f in mu_frames], # Convert to list for portability
            "durations": durations,
            "cum_times": cum_times,
            "total_duration": total_duration,
        }
        
    out_dir = Path(__file__).resolve().parent.parent / "backend" / "app" / "static" / "precomputed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "motion_precomputed.pt"
    
    # Save the dictionary using torch.save
    torch.save(output_dict, out_path)
    print(f"Precomputation complete! Saved precomputed motion records for {len(output_dict)} patients to {out_path}")

if __name__ == "__main__":
    main()
