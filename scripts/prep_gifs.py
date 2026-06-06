import os
import json
import pydicom
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

HU_WINDOWS = {
    "head": (40, 80),      # Brain: WL 40, WW 80
    "thorax": (-100, 1600), # Lung: WL -100, WW 1600
    "abdomen": (50, 350),   # Liver: WL 50, WW 350
    "pelvic": (0, 2000)     # Colonography: WL 0, WW 2000
}

OUTPUT_DIR = Path("/app/backend/app/static/precomputed")

def normalize_hu(img, center, width):
    lower = center - width // 2
    upper = center + width // 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)
    return (img * 255).astype(np.uint8)

def make_gif(ds_id):
    manifest_path = OUTPUT_DIR / f"{ds_id}_manifest.json"
    if not manifest_path.exists():
        print(f"Manifest for {ds_id} not found. Run prep_manifests.py first.")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Collect 3 patients to allow more slices per patient without bloated GIF file sizes
    selected_patients = []
    for pid, data in manifest.items():
        if data["slice_count"] >= 30:
            selected_patients.append(pid)
        if len(selected_patients) >= 3:
            break
    
    if not selected_patients:
        print(f"No suitable patients found in {ds_id} for GIF")
        return

    print(f"Creating slow-scroll GIF for {ds_id} using {len(selected_patients)} patients...")
    frames = []
    center, width = HU_WINDOWS.get(ds_id, (40, 400))
    
    for p_id in selected_patients:
        paths = manifest[p_id]["paths"]
        
        # Broader slice range to show more of the volume ("Full patient" feel)
        start_pct = 0.15
        end_pct = 0.85
            
        start_idx = int(len(paths) * start_pct)
        end_idx = int(len(paths) * end_pct)
        
        # Pick 12 slices per patient for a smoother scroll through the anatomy
        indices = np.linspace(start_idx, end_idx, 12, dtype=int)
        
        for idx in indices:
            s_path = paths[idx]
            try:
                ds = pydicom.dcmread(s_path)
                img = ds.pixel_array.astype(np.float32)
                img = img * getattr(ds, 'RescaleSlope', 1.0) + getattr(ds, 'RescaleIntercept', 0.0)

                # Standard Clinical Orientation Correction
                if hasattr(ds, "ImageOrientationPatient"):
                    iop = ds.ImageOrientationPatient
                    # Flip LR if Row X is negative
                    if iop[0] < 0:
                        img = np.fliplr(img)
                    # Flip UD if Col Y is negative
                    if iop[4] < 0:
                        img = np.flipud(img)

                norm = normalize_hu(img, center, width)
                pil_img = Image.fromarray(norm).resize((256, 256), resample=Image.BILINEAR)
                frames.append(pil_img)
            except Exception as e:
                print(f"Error loading slice {s_path}: {e}")
                continue
    
    if frames:
        gif_path = OUTPUT_DIR / f"{ds_id}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=180, # significantly slower (approx 5.5 fps)
            loop=0
        )
        print(f"Saved {gif_path}")

if __name__ == "__main__":
    for ds_id in HU_WINDOWS.keys():
        make_gif(ds_id)
