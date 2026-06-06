import os
import json
import pydicom
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Constants for normalization (same as in datasets.py)
HU_WINDOWS = {
    "head": (40, 120),
    "thorax": (-600, 1500),
    "abdomen": (50, 400),
    "pelvic": (50, 400)
}

ROOTS = {
    "head": "/data/cq500",
    "thorax": "/data/LIDC/LIDC-IDRI",
    "abdomen": "/data/LIHC/TCGA-LIHC",
    "pelvic": "/data/ACRIN/manifest-sFI3R7DS3069120899390652954/CT COLONOGRAPHY"
}

# The scripts runs inside the container or on host.
# Inside container: /app/backend/app/static/precomputed
# On host: /home/staticct/matt/workspace/static_ct_recon_demo/backend/app/static/precomputed
# Since /data is mounted, I'll run it inside the container for speed/consistency.
OUTPUT_DIR = Path("/app/backend/app/static/precomputed")

def normalize_hu(img, center, width):
    lower = center - width // 2
    upper = center + width // 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)
    return (img * 255).astype(np.uint8)

def process_dataset(ds_id, root_path):
    print(f"Processing {ds_id} at {root_path}...")
    root = Path(root_path)
    if not root.exists():
        print(f"Skipping {ds_id}: path not found")
        return

    # Filter for directories
    patient_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    manifest = {}
    
    # We will pick the first valid patient to generate the dataset button GIF
    button_gif_created = False
    
    for p_dir in tqdm(patient_dirs, desc=f"Scanning {ds_id}"):
        p_id = p_dir.name
        # Find CT slices
        slices = []
        for s_file in p_dir.rglob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(s_file), stop_before_pixels=True)
                if getattr(ds, "Modality", "") == "CT":
                    inst = int(getattr(ds, "InstanceNumber", 0))
                    slices.append((s_file, inst))
            except:
                continue
        
        if not slices:
            continue
            
        # Sort by instance number for axial sequence
        slices.sort(key=lambda x: x[1])
        valid_paths = [str(x[0]) for x in slices]
        
        # Save manifest entry
        manifest[p_id] = {
            "slice_count": len(valid_paths),
            "paths": valid_paths
        }
        
        # Create button GIF from mid-section of the first valid patient in this dataset
        if not button_gif_created and len(slices) >= 15:
            mid = len(slices) // 2
            subset = slices[mid-7 : mid+8] # 15 frames
            frames = []
            center, width = HU_WINDOWS.get(ds_id, (40, 400))
            
            for s_path, _ in subset:
                try:
                    ds = pydicom.dcmread(str(s_path))
                    img = ds.pixel_array.astype(np.float32)
                    img = img * getattr(ds, 'RescaleSlope', 1.0) + getattr(ds, 'RescaleIntercept', 0.0)
                    norm = normalize_hu(img, center, width)
                    # Downscale for performance/button size
                    pil_img = Image.fromarray(norm).resize((256, 256), resample=Image.BILINEAR)
                    frames.append(pil_img)
                except:
                    continue
            
            if frames:
                gif_path = OUTPUT_DIR / f"{ds_id}.gif"
                gif_path.parent.mkdir(parents=True, exist_ok=True)
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=60, # ~16 fps for button
                    loop=0
                )
                button_gif_created = True
                print(f"Created GIF for {ds_id} button")

    # Save manifest for the whole dataset
    with open(OUTPUT_DIR / f"{ds_id}_manifest.json", "w") as f:
        json.dump(manifest, f)
    print(f"Saved manifest for {ds_id}")

if __name__ == "__main__":
    for ds_id, root in ROOTS.items():
        process_dataset(ds_id, root)
