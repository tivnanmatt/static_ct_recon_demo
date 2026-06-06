import os
import json
import pydicom
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

HU_WINDOWS = {
    "head": (40, 120),
    "thorax": (-600, 1500),
    "abdomen": (50, 400),
    "pelvic": (50, 400)
}

OUTPUT_DIR = Path("/app/backend/app/static/precomputed")

def normalize_hu(img, center, width):
    lower = center - width // 2
    upper = center + width // 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)
    return (img * 255).astype(np.uint8)

def generate_gif(ds_id):
    manifest_path = OUTPUT_DIR / f"{ds_id}_manifest.json"
    if not manifest_path.exists():
        print(f"No manifest for {ds_id}")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Pick a patient with enough slices
    target_p_id = None
    # Sort keys to ensure deterministic choice
    sorted_p_ids = sorted(manifest.keys())
    for p_id in sorted_p_ids:
        if manifest[p_id]["slice_count"] >= 30:
            target_p_id = p_id
            break
    
    if not target_p_id and sorted_p_ids:
        target_p_id = sorted_p_ids[0]

    if not target_p_id:
        print(f"No patients found in {ds_id} manifest")
        return

    info = manifest[target_p_id]
    paths = info["paths"]
    mid = len(paths) // 2
    subset = paths[max(0, mid-15) : min(len(paths), mid+15)]
    
    frames = []
    center, width = HU_WINDOWS.get(ds_id, (40, 400))
    
    print(f"Creating GIF for {ds_id} using {target_p_id}...")
    for s_path in tqdm(subset):
        try:
            ds = pydicom.dcmread(s_path)
            img = ds.pixel_array.astype(np.float32)
            img = img * getattr(ds, 'RescaleSlope', 1.0) + getattr(ds, 'RescaleIntercept', 0.0)
            norm = normalize_hu(img, center, width)
            pil_img = Image.fromarray(norm).resize((256, 256), resample=Image.BILINEAR)
            frames.append(pil_img)
        except Exception as e:
            print(f"Error loading {s_path}: {e}")
            continue
    
    if frames:
        gif_path = OUTPUT_DIR / f"{ds_id}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=60,
            loop=0
        )
        print(f"Saved {gif_path}")

if __name__ == "__main__":
    for ds_id in HU_WINDOWS.keys():
        generate_gif(ds_id)
