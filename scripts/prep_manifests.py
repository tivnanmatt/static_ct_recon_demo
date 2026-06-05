import json
import pydicom
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOTS = {
    "head": "/data/cq500",
    "thorax": "/data/LIDC/LIDC-IDRI",
    "abdomen": "/data/LIHC/TCGA-LIHC",
    "pelvic": "/data/ACRIN/manifest-sFI3R7DS3069120899390652954/CT COLONOGRAPHY"
}

OUTPUT_DIR = Path("/app/backend/app/static/precomputed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCOUT_KEYWORDS = (
    "SCOUT",
    "LOCALIZER",
    "TOPO",
    "TOPOGRAM",
    "SURVIEW",
    "SCANOGRAM",
    "PILOT",
)


def clean_val(val):
    if val is None:
        return "Unknown"
    if hasattr(val, "real"):
        val_str = str(val)
        return float(val) if "." in val_str else int(val)
    if isinstance(val, (pydicom.multival.MultiValue, list, tuple)):
        return [str(x) for x in val]
    return str(val)


def safe_upper_str(val):
    if val is None:
        return ""
    if isinstance(val, (pydicom.multival.MultiValue, list, tuple)):
        return "\\".join(str(x) for x in val).upper()
    return str(val).upper()


def get_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def get_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def get_vector(value, length):
    try:
        vec = np.asarray([float(x) for x in value], dtype=np.float64)
    except Exception:
        return None
    if vec.shape != (length,):
        return None
    return vec


def get_axial_geometry(ds):
    if getattr(ds, "Modality", "") != "CT":
        return None

    rows = get_int(getattr(ds, "Rows", None))
    cols = get_int(getattr(ds, "Columns", None))
    if rows is None or cols is None or rows != cols:
        return None

    pixel_spacing = get_vector(getattr(ds, "PixelSpacing", None), 2)
    if pixel_spacing is None or np.any(pixel_spacing <= 0):
        return None

    iop = get_vector(getattr(ds, "ImageOrientationPatient", None), 6)
    if iop is None:
        return None

    row_dir = iop[:3]
    col_dir = iop[3:]
    normal = np.cross(row_dir, col_dir)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return None
    normal = normal / norm

    # Keep only near-axial slices and reject coronal/sagittal scouts.
    if abs(normal[2]) < 0.9:
        return None

    ipp = get_vector(getattr(ds, "ImagePositionPatient", None), 3)
    if ipp is None:
        slice_position = get_float(getattr(ds, "SliceLocation", None))
        if slice_position is None:
            return None
    else:
        slice_position = float(np.dot(ipp, normal))

    return {
        "rows": rows,
        "cols": cols,
        "pixel_spacing": pixel_spacing,
        "iop": iop,
        "slice_position": slice_position,
    }


def is_ct_scout(ds):
    if safe_upper_str(getattr(ds, "Modality", "")) != "CT":
        return False

    image_type = safe_upper_str(getattr(ds, "ImageType", ""))
    if "LOCALIZER" in image_type:
        return True

    combined = " ".join(
        [
            safe_upper_str(getattr(ds, "SeriesDescription", "")),
            safe_upper_str(getattr(ds, "ProtocolName", "")),
        ]
    )
    return any(keyword in combined for keyword in SCOUT_KEYWORDS)


def compatible_with_group(group, geom):
    return (
        group["rows"] == geom["rows"]
        and group["cols"] == geom["cols"]
        and np.allclose(group["pixel_spacing"], geom["pixel_spacing"], atol=1e-3)
        and np.allclose(np.abs(group["iop"]), np.abs(geom["iop"]), atol=1e-3)
    )


def build_metadata(ds, patient_id, study_uid, series_uid, geom):
    return {
        "PatientID": patient_id,
        "StudyUID": study_uid,
        "SeriesUID": series_uid,
        "SeriesDescription": clean_val(getattr(ds, "SeriesDescription", "Unknown")),
        "ProtocolName": clean_val(getattr(ds, "ProtocolName", "Unknown")),
        "StudyDescription": clean_val(getattr(ds, "StudyDescription", "Unknown")),
        "Manufacturer": clean_val(getattr(ds, "Manufacturer", "Unknown")),
        "Model": clean_val(getattr(ds, "ManufacturerModelName", "Unknown")),
        "KVP": clean_val(getattr(ds, "KVP", "Unknown")),
        "Exposure": clean_val(getattr(ds, "Exposure", "Unknown")),
        "mAs": clean_val(getattr(ds, "ExposureTime", "Unknown")),
        "TubeCurrent": clean_val(getattr(ds, "XRayTubeCurrent", "Unknown")),
        "Filter": clean_val(getattr(ds, "ConvolutionKernel", "Unknown")),
        "SliceThickness": clean_val(getattr(ds, "SliceThickness", "Unknown")),
        "PixelSpacing": [float(geom["pixel_spacing"][0]), float(geom["pixel_spacing"][1])],
        "Rows": geom["rows"],
        "Columns": geom["cols"],
        "RescaleIntercept": clean_val(getattr(ds, "RescaleIntercept", "Unknown")),
        "RescaleSlope": clean_val(getattr(ds, "RescaleSlope", "Unknown")),
        "WindowCenter": clean_val(getattr(ds, "WindowCenter", "Unknown")),
        "WindowWidth": clean_val(getattr(ds, "WindowWidth", "Unknown")),
        "ImageOrientationPatient": [float(x) for x in geom["iop"]],
        "Modality": "CT",
        "Ordering": "slice_position",
    }

def scan_patient(p_dir):
    p_id = p_dir.name
    series_groups = {}
    
    for s_file in p_dir.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(s_file), stop_before_pixels=True)
            if is_ct_scout(ds):
                continue
            geom = get_axial_geometry(ds)
            if geom is None:
                continue

            study_uid = str(getattr(ds, "StudyInstanceUID", "Unknown"))
            series_uid = str(getattr(ds, "SeriesInstanceUID", "Unknown"))
            group_key = (study_uid, series_uid)
            group = series_groups.get(group_key)
            if group is None:
                group = {
                    "study_uid": study_uid,
                    "series_uid": series_uid,
                    "rows": geom["rows"],
                    "cols": geom["cols"],
                    "pixel_spacing": geom["pixel_spacing"],
                    "iop": geom["iop"],
                    "records": [],
                    "metadata": build_metadata(ds, p_id, study_uid, series_uid, geom),
                }
                series_groups[group_key] = group

            if not compatible_with_group(group, geom):
                continue

            group["records"].append({
                "path": str(s_file),
                "instance": get_int(getattr(ds, "InstanceNumber", None), default=-1),
                "position": geom["slice_position"],
            })
        except Exception:
            continue
    
    if not series_groups:
        return p_id, None, None

    candidate_groups = []
    for group in series_groups.values():
        if not group["records"]:
            continue
        records = sorted(group["records"], key=lambda item: (item["position"], item["instance"], item["path"]))
        deduped_records = []
        last_position = None
        for record in records:
            if last_position is not None and abs(record["position"] - last_position) < 1e-3:
                continue
            deduped_records.append(record)
            last_position = record["position"]
        if len(deduped_records) < 2:
            continue
        candidate_groups.append((len(deduped_records), group["series_uid"], group, deduped_records))

    if not candidate_groups:
        return p_id, None, None

    _, _, best_group, final_records = max(candidate_groups, key=lambda item: (item[0], item[1]))
    
    patient_data = {
        "slice_count": len(final_records),
        "paths": [record["path"] for record in final_records],
        "instances": [record["instance"] for record in final_records],
        "positions": [record["position"] for record in final_records],
        "metadata": best_group["metadata"],
    }
    
    metadata = dict(best_group["metadata"])
    metadata["SliceCount"] = len(final_records)
    
    return p_id, patient_data, metadata

def process_dataset(ds_id, root_path):
    print(f"Scanning {ds_id} manifest...")
    root = Path(root_path)
    if not root.exists():
        print(f"Skipping {ds_id}: path not found")
        return

    patient_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    manifest = {}
    rows = []
    
    # Increased worker count for faster scanning
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(scan_patient, d): d for d in patient_dirs}
        for future in tqdm(as_completed(futures), total=len(futures), desc=ds_id):
            p_id, data, meta = future.result()
            if data:
                manifest[p_id] = data
                meta["Dataset"] = ds_id
                rows.append(meta)

    # Save JSON manifest for quick app loading
    with open(OUTPUT_DIR / f"{ds_id}_manifest.json", "w") as f:
        json.dump(manifest, f)
        
    # Save CSV for pandas searching
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / f"{ds_id}_metadata.csv", index=False)
    
    print(f"Saved {len(manifest)} patients for {ds_id}")

if __name__ == "__main__":
    for ds_id, root in ROOTS.items():
        process_dataset(ds_id, root)
