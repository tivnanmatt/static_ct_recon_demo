import json
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Optional


PRECOMPUTED_DIR = Path(__file__).resolve().parent.parent / "static" / "precomputed"

class MedicalCTDataset(Dataset):
    """
    Base class for clinical CT datasets.
    Manages loading of axial slices and patient metadata.
    """
    def __init__(self, root_dir: str, dataset_name: str, window_center: int = 40, window_width: int = 400, manifest_id: Optional[str] = None):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.window_center = window_center
        self.window_width = window_width
        self.manifest_path = PRECOMPUTED_DIR / f"{manifest_id}_manifest.json" if manifest_id else None
        self.manifest = self._load_manifest()
        self.patients = self._find_patients()

    def _load_manifest(self) -> Optional[Dict[str, Dict]]:
        if self.manifest_path is None or not self.manifest_path.exists():
            return None
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
        
    def _find_patients(self) -> List[str]:
        if self.manifest:
            return sorted(self.manifest.keys())
        if not self.root_dir.exists():
            return []
        return sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])

    def get_patient_slices(self, patient_id: str) -> List[Path]:
        if self.manifest and patient_id in self.manifest:
            return [Path(path_str) for path_str in self.manifest[patient_id].get("paths", [])]

        patient_path = self.root_dir / patient_id
        all_slices = sorted(list(patient_path.rglob("*.dcm")))
        ct_slices = []
        for s in all_slices:
            try:
                # Fast check for CT modality and instance number
                ds = pydicom.dcmread(str(s), stop_before_pixels=True)
                if getattr(ds, "Modality", "") == "CT":
                    # Try to get instance number for sorting
                    instance = getattr(ds, "InstanceNumber", 0)
                    try:
                        instance = int(instance)
                    except:
                        instance = 0
                    ct_slices.append((s, instance))
            except Exception:
                continue
        
        ct_slices.sort(key=lambda x: x[1])
        return [s[0] for s in ct_slices]

    def get_slice_info(self, slice_path: Path) -> Dict:
        try:
            ds = pydicom.dcmread(str(slice_path), stop_before_pixels=True)
            return {
                "PatientName": str(getattr(ds, "PatientName", "Anonymous")),
                "PatientID": str(getattr(ds, "PatientID", "N/A")),
                "PatientAge": str(getattr(ds, "PatientAge", "N/A")),
                "PatientSex": str(getattr(ds, "PatientSex", "N/A")),
                "StudyDate": str(getattr(ds, "StudyDate", "N/A")),
                "Modality": str(getattr(ds, "Modality", "CT")),
                "Manufacturer": str(getattr(ds, "Manufacturer", "N/A")),
                "InstitutionName": str(getattr(ds, "InstitutionName", "N/A")),
                "StudyInstanceUID": str(getattr(ds, "StudyInstanceUID", "N/A")),
                "SeriesInstanceUID": str(getattr(ds, "SeriesInstanceUID", "N/A")),
                "SeriesDescription": str(getattr(ds, "SeriesDescription", "N/A")),
                "KVP": str(getattr(ds, "KVP", "N/A")),
                "XRayTubeCurrent": str(getattr(ds, "XRayTubeCurrent", "N/A")),
                "Exposure": str(getattr(ds, "Exposure", "N/A")),
                "SliceThickness": str(getattr(ds, "SliceThickness", "N/A")),
                "PixelSpacing": str(getattr(ds, "PixelSpacing", "N/A")),
                "WindowCenter": str(getattr(ds, "WindowCenter", self.window_center)),
                "WindowWidth": str(getattr(ds, "WindowWidth", self.window_width)),
            }
        except Exception as e:
            return {"error": str(e)}

    def _load_slice_with_header(self, slice_path: Path):
        ds = pydicom.dcmread(str(slice_path))
        img = ds.pixel_array.astype(np.float32)

        rescale_slope = getattr(ds, 'RescaleSlope', 1.0)
        rescale_intercept = getattr(ds, 'RescaleIntercept', 0.0)
        img = img * rescale_slope + rescale_intercept

        if hasattr(ds, "ImageOrientationPatient"):
            iop = ds.ImageOrientationPatient
            if iop[0] < 0:
                img = np.fliplr(img)
            if iop[4] < 0:
                img = np.flipud(img)

        return img, ds

    def load_slice(self, slice_path: Path) -> np.ndarray:
        img, _ = self._load_slice_with_header(slice_path)
        return img

    def _resample_to_common_grid(self, hu_img: np.ndarray, row_spacing_mm: float, col_spacing_mm: float) -> np.ndarray:
        from scipy.ndimage import zoom

        target_size = 256
        target_spacing_mm = 1.6

        zoom_h = float(row_spacing_mm) / target_spacing_mm
        zoom_w = float(col_spacing_mm) / target_spacing_mm
        resampled_hu = zoom(hu_img, (zoom_h, zoom_w), order=1)

        output = np.full((target_size, target_size), -1000.0, dtype=np.float32)

        src_h, src_w = resampled_hu.shape
        crop_h = min(src_h, target_size)
        crop_w = min(src_w, target_size)

        src_h_start = max((src_h - target_size) // 2, 0)
        src_w_start = max((src_w - target_size) // 2, 0)
        dst_h_start = max((target_size - src_h) // 2, 0)
        dst_w_start = max((target_size - src_w) // 2, 0)

        output[
            dst_h_start:dst_h_start + crop_h,
            dst_w_start:dst_w_start + crop_w,
        ] = resampled_hu[
            src_h_start:src_h_start + crop_h,
            src_w_start:src_w_start + crop_w,
        ]
        return output

    def get_processed_slice(self, slice_path: Path) -> np.ndarray:
        """
        Processes a DICOM slice for simulation:
        1. Rescale to HU
        2. Interpolate to 256x256 (1.6mm spacing)
        3. Convert HU to Mu (Linear Attenuation) @ 60keV
        4. Clip at zero
        """
        # 1. Load raw HU
        hu_img, ds = self._load_slice_with_header(slice_path)
        
        # Clip to -1000 (air) to handle non-physical background values (e.g., -3024)
        hu_img = np.maximum(-1000.0, hu_img)

        pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
        try:
            row_spacing = float(pixel_spacing[0])
            col_spacing = float(pixel_spacing[1])
        except Exception:
            row_spacing = 1.0
            col_spacing = 1.0

        # 2. Re-interpolate to the common 256x256 grid using physical in-plane spacing.
        resampled_hu = self._resample_to_common_grid(hu_img, row_spacing, col_spacing)
        
        # 3. HU to Attenuation at 60keV (mu_water ~ 0.0183 mm^-1)
        mu_water = 0.0183
        mu_img = (resampled_hu + 1000.0) * mu_water / 1000.0
        
        # 4. Clip at zero
        return np.maximum(0, mu_img)

class HeadCTDataset(MedicalCTDataset):
    def __init__(self, root_dir: str = "/data/cq500"):
        super().__init__(root_dir, "Head CT (CQ500)", window_center=40, window_width=120, manifest_id="head")

class ThoraxCTDataset(MedicalCTDataset):
    def __init__(self, root_dir: str = "/data/LIDC/LIDC-IDRI"):
        super().__init__(root_dir, "Thorax CT (LIDC)", window_center=-600, window_width=1500, manifest_id="thorax")

class AbdomenCTDataset(MedicalCTDataset):
    def __init__(self, root_dir: str = "/data/LIHC/TCGA-LIHC"):
        super().__init__(root_dir, "Abdomen CT (LIHC)", window_center=50, window_width=350, manifest_id="abdomen")

class PelvicCTDataset(MedicalCTDataset):
    def __init__(self, root_dir: str = "/data/ACRIN/manifest-sFI3R7DS3069120899390652954/CT COLONOGRAPHY"):
        super().__init__(root_dir, "Pelvic CT (ACRIN)", window_center=50, window_width=400, manifest_id="pelvic")

# Registry of datasets for the UI
DATASET_REGISTRY = {
    "head": HeadCTDataset(),
    "thorax": ThoraxCTDataset(),
    "abdomen": AbdomenCTDataset(),
    "pelvic": PelvicCTDataset()
}
