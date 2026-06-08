import base64
import io
import json
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from matplotlib.figure import Figure
from scipy.ndimage import zoom

from api.deep_learning.model_registry import get_best_checkpoint_path, get_run_config_path
from api.simulation.projector import MU_WATER_60KEV, SimulationManager
from scripts.prep_DLR import DLRUNet, SpectralFeatureBuilder, exposure_mas_to_diffusion_time, predict_reconstruction


router = APIRouter()


def mu_to_hu(mu_img: np.ndarray) -> np.ndarray:
    return np.maximum(mu_img, 0.0) * (1000.0 / MU_WATER_60KEV) - 1000.0


def hu_to_mu(hu_img: np.ndarray) -> np.ndarray:
    return np.maximum((hu_img + 1000.0) * MU_WATER_60KEV / 1000.0, 0.0)


def encode_hu_png_bytes(hu_img: np.ndarray, win_min: float, win_max: float, figsize: Tuple[float, float] = (4, 4), dpi: int = 100) -> bytes:
    buf = io.BytesIO()
    fig = Figure(figsize=figsize, dpi=dpi, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(hu_img, cmap="gray", vmin=win_min, vmax=win_max, interpolation="nearest")
    ax.axis("off")
    fig.savefig(buf, format="png", facecolor="black", dpi=dpi)
    return buf.getvalue()


def encode_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")


def format_sse(payload: Dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class MotionScopeManager:
    PHASE_PATTERN = r"Gated,\s*([0-9]+(?:\.[0-9]+)?)%"

    def __init__(self, root_candidates: Optional[List[Path]] = None):
        self.root_dir = self._resolve_root(root_candidates)
        # Load precomputed motion data if available
        self.precomputed_path = Path(__file__).resolve().parent.parent / "static" / "precomputed" / "motion_precomputed.pt"
        self.precomputed = {}
        if self.precomputed_path.exists():
            try:
                self.precomputed = torch.load(self.precomputed_path)
                print(f"DEBUG [MotionScopeManager]: Loaded precomputed motion records for {len(self.precomputed)} patients.")
            except Exception as e:
                print(f"DEBUG [MotionScopeManager]: Failed to load precomputed pt: {e}")

        self.patient_cache: Dict[str, Dict[str, object]] = {}
        self.slice_cache: Dict[Tuple[str, float, int], np.ndarray] = {}
        self.playback_cache: Dict[Tuple[str, int, float, float, bool], List[Dict[str, object]]] = {}
        self.dlr_cache: Dict[Tuple[int, str], Dict[str, object]] = {}
        self.lock = Lock()
        self.recon_executor = ThreadPoolExecutor(max_workers=1)

        self.sim_device = self._pick_device(1)
        self.recon_device = self._pick_device(0)
        self.sim_manager = SimulationManager(device=self.sim_device)
        self.recon_manager = SimulationManager(device=self.recon_device)

    def _resolve_root(self, root_candidates: Optional[List[Path]]) -> Path:
        candidates = root_candidates or [
            Path("/data/4d_cardiac_gif"),
            Path("/home/staticct/data/4d_cardiac_gif"),
            Path("/workspace/static_ct_recon_demo/data/4d_cardiac_gif"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _pick_device(self, preferred_index: int) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        visible = torch.cuda.device_count()
        if visible > preferred_index:
            return f"cuda:{preferred_index}"
        return "cuda:0"

    def device_summary(self) -> Dict[str, object]:
        return {
            "simulation": str(self.sim_device),
            "reconstruction": str(self.recon_device),
            "dual_gpu": str(self.sim_device).startswith("cuda") and str(self.recon_device).startswith("cuda") and self.sim_device != self.recon_device,
            "visible_cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

    def list_patients(self) -> List[str]:
        if self.precomputed:
            return sorted(list(self.precomputed.keys()))
        if not self.root_dir.exists():
            return []
        return sorted([path.stem for path in self.root_dir.glob("*.gif")])

    def _load_gif(self, patient_id: str) -> Tuple[List[np.ndarray], List[float], float]:
        from PIL import Image, ImageSequence
        gif_path = self.root_dir / f"{patient_id}.gif"
        if not gif_path.exists():
            raise FileNotFoundError(f"Unknown motion patient: {patient_id}")
        
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
        
        total_duration = sum(durations)
        return mu_frames, durations, total_duration

    def get_patient_record(self, patient_id: str) -> Dict[str, object]:
        with self.lock:
            cached = self.patient_cache.get(patient_id)
        if cached is not None:
            return cached

        if self.precomputed and patient_id in self.precomputed:
            rec = self.precomputed[patient_id]
            mu_frames = [np.array(f, dtype=np.float32) for f in rec["mu_frames"]]
            record = {
                "patient_id": patient_id,
                "phases": rec["phases"],
                "slice_count": rec["slice_count"],
                "default_wl": rec["default_wl"],
                "default_ww": rec["default_ww"],
                "mu_frames": mu_frames,
                "durations": rec["durations"],
                "cum_times": rec["cum_times"],
                "total_duration": rec["total_duration"],
                "devices": self.device_summary(),
            }
            with self.lock:
                self.patient_cache[patient_id] = record
            return record

        try:
            mu_frames, durations, total_duration = self._load_gif(patient_id)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Unknown motion patient: {patient_id}") from error

        N = len(mu_frames)
        phases = [float((i / N) * 100.0) for i in range(N)]
        
        cum_times = [0.0]
        curr = 0.0
        for d in durations:
            curr += d
            cum_times.append(curr)

        record = {
            "patient_id": patient_id,
            "phases": phases,
            "slice_count": 1,
            "default_wl": -500.0,
            "default_ww": 1500.0,
            "mu_frames": mu_frames,
            "durations": durations,
            "cum_times": cum_times,
            "total_duration": total_duration,
            "devices": self.device_summary(),
        }

        with self.lock:
            self.patient_cache[patient_id] = record
        return record

    def get_processed_phase_slice(self, patient_id: str, phase: float, slice_index: int) -> np.ndarray:
        record = self.get_patient_record(patient_id)
        mu_frames = record["mu_frames"]
        phases_list = record["phases"]
        closest_index = min(range(len(mu_frames)), key=lambda i: abs(phases_list[i] - phase))
        return mu_frames[closest_index]

    def interpolate_phase_mu(self, patient_id: str, slice_index: int, time_seconds: float, overlay_clock: bool = False) -> np.ndarray:
        record = self.get_patient_record(patient_id)
        mu_frames = record["mu_frames"]
        cum_times = record["cum_times"]
        durations = record["durations"]
        total_duration = record["total_duration"]
        N = len(mu_frames)

        # periodic modulo subtraction
        t_cycle = time_seconds % max(total_duration, 1e-6)
        
        alpha = 0.0
        idx_lower = 0
        idx_upper = 0

        for i in range(N):
            if cum_times[i] <= t_cycle <= cum_times[i+1]:
                idx_lower = i
                idx_upper = (i + 1) % N
                alpha = (t_cycle - cum_times[i]) / max(durations[i], 1e-6)
                break
        
        lower_mu = mu_frames[idx_lower]
        upper_mu = mu_frames[idx_upper]
        interpolated = (1.0 - alpha) * lower_mu + alpha * upper_mu
        
        if overlay_clock:
            interpolated = self._overlay_clock(interpolated, time_seconds)
        return interpolated.astype(np.float32)

    def _overlay_clock(self, mu_img: np.ndarray, time_seconds: float) -> np.ndarray:
        hu_img = mu_to_hu(mu_img)
        output = hu_img.copy()
        height, width = output.shape
        radius = int(min(height, width) * 0.08)
        center_y = int(radius + 2)
        center_x = int(width / 2)

        yy, xx = np.ogrid[:height, :width]
        dist = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        frame_mask = np.logical_and(dist >= radius - 1.5, dist <= radius + 1.5)

        angle = 2.0 * math.pi * (time_seconds % 1.0) - math.pi / 2.0
        end_x = center_x + radius * math.cos(angle)
        end_y = center_y + radius * math.sin(angle)
        hand_dist = np.abs((end_y - center_y) * xx - (end_x - center_x) * yy + end_x * center_y - end_y * center_x) / max(math.hypot(end_x - center_x, end_y - center_y), 1e-6)
        hand_proj = ((xx - center_x) * (end_x - center_x) + (yy - center_y) * (end_y - center_y)) / max((end_x - center_x) ** 2 + (end_y - center_y) ** 2, 1e-6)
        hand_mask = np.logical_and.reduce((hand_dist <= 1.5, hand_proj >= 0.0, hand_proj <= 1.0))

        output[frame_mask] = 750.0
        output[hand_mask] = 750.0
        return hu_to_mu(output)

    def render_mu_data_url(self, mu_img: np.ndarray, ww: float, wl: float) -> str:
        hu_img = mu_to_hu(mu_img)
        return encode_data_url(encode_hu_png_bytes(hu_img, wl - ww / 2.0, wl + ww / 2.0))

    def render_sinogram_data_url(self, sinogram: np.ndarray) -> str:
        fig = Figure(figsize=(4, 4), dpi=100, facecolor="black")
        ax = fig.add_axes([0, 0, 1, 1])
        vmax = float(max(1.0, np.percentile(sinogram, 99.0)))
        ax.imshow(sinogram, cmap="gray", vmin=0.0, vmax=vmax, aspect="auto")
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="black", dpi=100)
        return encode_data_url(buf.getvalue())

    def get_playback_frames(self, patient_id: str, slice_index: int, ww: float, wl: float, overlay_clock: bool = False) -> List[Dict[str, object]]:
        cache_key = (patient_id, int(slice_index), float(ww), float(wl), bool(overlay_clock))
        with self.lock:
            cached = self.playback_cache.get(cache_key)
        if cached is not None:
            return cached

        record = self.get_patient_record(patient_id)
        frames = []
        for i, phase in enumerate(record["phases"]):
            phase_time = record["cum_times"][i]
            mu_img = self.interpolate_phase_mu(patient_id, slice_index, phase_time, overlay_clock=overlay_clock)
            frames.append({
                "phase": phase,
                "time_seconds": phase_time,
                "image": self.render_mu_data_url(mu_img, ww, wl),
            })

        with self.lock:
            self.playback_cache[cache_key] = frames
        return frames

    def _active_sinogram_vector(self, manager: SimulationManager, n_source: int, full_sinogram: np.ndarray) -> np.ndarray:
        manager._warmup_projectors(n_source)
        geometry = manager.geometry_cache[n_source]
        pieces = []
        for source_index in range(n_source):
            mask = geometry["source_module_mask"][source_index]
            row = full_sinogram[source_index]
            for module_index, active in enumerate(mask):
                if active:
                    start = module_index * 48
                    pieces.append(row[start:start + 48])
        return np.concatenate(pieces, axis=0).astype(np.float32)

    def get_dlr_assets(self, n_source: int) -> Dict[str, object]:
        cache_key = (n_source, str(self.recon_device))
        with self.lock:
            cached = self.dlr_cache.get(cache_key)
        if cached is not None:
            return cached

        config_path = get_run_config_path("main", n_source)
        if not config_path.exists():
            raise FileNotFoundError(f"Missing DLR config for n_source={n_source}: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            run_config = json.load(handle)

        checkpoint_path = get_best_checkpoint_path("main", n_source)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing DLR checkpoint for n_source={n_source}: {checkpoint_path}")

        device = torch.device(self.recon_device)
        builder = SpectralFeatureBuilder(n_source, device=device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        base_channels = int(checkpoint.get("config", {}).get("base_channels", run_config["base_channels"]))
        model = DLRUNet(base_channels=base_channels).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        assets = {
            "builder": builder,
            "model": model,
            "checkpoint_path": str(checkpoint_path),
        }
        with self.lock:
            self.dlr_cache[cache_key] = assets
        return assets

    def run_reconstruction(self, method: str, n_source: int, full_sinogram: np.ndarray, exposure_mas: float, ww: float, wl: float) -> Dict[str, object]:
        start_time = time.time()
        method_key = method.lower()
        if method_key == "flash-fbp":
            recon_hu = self.recon_manager.reconstruct_step(n_source, full_sinogram, step="filter")
            label = "FlashFBP"
            checkpoint_path = None
        else:
            assets = self.get_dlr_assets(n_source)
            builder = assets["builder"]
            model = assets["model"]
            active_sinogram = self._active_sinogram_vector(self.recon_manager, n_source, full_sinogram)
            noisy_sinogram = torch.from_numpy(active_sinogram).to(device=builder.device, dtype=torch.float32)
            measurement_components = builder.build_measurement_components_from_sinogram(noisy_sinogram)
            channels = builder.build_input_channels(measurement_components).unsqueeze(0)
            diffusion_time = exposure_mas_to_diffusion_time(torch.tensor([float(exposure_mas)], device=builder.device, dtype=channels.dtype)).to(dtype=channels.dtype)
            with torch.inference_mode():
                final_reconstruction = predict_reconstruction(model, channels, diffusion_time, builder).squeeze(0).squeeze(0)
            recon_hu = np.clip(mu_to_hu(final_reconstruction.detach().cpu().numpy()), -1000.0, 1500.0)
            label = "NeuralSpark"
            checkpoint_path = assets["checkpoint_path"]

        image = encode_data_url(encode_hu_png_bytes(recon_hu, wl - ww / 2.0, wl + ww / 2.0))
        return {
            "method": method_key,
            "label": label,
            "image": image,
            "elapsed_ms": (time.time() - start_time) * 1000.0,
            "checkpoint_path": checkpoint_path,
        }

    def stream_simulation(
        self,
        patient_id: str,
        slice_index: int,
        n_source: int,
        exposure_mas: float,
        rotations: int,
        rotations_per_second: float,
        reconstruction_method: str,
        ww: float,
        wl: float,
        overlay_clock: bool,
    ):
        self.sim_manager._warmup_projectors(n_source)
        self.recon_manager._warmup_projectors(n_source)

        total_views = int(n_source) * int(rotations)
        current_sinogram = np.zeros((int(n_source), 2304), dtype=np.float32)

        yield {
            "event": "meta",
            "patient_id": patient_id,
            "slice_index": int(slice_index),
            "n_source": int(n_source),
            "rotations": int(rotations),
            "rotations_per_second": float(rotations_per_second),
            "reconstruction_method": reconstruction_method,
            "devices": self.device_summary(),
        }

        for global_view_index in range(total_views):
            rotation_index = global_view_index // int(n_source)
            view_index = global_view_index % int(n_source)
            acquisition_time = global_view_index / max(float(n_source) * float(rotations_per_second), 1e-6)
            phase_mu = self.interpolate_phase_mu(patient_id, slice_index, acquisition_time, overlay_clock=overlay_clock)
            current_sinogram[view_index] = self.sim_manager.project_view(phase_mu, int(n_source), int(view_index), m_as=0.1)

            # Generate geometry base on-the-fly with standard static detectors + patient
            geo_base_bytes = self.sim_manager.generate_geometry_base(n_source, mu_to_hu(phase_mu), ww, wl)
            geometry_base_url = "data:image/png;base64," + geo_base_bytes

            payload = {
                "event": "view",
                "rotation_index": int(rotation_index),
                "view_index": int(view_index),
                "progress": float((global_view_index + 1) / max(total_views, 1)),
                "time_seconds": float(acquisition_time),
                "phase_percent": float((acquisition_time % 1.0) * 100.0),
                "geometry_base": geometry_base_url,
                "sinogram_image": self.render_sinogram_data_url(current_sinogram),
                "reconstruction": None,
            }

            if view_index == int(n_source) - 1:
                # Synchronously run and block for reconstruction - sequential waiting
                reconstruction = self.run_reconstruction(
                    reconstruction_method,
                    int(n_source),
                    current_sinogram.copy(),
                    0.1,
                    float(ww),
                    float(wl),
                )
                reconstruction["rotation_index"] = int(rotation_index)
                payload["reconstruction"] = reconstruction
                current_sinogram = np.zeros((int(n_source), 2304), dtype=np.float32)

            yield payload

        yield {
            "event": "complete",
            "playback_frames": self.get_playback_frames(patient_id, slice_index, ww, wl, overlay_clock),
        }


motion_scope_manager = MotionScopeManager()


@router.get("/api/motion/patients")
async def get_motion_patients():
    patients = motion_scope_manager.list_patients()
    return {
        "patients": patients,
        "count": len(patients),
        "devices": motion_scope_manager.device_summary(),
    }


@router.get("/api/motion/patient/{patient_id}")
async def get_motion_patient(patient_id: str):
    try:
        record = motion_scope_manager.get_patient_record(patient_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "patient_id": record["patient_id"],
        "phase_percents": record["phases"],
        "slice_count": record["slice_count"],
        "window_width": record["default_ww"],
        "window_level": record["default_wl"],
        "devices": record["devices"],
    }


@router.get("/api/motion/preview/{patient_id}/{slice_index}/{phase_percent}")
async def get_motion_preview(
    patient_id: str,
    slice_index: int,
    phase_percent: float,
    ww: Optional[float] = None,
    wl: Optional[float] = None,
    overlay_clock: bool = False,
):
    try:
        record = motion_scope_manager.get_patient_record(patient_id)
        phase_time = float(phase_percent) / 100.0
        mu_img = motion_scope_manager.interpolate_phase_mu(patient_id, slice_index, phase_time, overlay_clock=overlay_clock)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    target_ww = float(ww if ww is not None else record["default_ww"])
    target_wl = float(wl if wl is not None else record["default_wl"])
    png_bytes = encode_hu_png_bytes(mu_to_hu(mu_img), target_wl - target_ww / 2.0, target_wl + target_ww / 2.0)
    return Response(content=png_bytes, media_type="image/png")


@router.get("/api/motion/phase-frames/{patient_id}/{slice_index}")
async def get_motion_phase_frames(
    patient_id: str,
    slice_index: int,
    ww: Optional[float] = None,
    wl: Optional[float] = None,
    overlay_clock: bool = False,
):
    try:
        record = motion_scope_manager.get_patient_record(patient_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    target_ww = float(ww if ww is not None else record["default_ww"])
    target_wl = float(wl if wl is not None else record["default_wl"])
    return {
        "patient_id": patient_id,
        "slice_index": int(slice_index),
        "frames": motion_scope_manager.get_playback_frames(patient_id, slice_index, target_ww, target_wl, overlay_clock),
    }


@router.get("/api/motion/reconstruct-phase/{patient_id}/{slice_index}/{phase_percent}")
async def get_motion_reconstruct_phase(
    patient_id: str,
    slice_index: int,
    phase_percent: float,
    n_source: int = 80,
    recon_method: str = "flashfbp",
    rotations_per_second: float = 4.0,
    ww: Optional[float] = None,
    wl: Optional[float] = None,
    overlay_clock: bool = False,
):
    try:
        record = motion_scope_manager.get_patient_record(patient_id)
        phase_time = float(phase_percent) / 100.0
        
        target_ww = float(ww if ww is not None else record["default_ww"])
        target_wl = float(wl if wl is not None else record["default_wl"])
        n_source = int(n_source)
        rotations_per_second = float(rotations_per_second)
        recon_method = recon_method.lower()
        if recon_method in {"flashfbp", "flash-fbp"}:
            recon_method = "flash-fbp"
        else:
            recon_method = "neuralspark"

        cache_key = (patient_id, int(slice_index), float(phase_percent), recon_method, n_source, rotations_per_second, target_ww, target_wl, bool(overlay_clock))
        if not hasattr(motion_scope_manager, "phase_recon_cache"):
            motion_scope_manager.phase_recon_cache = {}
            
        if cache_key in motion_scope_manager.phase_recon_cache:
            return motion_scope_manager.phase_recon_cache[cache_key]

        # Simulate patient motion blur: project view-by-view, interpolating in time
        T = 1.0 / max(float(rotations_per_second), 1e-6)
        start_time = phase_time - T / 2.0
        
        motion_scope_manager.sim_manager._warmup_projectors(n_source)
        current_sinogram = np.zeros((n_source, 2304), dtype=np.float32)
        for view_index in range(n_source):
            view_time = start_time + (view_index / n_source) * T
            phase_mu = motion_scope_manager.interpolate_phase_mu(patient_id, slice_index, view_time, overlay_clock=overlay_clock)
            current_sinogram[view_index] = motion_scope_manager.sim_manager.project_view(phase_mu, n_source, view_index, m_as=100.0)

        recon_res = motion_scope_manager.run_reconstruction(
            recon_method,
            n_source,
            current_sinogram,
            100.0,
            target_ww,
            target_wl,
        )

        res = {
            "image": recon_res["image"],
            "elapsed_ms": recon_res["elapsed_ms"],
        }
        motion_scope_manager.phase_recon_cache[cache_key] = res
        return res
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/motion/simulate-stream/{patient_id}/{slice_index}")
async def get_motion_simulation_stream(
    patient_id: str,
    slice_index: int,
    n_source: int = 80,
    exposure_mas: float = 100.0,
    num_rotations: int = 2,
    rotations_per_second: float = 4.0,
    recon_method: str = "flashfbp",
    ww: Optional[float] = None,
    wl: Optional[float] = None,
    overlay_clock: bool = False,
):
    if n_source not in {80, 240}:
        raise HTTPException(status_code=400, detail="Supported views are 80 and 240")
    if rotations_per_second not in {4.0, 20.0, 80.0}:
        raise HTTPException(status_code=400, detail="Supported rotation rates are 4, 20, and 80")
    if num_rotations < 1:
        raise HTTPException(status_code=400, detail="Number of rotations must be at least 1")

    recon_method = recon_method.lower()
    if recon_method in {"flashfbp", "flash-fbp"}:
        recon_method = "flash-fbp"
    elif recon_method in {"neuralspark", "neural-spark"}:
        recon_method = "neuralspark"
    else:
        raise HTTPException(status_code=400, detail="Supported reconstruction methods are flashfbp and neuralspark")

    try:
        record = motion_scope_manager.get_patient_record(patient_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    target_ww = float(ww if ww is not None else record["default_ww"])
    target_wl = float(wl if wl is not None else record["default_wl"])

    def iterator():
        for payload in motion_scope_manager.stream_simulation(
            patient_id,
            slice_index,
            n_source,
            exposure_mas,
            num_rotations,
            rotations_per_second,
            recon_method,
            target_ww,
            target_wl,
            overlay_clock,
        ):
            yield format_sse(payload)

    return StreamingResponse(iterator(), media_type="text/event-stream")
