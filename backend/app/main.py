import base64
import io
import json
import asyncio
from pathlib import Path
from threading import Event, Lock
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import sys
import os
sys.path.append(os.path.dirname(__file__))

REPO_ROOT_CANDIDATES = [
    Path(__file__).resolve().parents[2],
    Path("/workspace/static_ct_recon_demo"),
]
for repo_root in REPO_ROOT_CANDIDATES:
    if repo_root.exists() and str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

from api.datasets import DATASET_REGISTRY
from api.deep_learning.model_registry import get_best_checkpoint_path, get_run_config_path, get_model_dir
from api.simulation.projector import MU_WATER_60KEV, sim_manager
from api.simulation.filter_plot import generate_filter_graph_png
from scripts.prep_DLR import DLRUNet, SpectralFeatureBuilder, exposure_mas_to_diffusion_time, predict_reconstruction
from scripts.prep_diffusion_test import solve_heun_null_space, solve_euler_null_space, solve_langevin_walk_only, solve_combined_diffusion_langevin

iterative_cancel_events: Dict[str, Event] = {}
iterative_cancel_lock = Lock()
dlr_runtime_cache: Dict[tuple, Dict[str, object]] = {}
dlr_runtime_lock = Lock()


def register_iterative_cancel(job_id: str) -> Event:
    with iterative_cancel_lock:
        event = Event()
        iterative_cancel_events[job_id] = event
        return event


def get_iterative_cancel(job_id: str) -> Optional[Event]:
    with iterative_cancel_lock:
        return iterative_cancel_events.get(job_id)


def clear_iterative_cancel(job_id: Optional[str]) -> None:
    if not job_id:
        return
    with iterative_cancel_lock:
        iterative_cancel_events.pop(job_id, None)


def get_dlr_runtime_assets(n_source: int, mode: str = "standard", model_variant: str = "base", exposure_mas: float = 100.0) -> Dict[str, object]:
    cache_key = (n_source, mode, model_variant, exposure_mas)
    with dlr_runtime_lock:
        cached = dlr_runtime_cache.get(cache_key)
        if cached is not None:
            return cached

        checkpoint_dataset_id = "main"
        config_path = get_run_config_path(checkpoint_dataset_id, n_source)

        if not config_path.exists():
            raise FileNotFoundError(f"Missing DLR config for n_source={n_source}: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            run_config = json.load(handle)

        if mode == "generative":
            if model_variant == "finetuned":
                checkpoint_filename = f"diffusion_{n_source}_{float(exposure_mas)}mas.pt"
                checkpoint_path = get_model_dir(checkpoint_dataset_id, n_source) / checkpoint_filename
                if not checkpoint_path.exists():
                    print(f"WARNING: Finetuned checkpoint {checkpoint_path} not found. Fallback to standard base.")
                    checkpoint_path = get_model_dir(checkpoint_dataset_id, n_source) / f"diffusion_{n_source}.pt"
            else:
                checkpoint_path = get_model_dir(checkpoint_dataset_id, n_source) / f"diffusion_{n_source}.pt"
                
            gen_config_path = get_model_dir(checkpoint_dataset_id, n_source) / f"diffusion_config_{n_source}.json"
            if gen_config_path.exists():
                with gen_config_path.open("r", encoding="utf-8") as ghandle:
                    run_config.update(json.load(ghandle))
        else:
            checkpoint_path = get_best_checkpoint_path(checkpoint_dataset_id, n_source)
            
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing DLR checkpoint for {mode} n_source={n_source}: {checkpoint_path}")

        device = sim_manager.device
        builder = SpectralFeatureBuilder(n_source, device=device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Diffusion models might have different base_channels
        base_channels = int(checkpoint.get("config", {}).get("base_channels", run_config["base_channels"]))
        
        model = DLRUNet(base_channels=base_channels).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        cached = {
            "builder": builder,
            "model": model,
            "run_config": run_config,
            "checkpoint_path": checkpoint_path,
        }
        dlr_runtime_cache[cache_key] = cached
        return cached

app = FastAPI(
    title="Static CT Demo Backend",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url=None,
)

@app.on_event("startup")
def pre_warmup_all_projectors():
    print("DEBUG [Startup]: Pre-warming all projectors (80 and 240 views)...")
    try:
        sim_manager._warmup_projectors(80)
        sim_manager._warmup_projectors(240)
        print("DEBUG [Startup]: Projectors loaded successfully.")
    except Exception as e:
        print(f"DEBUG [Startup Error]: Failed to pre-warm projectors: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Serve the precomputed MotionScope animation GIFs/MP4s (outputs/motion_animations).
# The directory lives outside the deployed `app` package, so its location differs between
# a host run (sibling of the repo root) and the container (only `backend/` is bind-mounted
# into /app, while the full workspace is mounted at /workspace). Probe known locations and
# allow an explicit MOTION_ANIM_DIR override.
def _resolve_motion_anim_dir() -> Optional[Path]:
    candidates = []
    env_dir = os.environ.get("MOTION_ANIM_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates += [
        Path(__file__).resolve().parents[2] / "outputs" / "motion_animations",  # host repo-root/outputs
        Path("/workspace/static_ct_recon_demo/outputs/motion_animations"),       # container /workspace mount
        Path("/app/outputs/motion_animations"),                                   # baked into image
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


MOTION_ANIM_DIR = _resolve_motion_anim_dir()
if MOTION_ANIM_DIR is not None:
    app.mount("/motion-animations", StaticFiles(directory=MOTION_ANIM_DIR), name="motion-animations")
    print(f"[motion-scope] Serving precomputed animations from {MOTION_ANIM_DIR}")
else:
    print("[motion-scope] WARNING: motion_animations directory not found; GIFs will 404")


def mu_to_hu(mu_img: np.ndarray) -> np.ndarray:
    return np.maximum(mu_img, 0.0) * (1000.0 / MU_WATER_60KEV) - 1000.0


def hu_std_to_atten(hu_std: float) -> float:
    """Convert an HU noise standard deviation to attenuation-units std.

    Matches the diffusion training scale (scripts/prep_diffusion_train.py, where
    SIGMA_MIN/SIGMA_MAX = hu_to_atten(1)/hu_to_atten(1000)), so the generative
    sampler runs with the same noise range the model was trained on.
    """
    return hu_std * MU_WATER_60KEV / 1000.0


def encode_hu_png(hu_img: np.ndarray, win_min: float, win_max: float, figsize=(4, 4), dpi: int = 100) -> bytes:
    from matplotlib.figure import Figure

    buf = io.BytesIO()
    fig = Figure(figsize=figsize, dpi=dpi, facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(hu_img, cmap='gray', vmin=win_min, vmax=win_max, interpolation='nearest')
    ax.axis('off')
    fig.savefig(buf, format='png', facecolor='black', dpi=dpi)
    return buf.getvalue()


def encode_colormap_png(img: np.ndarray, vmin: float, vmax: float, cmap: str = 'inferno', figsize=(4, 4), dpi: int = 100) -> bytes:
    """Encode a 2D array as a colormapped PNG (same layout as encode_hu_png)."""
    from matplotlib.figure import Figure

    buf = io.BytesIO()
    fig = Figure(figsize=figsize, dpi=dpi, facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.axis('off')
    fig.savefig(buf, format='png', facecolor='black', dpi=dpi)
    return buf.getvalue()


def compute_hallucination_map_b64(x0_tensor, fov_mask=None) -> str:
    """Per-pixel log-variance ('hallucination') map across the sample dimension of
    x0 ([M,1,H,W]). The colormap is autoscaled to the 10th-90th percentile of the
    in-FOV log-variance (inferno)."""
    arr = x0_tensor.detach().cpu().numpy().astype(np.float64)
    if arr.ndim == 4:
        arr = arr[:, 0]  # [M, H, W]
    # Variance across samples in HU^2 (the constant HU offset cancels under variance).
    hu_scale = 1000.0 / MU_WATER_60KEV
    var = np.var(arr, axis=0) * (hu_scale ** 2)
    log_var = np.log(var + 1e-6)
    if fov_mask is not None:
        fov = fov_mask.detach().cpu().numpy() > 0.5
    else:
        fov = np.ones(log_var.shape, dtype=bool)
    vals = log_var[fov]
    if vals.size == 0:
        vals = log_var.ravel()
    vmin = float(np.percentile(vals, 10))
    vmax = float(np.percentile(vals, 90))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(log_var))
        vmax = float(np.max(log_var)) + 1e-6
    display = np.where(fov, log_var, vmin)  # outside the FOV -> floor (dark)
    return base64.b64encode(encode_colormap_png(display, vmin, vmax, cmap='inferno')).decode('utf-8')


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Backend is running"}

@app.get("/api/datasets")
async def get_datasets():
    return [
        {
            "id": k, 
            "name": v.dataset_name, 
            "patients": v.patients,
            "default_ww": v.window_width,
            "default_wl": v.window_center
        }
        for k, v in DATASET_REGISTRY.items()
    ]

@app.get("/api/patients/{dataset_id}")
async def get_patients(dataset_id: str):
    if dataset_id not in DATASET_REGISTRY:
        return []
    return DATASET_REGISTRY[dataset_id].patients

@app.get("/api/slices/{dataset_id}/{patient_id}")
async def get_slices(dataset_id: str, patient_id: str):
    if dataset_id not in DATASET_REGISTRY:
        return []
    slices = DATASET_REGISTRY[dataset_id].get_patient_slices(patient_id)
    return [str(s.relative_to(DATASET_REGISTRY[dataset_id].root_dir)) for s in slices]

@app.get("/api/preview/{dataset_id}/{patient_id}/{slice_index}")
async def get_preview(dataset_id: str, patient_id: str, slice_index: int, ww: Optional[float] = None, wl: Optional[float] = None):
    print(f"DEBUG: Preview requested for {dataset_id}/{patient_id}/{slice_index} (WW: {ww}, WL: {wl})")
    if dataset_id not in DATASET_REGISTRY:
        return {"error": "Dataset not found"}
    
    ds_obj = DATASET_REGISTRY[dataset_id]
    
    # Use provided WW/WL or fall back to dataset defaults
    target_ww = ww if ww is not None else ds_obj.window_width
    target_wl = wl if wl is not None else ds_obj.window_center

    slices = DATASET_REGISTRY[dataset_id].get_patient_slices(patient_id)
    if slice_index < 0 or slice_index >= len(slices):
        return {"error": "Slice index out of range"}

    processed_mu = ds_obj.get_processed_slice(slices[slice_index])
    processed_hu = mu_to_hu(processed_mu)

    win_min = target_wl - (target_ww / 2)
    win_max = target_wl + (target_ww / 2)

    png_bytes = encode_hu_png(processed_hu, win_min, win_max)
    return Response(content=png_bytes, media_type="image/png")

@app.get("/api/dicom-info/{dataset_id}/{patient_id}/{slice_index}")
async def get_dicom_info(dataset_id: str, patient_id: str, slice_index: int):
    if dataset_id not in DATASET_REGISTRY:
        return {"error": "Dataset not found"}
    
    ds_obj = DATASET_REGISTRY[dataset_id]
    slices = ds_obj.get_patient_slices(patient_id)
    if slice_index < 0 or slice_index >= len(slices):
        return {"error": "Slice index out of range"}
    
    info = ds_obj.get_slice_info(slices[slice_index])
    return info

@app.get("/api/stage-patient/{dataset_id}/{patient_id}/{slice_index}")
async def stage_patient(dataset_id: str, patient_id: str, slice_index: int):
    if dataset_id not in DATASET_REGISTRY:
        return {"status": "error", "message": "Dataset not found"}
    
    ds_obj = DATASET_REGISTRY[dataset_id]
    slices = ds_obj.get_patient_slices(patient_id)
    if slice_index < 0 or slice_index >= len(slices):
        return {"status": "error", "message": "Slice index out of range"}
    
    # Process slice to 256x256 mu (linear attenuation)
    mu_img = ds_obj.get_processed_slice(slices[slice_index])
    
    # Cache processed image for simulation
    sim_manager.sinogram_cache[("processed_img", dataset_id, patient_id, slice_index)] = mu_img
    
    return {"status": "success", "message": f"Patient {patient_id} staged at 256x256 (1.6mm)"}

@app.get("/api/simulate/prepare/{n_source}")
async def prepare_simulation(n_source: int):
    """Warmup projectors and measure timings."""
    sim_manager._warmup_projectors(n_source)
    timings = sim_manager.timings.get(n_source)
    return {
        "status": "success", 
        "timings": timings,
        "message": f"GPU Warmup Complete. Full System Matrices Loaded."
    }

@app.get("/api/simulate/full-forward/{dataset_id}/{patient_id}/{slice_index}/{n_source}")
async def simulate_full_forward(dataset_id: str, patient_id: str, slice_index: int, n_source: int, ww: Optional[float] = None, wl: Optional[float] = None, mAs: Optional[float] = 300.0):
    if dataset_id not in DATASET_REGISTRY:
        return {"error": "Dataset not found"}
    
    ds_obj = DATASET_REGISTRY[dataset_id]
    slices = ds_obj.get_patient_slices(patient_id)
    
    mu_img = sim_manager.sinogram_cache.get(("processed_img", dataset_id, patient_id, slice_index))
    if mu_img is None:
        mu_img = ds_obj.get_processed_slice(slices[slice_index])
        sim_manager.sinogram_cache[("processed_img", dataset_id, patient_id, slice_index)] = mu_img

    # 1. Full Project
    full_sino = sim_manager.project_full(mu_img, n_source, m_as=mAs)
    sino_key = (dataset_id, patient_id, slice_index, n_source)
    sim_manager.set_sinogram(sino_key, full_sino)

    # 2. Generate Base Geometry Image
    raw_img = ds_obj.load_slice(slices[slice_index])
    geo_base_b64 = sim_manager.generate_geometry_base(n_source, raw_img, ww, wl)

    # 3. Generate Full Sinogram Image
    import matplotlib.pyplot as plt
    import io
    import base64
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 8), dpi=100)
    fig.patch.set_facecolor('black')
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    ax.imshow(full_sino, cmap='gray', vmin=0.0, vmax=8.0, aspect='auto')

    # Draw Axes with visible tick labels representing:
    # y-axis (Sources, increasing downward)
    # x-axis (Detector Columns)
    ax.set_xlabel("Detector Column Index", color='white', fontsize=12)
    ax.set_ylabel("Source Index (Increasing Downward)", color='white', fontsize=12)
    ax.tick_params(colors='white', which='both', labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add Twin X and Twin Y Axes for Angles
    # Target range: Source Angles and Detector Angles from 0 to 360 degrees
    # Set secondary X axis on top
    ax_top = ax.twiny()
    ax_top.set_xlim(0, 360)
    ax_top.set_xlabel("Detector Angle (Degrees)", color='white', fontsize=12)
    ax_top.tick_params(colors='white', which='both', labelsize=10)
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    # Set secondary Y axis on right
    ax_right = ax.twinx()
    ax_right.set_ylim(360, 0) # Increasing downward to match source indices
    ax_right.set_ylabel("Source Angle (Degrees)", color='white', fontsize=12)
    ax_right.tick_params(colors='white', which='both', labelsize=10)
    for spine in ax_right.spines.values():
        spine.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='black')
    sino_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Generate Sinogram Black Mask (Transparent except for solid black inside plot area)
    fig_mask = Figure(figsize=(8, 8), dpi=100)
    fig_mask.patch.set_alpha(0.0)
    ax_mask = fig_mask.add_axes([0.15, 0.15, 0.7, 0.7])
    ax_mask.imshow(0 * full_sino, cmap='gray', vmin=0.0, vmax=1.0, aspect='auto')
    ax_mask.axis('off')

    buf_mask = io.BytesIO()
    fig_mask.savefig(buf_mask, format='png', facecolor='none', edgecolor='none')
    mask_b64 = base64.b64encode(buf_mask.getvalue()).decode('utf-8')

    return {
        "geometry_base": geo_base_b64,
        "sinogram_full": sino_b64,
        "sinogram_mask": mask_b64,
        "n_source": n_source
    }

@app.get("/api/simulate/overlay/{n_source}/{view_idx}")
async def simulate_overlay(n_source: int, view_idx: int):
    """Returns a transparent PNG overlay for the given view."""
    overlay_b64 = sim_manager.generate_geometry_overlay(n_source, view_idx)
    return {"overlay": overlay_b64}

@app.get("/api/simulate/view/{dataset_id}/{patient_id}/{slice_index}/{n_source}/{view_idx}")
async def simulate_view(dataset_id: str, patient_id: str, slice_index: int, n_source: int, view_idx: int, ww: Optional[float] = None, wl: Optional[float] = None, mAs: Optional[float] = 300.0):
    if dataset_id not in DATASET_REGISTRY:
        return {"error": "Dataset not found"}
    
    ds_obj = DATASET_REGISTRY[dataset_id]
    slices = ds_obj.get_patient_slices(patient_id)
    
    # Load processed image (mu) from cache or process on the fly
    mu_img = sim_manager.sinogram_cache.get(("processed_img", dataset_id, patient_id, slice_index))
    if mu_img is None:
        mu_img = ds_obj.get_processed_slice(slices[slice_index])
        sim_manager.sinogram_cache[("processed_img", dataset_id, patient_id, slice_index)] = mu_img

    # Key for the full sinogram
    sino_key = (dataset_id, patient_id, slice_index, n_source)
    full_sino = sim_manager.get_sinogram(sino_key)
    
    # Project (expects mu_img)
    view_data = sim_manager.project_view(mu_img, n_source, view_idx, m_as=mAs)
    
    if full_sino is None:
        full_sino = np.zeros((n_source, len(view_data)))
    
    full_sino[view_idx] = view_data
    sim_manager.set_sinogram(sino_key, full_sino)

    # Load raw image for display (with WW/WL)
    raw_img = ds_obj.load_slice(slices[slice_index])
    
    # PARALLEL RENDERING
    import asyncio
    loop = asyncio.get_event_loop()
    geo_task = loop.run_in_executor(sim_manager.executor, sim_manager.generate_geometry_frame, n_source, view_idx, raw_img, ww, wl, mAs)
    sino_task = loop.run_in_executor(sim_manager.executor, sim_manager.generate_sinogram_frame, full_sino, view_idx, mAs)
    
    geo_b64, sino_b64 = await asyncio.gather(geo_task, sino_task)
    
    # Get current timing info
    timings = sim_manager.timings.get(n_source, {})
    
    return {
        "view_idx": view_idx,
        "geometry_image": geo_b64,
        "sinogram_image": sino_b64,
        "progress": float((view_idx + 1) / n_source * 100),
        "timings": timings
    }

@app.get("/api/filter/plot")
async def get_filter_plot(w_low: float = 1.0, w_mid: float = 0.5, w_high: float = 0.2):
    """Returns the base64 encoded PNG 1D plot of the FBP filter bands configuration."""
    try:
        img_b64 = generate_filter_graph_png(w_low, w_mid, w_high)
        return {"image": f"data:image/png;base64,{img_b64}"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/reconstruct/fbp/{dataset_id}/{patient_id}/{slice_index}/{n_source}")
async def reconstruct_fbp(
    dataset_id: str,
    patient_id: str,
    slice_index: int,
    n_source: int,
    step: str = 'filter',
    ww: Optional[float] = None,
    wl: Optional[float] = None,
    w_low: Optional[float] = None,
    w_mid: Optional[float] = None,
    w_high: Optional[float] = None
):
    sino_key = (dataset_id, patient_id, slice_index, n_source)
    full_sino = sim_manager.get_sinogram(sino_key)
    
    if full_sino is None:
        return {"error": "Sinogram not found. Run simulation first."}
    
    # Run Reconstruction step (laminogram or filter)
    import time
    start = time.time()
    recon_hu = sim_manager.reconstruct_step(
        n_source, full_sino, step=step,
        w_low=w_low, w_mid=w_mid, w_high=w_high
    )
    recon_time_ms = (time.time() - start) * 1000.0
    
    print(f"DEBUG [main.py]: step={step}, hu_mean={recon_hu.mean():.1f}, hu_std={recon_hu.std():.1f}, hu_min={recon_hu.min():.1f}, hu_max={recon_hu.max():.1f}")

    # Windowing for display
    if step == 'unfiltered' or step == 'laminogram' or step == 'unfiltered-bp':
        # For unfiltered bp, use min/max range for visualization to ensure visibility
        # Consistent with debug_fbp auto-windowing
        win_min = float(recon_hu.min())
        win_max = float(recon_hu.max())
    else:
        ds_obj = DATASET_REGISTRY.get(dataset_id)
        target_ww = ww if ww is not None else (ds_obj.window_width if ds_obj else 400)
        target_wl = wl if wl is not None else (ds_obj.window_center if ds_obj else 50)
        win_min = target_wl - (target_ww / 2)
        win_max = target_wl + (target_ww / 2)

    img_b64 = base64.b64encode(encode_hu_png(recon_hu, win_min, win_max)).decode('utf-8')
    
    return {
        "reconstruction_image": f"data:image/png;base64,{img_b64}",
        "recon_time_ms": recon_time_ms,
        "step": step
    }


@app.get("/api/reconstruct/runtime/{dataset_id}/{patient_id}/{slice_index}/{n_source}")
async def reconstruct_runtime(
    dataset_id: str,
    patient_id: str,
    slice_index: int,
    n_source: int,
    initial_step: str = 'unfiltered',
    final_step: str = 'filter',
    ww: Optional[float] = None,
    wl: Optional[float] = None,
    w_low: Optional[float] = None,
    w_mid: Optional[float] = None,
    w_high: Optional[float] = None
):
    sino_key = (dataset_id, patient_id, slice_index, n_source)
    full_sino = sim_manager.get_sinogram(sino_key)
    if full_sino is None:
        return {"error": "Sinogram not found. Run simulation first."}

    import time

    initial_start = time.time()
    initial_hu = sim_manager.reconstruct_step(
        n_source, full_sino, step=initial_step,
        w_low=w_low, w_mid=w_mid, w_high=w_high
    )
    initial_time_ms = (time.time() - initial_start) * 1000.0

    final_start = time.time()
    final_hu = sim_manager.reconstruct_step(
        n_source, full_sino, step=final_step,
        w_low=w_low, w_mid=w_mid, w_high=w_high
    )
    final_time_ms = (time.time() - final_start) * 1000.0

    ds_obj = DATASET_REGISTRY.get(dataset_id)
    target_ww = ww if ww is not None else (ds_obj.window_width if ds_obj else 400)
    target_wl = wl if wl is not None else (ds_obj.window_center if ds_obj else 50)
    win_min = target_wl - (target_ww / 2)
    win_max = target_wl + (target_ww / 2)

    if initial_step in {'unfiltered', 'laminogram', 'unfiltered-bp'}:
        initial_min = float(initial_hu.min())
        initial_max = float(initial_hu.max())
    else:
        initial_min = win_min
        initial_max = win_max

    initial_b64 = base64.b64encode(encode_hu_png(initial_hu, initial_min, initial_max)).decode('utf-8')
    final_b64 = base64.b64encode(encode_hu_png(final_hu, win_min, win_max)).decode('utf-8')

    return {
        "initial_image": f"data:image/png;base64,{initial_b64}",
        "final_image": f"data:image/png;base64,{final_b64}",
        "initial_step": initial_step,
        "final_step": final_step,
        "initial_time_ms": initial_time_ms,
        "final_time_ms": final_time_ms,
    }


def get_active_sinogram_tensor(full_sino: np.ndarray, n_source: int, device: torch.device) -> torch.Tensor:
    sim_manager._warmup_projectors(n_source)
    geom = sim_manager.geometry_cache[n_source]
    pieces = []
    for i in range(n_source):
        mask = geom["source_module_mask"][i]
        row = full_sino[i]
        for m_idx, active in enumerate(mask):
            if active:
                pieces.append(row[m_idx * 48 : (m_idx + 1) * 48])
    active_np = np.concatenate(pieces).astype(np.float32)
    return torch.from_numpy(active_np).to(device)


@app.get("/api/reconstruct/dlr/{dataset_id}/{patient_id}/{slice_index}/{n_source}")
async def reconstruct_dlr(dataset_id: str, patient_id: str, slice_index: int, n_source: int, exposure_mas: float = 100.0, ww: Optional[float] = None, wl: Optional[float] = None):
    sino_key = (dataset_id, patient_id, slice_index, n_source)
    full_sino = sim_manager.get_sinogram(sino_key)
    if full_sino is None:
        return {"error": "Sinogram not found. Run simulation first."}

    ds_obj = DATASET_REGISTRY.get(dataset_id)
    if ds_obj is None:
        return {"error": f"Unknown dataset: {dataset_id}"}

    slices = ds_obj.get_patient_slices(patient_id)
    if slice_index < 0 or slice_index >= len(slices):
        return {"error": "Slice index out of range"}

    assets = get_dlr_runtime_assets(n_source, mode="standard")
    builder = assets["builder"]
    model = assets["model"]
    device = sim_manager.device

    import time

    fbp_start = time.time()
    fbp_hu = sim_manager.reconstruct_step(n_source, full_sino, step='filter')
    initial_time_ms = (time.time() - fbp_start) * 1000.0

    active_sino_tensor = get_active_sinogram_tensor(full_sino, n_source, device)
    measurement_components = builder.build_measurement_components_from_sinogram(active_sino_tensor)

    channels = builder.build_input_channels(measurement_components).unsqueeze(0)
    with torch.inference_mode():
        diffusion_time = exposure_mas_to_diffusion_time(
            torch.tensor([float(exposure_mas)], device=device, dtype=channels.dtype)
        ).to(dtype=channels.dtype)
        final_start = time.time()
        final_reconstruction = predict_reconstruction(model, channels, diffusion_time, builder).squeeze(0).squeeze(0)
        final_time_ms = (time.time() - final_start) * 1000.0

    target_ww = ww if ww is not None else (ds_obj.window_width if ds_obj else 400)
    target_wl = wl if wl is not None else (ds_obj.window_center if ds_obj else 50)
    win_min = target_wl - (target_ww / 2)
    win_max = target_wl + (target_ww / 2)

    # pinv_np = ((pinv.detach().cpu().numpy() * 1000.0 / MU_WATER_60KEV) - 1000.0).clip(-1000, 1500)
    final_np = ((final_reconstruction.detach().cpu().numpy() * 1000.0 / MU_WATER_60KEV) - 1000.0).clip(-1000, 1500)

    fbp_b64 = base64.b64encode(encode_hu_png(fbp_hu, win_min, win_max)).decode('utf-8')
    final_b64 = base64.b64encode(encode_hu_png(final_np, win_min, win_max)).decode('utf-8')

    return {
        "initial_image": f"data:image/png;base64,{fbp_b64}",
        "final_image": f"data:image/png;base64,{final_b64}",
        "initial_step": "fbp",
        "final_step": "dlr",
        "initial_time_ms": initial_time_ms,
        "final_time_ms": final_time_ms,
        "checkpoint_path": str(assets["checkpoint_path"]),
        "exposure_mas": exposure_mas,
    }

@app.get("/api/reconstruct/iterative/{dataset_id}/{patient_id}/{slice_index}/{n_source}")
async def reconstruct_iterative(
    request: Request,
    dataset_id: str, patient_id: str, slice_index: int, n_source: int,
    iters: int = 50, tv: float = 0.01, lr: float = 1.0, precond: bool = True,
    ww: Optional[float] = None, wl: Optional[float] = None, job_id: Optional[str] = None
):
    print(f"DEBUG [main.py]: reconstruct_iterative request: ds={dataset_id}, p={patient_id}, s={slice_index}, n={n_source}, iters={iters}, tv={tv}, lr={lr}, precond={precond}")
    sino_key = (dataset_id, patient_id, slice_index, n_source)
    full_sino = sim_manager.get_sinogram(sino_key)
    if full_sino is None:
        print(f"DEBUG [main.py]: sinogram not found for key {sino_key}")
        return {"error": "Sinogram not found"}

    import json
    from matplotlib.figure import Figure

    cancel_event = register_iterative_cancel(job_id) if job_id else None

    async def event_generator():
        iter_generator = None
        try:
            print("DEBUG [main.py]: event_generator starting")
            iter_generator = sim_manager.run_iterative_recon(
                n_source,
                full_sino,
                num_iters=iters,
                tv_weight=tv,
                lr=lr,
                use_precond=precond,
                stop_requested=(cancel_event.is_set if cancel_event is not None else None),
            )

            while True:
                if cancel_event is not None and cancel_event.is_set():
                    print(f"DEBUG [main.py]: iterative cancel requested for job_id={job_id}")
                    break
                if await request.is_disconnected():
                    print("DEBUG [main.py]: client disconnected from iterative stream")
                    break

                try:
                    update = next(iter_generator)
                except StopIteration:
                    break

                hu_img = update["image"]
                iteration = update["iteration"]
                print(f"DEBUG [main.py]: yielding iteration {iteration}")
                
                # Window/Level
                if ww is not None and wl is not None:
                    win_min, win_max = wl - ww/2, wl + ww/2
                else:
                    win_min, win_max = -200, 400 # Default soft tissue
                
                # Image Encoding
                img_b64 = base64.b64encode(encode_hu_png(hu_img, win_min, win_max)).decode('utf-8')
                
                # Loss Plot Encoding (3 Subplots)
                loss_buf = io.BytesIO()
                fig_loss = Figure(figsize=(4, 8), dpi=100, facecolor='black')
                
                history_post = update.get("loss_history", [])
                history_lik = update.get("likelihood_history", [])
                history_prior = update.get("prior_history", [])
                iters_arr = range(1, len(history_post)+1) if history_post else []

                # 1. Likelihood
                ax1 = fig_loss.add_subplot(311)
                ax1.set_facecolor('black')
                if history_lik:
                    ax1.plot(iters_arr, history_lik, color='#FF5733', linewidth=1.5)
                ax1.set_title("Log Likelihood", color='#FF5733', fontsize=10)
                ax1.tick_params(colors='white', labelsize=7)
                ax1.grid(True, alpha=0.1, color='white')

                # 2. Prior
                ax2 = fig_loss.add_subplot(312)
                ax2.set_facecolor('black')
                if history_prior:
                    ax2.plot(iters_arr, history_prior, color='#75FF33', linewidth=1.5)
                ax2.set_title("Log Prior (TV)", color='#75FF33', fontsize=10)
                ax2.tick_params(colors='white', labelsize=7)
                ax2.grid(True, alpha=0.1, color='white')

                # 3. Posterior
                ax3 = fig_loss.add_subplot(313)
                ax3.set_facecolor('black')
                if history_post:
                    ax3.plot(iters_arr, history_post, color='#52DAF2', linewidth=2)
                ax3.set_title("Log Posterior", color='#52DAF2', fontsize=10)
                ax3.set_xlabel("Iteration", color='white', fontsize=8)
                ax3.tick_params(colors='white', labelsize=7)
                ax3.grid(True, alpha=0.1, color='white')
                
                fig_loss.tight_layout()
                fig_loss.savefig(loss_buf, format='png', facecolor='black')
                loss_b64 = base64.b64encode(loss_buf.getvalue()).decode('utf-8')
                
                payload = {
                    "iteration": iteration,
                    "total": iters,
                    "image": f"data:image/png;base64,{img_b64}",
                    "loss_plot": f"data:image/png;base64,{loss_b64}"
                }
                yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if iter_generator is not None:
                iter_generator.close()
            clear_iterative_cancel(job_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/reconstruct/iterative/stop/{job_id}")
async def stop_iterative(job_id: str):
    cancel_event = get_iterative_cancel(job_id)
    if cancel_event is None:
        return {"status": "not_found", "job_id": job_id}
    cancel_event.set()
    return {"status": "cancelling", "job_id": job_id}

@app.get("/api/reconstruct/generative/{dataset_id}/{patient_id}/{slice_index}/{n_source}")
async def reconstruct_generative(
    request: Request,
    dataset_id: str, patient_id: str, slice_index: int, n_source: int,
    steps: int = 20, sigma_max: float = 0.5, sigma_min: float = 0.001, solver: str = "heun", temperature: float = 0.0,
    exposure_mas: float = 100.0,
    ww: Optional[float] = None, wl: Optional[float] = None, job_id: Optional[str] = None,
    num_samples: int = 4, langevin_steps: int = 100,
    model_variant: str = "base"
):
    if dataset_id not in DATASET_REGISTRY:
        return {"error": "Dataset not found"}
    ds_obj = DATASET_REGISTRY[dataset_id]
    slices = ds_obj.get_patient_slices(patient_id)
    if slice_index < 0 or slice_index >= len(slices):
        return {"error": "Slice index out of range"}

    sino_key = (dataset_id, patient_id, slice_index, n_source)
    full_sino = sim_manager.get_sinogram(sino_key)
    if full_sino is None:
        return {"error": "Sinogram not found"}

    # Ground-truth mu image (numpy) for the comparison panel, mirroring the other recon endpoints.
    gt_mu_img = sim_manager.sinogram_cache.get(("processed_img", dataset_id, patient_id, slice_index))
    if gt_mu_img is None:
        gt_mu_img = ds_obj.get_processed_slice(slices[slice_index])
        sim_manager.sinogram_cache[("processed_img", dataset_id, patient_id, slice_index)] = gt_mu_img

    # Use same cancel mechanism as iterative
    cancel_event = register_iterative_cancel(job_id) if job_id else None

    async def event_generator():
        import json
        print(f"DEBUG [Generative]: Starting event_generator for {dataset_id}/{patient_id}/{slice_index} (Job: {job_id}, num_samples={num_samples}, langevin_steps={langevin_steps}, model_variant={model_variant})")
        try:
            device = sim_manager.device
            assets = get_dlr_runtime_assets(n_source, mode="generative", model_variant=model_variant, exposure_mas=exposure_mas)
            model = assets["model"]
            builder = assets["builder"]
            print(f"DEBUG [Generative]: Assets loaded. Device: {device}, using weight: {assets.get('checkpoint_path')}")

            # 2. Build Measurement Components
            active_sino_tensor = get_active_sinogram_tensor(full_sino, n_source, device)
            comps = builder.build_measurement_components_from_sinogram(active_sino_tensor)
            print("DEBUG [Generative]: Measurement components built.")

            # Start from measured null-space FBP plus null-space noise. The range-space pinv is fixed.
            pinv = comps["pinv"].unsqueeze(0).unsqueeze(0)
            measurement_null = comps["measurement_null"].unsqueeze(0).unsqueeze(0)
            full_fbp = comps["full_fbp"].unsqueeze(0).unsqueeze(0)

            M = max(1, num_samples)

            pinv_batched = pinv.repeat(M, 1, 1, 1)
            measurement_null_batched = measurement_null.repeat(M, 1, 1, 1)
            full_fbp_batched = full_fbp.repeat(M, 1, 1, 1)

            # Initialise the diffusion state from the measured null-space FBP plus sigma_max
            # null-space noise, exactly as the forward process in prep_diffusion_train.py
            # (xt_full = pinv + xt_null). The range anchor is the pseudoinverse (channel 0 = pinv),
            # matching how the new-input-style model was trained (range stays unfiltered).
            noise_null = builder.project_batch_null(torch.randn_like(pinv_batched)) * sigma_max
            xt_null_start = builder.project_batch_null(measurement_null_batched + noise_null)
            xt_start = pinv_batched + xt_null_start

            # BUILD INPUT STACK FOR SOLVER
            # Channels: [pinv, meas_null, full_fbp, null_xt, total_xt, coord_x, coord_y]
            inputs = builder.build_input_channels(comps).unsqueeze(0).repeat(M, 1, 1, 1)
            inputs[:, 3:4, ...] = xt_null_start
            inputs[:, 4:5, ...] = xt_start

            # W/L for UI
            target_ww = ww if ww is not None else ds_obj.window_width
            target_wl = wl if wl is not None else ds_obj.window_center
            win_min = target_wl - (target_ww / 2)
            win_max = target_wl + (target_ww / 2)

            def to_b64_index(mu_tensor, idx=0):
                hu = mu_to_hu(mu_tensor.detach().cpu().numpy()[idx,0])
                return base64.b64encode(encode_hu_png(hu, win_min, win_max)).decode('utf-8')

            def component_to_b64_index(component_tensor, idx=0):
                hu = component_tensor.detach().cpu().numpy()[idx,0] * (1000.0 / MU_WATER_60KEV)
                component_span = max(100.0, float(np.percentile(np.abs(hu), 99)))
                return base64.b64encode(encode_hu_png(hu, -component_span, component_span)).decode('utf-8')

            def to_b64(mu_tensor):
                return to_b64_index(mu_tensor, 0)

            def component_to_b64(component_tensor):
                return component_to_b64_index(component_tensor, 0)

            gt_mu = torch.from_numpy(np.ascontiguousarray(gt_mu_img)).to(device).float().unsqueeze(0).unsqueeze(0)
            gt_b64 = "data:image/png;base64," + to_b64(gt_mu)
            pinv_b64 = "data:image/png;base64," + to_b64(pinv)
            full_fbp_b64 = "data:image/png;base64," + to_b64(full_fbp)

            # Choose solver - now uses unified combined solve
            steps_computed = steps + langevin_steps
            print(f"DEBUG [Generative]: Launching combined solver {solver} with steps_diff={steps}, steps_lang={langevin_steps}, sigma_max={sigma_max}, sigma_min={sigma_min}, temperature={temperature}")
            gen = solve_combined_diffusion_langevin(
                model=model,
                builder=builder,
                inputs=inputs,
                num_steps_diff=steps,
                num_steps_lang=langevin_steps,
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                rho=7.0,
                temperature=temperature,
                solver=solver,
                device=device
            )

            for update in gen:
                # BREAKING THE HEAVY GPU BLOCK
                # Periodically check for disconnection and yield to the loop
                if update["step"] % 1 == 0:
                    if (cancel_event is not None and cancel_event.is_set()) or await request.is_disconnected():
                        print(f"DEBUG [Generative]: Cancel requested or disconnected for job_id={job_id}")
                        torch.cuda.empty_cache()
                        break
                    await asyncio.sleep(0.01)

                if M > 1:
                    null_hat_list = ["data:image/png;base64," + component_to_b64_index(update["null_hat"], b) for b in range(M)]
                    xt_list = ["data:image/png;base64," + to_b64_index(update["xt"], b) for b in range(M)]
                    x0_list = ["data:image/png;base64," + to_b64_index(update["x0"], b) for b in range(M)]
                    
                    mean_null_hat_tensor = update["null_hat"].mean(dim=0, keepdim=True)
                    mean_xt_tensor = update["xt"].mean(dim=0, keepdim=True)
                    mean_x0_tensor = update["x0"].mean(dim=0, keepdim=True)
                    
                    mean_null_hat_b64 = "data:image/png;base64," + component_to_b64(mean_null_hat_tensor)
                    mean_xt_b64 = "data:image/png;base64," + to_b64(mean_xt_tensor)
                    mean_x0_b64 = "data:image/png;base64," + to_b64(mean_x0_tensor)

                    null_b64 = null_hat_list[0]
                    xt_b64 = xt_list[0]
                    x0_b64 = x0_list[0]

                    # Hallucination map: log-variance across the M posterior samples.
                    hallucination_b64 = "data:image/png;base64," + compute_hallucination_map_b64(update["x0"], builder.fov_mask)
                else:
                    null_hat_list = None
                    xt_list = None
                    x0_list = None
                    mean_null_hat_b64 = None
                    mean_xt_b64 = None
                    mean_x0_b64 = None
                    hallucination_b64 = None  # needs >1 sample
                    null_b64 = "data:image/png;base64," + component_to_b64_index(update["null_hat"], 0)
                    xt_b64 = "data:image/png;base64," + to_b64_index(update["xt"], 0)
                    x0_b64 = "data:image/png;base64," + to_b64_index(update["x0"], 0)
                
                payload = {
                    'step': int(update['step']),
                    'total_steps': int(update.get('total_steps', steps_computed)),
                    'sigma': float(update['sigma']),
                    'null_hat': null_b64,
                    'xt': xt_b64,
                    'x0': x0_b64,
                    'gt': gt_b64,
                    'pinv': pinv_b64,
                    'full_fbp': full_fbp_b64,
                    'null_hat_list': null_hat_list,
                    'xt_list': xt_list,
                    'x0_list': x0_list,
                    'hallucination_map': hallucination_b64,
                    'mean_null_hat': mean_null_hat_b64,
                    'mean_xt': mean_xt_b64,
                    'mean_x0': mean_x0_b64
                }
                msg = "data: " + json.dumps(payload) + "\n\n"
                if update["step"] % 10 == 0 or update["step"] == 1:
                    print(f"DEBUG [Generative]: Yielding Step {update['step']}/{steps_computed}, Sigma: {update['sigma']:.4f}")
                yield msg
                # Yielding control to the event loop
                await asyncio.sleep(0.01)

            print(f"DEBUG [Generative]: event_generator completed for {dataset_id}/{patient_id}/{slice_index}")
            # Clean up after successful run
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
             print(f"DEBUG [Generative]: event_generator finished for {dataset_id}/{patient_id}/{slice_index}")
             if job_id:
                clear_iterative_cancel(job_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/reconstruct/full-sinogram/{dataset_id}/{patient_id}/{slice_index}/{n_source}")
async def get_full_sinogram_image(dataset_id: str, patient_id: str, slice_index: int, n_source: int):
    sino_key = (dataset_id, patient_id, slice_index, n_source)
    full_sino = sim_manager.get_sinogram(sino_key)
    if full_sino is None:
        return {"error": "Sinogram not found"}
    
    import matplotlib.pyplot as plt
    import io
    import base64
    
    # High resolution DPI for clinical sinogram
    fig = plt.figure(figsize=(8, 8), dpi=100)
    fig.patch.set_facecolor('black')
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    ax.imshow(full_sino, cmap='gray', vmin=0.0, vmax=8.0, aspect='auto')

    # Draw Axes with visible tick labels representing:
    # y-axis (Sources, increasing downward)
    # x-axis (Detector Columns)
    ax.set_xlabel("Detector Column Index", color='white', fontsize=12)
    ax.set_ylabel("Source Index (Increasing Downward)", color='white', fontsize=12)
    ax.tick_params(colors='white', which='both', labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add Twin X and Twin Y Axes for Angles
    # Target range: Source Angles and Detector Angles from 0 to 360 degrees
    # Set secondary X axis on top
    ax_top = ax.twiny()
    ax_top.set_xlim(0, 360)
    ax_top.set_xlabel("Detector Angle (Degrees)", color='white', fontsize=12)
    ax_top.tick_params(colors='white', which='both', labelsize=10)
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    # Set secondary Y axis on right
    ax_right = ax.twinx()
    ax_right.set_ylim(360, 0) # Increasing downward to match source indices
    ax_right.set_ylabel("Source Angle (Degrees)", color='white', fontsize=12)
    ax_right.tick_params(colors='white', which='both', labelsize=10)
    for spine in ax_right.spines.values():
        spine.set_visible(False)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='black')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return {"sinogram_image": f"data:image/png;base64,{img_b64}"}

@app.get("/api/simulate/gif/{dataset_id}/{patient_id}/{slice_index}/{n_source}")
async def get_simulation_gif(dataset_id: str, patient_id: str, slice_index: int, n_source: int, ww: Optional[float] = None, wl: Optional[float] = None, mAs: Optional[float] = 300.0):
    if dataset_id not in DATASET_REGISTRY:
        return {"error": "Dataset not found"}
    
    ds_obj = DATASET_REGISTRY[dataset_id]
    slices = ds_obj.get_patient_slices(patient_id)
    if slice_index < 0 or slice_index >= len(slices):
        return {"error": "Slice index out of range"}
    
    sino_key = (dataset_id, patient_id, slice_index, n_source)
    full_sino = sim_manager.get_sinogram(sino_key)
    
    if full_sino is None:
        return {"error": "Sinogram not found in cache. Run simulation first."}
    
    # Load processed image (mu) or process on the fly
    mu_img = sim_manager.sinogram_cache.get(("processed_img", dataset_id, patient_id, slice_index))
    if mu_img is None:
        mu_img = ds_obj.get_processed_slice(slices[slice_index])
    
    # Actually, the visualization uses the RAW image for background but mu for simulation correctness
    raw_img = ds_obj.load_slice(slices[slice_index])
    
    # Debug logging
    print(f"DEBUG [Simulation]: Starting GIF generation for {patient_id} slice {slice_index} ({n_source} views)")
    
    gif_url = sim_manager.generate_simulation_gif(dataset_id, patient_id, slice_index, n_source, raw_img, full_sino, ww, wl, m_as=mAs)
    
    print(f"DEBUG [Simulation]: GIF generation complete: {gif_url}")
    
    return {"gif_url": gif_url}

from api.simulation.eval_plots import generate_comparison_violin_plots, generate_method_square_plot_png, compute_metric_ylimits
from api.evaluation.metrics import compute_metrics

@app.get("/api/evaluation/reconstructions/{dataset_id}/{patient_id}/{slice_index}/{n_source}")
async def get_evaluation_reconstructions(
    dataset_id: str, patient_id: str, slice_index: int, n_source: int,
    ww: Optional[float] = None, wl: Optional[float] = None, exposure_mas: float = 100.0
):
    device = sim_manager.device
    ds_obj = DATASET_REGISTRY.get(dataset_id)
    if ds_obj is None:
        return {"error": "Dataset not found"}
        
    slices = ds_obj.get_patient_slices(patient_id)
    if slice_index < 0 or slice_index >= len(slices):
        return {"error": "Slice index out of range"}
        
    sino_key = (dataset_id, patient_id, slice_index, n_source)
    full_sino = sim_manager.get_sinogram(sino_key)
    if full_sino is None:
        mu_img = sim_manager.sinogram_cache.get(("processed_img", dataset_id, patient_id, slice_index))
        if mu_img is None:
            mu_img = ds_obj.get_processed_slice(slices[slice_index])
            sim_manager.sinogram_cache[("processed_img", dataset_id, patient_id, slice_index)] = mu_img
        full_sino = sim_manager.project_full(mu_img, n_source, m_as=exposure_mas)
        sim_manager.set_sinogram(sino_key, full_sino)
    else:
        mu_img = sim_manager.sinogram_cache.get(("processed_img", dataset_id, patient_id, slice_index))
        if mu_img is None:
            mu_img = ds_obj.get_processed_slice(slices[slice_index])
            sim_manager.sinogram_cache[("processed_img", dataset_id, patient_id, slice_index)] = mu_img

    target_ww = ww if ww is not None else ds_obj.window_width
    target_wl = wl if wl is not None else ds_obj.window_center
    win_min = target_wl - (target_ww / 2)
    win_max = target_wl + (target_ww / 2)

    import time
    # 1. Ground Truth
    gt_hu = (mu_img * 1000.0 / MU_WATER_60KEV) - 1000.0
    gt_b64 = base64.b64encode(encode_hu_png(gt_hu, win_min, win_max)).decode('utf-8')

    # 2. FlashFBP
    fbp_start = time.time()
    fbp_hu = sim_manager.reconstruct_step(n_source, full_sino, step='filter')
    fbp_time = (time.time() - fbp_start) * 1000.0
    fbp_b64 = base64.b64encode(encode_hu_png(fbp_hu, win_min, win_max)).decode('utf-8')

    # 3. FidelityMBIR
    mbir_start = time.time()
    iter_gen = sim_manager.run_iterative_recon(
        n_source, full_sino, num_iters=25, tv_weight=4.0, lr=0.1, use_precond=True
    )
    mbir_hu = fbp_hu
    for update in iter_gen:
        mbir_hu = update["image"]
    mbir_time = (time.time() - mbir_start) * 1000.0
    mbir_b64 = base64.b64encode(encode_hu_png(mbir_hu, win_min, win_max)).decode('utf-8')

    # 4. NeuralSpark
    dlr_start = time.time()
    assets_dlr = get_dlr_runtime_assets(n_source, mode="standard")
    model_dlr = assets_dlr["model"]
    builder_dlr = assets_dlr["builder"]
    active_sino_tensor = get_active_sinogram_tensor(full_sino, n_source, device)
    measurement_components = builder_dlr.build_measurement_components_from_sinogram(active_sino_tensor)
    channels = builder_dlr.build_input_channels(measurement_components).unsqueeze(0)
    with torch.inference_mode():
        diffusion_time = exposure_mas_to_diffusion_time(
            torch.tensor([float(exposure_mas)], device=device, dtype=channels.dtype)
        ).to(dtype=channels.dtype)
        final_reconstruction = predict_reconstruction(model_dlr, channels, diffusion_time, builder_dlr).squeeze(0).squeeze(0)
        dlr_hu = ((final_reconstruction.detach().cpu().numpy() * 1000.0 / MU_WATER_60KEV) - 1000.0).clip(-1000, 1500)
    dlr_time = (time.time() - dlr_start) * 1000.0
    dlr_b64 = base64.b64encode(encode_hu_png(dlr_hu, win_min, win_max)).decode('utf-8')

    # 5. GenerativeVision
    gen_start = time.time()
    assets_gen = get_dlr_runtime_assets(n_source, mode="generative", model_variant="base", exposure_mas=exposure_mas)
    model_gen = assets_gen["model"]
    builder_gen = assets_gen["builder"]
    # Re-use measurement_components built from sinogram for generative too
    inputs = builder_gen.build_input_channels(measurement_components).unsqueeze(0)
    xt_null_start = builder_gen.project_batch_null(measurement_components["measurement_null"].unsqueeze(0).unsqueeze(0))
    xt_start = measurement_components["pinv"].unsqueeze(0).unsqueeze(0) + xt_null_start
    inputs[:, 3:4, ...] = xt_null_start
    inputs[:, 4:5, ...] = xt_start
    sigma_max = hu_std_to_atten(1000.0)
    sigma_min = hu_std_to_atten(1.0)
    gen_iter = solve_combined_diffusion_langevin(
        model=model_gen,
        builder=builder_gen,
        inputs=inputs,
        num_steps_diff=10,
        num_steps_lang=0,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        rho=7.0,
        temperature=0.0,
        solver="heun",
        device=device
    )
    gen_hu = dlr_hu
    for update in gen_iter:
        gen_hu = ((update["x0"].squeeze(0).squeeze(0).detach().cpu().numpy() * 1000.0 / MU_WATER_60KEV) - 1000.0).clip(-1000, 1500)
    gen_time = (time.time() - gen_start) * 1000.0
    gen_b64 = base64.b64encode(encode_hu_png(gen_hu, win_min, win_max)).decode('utf-8')

    return {
        "gt": f"data:image/png;base64,{gt_b64}",
        "fbp": f"data:image/png;base64,{fbp_b64}",
        "fbp_time": fbp_time,
        "mbir": f"data:image/png;base64,{mbir_b64}",
        "mbir_time": mbir_time,
        "dlr": f"data:image/png;base64,{dlr_b64}",
        "dlr_time": dlr_time,
        "gen": f"data:image/png;base64,{gen_b64}",
        "gen_time": gen_time
    }


@app.get("/api/evaluation/run")
async def run_evaluation_benchmark(
    request: Request,
    scope: str,
    num_patients: int,
    dataset_id: str,
    patient_id: str,
    slice_index: int,
    n_source: int,
    exposure_mas: float = 100.0,
    ww: Optional[float] = None,
    wl: Optional[float] = None,
    # FlashFBP filter gains (from the FlashFBP stage controls)
    w_low: Optional[float] = None,
    w_mid: Optional[float] = None,
    w_high: Optional[float] = None,
    # FidelityMBIR settings (from the iterative stage controls)
    iters: int = 25,
    tv: float = 4.0,
    lr: float = 0.1,
    precond: bool = True,
    # GenerativeVision settings (from the GenerativeVision stage controls)
    steps: int = 10,
    langevin_steps: int = 0,
    sigma_max: Optional[float] = None,
    sigma_min: Optional[float] = None,
    solver: str = "heun",
    temperature: float = 0.0,
    num_samples: int = 1,
    model_variant: str = "base",
):
    # Retrieve slices to evaluate
    eval_slices = []
    if scope == "single_patient":
        eval_slices = [(dataset_id, patient_id, slice_index)]
    elif scope == "single_dataset":
        if dataset_id not in DATASET_REGISTRY:
            return {"error": "Dataset not found"}
        ds = DATASET_REGISTRY[dataset_id]
        p_ids = sorted(ds.patients)
        for i in range(min(num_patients, len(p_ids))):
            p = p_ids[i]
            sl_list = ds.get_patient_slices(p)
            if sl_list:
                eval_slices.append((dataset_id, p, len(sl_list) // 2))
    elif scope == "all_datasets":
        all_candidates = []
        for d_id in sorted(DATASET_REGISTRY.keys()):
            cur_ds = DATASET_REGISTRY[d_id]
            for p in sorted(cur_ds.patients):
                sl_list = cur_ds.get_patient_slices(p)
                if sl_list:
                    all_candidates.append((d_id, p, len(sl_list) // 2))
        eval_slices = all_candidates[:num_patients]

    async def event_generator():
        try:
            device = sim_manager.device
            assets_dlr = get_dlr_runtime_assets(n_source, mode="standard")
            model_dlr = assets_dlr["model"]
            builder_dlr = assets_dlr["builder"]

            assets_gen = get_dlr_runtime_assets(n_source, mode="generative", model_variant=model_variant, exposure_mas=exposure_mas)
            model_gen = assets_gen["model"]
            builder_gen = assets_gen["builder"]

            # Generative noise range: use the UI values if provided, else the training-scale defaults.
            gen_sigma_max = sigma_max if sigma_max is not None else hu_std_to_atten(1000.0)
            gen_sigma_min = sigma_min if sigma_min is not None else hu_std_to_atten(1.0)

            acc_data = {
                "fbp": {"rmse_hu": [], "psnr": [], "ssim": [], "lpips": []},
                "mbir": {"rmse_hu": [], "psnr": [], "ssim": [], "lpips": []},
                "dlr": {"rmse_hu": [], "psnr": [], "ssim": [], "lpips": []},
                "generative": {"rmse_hu": [], "psnr": [], "ssim": [], "lpips": []}
            }

            total = len(eval_slices)
            for index, (d_id, p_id, s_idx) in enumerate(eval_slices):
                if await request.is_disconnected():
                    break

                cur_ds = DATASET_REGISTRY[d_id]
                slices = cur_ds.get_patient_slices(p_id)
                mu_img = cur_ds.get_processed_slice(slices[s_idx])

                sino_key = (d_id, p_id, s_idx, n_source)
                full_sino = sim_manager.get_sinogram(sino_key)
                if full_sino is None:
                    full_sino = sim_manager.project_full(mu_img, n_source, m_as=exposure_mas)
                    sim_manager.set_sinogram(sino_key, full_sino)

                # Set window scaling min/max
                target_ww = ww if ww is not None else cur_ds.window_width
                target_wl = wl if wl is not None else cur_ds.window_center
                win_min = target_wl - (target_ww / 2)
                win_max = target_wl + (target_ww / 2)

                gt_hu = (mu_img * 1000.0 / MU_WATER_60KEV) - 1000.0

                # 1. FlashFBP — apply the FlashFBP stage's filter gains.
                fbp_hu = sim_manager.reconstruct_step(
                    n_source, full_sino, step='filter', w_low=w_low, w_mid=w_mid, w_high=w_high
                )
                m_fbp = compute_metrics(fbp_hu, gt_hu, win_min, win_max, device=device)

                # 2. FidelityMBIR — apply the iterative stage's iterations / TV / LR / precond.
                iter_gen = sim_manager.run_iterative_recon(
                    n_source, full_sino, num_iters=iters, tv_weight=tv, lr=lr, use_precond=precond
                )
                mbir_hu = fbp_hu
                for update in iter_gen:
                    mbir_hu = update["image"]
                m_mbir = compute_metrics(mbir_hu, gt_hu, win_min, win_max, device=device)

                # 3. NeuralSpark (DLR)
                # Recompute the FBP from the acquired sinogram with the builder's training
                # filter (SVD pinv + optimal 2D ramp H_ramp), identical to prep_DLR.py and
                # prep_diffusion_train.py. The SAME measurement_components (pinv,
                # measurement_null, full_fbp) feed BOTH NeuralSpark and GenerativeVision.
                active_sino_tensor = get_active_sinogram_tensor(full_sino, n_source, device)
                measurement_components = builder_dlr.build_measurement_components_from_sinogram(active_sino_tensor)
                channels = builder_dlr.build_input_channels(measurement_components).unsqueeze(0)
                with torch.inference_mode():
                    diffusion_time = exposure_mas_to_diffusion_time(
                        torch.tensor([float(exposure_mas)], device=device, dtype=channels.dtype)
                    ).to(dtype=channels.dtype)
                    final_reconstruction = predict_reconstruction(model_dlr, channels, diffusion_time, builder_dlr).squeeze(0).squeeze(0)
                    dlr_hu = ((final_reconstruction.detach().cpu().numpy() * 1000.0 / MU_WATER_60KEV) - 1000.0).clip(-1000, 1500)
                m_dlr = compute_metrics(dlr_hu, gt_hu, win_min, win_max, device=device)

                # 4. GenerativeVision — use the GenerativeVision stage settings (steps,
                # langevin steps, sigma range, temperature, solver, model variant, num_samples).
                # Same builder FBP/filter as NeuralSpark (shared measurement_components); the
                # diffusion state is initialised from the measured null-space FBP plus sigma_max
                # null-space noise, exactly as the main generative endpoint / prep_diffusion_train.py.
                M = max(1, int(num_samples))
                pinv_t = measurement_components["pinv"].unsqueeze(0).unsqueeze(0).repeat(M, 1, 1, 1)
                measurement_null_t = measurement_components["measurement_null"].unsqueeze(0).unsqueeze(0).repeat(M, 1, 1, 1)
                # Init xt_full = pinv + xt_null at sigma_max, matching prep_diffusion_train.py.
                # Range anchor stays the pseudoinverse (channel 0 = pinv), as the new-input-style
                # model was trained.
                noise_null = builder_gen.project_batch_null(torch.randn_like(pinv_t)) * gen_sigma_max
                xt_null_start = builder_gen.project_batch_null(measurement_null_t + noise_null)
                xt_start = pinv_t + xt_null_start
                inputs = builder_gen.build_input_channels(measurement_components).unsqueeze(0).repeat(M, 1, 1, 1)
                inputs[:, 3:4, ...] = xt_null_start
                inputs[:, 4:5, ...] = xt_start
                gen_iter = solve_combined_diffusion_langevin(
                    model=model_gen,
                    builder=builder_gen,
                    inputs=inputs,
                    num_steps_diff=int(steps),
                    num_steps_lang=int(langevin_steps),
                    sigma_max=gen_sigma_max,
                    sigma_min=gen_sigma_min,
                    rho=7.0,
                    temperature=float(temperature),
                    solver=solver,
                    device=device
                )
                gen_x0 = None
                for update in gen_iter:
                    gen_x0 = update["x0"]
                if gen_x0 is not None:
                    # Posterior mean across the M samples (matches the GenerativeVision stage).
                    gen_mu = gen_x0.mean(dim=0).squeeze(0)
                    gen_hu = ((gen_mu.detach().cpu().numpy() * 1000.0 / MU_WATER_60KEV) - 1000.0).clip(-1000, 1500)
                else:
                    gen_hu = dlr_hu
                m_gen = compute_metrics(gen_hu, gt_hu, win_min, win_max, device=device)

                # Record
                for k, v in m_fbp.items():
                    acc_data["fbp"][k].append(v)
                for k, v in m_mbir.items():
                    acc_data["mbir"][k].append(v)
                for k, v in m_dlr.items():
                    acc_data["dlr"][k].append(v)
                for k, v in m_gen.items():
                    acc_data["generative"][k].append(v)

                # Per-method 2x2 violin panels (RMSE HU, PSNR, SSIM, LPIPS), accumulating
                # cumulatively across every sampled patient. Shared per-metric y-limits make
                # the same metric directly comparable across all method panels.
                y_limits = compute_metric_ylimits(acc_data)
                method_plots = {
                    "fbp": generate_method_square_plot_png(acc_data, "fbp", "#52daf2", y_limits),
                    "mbir": generate_method_square_plot_png(acc_data, "mbir", "#75ff33", y_limits),
                    "dlr": generate_method_square_plot_png(acc_data, "dlr", "#ffb733", y_limits),
                    "gen": generate_method_square_plot_png(acc_data, "generative", "#ff5733", y_limits),
                }

                # Stream the reconstructions for every sampled patient so the images update
                # live as the benchmark progresses.
                payload = {
                    "index": index + 1,
                    "total": total,
                    "patient_id": p_id,
                    "plots": method_plots,
                    "reconstructions": {
                        "gt": base64.b64encode(encode_hu_png(gt_hu, win_min, win_max)).decode('utf-8'),
                        "fbp": base64.b64encode(encode_hu_png(fbp_hu, win_min, win_max)).decode('utf-8'),
                        "mbir": base64.b64encode(encode_hu_png(mbir_hu, win_min, win_max)).decode('utf-8'),
                        "dlr": base64.b64encode(encode_hu_png(dlr_hu, win_min, win_max)).decode('utf-8'),
                        "gen": base64.b64encode(encode_hu_png(gen_hu, win_min, win_max)).decode('utf-8'),
                    }
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0.01)

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

LANDING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script>
        // Constants for conversion (Water mu at 60keV)
        const MU_WATER_60KEV = 0.0183;

        /**
         * Converts attenuation linear scale to Hounsfield Units (HU).
         * For standard deviation (scale only), shifts are ignored.
         */
        function atten_to_HU(atten, scaleOnly = false) {
            const hu = atten * (1000.0 / MU_WATER_60KEV);
            return scaleOnly ? hu : hu - 1000.0;
        }

        /**
         * Converts Hounsfield Units (HU) to attenuation linear scale.
         * For standard deviation (scale only), shifts are ignored.
         */
        function HU_to_atten(hu, scaleOnly = false) {
            const atten = scaleOnly ? hu : hu + 1000.0;
            return atten * (MU_WATER_60KEV / 1000.0);
        }
    </script>
    <title>CT Reconstruction Laboratory</title>
    <link rel="stylesheet" href="/static/site/landing.css?v=20260607-fbp-redo-3">
</head>
<body>
    <main class="kiosk-shell">
        <header class="page-header" style="display: flex; justify-content: center; align-items: center; padding: 1.5rem 2.5rem 1rem; position: relative;">
            <h1 class="page-title" style="margin: 0; text-align: center;">Static CT Reconstruction Algorithms</h1>
        </header>

        <section class="workflow-shell">
            <aside class="sidebar">
                <div class="brand-stack">
                    <nav class="stage-nav" aria-label="Workflow stages">
                    <button class="stage-button active" data-stage-button="load-patient" type="button" aria-label="Load Patient">
                        <span class="stage-button-art">
                            <img class="stage-button-icon" src="/static/site/workflow-icons/loadpatient_icon.png" alt="Load Patient icon">
                        </span>
                    </button>
                    <button class="stage-button" data-stage-button="simulate-ct-data" disabled type="button" aria-label="Simulate CT Data">
                        <span class="stage-button-art">
                            <img class="stage-button-icon" src="/static/site/workflow-icons/simulation_icon.png" alt="Simulate CT Data icon">
                        </span>
                    </button>
                    <button class="stage-button" data-stage-button="eigen-fbp-recon" disabled type="button" aria-label="FlashFBP">
                        <span class="stage-button-art">
                            <img class="stage-button-icon" src="/static/site/workflow-icons/flashFBP_icon.png" alt="FlashFBP icon">
                        </span>
                    </button>
                    <button class="stage-button" data-stage-button="model-based-iterative-recon" disabled type="button" aria-label="FidelityMBIR">
                        <span class="stage-button-art">
                            <img class="stage-button-icon" src="/static/site/workflow-icons/fidelity_mbir_icon.png" alt="FidelityMBIR icon">
                        </span>
                    </button>
                    <button class="stage-button" data-stage-button="deep-learning-recon" disabled type="button" aria-label="NeuralSpark">
                        <span class="stage-button-art">
                            <img class="stage-button-icon" src="/static/site/workflow-icons/neuralspark_icon.png" alt="NeuralSpark icon">
                        </span>
                    </button>
                    <button class="stage-button" data-stage-button="generative-ai-recon" disabled type="button" aria-label="GenerativeVision">
                        <span class="stage-button-art">
                            <img class="stage-button-icon" src="/static/site/workflow-icons/generativevision_icon.png" alt="GenerativeVision icon">
                        </span>
                    </button>
                    <button class="stage-button" data-stage-button="evaluation" disabled type="button" aria-label="Evaluation">
                        <span class="stage-button-art">
                            <img class="stage-button-icon" src="/static/site/workflow-icons/evaluation_icon.png" alt="Evaluation icon">
                        </span>
                    </button>
                    <button class="stage-button" data-stage-button="motion-scope" disabled type="button" aria-label="MotionScope">
                        <span class="stage-button-art">
                            <img class="stage-button-icon" src="/static/site/workflow-icons/motioncope_icon.png" alt="MotionScope icon">
                        </span>
                    </button>
                    </nav>
                </div>
            </aside>

            <section class="content">
                <section class="stage-area">
                <article class="stage-panel active" data-stage-panel="load-patient">
                    <div class="stage-header" style="margin-bottom: 0.5rem;">
                        <div>
                            <h2 class="stage-title" style="margin: 0;">Load Patient</h2>
                        </div>
                    </div>
                    <div class="stage-figure-shell">
                        <div class="dataset-grid">
                            <button class="dataset-card" data-dataset="head">
                                <div class="dataset-preview" id="preview-head"></div>
                                <div class="dataset-info">
                                    <span class="dataset-label">Head CT</span>
                                    <span class="dataset-source">CQ500</span>
                                </div>
                            </button>
                            <button class="dataset-card" data-dataset="thorax">
                                <div class="dataset-preview" id="preview-thorax"></div>
                                <div class="dataset-info">
                                    <span class="dataset-label">Thorax CT</span>
                                    <span class="dataset-source">LIDC</span>
                                </div>
                            </button>
                            <button class="dataset-card" data-dataset="abdomen">
                                <div class="dataset-preview" id="preview-abdomen"></div>
                                <div class="dataset-info">
                                    <span class="dataset-label">Abdomen CT</span>
                                    <span class="dataset-source">LIHC</span>
                                </div>
                            </button>
                            <button class="dataset-card" data-dataset="pelvic">
                                <div class="dataset-preview" id="preview-pelvic"></div>
                                <div class="dataset-info">
                                    <span class="dataset-label">Pelvic CT</span>
                                    <span class="dataset-source">ACRIN</span>
                                </div>
                            </button>
                        </div>
                        <div class="selection-controls" id="patient-selection-controls" style="display: none; flex-direction: column; gap: 0.5rem; margin-top: 0.5rem;">
                            <!-- Top section: Patient/Slice Sliders next to DICOM metadata -->
                            <div style="display: grid; grid-template-columns: 1.8fr 1fr; gap: 1.5rem; align-items: stretch; width: 100%;">
                                <div class="controls-column" style="display: flex; flex-direction: column; gap: 0.35rem; justify-content: center;">
                                    <div class="control-row">
                                        <label>Patient</label>
                                        <input type="range" id="patient-slider" min="0" max="0" value="0">
                                        <span id="patient-id-display">N/A</span>
                                    </div>
                                    <div class="control-row">
                                        <label>Axial Slice</label>
                                        <input type="range" id="slice-slider" min="0" max="0" value="0">
                                        <div class="slice-vitals">
                                            <span id="slice-index-display">0</span>
                                            <span class="vitals-separator">|</span>
                                            <span id="slice-pos-display">0.0 mm</span>
                                            <span class="vitals-separator">|</span>
                                            <span id="slice-inst-display">Inst: 1</span>
                                        </div>
                                    </div>
                                    <div class="control-row">
                                        <label>Window Level</label>
                                        <input type="range" id="window-level-slider" min="-1000" max="1500" value="50">
                                        <span id="wl-display">50</span>
                                    </div>
                                    <div class="control-row">
                                        <label>Window Width</label>
                                        <input type="range" id="window-width-slider" min="1" max="2500" value="350">
                                        <span id="ww-display">350</span>
                                    </div>
                                    <div class="control-row-col" style="margin-top: 2px;">
                                        <label style="font-weight: 600; font-size: 0.8rem; color: var(--arpa-h-primary-700); text-transform: uppercase;">Window Presets</label>
                                        <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px;">
                                            <button class="button button-ghost" id="wlpreset-soft" type="button" data-ww="400" data-wl="40" style="padding: 5px 10px; font-size: 11px; min-width: auto; height: 30px; flex: 1 1 80px;">Soft Tissue</button>
                                            <button class="button button-ghost" id="wlpreset-lung" type="button" data-ww="1500" data-wl="-600" style="padding: 5px 10px; font-size: 11px; min-width: auto; height: 30px; flex: 1 1 80px;">Lung</button>
                                            <button class="button button-ghost" id="wlpreset-brain" type="button" data-ww="80" data-wl="40" style="padding: 5px 10px; font-size: 11px; min-width: auto; height: 30px; flex: 1 1 80px;">Brain</button>
                                            <button class="button button-ghost" id="wlpreset-bone" type="button" data-ww="1800" data-wl="400" style="padding: 5px 10px; font-size: 11px; min-width: auto; height: 30px; flex: 1 1 80px;">Bone</button>
                                            <button class="button button-ghost" id="wlpreset-mediastinum" type="button" data-ww="350" data-wl="50" style="padding: 5px 10px; font-size: 11px; min-width: auto; height: 30px; flex: 1 1 80px;">Mediastinum</button>
                                            <button class="button button-ghost" id="wlpreset-liver" type="button" data-ww="150" data-wl="30" style="padding: 5px 10px; font-size: 11px; min-width: auto; height: 30px; flex: 1 1 80px;">Liver</button>
                                        </div>
                                    </div>
                                </div>
                                <div class="metadata-panel" style="padding: 0.8rem; margin: 0;">
                                    <h3 style="margin: 0 0 0.4rem 0; font-size: 0.8rem;">DICOM METADATA</h3>
                                    <div class="metadata-scroll-container" style="padding: 0.4rem; height: calc(100% - 1.2rem); overflow-y: auto;">
                                        <div id="dicom-metadata-text" class="metadata-dump" style="font-size: 0.85rem; line-height: 1.4;"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="stage-footer">
                        <div class="stage-actions">
                            <div class="action-group">
                                <button class="button button-primary" id="load-patient-button" disabled type="button">Load Patient</button>
                                <button class="button button-ghost" data-next-stage="load-patient" id="next-to-simulate" disabled type="button">Next: Simulate CT Data</button>
                            </div>
                        </div>
                        <div class="progress-block">
                            <div class="progress-label">Patient loading status</div>
                            <div class="progress-track"><div class="progress-fill" data-progress-fill="load-patient"></div></div>
                            <div class="progress-status" data-progress-status="load-patient">Idle</div>
                        </div>
                    </div>
                </article>

                <article class="stage-panel" data-stage-panel="simulate-ct-data">
                    <div class="stage-header" style="margin-bottom: 0.5rem;">
                        <div>
                            <h2 class="stage-title" style="margin: 0;">Simulate CT Data</h2>
                        </div>
                    </div>
                    <div class="stage-figure-shell">
                        <!-- Figures row at the top -->
                        <div class="reconstruction-layout-grid" style="grid-template-columns: repeat(3, minmax(0, 1fr)) !important;">
                            <!-- Column 1: Loaded Patient Canvas -->
                            <div class="recon-view" id="sim-display-patient-source">
                                <label>Loaded Patient Slice</label>
                                <div class="stage-figure-box" style="margin: 0 auto;">
                                    <canvas id="sim-patient-canvas" style="width: 100%; height: 100%; background: black; border: none;"></canvas>
                                </div>
                            </div>
                            <!-- Column 2: System Geometry -->
                            <div class="recon-view" id="sim-display-geometry">
                                <label>System Geometry & Rays</label>
                                <div class="stage-figure-box" style="margin: 0 auto;">
                                    <canvas id="geometry-canvas" style="width: 100%; height: 100%; background: black; border: none;"></canvas>
                                </div>
                            </div>
                            <!-- Column 3: Sinogram Preview -->
                            <div class="recon-view" id="sim-display-projection">
                                <label>Sinogram Preview</label>
                                <div class="stage-figure-box" style="margin: 0 auto;">
                                    <canvas id="sinogram-canvas" style="width: 100%; height: 100%; background: black; border: none;"></canvas>
                                </div>
                            </div>
                        </div>

                        <!-- Controls panel at the bottom -->
                        <div class="control-panel-wrapper" style="margin-bottom: 0;">
                            <div class="control-panel compact-generative-panel" style="padding: 0.75rem 1.25rem !important;">
                                <div class="control-rows-container">
                                    <div class="control-row">
                                        <label>System Geometry</label>
                                        <select id="sim-geometry-select" style="padding: 6px 12px; border-radius: 6px; border: 1px solid rgba(0, 27, 94, 0.2); background: white; font-size: 0.95rem; font-family: var(--font-family); color: var(--scale-900); flex-grow: 1; outline: none; height: 32px; font-weight: 500;">
                                            <option value="80">80-View Static CT</option>
                                            <option value="240">240-View Static CT</option>
                                        </select>
                                    </div>
                                    <div class="control-row">
                                        <label>Exposure (mAs)</label>
                                        <select id="sim-exposure-select" style="padding: 6px 12px; border-radius: 6px; border: 1px solid rgba(0, 27, 94, 0.2); background: white; font-size: 0.95rem; font-family: var(--font-family); color: var(--scale-900); flex-grow: 1; outline: none; height: 32px; font-weight: 500;">
                                            <option value="1">1 mAs (I0=1e5)</option>
                                            <option value="10">10 mAs (I0=1e6)</option>
                                            <option value="100" selected>100 mAs (I0=1e7)</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="stage-footer">
                        <div class="stage-actions">
                            <div class="action-group">
                                <button class="button button-primary" data-run-button="simulate-ct-data" disabled type="button">Simulate CT Data</button>
                                <button class="button button-stop" data-stop-button="simulate-ct-data" disabled type="button">Stop</button>
                                <button class="button button-ghost" data-next-stage="simulate-ct-data" disabled type="button">Next: FlashFBP</button>
                            </div>
                        </div>
                        <div class="progress-block">
                            <div class="progress-label">Simulation status</div>
                            <div class="progress-track"><div class="progress-fill" data-progress-fill="simulate-ct-data"></div></div>
                            <div class="progress-status" data-progress-status="simulate-ct-data">Idle</div>
                            <div id="sim-gif-link-container" style="margin-top: 10px; display: none;">
                                <a id="sim-gif-link" href="#" target="_blank" style="color: #0078d4; text-decoration: underline; font-weight: 500;">Download Simulation GIF</a>
                            </div>
                        </div>
                    </div>
                        <div class="progress-block">
                            <div class="progress-label">Simulation status</div>
                            <div class="progress-track"><div class="progress-fill" data-progress-fill="simulate-ct-data"></div></div>
                            <div class="progress-status" data-progress-status="simulate-ct-data">Idle</div>
                            <div id="sim-gif-link-container" style="margin-top: 10px; display: none;">
                                <a id="sim-gif-link" href="#" target="_blank" style="color: #0078d4; text-decoration: underline; font-weight: 500;">Download Simulation GIF</a>
                            </div>
                        </div>
                    </div>
                </article>

                <article class="stage-panel" data-stage-panel="eigen-fbp-recon">
                    <div class="stage-header" style="margin-bottom: 0.5rem;">
                        <div>
                            <h2 class="stage-title" style="margin: 0;">FlashFBP</h2>
                        </div>
                    </div>
                    <div class="stage-figure-shell">
                        <div class="reconstruction-layout-grid">
                            <div class="recon-view">
                                <label>Ground Truth</label>
                                <div class="stage-figure-box" id="recon-box-gt" style="margin: 0 auto;"></div>
                            </div>
                            <div class="recon-view">
                                <label>Simulated Sinogram</label>
                                <div class="stage-figure-box" id="recon-box-sino" style="margin: 0 auto;"></div>
                                <div class="timing-display" id="timing-sino">--</div>
                            </div>
                            <div class="recon-view">
                                <label>Unfiltered Back Projection</label>
                                <div class="stage-figure-box" id="recon-box-unfiltered" aria-label="Unfiltered BP" style="margin: 0 auto;"></div>
                                <div class="timing-display" id="timing-unfiltered">-- ms</div>
                            </div>
                            <div class="recon-view">
                                <label>FlashFBP Reconstruction</label>
                                <div class="stage-figure-box" id="recon-box-filtered" aria-label="Filtered Recon" style="margin: 0 auto;"></div>
                                <div class="timing-display" id="timing-filtered">-- ms</div>
                            </div>
                        </div>

                        <div class="control-panel-wrapper" style="flex: 1.5 1 0% !important; margin-bottom: 0 !important; display: flex; flex-direction: column; min-height: 0;">
                            <!-- Control Panel (Full Width and multi-column flexible layout) -->
                            <div class="control-panel compact-generative-panel" style="padding: 1rem 1.25rem !important; flex: 1 1 0%; min-height: 0;">
                                <div class="generative-control-grid" style="display: grid; grid-template-columns: 1.2fr 1fr 1fr; gap: 0.75rem 1.5rem; width: 100%; height: 100%; min-height: 0;">
                                    <!-- Left Column: Preset buttons -->
                                    <div style="display: flex; flex-direction: column; gap: 0.5rem; border-right: 1px solid rgba(0, 27, 94, 0.1); padding-right: 15px; justify-content: center;">
                                        <div class="control-row-col">
                                            <label style="font-weight: 600; font-size: 0.85rem; color: var(--arpa-h-primary-700); text-transform: uppercase;">Filter Presets</label>
                                            <div style="display: flex; gap: 8px; margin-top: 6px;">
                                                <button class="button button-ghost active" id="preset-ramp-btn" type="button" style="padding: 6px 12px; font-size: 12px; min-width: auto; height: 32px; flex: 1; display: flex; align-items: center; justify-content: center; background-color: #0a63a8; color: white;">Ramp</button>
                                                <button class="button button-ghost" id="preset-sharp-btn" type="button" style="padding: 6px 12px; font-size: 12px; min-width: auto; height: 32px; flex: 1; display: flex; align-items: center; justify-content: center;">Sharp</button>
                                                <button class="button button-ghost" id="preset-soft-btn" type="button" style="padding: 6px 12px; font-size: 12px; min-width: auto; height: 32px; flex: 1; display: flex; align-items: center; justify-content: center;">Soft</button>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Middle Column: Filter Subband Sliders -->
                                    <div style="display: flex; flex-direction: column; gap: 0.6rem; justify-content: center;">
                                        <div class="control-row" style="grid-template-columns: 120px 1fr 50px !important;">
                                            <label>Low Freq Gain</label>
                                            <input type="range" id="fbp-low-slider" min="0" max="100" step="1" value="100">
                                            <span id="fbp-low-display" class="slider-val-badge">100%</span>
                                        </div>
                                        <div class="control-row" style="grid-template-columns: 120px 1fr 50px !important;">
                                            <label>Mid Freq Gain</label>
                                            <input type="range" id="fbp-mid-slider" min="0" max="100" step="1" value="100">
                                            <span id="fbp-mid-display" class="slider-val-badge">100%</span>
                                        </div>
                                        <div class="control-row" style="grid-template-columns: 120px 1fr 50px !important;">
                                            <label>High Freq Gain</label>
                                            <input type="range" id="fbp-high-slider" min="0" max="100" step="1" value="100">
                                            <span id="fbp-high-display" class="slider-val-badge">100%</span>
                                        </div>
                                    </div>

                                    <!-- Right Column: Active Filter Plot -->
                                    <div style="display: flex; flex-direction: column; gap: 0.4rem; justify-content: center; border-left: 1px solid rgba(0, 27, 94, 0.1); padding-left: 15px;">
                                        <label style="font-weight: 600; font-size: 0.85rem; color: var(--arpa-h-primary-700); text-transform: uppercase;">Active Filter Plot</label>
                                        <div class="stage-figure-box" id="recon-box-filter-plot" style="background: white; flex-grow: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; border-radius: 8px; border: 1px solid #444; min-height: 190px; width: 100%;">
                                            <img id="fbp-filter-plot-img" style="width: 100%; height: 100%; object-fit: contain;" src="" alt="Filter graph plot">
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="stage-footer">
                        <div class="stage-actions">
                            <div class="action-group">
                                <button class="button button-primary" data-run-button="eigen-fbp-recon" disabled type="button">Run FlashFBP</button>
                                <button class="button button-stop" data-stop-button="eigen-fbp-recon" disabled type="button">Stop</button>
                                <button class="button button-ghost" data-next-stage="eigen-fbp-recon" disabled type="button">Next: FidelityMBIR</button>
                            </div>
                        </div>
                        <div class="progress-block">
                        </div>
                        <div class="progress-block">
                            <div class="progress-label">FlashFBP status</div>
                            <div class="progress-track"><div class="progress-fill" data-progress-fill="eigen-fbp-recon"></div></div>
                            <div class="progress-status" data-progress-status="eigen-fbp-recon">Idle</div>
                        </div>
                    </div>
                </article>

                <article class="stage-panel" data-stage-panel="model-based-iterative-recon">
                    <div class="stage-header" style="margin-bottom: 0.5rem;">
                        <div>
                            <h2 class="stage-title" style="margin: 0;">FidelityMBIR</h2>
                        </div>
                    </div>
                    
                    <div class="stage-figure-shell">
                        <div class="iterative-layout-grid">
                            <div class="recon-view">
                                <label>Ground Truth</label>
                                <div class="stage-figure-box" id="iter-box-gt" style="margin: 0 auto;"></div>
                            </div>
                            <div class="recon-view">
                                <label>Simulated Sinogram</label>
                                <div class="stage-figure-box" id="iter-box-sino" style="margin: 0 auto;"></div>
                            </div>
                            <div class="recon-view">
                                <label>FlashFBP Initialization</label>
                                <div class="stage-figure-box" id="iter-box-init" style="margin: 0 auto;"></div>
                            </div>
                            <div class="recon-view">
                                <label>FidelityMBIR</label>
                                <div class="stage-figure-box" id="iter-box-live" style="margin: 0 auto;"></div>
                            </div>
                            <div class="recon-view recon-view-loss recon-view-tall">
                                <label>Loss Function</label>
                                <div class="stage-figure-box stage-figure-box-tall" id="iter-box-loss" style="margin: 0 auto;"></div>
                            </div>
                        </div>

                        <div class="control-panel-wrapper" style="margin-bottom: 0;">
                            <!-- Control Panel (Full Width) -->
                            <div class="control-panel compact-generative-panel" style="padding: 0.75rem 1.25rem !important;">
                                <div class="control-rows-container" style="display: grid !important; grid-template-columns: 1fr 1fr !important; gap: 0.5rem 1.5rem !important;">
                                    <div class="control-row" style="grid-template-columns: 130px 1fr 50px !important;">
                                        <label>Iterations</label>
                                        <input type="range" id="iter-count-slider" min="10" max="500" step="10" value="30">
                                        <span id="iter-count-display">30</span>
                                    </div>
                                    <div class="control-row" style="grid-template-columns: 130px 1fr 50px !important;">
                                        <label>TV Strength</label>
                                        <input type="range" id="tv-strength-slider" min="-6" max="10" step="0.5" value="4.0">
                                        <span id="tv-strength-display">10000</span>
                                    </div>
                                    <div class="control-row" style="grid-template-columns: 130px 1fr 50px !important;">
                                        <label>Step Size / LR</label>
                                        <input type="range" id="lr-slider" min="-8" max="1" step="0.25" value="-1">
                                        <span id="lr-display">0.1</span>
                                    </div>
                                    <div class="control-row" style="grid-template-columns: 130px 1fr 50px !important;">
                                        <label>Eigen Precond.</label>
                                        <div class="checkbox-container">
                                            <input type="checkbox" id="use-precond-check" checked>
                                        </div>
                                        <span></span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="stage-footer">
                        <div class="stage-actions">
                            <div class="action-group">
                                <button class="button button-primary" data-run-button="model-based-iterative-recon" disabled type="button">Run FidelityMBIR</button>
                                <button class="button button-stop" data-stop-button="model-based-iterative-recon" disabled type="button">Stop</button>
                                <button class="button button-ghost" data-next-stage="model-based-iterative-recon" disabled type="button">Next: NeuralSpark</button>
                            </div>
                        </div>
                        <div class="progress-block">
                            <div class="progress-label">FidelityMBIR progress</div>
                            <div class="progress-track"><div class="progress-fill" data-progress-fill="model-based-iterative-recon"></div></div>
                            <div class="progress-status" data-progress-status="model-based-iterative-recon">Idle</div>
                        </div>
                    </div>
                </article>

                <article class="stage-panel" data-stage-panel="deep-learning-recon">
                    <div class="stage-header" style="margin-bottom: 0.5rem;">
                        <div>
                            <h2 class="stage-title" style="margin: 0;">NeuralSpark</h2>
                        </div>
                    </div>
                    <div class="stage-figure-shell">
                        <div class="reconstruction-layout-grid">
                            <div class="recon-view">
                                <label>Ground Truth</label>
                                <div class="stage-figure-box" id="dlr-box-gt" style="margin: 0 auto;"></div>
                            </div>
                            <div class="recon-view">
                                <label>Simulated Sinogram</label>
                                <div class="stage-figure-box" id="dlr-box-sino" style="margin: 0 auto;"></div>
                            </div>
                            <div class="recon-view">
                                <label>FlashFBP Initialization</label>
                                <div class="stage-figure-box" id="dlr-box-init" style="margin: 0 auto;"></div>
                                <div class="timing-display" id="timing-dlr-init">-- ms</div>
                            </div>
                            <div class="recon-view">
                                <label>NeuralSpark Final Reconstruction</label>
                                <div class="stage-figure-box" id="dlr-box-final" style="margin: 0 auto;"></div>
                                <div class="timing-display" id="timing-dlr-final">-- ms</div>
                            </div>
                        </div>
                        <!-- Empty spacer (no controls here) so images match the FlashFBP page height -->
                        <div class="control-panel-wrapper" aria-hidden="true"></div>
                    </div>
                    <div class="stage-footer">
                        <div class="stage-actions">
                            <div class="action-group">
                                <button class="button button-primary" data-run-button="deep-learning-recon" disabled type="button">Run NeuralSpark</button>
                                <button class="button button-stop" data-stop-button="deep-learning-recon" disabled type="button">Stop</button>
                                <button class="button button-ghost" data-next-stage="deep-learning-recon" disabled type="button">Next: GenerativeVision</button>
                            </div>
                        </div>
                        <div class="progress-block">
                            <div class="progress-label">NeuralSpark status</div>
                            <div class="progress-track"><div class="progress-fill" data-progress-fill="deep-learning-recon"></div></div>
                            <div class="progress-status" data-progress-status="deep-learning-recon">Idle</div>
                        </div>
                    </div>
                </article>

                <article class="stage-panel" data-stage-panel="generative-ai-recon">
                    <div class="stage-header" style="margin-bottom: 0.5rem;">
                        <div>
                            <h2 class="stage-title" style="margin: 0;">GenerativeVision</h2>
                        </div>
                    </div>

                    <div class="stage-figure-shell">
                        <div class="reconstruction-layout-grid">
                            <div class="recon-view">
                                <label>Ground Truth</label>
                                <div class="stage-figure-box" id="gen-box-gt" style="margin: 0 auto;"></div>
                            </div>
                            <div class="recon-view">
                                <label>FlashFBP Initialization</label>
                                <div class="stage-figure-box" id="gen-box-full-fbp" style="margin: 0 auto;"></div>
                            </div>
                            <div class="recon-view">
                                <label>Generative Process</label>
                                <div class="stage-figure-box" id="gen-box-xt" style="margin: 0 auto;"></div>
                            </div>
                            <div class="recon-view">
                                <label>Generative Reconstruction</label>
                                <div class="stage-figure-box" id="gen-box-x0" style="margin: 0 auto;"></div>
                            </div>
                        </div>

                        <div class="control-panel-wrapper">
                            <!-- Control Panel (Full Width and multi-column flexible layout) -->
                            <div class="control-panel compact-generative-panel">
                                <!-- Slide & Radio section inside Control Panel arranged dynamically -->
                                <div class="generative-control-grid" style="display: grid; grid-template-columns: 1.2fr 1fr 1fr; gap: 0.75rem 1.5rem; width: 100%;">
                                    <!-- Left Column: Config Options -->
                                    <div style="display: flex; flex-direction: column; gap: 0.5rem; border-right: 1px solid rgba(0, 27, 94, 0.1); padding-right: 15px;">
                                        <div class="control-row-col">
                                            <label style="font-weight: 600; font-size: 0.85rem; color: var(--arpa-h-primary-700); text-transform: uppercase;">Model Variant</label>
                                            <div class="radio-group-horizontal-compact" style="display: flex; gap: 1rem; margin-top: 2px;">
                                                <label class="compact-radio-label">
                                                    <input type="radio" name="diffusion-model-variant" value="base">
                                                    <span>Base</span>
                                                </label>
                                                <label class="compact-radio-label">
                                                    <input type="radio" name="diffusion-model-variant" value="finetuned" checked>
                                                    <span>Finetuned</span>
                                                </label>
                                            </div>
                                        </div>
                                        
                                        <div class="control-row-col" style="margin-top: 4px;">
                                            <label style="font-weight: 600; font-size: 0.85rem; color: var(--arpa-h-primary-700); text-transform: uppercase;">Solver Mode</label>
                                            <div class="radio-group-horizontal-compact" style="display: flex; gap: 1rem; margin-top: 2px;">
                                                <label class="compact-radio-label">
                                                    <input type="radio" name="diffusion-solver" value="heun" checked>
                                                    <span>Heun</span>
                                                </label>
                                                <label class="compact-radio-label">
                                                    <input type="radio" name="diffusion-solver" value="euler">
                                                    <span>Euler</span>
                                                </label>
                                            </div>
                                        </div>

                                        <div class="control-row-col" style="margin-top: 4px;">
                                            <label style="font-weight: 600; font-size: 0.85rem; color: var(--arpa-h-primary-700); text-transform: uppercase;">Display Mode</label>
                                            <div class="radio-group-horizontal-compact" style="display: flex; gap: 1rem; margin-top: 2px;">
                                                <label class="compact-radio-label">
                                                    <input type="radio" name="display-mode" value="sample" checked>
                                                    <span>Sample</span>
                                                </label>
                                                <label class="compact-radio-label">
                                                    <input type="radio" name="display-mode" value="animation">
                                                    <span>Animation</span>
                                                </label>
                                                <label class="compact-radio-label">
                                                    <input type="radio" name="display-mode" value="mean">
                                                    <span>Mean</span>
                                                </label>
                                                <label class="compact-radio-label">
                                                    <input type="radio" name="display-mode" value="hallucination">
                                                    <span>Hallucination Map</span>
                                                </label>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Middle Column: Diffusion Time & Noise Sliders -->
                                    <div style="display: flex; flex-direction: column; gap: 0.4rem; justify-content: center;">
                                        <div class="control-row" style="grid-template-columns: 120px 1fr 40px !important;">
                                            <label>Diffusion Steps</label>
                                            <input type="range" id="diffusion-steps-slider" min="5" max="50" step="5" value="50">
                                            <span id="diffusion-steps-display" class="slider-val-badge">50</span>
                                        </div>
                                        <div class="control-row" style="grid-template-columns: 120px 1fr 40px !important;">
                                            <label>Max Null Noise</label>
                                            <input type="range" id="sigma-max-slider" min="0" max="3" step="0.05" value="2.7">
                                            <span id="sigma-max-display" class="slider-val-badge">500</span>
                                        </div>
                                        <div class="control-row" style="grid-template-columns: 120px 1fr 40px !important;">
                                            <label>Min Null Noise</label>
                                            <input type="range" id="sigma-min-slider" min="0" max="3" step="0.05" value="1.0">
                                            <span id="sigma-min-display" class="slider-val-badge">10</span>
                                        </div>
                                        <div class="control-row" style="grid-template-columns: 120px 1fr 40px !important;">
                                            <label>Num Samples</label>
                                            <input type="range" id="num-samples-slider" min="1" max="16" step="1" value="1">
                                            <span id="num-samples-display" class="slider-val-badge">1</span>
                                        </div>
                                    </div>

                                    <!-- Right Column: Langevin Sliders -->
                                    <div style="display: flex; flex-direction: column; gap: 0.4rem; justify-content: center; border-left: 1px solid rgba(0, 27, 94, 0.1); padding-left: 15px;">
                                        <div class="control-row" style="grid-template-columns: 110px 1fr 40px !important;">
                                            <label>Langevin Steps</label>
                                            <input type="range" id="langevin-steps-slider" min="0" max="500" step="10" value="0">
                                            <span id="langevin-steps-display" class="slider-val-badge">0</span>
                                        </div>
                                        <div class="control-row" style="grid-template-columns: 110px 1fr 40px !important;">
                                            <label>Langevin Temp</label>
                                            <input type="range" id="diffusion-temperature-slider" min="0" max="5" step="0.05" value="5.0">
                                            <span id="diffusion-temperature-display" class="slider-val-badge">5.00</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="stage-footer">
                        <div class="stage-actions">
                            <div class="action-group">
                                <button class="button button-primary" data-run-button="generative-ai-recon" disabled type="button">Run GenerativeVision</button>
                                <button class="button button-stop" data-stop-button="generative-ai-recon" disabled type="button">Stop</button>
                            </div>
                        </div>
                        <div class="progress-block">
                            <div class="progress-label">GenerativeVision status</div>
                            <div class="progress-track"><div class="progress-fill" data-progress-fill="generative-ai-recon"></div></div>
                            <div class="progress-status" data-progress-status="generative-ai-recon">Idle</div>
                        </div>
                    </div>
                </article>

                <article class="stage-panel" data-stage-panel="evaluation">
                    <div class="stage-header" style="margin-bottom: 0.5rem;">
                        <div>
                            <h2 class="stage-title" style="margin: 0;">Evaluation Benchmarking</h2>
                        </div>
                    </div>
                    
                    <div class="stage-figure-shell">
                        <!-- 5 columns (one per method) x 2 rows: reconstructions on top, metric panels below -->
                        <div class="eval-grid">
                            <!-- Column headers (method names) -->
                            <div class="eval-col-head">Ground Truth</div>
                            <div class="eval-col-head">FlashFBP</div>
                            <div class="eval-col-head">FidelityMBIR</div>
                            <div class="eval-col-head">NeuralSpark</div>
                            <div class="eval-col-head">GenerativeVision</div>

                            <!-- Row 1: reconstructions -->
                            <div class="eval-box eval-box-img" id="eval-box-gt"></div>
                            <div class="eval-box eval-box-img" id="eval-box-fbp"></div>
                            <div class="eval-box eval-box-img" id="eval-box-mbir"></div>
                            <div class="eval-box eval-box-img" id="eval-box-dlr"></div>
                            <div class="eval-box eval-box-img" id="eval-box-gen"></div>

                            <!-- Row 2: per-method 2x2 metric panels (PSNR, SSIM, LPIPS, RMSE HU) -->
                            <div class="eval-box eval-box-plot eval-box-empty">Reference &mdash; no error metrics</div>
                            <div class="eval-box eval-box-plot"><img id="eval-plot-fbp" src="" alt="FlashFBP benchmark metrics"></div>
                            <div class="eval-box eval-box-plot"><img id="eval-plot-mbir" src="" alt="FidelityMBIR benchmark metrics"></div>
                            <div class="eval-box eval-box-plot"><img id="eval-plot-dlr" src="" alt="NeuralSpark benchmark metrics"></div>
                            <div class="eval-box eval-box-plot"><img id="eval-plot-gen" src="" alt="GenerativeVision benchmark metrics"></div>
                        </div>

                        <!-- Bottom Row: scope selector, conditional sample-size slider, and run button -->
                        <div class="control-panel-wrapper" style="margin-bottom: 0;">
                            <div class="control-panel compact-generative-panel" style="padding: 0.85rem 1.25rem !important;">
                                <div style="display: grid; grid-template-columns: auto 1fr auto; gap: 0.75rem 1.75rem; width: 100%; align-items: center;">
                                    <div class="control-row-col">
                                        <label style="font-weight: 600; font-size: 0.85rem; color: var(--arpa-h-primary-700); text-transform: uppercase;">Benchmark Scope</label>
                                        <div class="radio-group-horizontal-compact" style="display: flex; gap: 1rem; margin-top: 2px;">
                                            <label class="compact-radio-label">
                                                <input type="radio" name="eval-scope" value="single_patient" checked>
                                                <span>Single Patient</span>
                                            </label>
                                            <label class="compact-radio-label">
                                                <input type="radio" name="eval-scope" value="single_dataset">
                                                <span>Single Dataset</span>
                                            </label>
                                            <label class="compact-radio-label">
                                                <input type="radio" name="eval-scope" value="all_datasets">
                                                <span>All Datasets</span>
                                            </label>
                                        </div>
                                    </div>

                                    <div class="control-row" id="eval-num-patients-row" style="display: none; grid-template-columns: 150px 1fr 50px !important; align-items: center;">
                                        <label style="font-size: 0.85rem; font-weight: 600; margin: 0;">Patients to sample (N)</label>
                                        <input type="range" id="eval-num-patients-slider" min="1" max="100" step="1" value="10">
                                        <span id="eval-num-patients-display" class="slider-val-badge">10</span>
                                    </div>

                                    <div style="display: flex; gap: 0.5rem; align-items: center;">
                                        <button class="button button-primary" id="btn-run-evaluation" type="button" style="padding: 10px 22px; font-weight: 700; white-space: nowrap;">Run Benchmark</button>
                                        <button class="button button-stop" data-stop-button="evaluation" disabled type="button" style="padding: 10px 18px; font-weight: 700; white-space: nowrap;">Stop</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="stage-footer">
                        <div class="stage-actions">
                            <div class="action-group">
                                <!-- Standard action buttons -->
                            </div>
                        </div>
                        <div class="progress-block">
                            <div class="progress-label">Evaluation Benchmarking status</div>
                            <div class="progress-track"><div class="progress-fill" data-progress-fill="evaluation"></div></div>
                            <div class="progress-status" data-progress-status="evaluation">Idle</div>
                        </div>
                    </div>
                </article>

                <article class="stage-panel" data-stage-panel="motion-scope">
                    <div class="stage-header" style="margin-bottom: 0.5rem;">
                        <div>
                            <h2 class="stage-title" style="margin: 0;">MotionScope</h2>
                        </div>
                    </div>

                    <div class="stage-figure-shell">
                        <div class="reconstruction-layout-grid">
                            <div class="recon-view">
                                <label>Ground Truth</label>
                                <div class="stage-figure-box" id="motion-box-gt" style="margin: 0 auto;"></div>
                            </div>
                            <div class="recon-view">
                                <label>4 Rotations Per Second</label>
                                <div class="stage-figure-box" id="motion-box-rps4" style="margin: 0 auto;"></div>
                            </div>
                            <div class="recon-view">
                                <label>20 Rotations Per Second</label>
                                <div class="stage-figure-box" id="motion-box-rps20" style="margin: 0 auto;"></div>
                            </div>
                            <div class="recon-view">
                                <label>80 Rotations Per Second</label>
                                <div class="stage-figure-box" id="motion-box-rps80" style="margin: 0 auto;"></div>
                            </div>
                        </div>

                        <div class="control-panel-wrapper">
                            <div class="control-panel compact-generative-panel">
                                <div class="generative-control-grid" style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.75rem 1.5rem; width: 100%;">
                                    <div class="control-row-col">
                                        <label style="font-weight: 600; font-size: 0.85rem; color: var(--arpa-h-primary-700); text-transform: uppercase;">Acquisition Views</label>
                                        <div class="radio-group-horizontal-compact" style="display: flex; gap: 1rem; margin-top: 2px;">
                                            <label class="compact-radio-label">
                                                <input type="radio" name="motion-views" value="80" checked>
                                                <span>80 Views</span>
                                            </label>
                                            <label class="compact-radio-label">
                                                <input type="radio" name="motion-views" value="240">
                                                <span>240 Views</span>
                                            </label>
                                        </div>
                                    </div>

                                    <div class="control-row-col" style="border-left: 1px solid rgba(0, 27, 94, 0.1); padding-left: 15px;">
                                        <label style="font-weight: 600; font-size: 0.85rem; color: var(--arpa-h-primary-700); text-transform: uppercase;">Display</label>
                                        <div class="radio-group-horizontal-compact" style="display: flex; gap: 1rem; margin-top: 2px;">
                                            <label class="compact-radio-label">
                                                <input type="radio" name="motion-clock" value="no" checked>
                                                <span>Original Image</span>
                                            </label>
                                            <label class="compact-radio-label">
                                                <input type="radio" name="motion-clock" value="yes">
                                                <span>Overlay 1Hz Clock</span>
                                            </label>
                                        </div>
                                    </div>

                                    <div class="control-row-col" style="border-left: 1px solid rgba(0, 27, 94, 0.1); padding-left: 15px;">
                                        <label style="font-weight: 600; font-size: 0.85rem; color: var(--arpa-h-primary-700); text-transform: uppercase;">Reconstruction Method</label>
                                        <div class="radio-group-horizontal-compact" style="display: flex; gap: 1rem; margin-top: 2px;">
                                            <label class="compact-radio-label">
                                                <input type="radio" name="motion-method" value="flashfbp" checked>
                                                <span>FlashFBP</span>
                                            </label>
                                            <label class="compact-radio-label">
                                                <input type="radio" name="motion-method" value="neuralspark">
                                                <span>NeuralSpark</span>
                                            </label>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="stage-footer">
                        <div class="stage-actions">
                            <div class="action-group">
                            </div>
                        </div>
                        <div class="progress-block">
                            <div class="progress-label">MotionScope status</div>
                            <div class="progress-track"><div class="progress-fill" data-progress-fill="motion-scope"></div></div>
                            <div class="progress-status" data-progress-status="motion-scope">Idle</div>
                        </div>
                    </div>
                </article>
                </section>
            </section>
        </section>

        <section class="brand-banner" aria-label="Partner logos">
            <img src="/static/branding/arpa-h/logo-dark-blue-white-background.png" alt="ARPA-H logo" style="height: 60px;">
            <img src="/static/branding/ats/logo-color.png" class="logo-ats" alt="Advanced Tomography Systems logo" style="height: 90px; margin-left: 20px;">
            <img src="/static/branding/mgh/logo.png" alt="Massachusetts General Hospital logo" style="height: 60px;">
            <img src="/static/branding/hms/logo.png" class="logo-hms" alt="Harvard Medical School logo" style="height: 90px; margin-right: 20px;">
        </section>
    </main>
    <script src="/static/site/landing.js?v=20260607-fbp-redo-3"></script>
</body>
</html>
"""


PONG_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Static CT Pong Demo</title>
    <style>
        html, body {
            margin: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #222;
            font-family: sans-serif;
        }

        body {
            display: flex;
            align-items: center;
            justify-content: center;
            touch-action: none;
        }

        #game-container {
            width: 100vw;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        #game-title {
            position: fixed;
            top: 20px;
            color: white;
            font-size: 24px;
            text-align: center;
            width: 100%;
        }

        #scoreboard {
            position: fixed;
            top: 58px;
            color: white;
            font-size: 20px;
            text-align: center;
            width: 100%;
            letter-spacing: 0.08em;
        }

        canvas {
            border: 5px solid white;
            cursor: none;
            width: 90%;
            height: auto;
            max-width: 800px;
            aspect-ratio: 4 / 3;
        }
    </style>
</head>
<body>
    <div id="game-title">Static CT Pong Demo - Drag to Play</div>
    <div id="scoreboard">YOU 0 : 0 BOT</div>
    <div id="game-container">
        <canvas id="pong" width="800" height="600"></canvas>
    </div>
    <script>
        const canvas = document.getElementById("pong");
        const ctx = canvas.getContext("2d");
        const scoreboard = document.getElementById("scoreboard");
        const width = 800;
        const height = 600;
        const paddleHeight = 100;
        const paddleWidth = 20;
        const leftPaddleX = 10;
        const rightPaddleX = width - paddleWidth - 10;
        const ballRadius = 10;
        const botSpeed = 4.2;
        const speedBoost = 1.06;
        const restartDelayMs = 900;
        let playerY = 250;
        let botY = 250;
        let playerScore = 0;
        let botScore = 0;
        let roundPaused = false;
        let roundMessage = "";
        let dragging = false;
        let ball = createBall();

        function clamp(value, min, max) {
            return Math.max(min, Math.min(value, max));
        }

        function updateScoreboard() {
            scoreboard.textContent = `YOU ${playerScore} : ${botScore} BOT`;
        }

        function createBall(direction = Math.random() > 0.5 ? 1 : -1) {
            const speedX = 4 + Math.random() * 2.5;
            const speedY = (Math.random() * 4) - 2;
            return {
                x: width / 2,
                y: height / 2,
                vx: speedX * direction,
                vy: speedY === 0 ? 1.5 : speedY,
            };
        }

        function setPaddle(clientY) {
            const rect = canvas.getBoundingClientRect();
            const nextY = clientY - rect.top - paddleHeight / 2;
            playerY = clamp(nextY, 0, height - paddleHeight);
        }

        function accelerateBall(horizontalDirection, verticalOffsetScale) {
            const nextSpeedX = Math.max(4, Math.abs(ball.vx) * speedBoost);
            ball.vx = nextSpeedX * horizontalDirection;
            ball.vy = (ball.vy + verticalOffsetScale) * speedBoost;
        }

        function restartRound(scoredBy) {
            roundPaused = true;
            roundMessage = scoredBy === "player" ? "You scored" : "Bot scored";
            if (scoredBy === "player") {
                playerScore += 1;
            } else {
                botScore += 1;
            }
            updateScoreboard();
            const direction = scoredBy === "player" ? -1 : 1;
            window.setTimeout(() => {
                ball = createBall(direction);
                roundPaused = false;
                roundMessage = "";
            }, restartDelayMs);
        }

        function updateBot() {
            const targetY = ball.y - paddleHeight / 2;
            if (Math.abs(targetY - botY) <= botSpeed) {
                botY = targetY;
            } else if (targetY > botY) {
                botY += botSpeed;
            } else {
                botY -= botSpeed;
            }
            botY = clamp(botY, 0, height - paddleHeight);
        }

        function drawNet() {
            ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
            for (let y = 0; y < height; y += 32) {
                ctx.fillRect(width / 2 - 2, y, 4, 18);
            }
        }

        function drawMessage() {
            if (!roundMessage) {
                return;
            }
            ctx.fillStyle = "white";
            ctx.font = "28px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(roundMessage, width / 2, height / 2 - 24);
            ctx.textAlign = "start";
        }

        function draw() {
            if (!roundPaused) {
                ball.x += ball.vx;
                ball.y += ball.vy;
                updateBot();

                if (ball.y <= ballRadius || ball.y >= height - ballRadius) {
                    ball.vy *= -1;
                    ball.y = clamp(ball.y, ballRadius, height - ballRadius);
                }

                if (
                    ball.x - ballRadius <= leftPaddleX + paddleWidth &&
                    ball.x > leftPaddleX &&
                    ball.y >= playerY &&
                    ball.y <= playerY + paddleHeight
                ) {
                    accelerateBall(1, (ball.y - (playerY + paddleHeight / 2)) * 0.03);
                    ball.x = leftPaddleX + paddleWidth + ballRadius;
                }

                if (
                    ball.x + ballRadius >= rightPaddleX &&
                    ball.x < rightPaddleX + paddleWidth &&
                    ball.y >= botY &&
                    ball.y <= botY + paddleHeight
                ) {
                    accelerateBall(-1, (ball.y - (botY + paddleHeight / 2)) * 0.025);
                    ball.x = rightPaddleX - ballRadius;
                }

                if (ball.x < -ballRadius) {
                    restartRound("bot");
                }

                if (ball.x > width + ballRadius) {
                    restartRound("player");
                }
            }

            ctx.fillStyle = "black";
            ctx.fillRect(0, 0, width, height);
            drawNet();

            ctx.fillStyle = "white";
            ctx.fillRect(leftPaddleX, playerY, paddleWidth, paddleHeight);
            ctx.fillRect(rightPaddleX, botY, paddleWidth, paddleHeight);
            ctx.beginPath();
            ctx.arc(ball.x, ball.y, ballRadius, 0, Math.PI * 2);
            ctx.fill();
            drawMessage();

            requestAnimationFrame(draw);
        }

        canvas.addEventListener("mousedown", (event) => {
            dragging = true;
            setPaddle(event.clientY);
        });

        canvas.addEventListener("mousemove", (event) => {
            if (dragging) {
                setPaddle(event.clientY);
            }
        });

        window.addEventListener("mouseup", () => {
            dragging = false;
        });

        canvas.addEventListener("mouseleave", () => {
            dragging = false;
        });

        canvas.addEventListener("touchstart", (event) => {
            event.preventDefault();
            dragging = true;
            if (event.touches.length > 0) {
                setPaddle(event.touches[0].clientY);
            }
        }, { passive: false });

        canvas.addEventListener("touchmove", (event) => {
            event.preventDefault();
            if (dragging && event.touches.length > 0) {
                setPaddle(event.touches[0].clientY);
            }
        }, { passive: false });

        window.addEventListener("touchend", () => {
            dragging = false;
        }, { passive: false });

        window.addEventListener("touchcancel", () => {
            dragging = false;
        }, { passive: false });

        updateScoreboard();
        requestAnimationFrame(draw);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    def with_asset_versions(html: str) -> str:
        css_path = Path(__file__).parent / "static" / "site" / "landing.css"
        js_path = Path(__file__).parent / "static" / "site" / "landing.js"

        css_version = int(css_path.stat().st_mtime) if css_path.exists() else 0
        js_version = int(js_path.stat().st_mtime) if js_path.exists() else 0

        return (
            html
            .replace('/static/site/landing.css', f'/static/site/landing.css?v={css_version}')
            .replace('/static/site/landing.js', f'/static/site/landing.js?v={js_version}')
        )

    # Dynamically extract LANDING_HTML from main.py on disk to support hot-reloading 
    # the frontend UI on browser refresh without restarting the docker container,
    # thereby keeping all warmed up GPU projection matrices in memory!
    try:
        import os
        filepath = __file__
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            start_marker = '\nLANDING_HTML = """'
            start_idx = content.find(start_marker)
            if start_idx == -1:
                # Try fallback search without preceding newline
                start_marker = 'LANDING_HTML = """'
                start_idx = content.find(start_marker)
            
            if start_idx != -1:
                start_idx += len(start_marker)
                # Find the closing triple quotes
                end_idx = content.find('"""', start_idx)
                if end_idx != -1:
                    dynamic_html = content[start_idx:end_idx]
                    return HTMLResponse(content=with_asset_versions(dynamic_html))
    except Exception as e:
        print(f"DEBUG: Hot reloading LANDING_HTML failed: {e}")
    return HTMLResponse(content=with_asset_versions(LANDING_HTML))


@app.get("/pong/", response_class=HTMLResponse)
async def pong_game():
    return HTMLResponse(content=PONG_HTML)
