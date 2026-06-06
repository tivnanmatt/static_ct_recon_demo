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


def get_dlr_runtime_assets(n_source: int, mode: str = "standard") -> Dict[str, object]:
    cache_key = (n_source, mode)
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


def mu_to_hu(mu_img: np.ndarray) -> np.ndarray:
    return np.maximum(mu_img, 0.0) * (1000.0 / MU_WATER_60KEV) - 1000.0


def encode_hu_png(hu_img: np.ndarray, win_min: float, win_max: float, figsize=(4, 4), dpi: int = 100) -> bytes:
    from matplotlib.figure import Figure

    buf = io.BytesIO()
    fig = Figure(figsize=figsize, dpi=dpi, facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(hu_img, cmap='gray', vmin=win_min, vmax=win_max, interpolation='nearest')
    ax.axis('off')
    fig.savefig(buf, format='png', facecolor='black', dpi=dpi)
    return buf.getvalue()

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
    ax = fig.add_subplot(111)
    ax.imshow(full_sino, cmap='gray', vmin=0.0, vmax=8.0, aspect='auto')
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
    sino_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return {
        "geometry_base": geo_base_b64,
        "sinogram_full": sino_b64,
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

@app.get("/api/reconstruct/fbp/{dataset_id}/{patient_id}/{slice_index}/{n_source}")
async def reconstruct_fbp(dataset_id: str, patient_id: str, slice_index: int, n_source: int, step: str = 'filter', ww: Optional[float] = None, wl: Optional[float] = None):
    sino_key = (dataset_id, patient_id, slice_index, n_source)
    full_sino = sim_manager.get_sinogram(sino_key)
    
    if full_sino is None:
        return {"error": "Sinogram not found. Run simulation first."}
    
    # Run Reconstruction step (laminogram or filter)
    import time
    start = time.time()
    recon_hu = sim_manager.reconstruct_step(n_source, full_sino, step=step)
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
async def reconstruct_runtime(dataset_id: str, patient_id: str, slice_index: int, n_source: int, initial_step: str = 'unfiltered', final_step: str = 'filter', ww: Optional[float] = None, wl: Optional[float] = None):
    sino_key = (dataset_id, patient_id, slice_index, n_source)
    full_sino = sim_manager.get_sinogram(sino_key)
    if full_sino is None:
        return {"error": "Sinogram not found. Run simulation first."}

    import time

    initial_start = time.time()
    initial_hu = sim_manager.reconstruct_step(n_source, full_sino, step=initial_step)
    initial_time_ms = (time.time() - initial_start) * 1000.0

    final_start = time.time()
    final_hu = sim_manager.reconstruct_step(n_source, full_sino, step=final_step)
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

    target_mu_np = ds_obj.get_processed_slice(slices[slice_index]).astype(np.float32)
    target_mu = torch.from_numpy(target_mu_np).to(device=device, dtype=torch.float32)

    import time

    fbp_start = time.time()
    fbp_hu = sim_manager.reconstruct_step(n_source, full_sino, step='filter')
    initial_time_ms = (time.time() - fbp_start) * 1000.0

    init_start = time.time()
    measurement_components = builder.build_measurement_components(target_mu, i0=float(exposure_mas) * 1e5)
    # pinv = measurement_components["pinv"]
    # initial_time_ms = (time.time() - init_start) * 1000.0

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
    num_samples: int = 4, langevin_steps: int = 100
):
    sino_key = (dataset_id, patient_id, slice_index, n_source)
    full_sino = sim_manager.get_sinogram(sino_key)
    if full_sino is None:
        return {"error": "Sinogram not found"}

    # Use same cancel mechanism as iterative
    cancel_event = register_iterative_cancel(job_id) if job_id else None

    async def event_generator():
        import json
        print(f"DEBUG [Generative]: Starting event_generator for {dataset_id}/{patient_id}/{slice_index} (Job: {job_id}, num_samples={num_samples}, langevin_steps={langevin_steps})")
        try:
            device = sim_manager.device
            assets = get_dlr_runtime_assets(n_source, mode="generative")
            model = assets["model"]
            builder = assets["builder"]
            print(f"DEBUG [Generative]: Assets loaded. Device: {device}")

            # 1. Get GT for reference
            ds = DATASET_REGISTRY[dataset_id]
            slice_path = ds.get_patient_slices(patient_id)[slice_index]
            mu_np = ds.get_processed_slice(slice_path).astype(np.float32)
            gt_mu = torch.from_numpy(mu_np).to(device).unsqueeze(0).unsqueeze(0)

            # 2. Build Measurement Components
            comps = builder.build_measurement_components(gt_mu[0,0], i0=exposure_mas * 1e5)
            print("DEBUG [Generative]: Measurement components built.")

            # Start from measured null-space FBP plus null-space noise. The range-space pinv is fixed.
            pinv = comps["pinv"].unsqueeze(0).unsqueeze(0)
            measurement_null = comps["measurement_null"].unsqueeze(0).unsqueeze(0)
            full_fbp = comps["full_fbp"].unsqueeze(0).unsqueeze(0)

            M = max(1, num_samples)

            pinv_batched = pinv.repeat(M, 1, 1, 1)
            measurement_null_batched = measurement_null.repeat(M, 1, 1, 1)
            full_fbp_batched = full_fbp.repeat(M, 1, 1, 1)

            noise_null = builder.project_batch_null(torch.randn_like(pinv_batched)) * sigma_max
            xt_null_start = builder.project_batch_null(measurement_null_batched + noise_null)
            xt_start = pinv_batched + xt_null_start

            # BUILD INPUT STACK FOR SOLVER
            # Channels: [pinv, meas_null, full_fbp, null_xt, total_xt, coord_x, coord_y]
            inputs = builder.build_input_channels(comps).unsqueeze(0).repeat(M, 1, 1, 1)
            inputs[:, 3:4, ...] = xt_null_start
            inputs[:, 4:5, ...] = xt_start

            # W/L for UI
            target_ww = ww if ww is not None else ds.window_width
            target_wl = wl if wl is not None else ds.window_center
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
                else:
                    null_hat_list = None
                    xt_list = None
                    x0_list = None
                    mean_null_hat_b64 = None
                    mean_xt_b64 = None
                    mean_x0_b64 = None
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
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(full_sino, cmap='gray', vmin=0.0, vmax=8.0, aspect='auto')
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches=0, pad_inches=0)
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
    <link rel="stylesheet" href="/static/site/landing.css">
</head>
<body>
    <main class="kiosk-shell">
        <header class="page-header" style="display: flex; justify-content: space-between; align-items: center; padding: 1.5rem 2.5rem 1rem;">
            <h1 class="page-title" style="margin: 0;">CT Reconstruction Laboratory</h1>
            <img src="/static/branding/ats/logo-color.png" alt="Advanced Tomography Systems logo" style="height: 60px; object-fit: contain; flex: 0 0 auto;">
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
                    <button class="stage-button" data-stage-button="eigen-fbp-recon" disabled type="button" aria-label="EigenFBP">
                        <span class="stage-button-art">
                            <img class="stage-button-icon" src="/static/site/workflow-icons/eigenfbp_icon.png" alt="EigenFBP icon">
                        </span>
                    </button>
                    <button class="stage-button" data-stage-button="model-based-iterative-recon" disabled type="button" aria-label="HighFidelityMBIR">
                        <span class="stage-button-art">
                            <img class="stage-button-icon" src="/static/site/workflow-icons/mbir_icon.png" alt="HighFidelityMBIR icon">
                        </span>
                    </button>
                    <button class="stage-button" data-stage-button="deep-learning-recon" disabled type="button" aria-label="NeuralSpeed">
                        <span class="stage-button-art">
                            <img class="stage-button-icon" src="/static/site/workflow-icons/neuralspeed_icon.png" alt="NeuralSpeed icon">
                        </span>
                    </button>
                    <button class="stage-button" data-stage-button="generative-ai-recon" disabled type="button" aria-label="GenerativeVision">
                        <span class="stage-button-art">
                            <img class="stage-button-icon" src="/static/site/workflow-icons/generativevision_icon.png" alt="GenerativeVision icon">
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
                        <div class="selection-controls" id="patient-selection-controls" style="display: none; flex-direction: column; gap: 1rem; margin-top: 1rem;">
                            <!-- Top section: Patient/Slice Sliders next to DICOM metadata -->
                            <div style="display: grid; grid-template-columns: 1.8fr 1fr; gap: 1.5rem; align-items: stretch; width: 100%;">
                                <div class="controls-column" style="display: flex; flex-direction: column; gap: 1rem; justify-content: center;">
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
                                </div>
                                <div class="metadata-panel" style="max-height: 140px; min-height: 100px; padding: 0.8rem; margin: 0;">
                                    <h3 style="margin: 0 0 0.4rem 0; font-size: 0.8rem;">DICOM METADATA</h3>
                                    <div class="metadata-scroll-container" style="padding: 0.4rem; height: calc(100% - 1.2rem); overflow-y: auto;">
                                        <div id="dicom-metadata-text" class="metadata-dump" style="font-size: 0.85rem; line-height: 1.4;"></div>
                                    </div>
                                </div>
                            </div>
                            <!-- Bottom section: Window levels stacked vertically and matching widths -->
                            <div style="display: flex; flex-direction: column; gap: 1rem; width: 100%; border-top: 1px solid rgba(0, 0, 0, 0.05); padding-top: 1rem;">
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
                        <div class="simulation-layout">
                            <div class="sim-controls-panel">
                                <div class="control-group">
                                    <label>System Geometry</label>
                                    <select id="sim-geometry-select">
                                        <option value="80">80-View Static CT</option>
                                        <option value="240">240-View Static CT</option>
                                    </select>
                                </div>
                                <div class="control-group">
                                    <label>Exposure (mA)</label>
                                    <select id="sim-exposure-select">
                                        <option value="1">1 mA (I0=1e5)</option>
                                        <option value="100" selected>100 mA (I0=1e7)</option>
                                    </select>
                                </div>
                                
                                <button class="button button-primary" data-run-button="simulate-ct-data" disabled type="button">Simulate CT Data</button>
                                <button class="button button-stop" data-stop-button="simulate-ct-data" disabled type="button">Stop</button>
                            </div>
                            <div class="simulation-columns">
                                <div class="sim-display" id="sim-display-geometry">
                                    <label>System Geometry & Rays</label>
                                    <div class="canvas-container">
                                        <canvas id="geometry-canvas"></canvas>
                                    </div>
                                </div>
                                <div class="sim-display" id="sim-display-projection">
                                    <label>Sinogram Preview</label>
                                    <div class="canvas-container">
                                        <canvas id="sinogram-canvas"></canvas>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="stage-footer">
                        <div class="stage-actions">
                            <div class="action-group">
                                <button class="button button-ghost" data-next-stage="simulate-ct-data" disabled type="button">Next: EigenFBP</button>
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
                            <h2 class="stage-title" style="margin: 0;">EigenFBP</h2>
                        </div>
                    </div>
                    <div class="stage-figure-shell">
                        <div class="reconstruction-layout-grid">
                            <div class="recon-view">
                                <label>Ground Truth</label>
                                <div class="stage-figure-box" id="recon-box-gt"></div>
                                <div class="timing-display">Source Data</div>
                            </div>
                            <div class="recon-view">
                                <label>Simulated Sinogram</label>
                                <div class="stage-figure-box" id="recon-box-sino"></div>
                                <div class="timing-display" id="timing-sino">--</div>
                            </div>
                            <div class="recon-view">
                                <label>Laminogram (BP)</label>
                                <div class="stage-figure-box" id="recon-box-unfiltered" aria-label="Unfiltered BP"></div>
                                <div class="timing-display" id="timing-unfiltered">-- ms</div>
                            </div>
                            <div class="recon-view">
                                <label>EigenFBP</label>
                                <div class="stage-figure-box" id="recon-box-filtered" aria-label="Filtered Recon"></div>
                                <div class="timing-display" id="timing-filtered">4096-mode sparse eigen filter</div>
                            </div>
                        </div>
                    </div>
                    <div class="stage-footer">
                        <div class="stage-actions">
                            <div class="action-group">
                                <button class="button button-primary" data-run-button="eigen-fbp-recon" disabled type="button">Run EigenFBP</button>
                                <button class="button button-stop" data-stop-button="eigen-fbp-recon" disabled type="button">Stop</button>
                                <button class="button button-ghost" data-next-stage="eigen-fbp-recon" disabled type="button">Next: HighFidelityMBIR</button>
                            </div>
                            <div class="stage-hint">The app and training prep share the combined SVD basis: base 1024 modes plus 3 extension blocks for a total rank of 4096.</div>
                        </div>
                        <div class="progress-block">
                            <div class="progress-label">EigenFBP status</div>
                            <div class="progress-track"><div class="progress-fill" data-progress-fill="eigen-fbp-recon"></div></div>
                            <div class="progress-status" data-progress-status="eigen-fbp-recon">Idle</div>
                        </div>
                    </div>
                </article>

                <article class="stage-panel" data-stage-panel="model-based-iterative-recon">
                    <div class="stage-header" style="margin-bottom: 0.5rem;">
                        <div>
                            <h2 class="stage-title" style="margin: 0;">HighFidelityMBIR</h2>
                        </div>
                    </div>
                    
                    <div class="stage-figure-shell">
                        <div class="iterative-layout-grid">
                            <div class="recon-view">
                                <label>Ground Truth</label>
                                <div class="stage-figure-box" id="iter-box-gt"></div>
                            </div>
                            <div class="recon-view">
                                <label>Simulated Sinogram</label>
                                <div class="stage-figure-box" id="iter-box-sino"></div>
                            </div>
                            <div class="recon-view">
                                <label>EigenFBP Initialization</label>
                                <div class="stage-figure-box" id="iter-box-init"></div>
                            </div>
                            <div class="recon-view">
                                <label>HighFidelityMBIR</label>
                                <div class="stage-figure-box" id="iter-box-live"></div>
                            </div>
                            <div class="recon-view recon-view-loss recon-view-tall">
                                <label>Loss Function</label>
                                <div class="stage-figure-box stage-figure-box-tall" id="iter-box-loss"></div>
                            </div>
                        </div>
                    </div>

                    <div class="control-panel-wrapper">
                        <!-- Control Panel (Full Width) -->
                        <div class="control-panel">
                            <div class="control-row">
                                <label>Iterations</label>
                                <input type="range" id="iter-count-slider" min="10" max="500" step="10" value="100">
                                <span id="iter-count-display">100</span>
                            </div>
                            <div class="control-row">
                                <label>TV Strength</label>
                                <input type="range" id="tv-strength-slider" min="-6" max="10" step="0.5" value="-2.5">
                                <span id="tv-strength-display">0.003</span>
                            </div>
                            <div class="control-row">
                                <label>Step Size / LR</label>
                                <input type="range" id="lr-slider" min="-8" max="1" step="0.25" value="-1">
                                <span id="lr-display">0.1</span>
                            </div>
                            <div class="control-row">
                                <label>Eigen Precond.</label>
                                <div class="checkbox-container">
                                    <input type="checkbox" id="use-precond-check" checked>
                                </div>
                                <span></span>
                            </div>
                        </div>
                    </div>
                    <div class="stage-footer">
                        <div class="stage-actions">
                            <div class="action-group">
                                <button class="button button-primary" data-run-button="model-based-iterative-recon" disabled type="button">Run HighFidelityMBIR</button>
                                <button class="button button-stop" data-stop-button="model-based-iterative-recon" disabled type="button">Stop</button>
                                <button class="button button-ghost" data-next-stage="model-based-iterative-recon" disabled type="button">Next: NeuralSpeed</button>
                            </div>
                        </div>
                        <div class="progress-block">
                            <div class="progress-label">HighFidelityMBIR progress</div>
                            <div class="progress-track"><div class="progress-fill" data-progress-fill="model-based-iterative-recon"></div></div>
                            <div class="progress-status" data-progress-status="model-based-iterative-recon">Idle</div>
                        </div>
                    </div>
                </article>

                <article class="stage-panel" data-stage-panel="deep-learning-recon">
                    <div class="stage-header" style="margin-bottom: 0.5rem;">
                        <div>
                            <h2 class="stage-title" style="margin: 0;">NeuralSpeed</h2>
                        </div>
                    </div>
                    <div class="stage-figure-shell">
                        <div class="reconstruction-layout-grid">
                            <div class="recon-view">
                                <label>Ground Truth</label>
                                <div class="stage-figure-box" id="dlr-box-gt"></div>
                                <div class="timing-display">Source Data</div>
                            </div>
                            <div class="recon-view">
                                <label>Simulated Sinogram</label>
                                <div class="stage-figure-box" id="dlr-box-sino"></div>
                                <div class="timing-display" id="timing-dlr-sino">--</div>
                            </div>
                            <div class="recon-view">
                                <label>Full FBP Initialization</label>
                                <div class="stage-figure-box" id="dlr-box-init"></div>
                                <div class="timing-display" id="timing-dlr-init">-- ms</div>
                            </div>
                            <div class="recon-view">
                                <label>NeuralSpeed Final Recon</label>
                                <div class="stage-figure-box" id="dlr-box-final"></div>
                                <div class="timing-display" id="timing-dlr-final">-- ms</div>
                            </div>
                        </div>
                    </div>
                    <div class="stage-footer">
                        <div class="stage-actions">
                            <div class="action-group">
                                <button class="button button-primary" data-run-button="deep-learning-recon" disabled type="button">Run NeuralSpeed</button>
                                <button class="button button-stop" data-stop-button="deep-learning-recon" disabled type="button">Stop</button>
                                <button class="button button-ghost" data-next-stage="deep-learning-recon" disabled type="button">Next: GenerativeVision</button>
                            </div>
                            <div class="stage-hint">The shared backprojection work is reused. The displayed final timing reflects only the restoration step after the Full FBP initialization is ready.</div>
                        </div>
                        <div class="progress-block">
                            <div class="progress-label">NeuralSpeed status</div>
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
                                <div class="stage-figure-box" id="gen-box-gt"></div>
                            </div>
                            <div class="recon-view">
                                <label>Initial FBP (full)</label>
                                <div class="stage-figure-box" id="gen-box-full-fbp"></div>
                            </div>
                            <div class="recon-view">
                                <label>Generative Process</label>
                                <div class="stage-figure-box" id="gen-box-xt"></div>
                            </div>
                            <div class="recon-view">
                                <label>Generative Reconstruction</label>
                                <div class="stage-figure-box" id="gen-box-x0"></div>
                            </div>
                        </div>
                    </div>

                    <div class="control-panel-wrapper">
                        <!-- Control Panel (Full Width) -->
                        <div class="control-panel">
                            <!-- Slider Section inside Control Panel -->
                            <div class="control-rows-container">
                                <div class="control-row">
                                    <label>Diffusion Steps</label>
                                    <input type="range" id="diffusion-steps-slider" min="5" max="50" step="5" value="50">
                                    <span id="diffusion-steps-display" class="slider-val-badge">50</span>
                                </div>
                                <div class="control-row">
                                    <label>Solver Mode</label>
                                    <select id="diffusion-solver-select">
                                        <option value="heun">Heun (2nd Order)</option>
                                        <option value="euler">Euler (1st Order)</option>
                                    </select>
                                    <span></span>
                                </div>
                                <div class="control-row">
                                    <label>Max Null Noise (HU)</label>
                                    <input type="range" id="sigma-max-slider" min="0" max="3" step="0.05" value="3">
                                    <span id="sigma-max-display" class="slider-val-badge">1000</span>
                                </div>
                                <div class="control-row">
                                    <label>Min Null Noise (HU)</label>
                                    <input type="range" id="sigma-min-slider" min="0" max="3" step="0.05" value="0">
                                    <span id="sigma-min-display" class="slider-val-badge">1</span>
                                </div>
                                <div class="control-row">
                                    <label>Langevin Temp</label>
                                    <input type="range" id="diffusion-temperature-slider" min="0" max="5" step="0.05" value="0">
                                    <span id="diffusion-temperature-display" class="slider-val-badge">0.00</span>
                                </div>
                                <div class="control-row">
                                    <label>Langevin Steps</label>
                                    <input type="range" id="langevin-steps-slider" min="0" max="500" step="10" value="100">
                                    <span id="langevin-steps-display" class="slider-val-badge">100</span>
                                </div>
                                <div class="control-row">
                                    <label>Num Samples</label>
                                    <input type="range" id="num-samples-slider" min="1" max="16" step="1" value="4">
                                    <span id="num-samples-display" class="slider-val-badge">4</span>
                                </div>
                            </div>

                            <!-- Display Mode Selector Segmented Buttons inside Control Panel -->
                            <div class="display-mode-container">
                                <label>Image Display Mode (Ensemble options when Num Samples > 1)</label>
                                <div class="display-mode-buttons-row">
                                    <button type="button" class="display-mode-btn active" id="btn-mode-sample" data-mode="sample">A) Generative Sample</button>
                                    <button type="button" class="display-mode-btn" id="btn-mode-animation" data-mode="animation">B) Multi-Sample Animation</button>
                                    <button type="button" class="display-mode-btn" id="btn-mode-mean" data-mode="mean">C) Generative Mean</button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="stage-footer">
                        <div class="stage-actions">
                            <div class="action-group">
                                <button class="button button-primary" data-run-button="generative-ai-recon" disabled type="button">Run GenerativeVision</button>
                                <button class="button button-stop" data-stop-button="generative-ai-recon" disabled type="button">Stop</button>
                                <a class="button button-ghost" href="/pong/">Open Pong Demo</a>
                            </div>
                            <div class="stage-hint">Reverse diffusion proceeds only in the null space. The measured range space is preserved at every step.</div>
                        </div>
                        <div class="progress-block">
                            <div class="progress-label">GenerativeVision status</div>
                            <div class="progress-track"><div class="progress-fill" data-progress-fill="generative-ai-recon"></div></div>
                            <div class="progress-status" data-progress-status="generative-ai-recon">Idle</div>
                        </div>
                    </div>
                </article>
                </section>
            </section>
        </section>

        <section class="brand-banner" aria-label="Partner logos">
            <img src="/static/branding/arpa-h/logo-dark-blue-white-background.png" alt="ARPA-H logo">
            <img src="/static/branding/ats/logo-color.png" class="logo-ats" alt="Advanced Tomography Systems logo">
            <img src="/static/branding/mgh/logo.png" alt="Massachusetts General Hospital logo">
            <img src="/static/branding/hms/logo.png" class="logo-hms" alt="Harvard Medical School logo">
        </section>
    </main>
    <script src="/static/site/landing.js"></script>
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
                    return HTMLResponse(content=dynamic_html)
    except Exception as e:
        print(f"DEBUG: Hot reloading LANDING_HTML failed: {e}")
    return HTMLResponse(content=LANDING_HTML)


@app.get("/pong/", response_class=HTMLResponse)
async def pong_game():
    return HTMLResponse(content=PONG_HTML)
