# Static CT Clinical Reconstruction Demo

A GPU-accelerated kiosk demo for **static sparse-view CT**: it walks viewers from raw clinical DICOM slices through simulated photon-noise acquisition and four competing reconstruction methods — eigen-filtered FBP, model-based iterative reconstruction, learned image restoration, and a generative diffusion sampler. Branded for **Advanced Tomography Systems (ATS)** and partners (ARPA-H, PARADIGM, MGH, HMS).

> This software is a research demonstration platform and is **not** intended for clinical diagnosis.

## TODO: Project Reorganization (reorg only — preserve all behavior)

**Guiding rule:** this cleanup is *pure reorganization*. Do **not** change any algorithm, endpoint, parameter, or output. Every file move must be paired with an update to whatever references it (import roots, the `StaticFiles` mount, HTML asset URLs, `Path(__file__)`-relative constants) so the running app behaves identically. After each step: launch the container, load a patient, and run all six stages + Motion Scope to confirm nothing regressed.

Current pain points: `main.py` is ~2400 lines because the entire landing-page and Pong HTML are inline strings; the real frontend is split between that inline HTML and `backend/app/static/`; `frontend/` is a dead Vite+React Pong app; prep scripts are scattered (one is duplicated) and there is no single entrypoint; dev/debug utilities sit inside the app package; and several absolute paths are hardcoded.

### Tasks

**Frontend / backend split**
- [ ] Extract the inline `LANDING_HTML` string from `backend/app/main.py` into `frontend/index.html`. Keep the `/` route serving it **from disk on each request** so the existing hot-reload-without-restart (warm GPU projectors) behavior is preserved — only the source of the HTML changes, not the behavior.
- [ ] Extract the inline `PONG_HTML` into `frontend/pong.html` and serve `/pong/` from it. Then **delete the legacy `frontend/` Vite+React Pong app** (`package.json`, `vite.config.js`, `src/`) — it is unused; the inline Pong is the live one.
- [ ] Move `backend/app/static/site/{landing,motion_scope}.{js,css}`, `motion_scope.html`, `workflow-icons/`, and `branding/` into the new `frontend/` tree. Update the `StaticFiles` mount path and the `/static/...` URLs in the HTML to match (keep the same `/static` and `/api` URL prefixes so `landing.js`/`motion_scope.js` keep working unchanged, same-origin).
- [ ] Move `backend/app/static/branding/tokens.css` alongside the other frontend CSS.

**Consolidate prep scripts**
- [ ] Resolve the duplicated `prep_sim_animations.py` (one in `scripts/`, one in `backend/app/`, and they differ): keep a single canonical copy in `scripts/`, delete the other, and update any importer.
- [ ] Ensure **all** prep/training/asset scripts live under `scripts/` (they already mostly do) and that each still resolves the backend package via its `sys.path` shim after any moves.
- [ ] **Add `scripts/prep_all.py`** — a single orchestrator that runs the full pipeline in dependency order (`prep_manifests` → `prep_FBP` → `prep_FBP_extend` → `prep_fourier_ramp` → `prep_DLR` / `prep_diffusion_train` → `cleanup_dlr_weights`), with flags to skip stages and to choose dataset/geometry. It should only *call* the existing scripts/functions — no new training logic.

**Move dev/debug utilities out of the app package**
- [ ] Move `backend/app/{debug_gd,debug_pgd,debug_iterative,debug_svd,svd_visual_debug,make_gifs}.py` into `scripts/debug/` (or `scripts/`) so the deployed `app` package contains only runtime code. Fix the hardcoded `/app/backend/app/static/precomputed` path in `make_gifs.py` to a `Path(__file__)`-relative path while moving it.

**De-fragilize paths (careful — keep resolution identical)**
- [ ] Replace the absolute `/workspace/static_ct_recon_demo` paths baked into model `config.json` files and `main.py` with relative resolution, verifying the resolved location is unchanged.
- [ ] Keep `models/` (checkpoints) and `weights/` (SVD bases) where the code resolves them today (`Path(__file__).parent`-relative inside the `api` package) — moving them is optional and only allowed with matching code edits.

**Housekeeping**
- [ ] Move the 6 MB `backend/app/branding_sources/` (raw `.zip` + `.pdf`) out of the runtime package into a non-deployed `assets/` (or drop from the image) — it is source material, not served content.
- [ ] Tidy root: collect stray `*.log` (`training.log`, `training_progress.log`, `diffusion_train.log`) under `logs/`; confirm `logs/` and `outputs/` are gitignored.

### Proposed final structure

```
static_ct_recon_demo/
├── README.md
├── docker/                     # Dockerfile, docker-compose.yml, setup_dicom_decoders.sh
├── backend/                    # Python ONLY (the deployed app package)
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py             # FastAPI app + routing only (no inline HTML)
│       └── api/
│           ├── datasets.py
│           ├── motion.py
│           ├── simulation/     # projector.py, svd_extensions.py, filter_plot.py
│           │   └── weights/    # svd_80/, svd_240/  (kept: code resolves it here)
│           └── deep_learning/
│               ├── model_registry.py
│               └── models/     # main/, <dataset>/, archive/  (kept: code resolves it here)
├── frontend/                   # HTML / JS / CSS ONLY
│   ├── index.html              # extracted from main.py (served at /)
│   ├── pong.html               # extracted from main.py (served at /pong/)
│   ├── motion_scope.html
│   ├── css/                    # landing.css, motion_scope.css, tokens.css
│   ├── js/                     # landing.js, motion_scope.js
│   └── assets/                 # workflow-icons/, branding/
├── scripts/                    # ALL offline prep / training / diagnostics
│   ├── prep_all.py             # NEW orchestrator (runs the pipeline in order)
│   ├── prep_manifests.py  prep_dataset.py  prep_gifs.py  prep_sim_animations.py
│   ├── prep_FBP.py  prep_FBP_extend.py  prep_fourier_ramp.py  prep_projector.py
│   ├── prep_DLR.py  prep_DLR_example.py
│   ├── prep_diffusion_train.py  prep_diffusion_test.py
│   ├── cleanup_dlr_weights.py
│   └── debug/                  # debug_*.py, svd_visual_debug.py, make_gifs.py, *_timing.py
├── assets/                     # non-served source material (branding_sources/)
├── data/                       # manifests + preview GIFs (precomputed/)   [careful: update refs]
├── logs/                       # all *.log
└── outputs/                    # generated example artifacts
```

Notes on coherence: the backend keeps serving on one origin, so the absolute `/static/...` and `/api/...` URLs in the JS/HTML stay valid with no JS edits. `models/` and `weights/` stay inside the `api` package because the code resolves them via `Path(__file__)` — they are shown nested above rather than hoisted out, to avoid any functional change. The `data/` move is marked *careful* because `precomputed/` is currently under `backend/app/static/` and is referenced both as a served path and by prep scripts; only move it with matching reference updates.

## TODO: Evaluation tab (new feature)

A new **Evaluation** stage (icon already present at `static/site/workflow-icons/evaluation_icon.png` and `static/branding/ats/evaluation_icon.png`) that benchmarks all four reconstructors against the ground-truth slice using the **current slider settings from every other stage**, times each method, and reports image-quality metrics — for a single patient or across a population.

> Assumption: the request said "the metrics should be displayed both in …" (truncated). This plan displays them **both as a results table and as histograms** of the population distribution. Adjust if a different second view was intended.

### Behavior spec

- A single **Run Evaluation** button. No per-method controls on this page — it *reads the existing controls* from the other stages (see ID map below).
- Runs, in order, on the selected slice: **1) FlashFBP → 2) FidelityMBIR → 3) NeuralSpark → 4) GenerativeVision**. For GenerativeVision with `num_samples > 1`, the evaluated reconstruction is the **posterior mean** of the final samples.
- **Times each method** (wall-clock, with `torch.cuda.synchronize()` around GPU work so the numbers are real).
- Computes against ground truth: **RMSE in raw HU** (no windowing), and **PSNR / SSIM / LPIPS** on the image **windowed to [0,1]** using the active W/L. 
- A **Single patient / All patients** selector. When **All patients** is chosen, a **"Samples" slider (1–500)** appears (number of patients to draw, clamped to availability); the table and histograms update **live** as each patient finishes, showing **mean ± std** across the population (std only shown when N > 1).

### Backend

**1. Dependencies** — add to `backend/requirements.txt`:
```
torchmetrics
lpips
```
LPIPS downloads AlexNet weights on first use; pre-cache them in the Docker build so the kiosk works offline (e.g. add a build step that imports `LearnedPerceptualImagePatchSimilarity(net_type='alex')` once).

**2. New metrics module** — `backend/app/api/evaluation/metrics.py`:
```python
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

_METRICS = {}

def _get(device):
    if device not in _METRICS:
        _METRICS[device] = {
            "psnr": PeakSignalNoiseRatio(data_range=1.0).to(device),
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
            # normalize=True => expects inputs in [0,1]
            "lpips": LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device),
        }
    return _METRICS[device]

def _window_norm(hu, win_min, win_max):
    x = (hu - win_min) / max(1e-6, (win_max - win_min))
    return np.clip(x, 0.0, 1.0).astype(np.float32)

@torch.inference_mode()
def compute_metrics(recon_hu, gt_hu, win_min, win_max, device="cuda"):
    """RMSE in raw HU; PSNR/SSIM/LPIPS on window-normalized [0,1]."""
    rmse_hu = float(np.sqrt(np.mean((recon_hu - gt_hu) ** 2)))
    r = torch.from_numpy(_window_norm(recon_hu, win_min, win_max))[None, None].to(device)
    g = torch.from_numpy(_window_norm(gt_hu,    win_min, win_max))[None, None].to(device)
    m = _get(device)
    return {
        "rmse_hu": rmse_hu,
        "psnr":  float(m["psnr"](r, g)),
        "ssim":  float(m["ssim"](r, g)),
        # LPIPS needs 3 channels
        "lpips": float(m["lpips"](r.repeat(1, 3, 1, 1), g.repeat(1, 3, 1, 1))),
    }
```

**3. New endpoint** in `backend/app/main.py` (reuses the *exact* recon calls the existing stage endpoints use — see `reconstruct_fbp`, `reconstruct_iterative`, `reconstruct_dlr`, `reconstruct_generative`). Returns one patient's results; the frontend loops for the population so histograms can update live.

```python
from api.evaluation.metrics import compute_metrics

def _ensure_sinogram(dataset_id, patient_id, slice_index, n_source, mAs):
    """Return cached sinogram or forward-project it (mirrors simulate_stream_forward)."""
    key = (dataset_id, patient_id, slice_index, n_source)
    sino = sim_manager.get_sinogram(key)
    if sino is not None:
        return sino
    ds = DATASET_REGISTRY[dataset_id]
    slices = ds.get_patient_slices(patient_id)
    mu_img = ds.get_processed_slice(slices[slice_index])
    sim_manager._warmup_projectors(n_source)
    sino = np.zeros((n_source, 2304), dtype=np.float32)
    for v in range(n_source):
        sino[v] = sim_manager.project_view(mu_img, n_source, v, m_as=mAs)
    sim_manager.set_sinogram(key, sino)
    return sino

def _timed(fn):
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return out, (time.time() - t0) * 1000.0

@app.get("/api/evaluate/{dataset_id}/{patient_id}/{slice_index}/{n_source}")
async def evaluate_all(
    dataset_id: str, patient_id: str, slice_index: int, n_source: int,
    # FlashFBP
    bandA: float = 0.0, bandB: float = 0.0, bandC: float = 0.0,
    # FidelityMBIR
    iters: int = 50, tv: float = 0.01, lr: float = 1.0, precond: bool = True,
    # NeuralSpark + GenerativeVision
    exposure_mas: float = 100.0,
    # GenerativeVision
    steps: int = 20, sigma_max: float = 0.5, sigma_min: float = 0.001,
    solver: str = "heun", temperature: float = 0.0,
    num_samples: int = 4, langevin_steps: int = 100,
    ww: Optional[float] = None, wl: Optional[float] = None,
):
    ds = DATASET_REGISTRY[dataset_id]
    slices = ds.get_patient_slices(patient_id)
    device = sim_manager.device

    mu = ds.get_processed_slice(slices[slice_index]).astype(np.float32)
    gt_hu = mu_to_hu(mu)
    target_ww = ww if ww is not None else ds.window_width
    target_wl = wl if wl is not None else ds.window_center
    win_min, win_max = target_wl - target_ww / 2, target_wl + target_ww / 2

    sino = _ensure_sinogram(dataset_id, patient_id, slice_index, n_source, mAs=exposure_mas)
    results = {}

    # 1) FlashFBP
    fbp_hu, t = _timed(lambda: sim_manager.reconstruct_step(
        n_source, sino, step="filter", band_a=bandA, band_b=bandB, band_c=bandC))
    results["flashfbp"] = {**compute_metrics(fbp_hu, gt_hu, win_min, win_max, device), "time_ms": t}

    # 2) FidelityMBIR (run generator to completion, take final image)
    def _run_mbir():
        last = None
        for upd in sim_manager.run_iterative_recon(n_source, sino, num_iters=iters,
                                                   tv_weight=tv, lr=lr, use_precond=precond):
            last = upd
        return last["image"]
    mbir_hu, t = _timed(_run_mbir)
    results["fidelitymbir"] = {**compute_metrics(mbir_hu, gt_hu, win_min, win_max, device), "time_ms": t}

    # 3) NeuralSpark (mirror reconstruct_dlr)
    assets = get_dlr_runtime_assets(n_source, mode="standard")
    builder, model = assets["builder"], assets["model"]
    target_mu = torch.from_numpy(mu).to(device=device, dtype=torch.float32)
    def _run_ns():
        comps = builder.build_measurement_components(target_mu, i0=float(exposure_mas) * 1e5)
        ch = builder.build_input_channels(comps).unsqueeze(0)
        dt = exposure_mas_to_diffusion_time(
            torch.tensor([float(exposure_mas)], device=device, dtype=ch.dtype)).to(dtype=ch.dtype)
        with torch.inference_mode():
            pred = predict_reconstruction(model, ch, dt, builder).squeeze(0).squeeze(0)
        return mu_to_hu(pred.detach().cpu().numpy())
    ns_hu, t = _timed(_run_ns)
    results["neuralspark"] = {**compute_metrics(ns_hu, gt_hu, win_min, win_max, device), "time_ms": t}

    # 4) GenerativeVision (posterior mean of final x0; mirror reconstruct_generative setup)
    def _run_gv():
        comps = builder.build_measurement_components(target_mu, i0=exposure_mas * 1e5)
        M = max(1, num_samples)
        inputs = builder.build_input_channels(comps).unsqueeze(0).repeat(M, 1, 1, 1)
        # NOTE: replicate the xt_null_start / xt_start init from reconstruct_generative
        last = None
        for upd in solve_combined_diffusion_langevin(
                model=model, builder=builder, inputs=inputs,
                num_steps_diff=steps, num_steps_lang=langevin_steps,
                sigma_max=sigma_max, sigma_min=sigma_min, rho=7.0,
                temperature=temperature, solver=solver, device=device):
            last = upd
        post_mean = last["x0"].mean(dim=0, keepdim=True)        # posterior mean over M samples
        return mu_to_hu(post_mean.detach().cpu().numpy()[0, 0])
    gv_hu, t = _timed(_run_gv)
    results["generativevision"] = {**compute_metrics(gv_hu, gt_hu, win_min, win_max, device), "time_ms": t}

    return {"dataset_id": dataset_id, "patient_id": patient_id, "slice_index": slice_index,
            "n_source": n_source, "num_samples": num_samples, "results": results}
```
> Refactor note (optional): to avoid drift, extract the four recon bodies that already live in `reconstruct_fbp/iterative/dlr/generative` into shared helpers and call them from both the stage endpoints and `evaluate_all`. The exact `xt_null_start` / `xt_start` initialization for GenerativeVision must be copied from `reconstruct_generative` so results match the UI stage.

### Frontend

**1. Sidebar + panel** — add a stage button and panel mirroring the existing markup (`backend/app/static/site/` HTML, currently inline in `main.py` until the reorg extracts it):
```html
<button class="stage-button" data-stage-button="evaluation">
  <img class="stage-button-icon" src="/static/site/workflow-icons/evaluation_icon.png" alt="Evaluation icon">
  <span>Evaluation</span>
</button>

<section class="stage-panel" data-stage-panel="evaluation">
  <div class="eval-controls">
    <label><input type="radio" name="eval-scope" value="single" checked> Single patient</label>
    <label><input type="radio" name="eval-scope" value="all"> All patients</label>
    <div id="eval-samples-row" style="display:none">
      <label>Samples: <span id="eval-samples-display">25</span></label>
      <input type="range" id="eval-samples-slider" min="1" max="500" value="25">
    </div>
    <button id="run-evaluation-btn">Run Evaluation</button>
    <div data-progress-status="evaluation">Idle</div>
  </div>
  <table id="eval-table"><thead><tr>
    <th>Method</th><th>RMSE (HU)</th><th>PSNR (dB)</th><th>SSIM</th><th>LPIPS</th><th>Time (ms)</th>
  </tr></thead><tbody></tbody></table>
  <div id="eval-histograms"><canvas id="hist-rmse_hu"></canvas><canvas id="hist-psnr"></canvas>
    <canvas id="hist-ssim"></canvas><canvas id="hist-lpips"></canvas></div>
</section>
```
Register the panel in `switchStage()` and the stage-order map in `handleStageNavigation()` (add `'generative-ai-recon' -> 'evaluation'`).

**2. Settings harvester** — read the **same DOM IDs** the existing run functions read, so Evaluation uses identical settings (verified IDs):
```js
// Mirrors the conversions in the existing run* functions in landing.js
function collectEvalSettings() {
  const numSources = parseInt(document.getElementById('sim-geometry-select').value);
  const exposureMas = parseFloat(document.getElementById('sim-exposure-select').value);
  // FlashFBP bands: slider 0..100 -> (v/100)*40 - 40
  const band = id => (parseFloat(document.getElementById(id)?.value || 50) / 100) * 40 - 40;
  // FidelityMBIR
  const iters = document.getElementById('iter-count-slider').value;
  const tv = Math.pow(10, parseFloat(document.getElementById('tv-strength-slider').value));
  const lr = Math.pow(10, parseFloat(document.getElementById('lr-slider').value));
  const precond = document.getElementById('use-precond-check').checked;
  // GenerativeVision (HU_to_atten matches landing.js usage)
  const sigmaMax = HU_to_atten(Math.pow(10, parseFloat(document.getElementById('sigma-max-slider').value)), true);
  const sigmaMin = HU_to_atten(Math.pow(10, parseFloat(document.getElementById('sigma-min-slider').value)), true);
  return {
    numSources, exposureMas,
    bandA: band('filter-band-a'), bandB: band('filter-band-b'), bandC: band('filter-band-c'),
    iters, tv, lr, precond,
    steps: document.getElementById('diffusion-steps-slider').value,
    sigmaMax, sigmaMin,
    solver: document.querySelector('input[name="diffusion-solver"]:checked')?.value || 'heun',
    temperature: parseFloat(document.getElementById('diffusion-temperature-slider').value),
    numSamples: parseInt(document.getElementById('num-samples-slider')?.value || '4'),
    langevinSteps: parseInt(document.getElementById('langevin-steps-slider')?.value || '100'),
    ww: windowWidth, wl: windowLevel,
  };
}
```

**3. Run loop** (single patient → one call; all patients → loop, updating table + histograms live):
```js
const METHODS = ['flashfbp', 'fidelitymbir', 'neuralspark', 'generativevision'];
const METRICS = ['rmse_hu', 'psnr', 'ssim', 'lpips'];

async function runEvaluation() {
  const s = collectEvalSettings();
  const scope = document.querySelector('input[name="eval-scope"]:checked').value;
  const qs = `bandA=${s.bandA}&bandB=${s.bandB}&bandC=${s.bandC}` +
             `&iters=${s.iters}&tv=${s.tv}&lr=${s.lr}&precond=${s.precond}` +
             `&exposure_mas=${s.exposureMas}&steps=${s.steps}&sigma_max=${s.sigmaMax}` +
             `&sigma_min=${s.sigmaMin}&solver=${s.solver}&temperature=${s.temperature}` +
             `&num_samples=${s.numSamples}&langevin_steps=${s.langevinSteps}&ww=${s.ww}&wl=${s.wl}`;

  // Build the population: current patient, or first N patients (mid slice each)
  let population;
  if (scope === 'single') {
    const pIdx = document.getElementById('patient-slider').value;
    population = [{ pId: currentPatientIds[pIdx], sIdx: document.getElementById('slice-slider').value }];
  } else {
    const n = Math.min(parseInt(document.getElementById('eval-samples-slider').value), currentPatientIds.length);
    population = currentPatientIds.slice(0, n).map(pId => ({ pId, sIdx: 'mid' })); // resolve 'mid' server-side or via /api/patients
  }

  const acc = {}; // method -> metric -> [values]
  METHODS.forEach(m => acc[m] = Object.fromEntries(METRICS.concat('time_ms').map(k => [k, []])));

  for (let i = 0; i < population.length; i++) {
    const { pId, sIdx } = population[i];
    const status = document.querySelector('[data-progress-status="evaluation"]');
    status.textContent = `Evaluating ${i + 1} / ${population.length} (${pId})`;
    const r = await fetch(`/api/evaluate/${selectedDataset}/${pId}/${sIdx}/${s.numSources}?${qs}`).then(x => x.json());
    METHODS.forEach(m => METRICS.concat('time_ms').forEach(k => acc[m][k].push(r.results[m][k])));
    renderEvalTable(acc, population.length);     // mean ± std (std only when length>1)
    if (population.length > 1) renderHistograms(acc); // running histograms
  }
}

const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const std  = a => Math.sqrt(mean(a.map(v => (v - mean(a)) ** 2)));

function renderEvalTable(acc, total) {
  const body = document.querySelector('#eval-table tbody');
  body.innerHTML = METHODS.map(m => {
    const cell = k => total > 1 ? `${mean(acc[m][k]).toFixed(3)} ± ${std(acc[m][k]).toFixed(3)}`
                                : `${acc[m][k][0].toFixed(3)}`;
    return `<tr><td>${m}</td><td>${cell('rmse_hu')}</td><td>${cell('psnr')}</td>` +
           `<td>${cell('ssim')}</td><td>${cell('lpips')}</td><td>${cell('time_ms')}</td></tr>`;
  }).join('');
}
// renderHistograms(acc): for each metric, draw 4 overlaid per-method distributions on
// #hist-<metric> via canvas 2D (simple binning) — no new chart dependency required.
```
Wire `#run-evaluation-btn` → `runEvaluation`, and toggle `#eval-samples-row` visibility on the scope radios. The `'mid'` slice token should be resolved to `floor(slice_count/2)` either client-side (via `/api/patients` / `/api/slices`) or accepted by the endpoint.

### Tasks
- [ ] Add `torchmetrics` + `lpips` to `requirements.txt`; pre-cache LPIPS AlexNet weights in the Docker build.
- [ ] Add `backend/app/api/evaluation/metrics.py` (`compute_metrics`, window-normalize helper).
- [ ] Add `_ensure_sinogram`, `_timed`, and the `/api/evaluate/...` endpoint to `main.py`; copy the GenerativeVision `xt` init from `reconstruct_generative` so the posterior mean matches the stage.
- [ ] (Optional) Refactor the four recon bodies into shared helpers used by both the stage endpoints and `evaluate_all`.
- [ ] Add the Evaluation sidebar button + panel; register it in `switchStage()` and the nav order map.
- [ ] Add `collectEvalSettings()`, `runEvaluation()`, `renderEvalTable()`, `renderHistograms()` to `landing.js`.
- [ ] Add CSS for the eval table/histograms (`landing.css`).
- [ ] Add `torch.cuda.synchronize()` around the timed blocks for accurate per-method timing.
- [ ] Verify: single-patient run shows a 4-row table with RMSE(HU)/PSNR/SSIM/LPIPS + timings; all-patients run shows live-updating mean ± std and histograms.

## What "static CT" means here

Instead of a rotating gantry, the system models a fixed ring of **N stationary X-ray sources**. The demo supports an **80-view** and a **240-view** geometry (a `120` slot also exists in the model tree). The point of the demo is to show how reconstruction quality holds up under sparse, static acquisition and how successively more sophisticated algorithms recover the missing information.

## Architecture

- **Single container, unified backend + frontend.** A FastAPI/Uvicorn app (`recon-web-server`) serves both the REST/SSE API and the static kiosk UI from the same process.
- **CUDA throughout.** Forward/back-projection and reconstruction run on NVIDIA GPUs (compose reserves 2 GPUs; `motion.py` uses GPU0 for simulation and GPU1 for reconstruction).
- **CT engine** comes from the sibling `ct_laboratory` PyTorch library (CUDA intersection-based projectors), imported via `PYTHONPATH=/app:/ct_laboratory`.
- **Frontend** is hand-written vanilla JS/CSS served from `backend/app/static/site/`. The `frontend/` Vite+React app is a standalone Pong idle-screen game (served at `/pong/`), not the main UI.
- **Data grid.** Slices are resampled to a common `256 x 256` grid at `1.6 mm` in-plane spacing, and Hounsfield Units are converted to linear attenuation at 60 keV.

## The kiosk pipeline (6 stages)

The landing page (`landing.js`) drives a six-stage workflow. Each reconstruction stage has a product name:

1. **Load Patient** — pick dataset (Head/Thorax/Abdomen/Pelvic), patient, and slice; preview with window/level controls and DICOM metadata.
2. **Simulate CT Data** — one-shot GPU forward projection for all views with Poisson photon-count noise; the acquisition is animated and a single sinogram is cached for every downstream method. Exposure selectable at `1 / 10 / 100 mAs` (I0 = `1e5 / 1e6 / 1e7`).
3. **FlashFBP** — eigen-filtered FBP using a sparse SVD eigen filter (combined basis up to **4096 modes** = `4 x 1024`), with a tunable 3-band Fourier ramp (Ramp / Optimal / Bone / Soft / Tissue presets).
4. **FidelityMBIR** — model-based iterative reconstruction (MAP with a Total-Variation prior, preconditioned gradient descent); streams a live loss plot and intermediate images over SSE, with a stop control.
5. **NeuralSpark** — deep-learning restoration: a 2D U-Net maps a multi-channel FBP input to a clean attenuation image.
6. **GenerativeVision** — generative diffusion reconstruction (see below), with **Sample / Animation / Mean** display modes for posterior visualization.

A separate page, **Motion Scope** (`motion_scope.html`), simulates **4D acquisition through respiratory/cardiac motion**: views are acquired one-by-one as the gantry rotates while the patient state is interpolated across motion phases, and FlashFBP or NeuralSpark reconstructs each completed rotation in a pipelined stream.

## GenerativeVision: the generative reconstruction algorithm

GenerativeVision is a **score-based / EDM diffusion sampler operating in the null space** of the static-CT system matrix:

- The measured **range space is held fixed** to the FBP solution (exact data consistency via the SVD basis); diffusion only synthesizes the **unmeasured null-space** content.
- The denoiser is the same `DLRUNet` architecture used by NeuralSpark, conditioned on the FBP channels plus the current diffusion state and spatial coordinates, with **EDM preconditioning** over a learnable σ schedule.
- Sampling **alternates Heun 2nd-order ODE steps** (deterministic denoising) with optional **Langevin dynamics** at low σ (stochastic exploration / uncertainty), and yields intermediate images at each step for real-time display.
- Drawing **multiple samples** turns it into a posterior estimator: *Animation* cycles through samples, *Mean* averages them to reduce variance, *Sample* shows one draw.
- Tunable parameters: σ range, number of diffusion steps, solver, temperature, number of samples, and Langevin steps.

The diffusion model is trained by `scripts/prep_diffusion_train.py` (EDM training on null-space residuals) and exercised standalone by `scripts/prep_diffusion_test.py`.

## Repository file index

### Top level
- `README.md` — this document.
- `docker/` — container build and compose for `recon-web-server`.
- `backend/` — FastAPI app, API package, models, static UI, branding.
- `frontend/` — standalone Vite+React Pong idle-screen game (served at `/pong/`).
- `scripts/` — offline prep/training/benchmark scripts (run ahead of time).
- `logs/`, `outputs/` — training/timing logs and generated example artifacts.
- `*.log` (`training.log`, `training_progress.log`, `diffusion_train.log`) — run logs.

### `docker/`
- `Dockerfile` — multi-stage CUDA 12.6 image (Ubuntu 22.04, Python 3.10 venv, Torch cu126) building the `backend` target.
- `docker-compose.yml` — `recon-web-server` service: host networking, 2 NVIDIA GPUs, mounts workspace + `/home/staticct/data`.
- `setup_dicom_decoders.sh` — installs DICOM pixel-data decoders/codecs into the image.
- `timing_log.txt` — captured projector/recon timing output.

### `backend/`
- `requirements.txt` — Python deps (fastapi, uvicorn, torch, numpy, imageio, …).
- `__init__.py`, `app/__init__.py` — package markers.

### `backend/app/` — application core
- `main.py` — the FastAPI app. Serves the kiosk HTML at `/`, mounts `/static`, includes the motion router, and exposes the API groups: datasets/patients/slices/preview/dicom-info, `simulate/*` (prepare, full-forward, per-view overlay, gif), and `reconstruct/*` (`filter-plot`, `fbp`, `runtime`, `dlr`, `iterative` + `iterative/stop`, `generative`, `full-sinogram`). Iterative and generative recon stream via Server-Sent Events.
- `BRANDING.md` — brand names, logo/asset locations, sponsor display order, ARPA-H palette.
- `prep_sim_animations.py` — precomputes geometry-overlay frames (source positions, active detectors, rays) for the 80/240-view acquisition animation.
- `make_gifs.py` — generates simple 8-bit slice GIFs of a dataset for preview/testing.
- `svd_visual_debug.py` — compares P-INV (null-space zeroed) vs FBP (null filled with `1/σ_min²`) reconstructions to validate spectral filtering.
- `debug_gd.py` — pure gradient-descent recon (no preconditioner) for learning-rate sensitivity checks.
- `debug_pgd.py` — preconditioned gradient descent using the sparse-eigendecomposition preconditioner.
- `debug_iterative.py` — sweeps learning rates / TV strength for MAP iterative recon on real thorax data.
- `debug_svd.py` — validates the SVD filtering pipeline against the loaded weights.

### `backend/app/api/` — API package
- `datasets.py` — loads clinical DICOM (CQ500/LIDC/LIHC/ACRIN), parses + HU-rescales, resamples to `256×256 @ 1.6mm`, converts to attenuation, and caches manifests.
- `motion.py` — Motion Scope router: serves 4D/gated patient data and streams the view-by-view motion simulation + pipelined FlashFBP/NeuralSpark reconstruction (dual-GPU).
- `simulation/projector.py` — core physics engine: wraps `ct_laboratory` projectors, caches geometry/weights/sinograms, applies SVD spectral filtering for FBP, and runs preconditioned MAP iterative recon; defines HU→attenuation and the `SVDImageFilter`.
- `simulation/svd_extensions.py` — loads and merges the combined SVD basis (`S.pt`/`V.pt` + extension `.mat` files), sorted by singular value, for multi-mode filtering.
- `simulation/filter_plot.py` — renders the FBP ramp / adjoint-cutoff filter response as a base64 PNG for the UI.
- `deep_learning/model_registry.py` — locates trained checkpoints and `config.json` by `dataset_id` / `n_source`, across the `main` (canonical) and dataset-specific namespaces.

### `backend/app/static/`
- `site/landing.js` — kiosk controller for the 6-stage workflow; manages streaming visualization (SSE), filter tuning, and stage/abort state.
- `site/landing.css` — full kiosk stylesheet (compressed layout tuned for TV/kiosk zoom).
- `site/motion_scope.html` / `motion_scope.js` / `motion_scope.css` — the standalone 4D Motion Scope page, controller, and styles.
- `branding/tokens.css` — CSS color tokens (ARPA-H/ATS palettes).
- `precomputed/` — dataset manifests (`*_manifest.json`), metadata CSVs, and preview GIFs.
- `branding/` — sponsor logos and brand assets.

### `backend/app/api/deep_learning/models/`
- `main/n_source_<N>/` — canonical shared checkpoints per geometry (`80`, `120`, `240`).
- `<dataset>/n_source_<N>/` — legacy dataset-specific runs.
- `archive/pre_main_cleanup/` — archived pre-consolidation runs.
- Each run holds `main.pt`/`latest.pt`/`best.pt`, `config.json`, `manifest.json`, `checkpoints/epoch_XXXX.pt`, and `history.jsonl`.

### `backend/app/api/simulation/weights/`
- `svd_80/`, `svd_240/` — per-geometry SVD bases: `S.pt`, `V.pt`, extension `.mat` blocks, and the learned `optimal_2d_ramp_filter.pt`.

### `scripts/` — offline prep / training / diagnostics
Asset generation:
- `prep_dataset.py` — scans DICOM datasets for valid axial slices; emits initial per-dataset GIFs + manifest.
- `prep_manifests.py` — **canonical manifest builder** (parallel): validates geometry, drops scouts/localizers, dedupes by slice position, writes `*_manifest.json` + `*_metadata.csv`. Required by everything downstream.
- `prep_gifs.py` — higher-quality anatomical preview GIFs with per-region window/level.
- `prep_sim_animations.py` *(also in `backend/app/`)* — see above.

Reconstruction bases:
- `prep_FBP.py` — computes the SVD/eigendecomposition of `AᵀA` per geometry; writes `S.pt`/`V.pt`. **Required before DLR/diffusion.**
- `prep_FBP_extend.py` — computes additional residual eigenvector blocks (the 1024-mode extensions) to grow the basis toward rank 4096.
- `prep_fourier_ramp.py` — learns the optimal 2D Fourier ramp filter from GT-vs-FBP power spectra over ~150 slices.
- `prep_FBP_debug.py` — visual sanity check of unfiltered vs eigen-filtered FBP on a real slice.
- `prep_FBP_timing.py` — profiles back-projection vs sparse-filter timing.
- `prep_projector.py` — benchmarks forward/back projection across CPU/torch/CUDA at 80/240 sources.

Deep learning:
- `prep_DLR.py` — **main U-Net training** entrypoint (see Deep Learning section); multi-dataset, multi-exposure, region-weighted loss.
- `prep_DLR_example.py` — renders example range/null/full reconstructions (GT, FBP, DLR) and logs RMSE for a checkpoint.
- `prep_diffusion_train.py` — trains the EDM diffusion model on null-space residuals across the σ schedule.
- `prep_diffusion_test.py` — single-step EDM denoiser inference demo on synthetic noisy null-space signals.
- `cleanup_dlr_weights.py` — consolidates dataset runs into the canonical `main/` checkpoints, archives old runs, writes `manifest.json`.

## Prep pipeline run order

Run these inside the container before serving (only `prep_manifests.py` and `prep_FBP.py` are strictly required; the rest are optional assets/diagnostics):

1. `prep_manifests.py` — **required first**; all downstream work depends on the manifests.
2. `prep_gifs.py` *(optional)* — preview GIFs.
3. `prep_FBP.py` — **required**; SVD bases for every spectral method.
4. `prep_FBP_extend.py` → `prep_fourier_ramp.py` *(optional)* — grow the basis / learn the ramp filter.
5. `prep_DLR.py` and/or `prep_diffusion_train.py` — train learned models (can run in parallel once FBP weights exist).
6. `cleanup_dlr_weights.py` *(optional)* — promote trained runs to `main/`.
7. `prep_*_debug.py`, `prep_*_timing.py`, `prep_DLR_example.py`, `prep_diffusion_test.py` — diagnostics, any time after their inputs exist.

Example (manifests + GIFs in the running container):

```bash
cd /home/staticct/matt/workspace/static_ct_recon_demo/docker
docker compose exec recon-web-server sh -lc \
  "cd /workspace/static_ct_recon_demo && /opt/venv/bin/python -u scripts/prep_manifests.py"
docker compose exec recon-web-server sh -lc \
  "cd /workspace/static_ct_recon_demo && /opt/venv/bin/python -u scripts/prep_gifs.py"
```

## Deep Learning reconstruction (NeuralSpark)

Training entrypoint: `scripts/prep_DLR.py`.

- Ground truth is validated attenuation-domain CT on the common `256 × 256 @ 1.6 mm` grid.
- One simulated sinogram per slice (forward projection → Poisson count noise → robust log to line integrals).
- Seven attenuation-space input channels: pinv FBP (signal-only), measurement-space null component, full FBP (signal+null), diffusion-null placeholder, diffusion-full placeholder, normalized x, normalized y.
- A 2D U-Net maps the 7-channel input to one attenuation output channel.
- Loss: region-weighted MSE (soft tissue `10x`, bone `1x`, air `1x`).
- Exposure mixed at `1 / 10 / 100 mAs` (I0 = `1e5 / 1e6 / 1e7`).
- Canonical checkpoints live under `backend/app/api/deep_learning/models/main/n_source_<N>/`; the `240` main is the completed thorax run, the `80` main is currently an untrained initialized checkpoint.

Checkpoint layout: `main.pt` (canonical) / `latest.pt`+`best.pt` (legacy runs), `config.json`, `manifest.json`, `checkpoints/epoch_XXXX.pt`, `history.jsonl`.

## Datasets

Precomputed-first manifests for **Head (CQ500)**, **Thorax (LIDC)**, **Abdomen (LIHC)**, and **Pelvic (ACRIN)**. Validation keeps one CT study + one CT series per patient, rejects coronal/sagittal views and scout/localizer series, and orders slices by physical position.

## Next steps

1. Replace `/api/simulate/overlay/...` per-view rendering with precomputed transparent animation assets for 80/240-view playback.
2. Train the canonical `80`-view NeuralSpark checkpoint (currently untrained) and refresh the `main/` tree via `cleanup_dlr_weights.py`.
3. Add an `all_data` training option to `prep_DLR.py` to mix all validated datasets into one pool.
4. Run the long-running 80-view then 240-view training sweeps with verbose train/val logging.

---

This software is a research demonstration platform and is not intended for clinical diagnosis.
