# Static CT Clinical Reconstruction Demo

A GPU-accelerated kiosk demo for static sparse-view CT simulation, eigen-filtered FBP, iterative reconstruction, and learned image restoration.

## Current State

The application runs as a unified FastAPI/Uvicorn backend with a specialized frontend served from the same container. The current stack includes:

- CUDA-enabled projection and reconstruction inside the `recon-web-server` container.
- Precomputed-first dataset manifests for Head (CQ500), Thorax (LIDC), Abdomen (LIHC), and Pelvic (ACRIN).
- Manifest validation that keeps one CT study and one CT series per patient selection, rejects coronal/sagittal views and LIHC scout/localizer series, and orders slices by physical slice position.
- Slice preprocessing on a common `256 x 256` grid using DICOM pixel spacing to map into a `1.6 mm x 1.6 mm` in-plane reconstruction grid.
- Shared sparse eigen filtering across the app and prep scripts using the combined SVD basis in `backend/app/api/simulation/weights`, currently `3 x 1024 = 3072` modes when the two extension files are present.

## Data Preparation

Refresh manifests and preview GIFs inside the container:

```bash
# 1. Regenerate validated manifests
cd /home/staticct/matt/workspace/static_ct_recon_demo/docker
docker compose exec recon-web-server sh -lc "cd /workspace/static_ct_recon_demo && /opt/venv/bin/python -u /workspace/static_ct_recon_demo/scripts/prep_manifests.py"

# 2. Regenerate dataset preview GIFs
docker compose exec recon-web-server sh -lc "cd /workspace/static_ct_recon_demo && /opt/venv/bin/python -u /workspace/static_ct_recon_demo/scripts/prep_gifs.py"
```

## Deep Learning Reconstruction

The deep learning training entrypoint is `scripts/prep_DLR.py`.

Current training design:

- Ground truth starts from validated attenuation-domain CT slices on the common `256 x 256 x 1.6 mm` grid.
- A single simulated sinogram is produced by forward projection, nonlinear Poisson count noise, and robust logarithm conversion back to line integrals.
- Seven attenuation-space input channels are constructed:
	1. pinv FBP signal-only channel
	2. measurement-space null component
	3. full FBP signal-plus-null channel
	4. diffusion null placeholder
	5. diffusion full-image placeholder
	6. normalized x coordinate
	7. normalized y coordinate
- A 2D U-Net maps the 7-channel input to one attenuation output channel.
- The loss is weighted MSE with region-aware emphasis: soft tissue `10x`, bone `1x`, air `1x`.
- Exposure levels are mixed during training at `1 mAs`, `10 mAs`, and `100 mAs`, corresponding to `I0 = 1e5`, `1e6`, and `1e7`.
- Canonical shared main checkpoints are stored under `backend/app/api/deep_learning/models/main/n_source_<N>/`.
- The current canonical `240` main checkpoint is copied from the completed `thorax` run.
- The current canonical `80` main checkpoint is an untrained initialized checkpoint with the same network architecture, ready for future training.
- Legacy dataset-specific `240` runs remain under `backend/app/api/deep_learning/models/<dataset>/n_source_240/`.
- Older dataset-specific `80` runs are archived under `backend/app/api/deep_learning/models/archive/pre_main_cleanup/` so the active tree reflects that canonical `80` training has not started.

Checkpoint layout:

- `main.pt`: canonical shared checkpoint for the selected system geometry
- `config.json`: run configuration, feature-channel definitions, and model layout paths
- `manifest.json`: summary of the canonical `80` and `240` main artifacts
- Legacy dataset runs still use `latest.pt` and `best.pt`
- `checkpoints/epoch_XXXX.pt`: periodic saved checkpoints
- `history.jsonl`: epoch-by-epoch train and validation metrics

## Frontend Plan For DLR Image Restoration

The Stage 5 deep learning panel is intended to explain the learned FBP-restoration pipeline before runtime inference is exposed to users.

Planned visual story:

- Show the four attenuation-domain FBP channels as stacked input panels with visible separators to communicate channel concatenation.
- Feed the stacked channels into a U-Net graphic or placeholder network block.
- Show the one-channel deep learning reconstruction output.
- Show the simulation ground truth image beside the output.
- Connect output and ground truth with an explicit loss annotation such as `Loss Function Measures error w.r.t. simulation ground truth`.

If a U-Net asset is provided later, the placeholder network block can be replaced without changing the surrounding layout.

## Frontend Plan For Simulation Playback

The simulation stage should remain a one-shot physical simulation:

- run one forward projection and photon-noise simulation for all views
- cache that sinogram for Eigen FBP, iterative recon, and deep learning recon
- animate acquisition in the frontend using precomputed transparent PNG frames or transparent GIF playback instead of per-view matplotlib rendering
- keep the sinogram reveal synchronized with the staged animation and progress/status messaging

This preserves a single source of truth for the simulated data while making the UI substantially lighter.

## Next Steps

1. Replace the current `/api/simulate/overlay/...` per-view rendering path with precomputed transparent animation assets for `80`-view and `240`-view playback.
2. Expose a runtime loader for the saved deep learning checkpoints so Stage 5 can run learned restoration on the already-simulated sinogram.
3. Add an `all_data` training option to `prep_DLR.py` so all validated datasets can be mixed into one training pool when requested.
4. Launch the long-running `80`-view training sweep first, then the `240`-view sweep, with verbose logging of train/validation losses and timings.

---

This software is a research demonstration platform and is not intended for clinical diagnosis.
