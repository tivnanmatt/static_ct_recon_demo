import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "backend" / "app"))

from api.datasets import DATASET_REGISTRY  # noqa: E402
from api.deep_learning.model_registry import (  # noqa: E402
    CANONICAL_MODEL_NAMESPACE,
    get_best_checkpoint_path,
    get_checkpoint_dir,
    get_latest_checkpoint_path,
    get_model_dir,
    get_run_config_path,
    is_canonical_model_id,
)
from api.simulation.projector import MU_WATER_60KEV  # noqa: E402
from api.simulation.svd_extensions import load_combined_svd_weights  # noqa: E402
from ct_laboratory import StaticCTProjector2D, build_uniform_static_2d_geometry  # noqa: E402


DEFAULT_EXPOSURES_MAS = (1.0, 10.0, 100.0)
WEIGHTS_DIR = ROOT / "backend" / "app" / "api" / "simulation" / "weights"
PRECOMPUTED_DIR = ROOT / "backend" / "app" / "static" / "precomputed"
CHANNEL_DESCRIPTIONS = [
    "pinv_fbp_signal_only",
    "measurement_null_component",
    "full_fbp_signal_plus_null",
    "diffusion_null_xt_placeholder",
    "diffusion_total_xt_placeholder",
    "coord_x_position",
    "coord_y_position",
]
DISPLAY_NULL_STD_MULTIPLIER = 2.0
TARGET_MODE_NULL_RESIDUAL_MSE = "null_residual_mse"


def hu_to_atten_value(hu_value: float) -> float:
    return max(0.0, (hu_value + 1000.0) * MU_WATER_60KEV / 1000.0)


AIR_UPPER_MU = hu_to_atten_value(-500.0)
SOFT_LOWER_MU = hu_to_atten_value(-200.0)
SOFT_UPPER_MU = hu_to_atten_value(300.0)
BONE_LOWER_MU = hu_to_atten_value(300.0)


@dataclass
class TrainingConfig:
    dataset_id: str
    training_dataset_ids: Tuple[str, ...]
    n_source: int
    device: str
    epochs: int
    train_steps_per_epoch: int
    val_steps_per_epoch: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    checkpoint_every: int
    exposures_mas: Tuple[float, ...]
    val_fraction: float
    seed: int
    base_channels: int
    air_weight: float
    soft_tissue_weight: float
    bone_weight: float
    warmup_fraction: float
    min_lr_scale: float


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class DLRUNet(nn.Module):
    def __init__(self, in_channels: int = len(CHANNEL_DESCRIPTIONS), out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        self.bottleneck = DownBlock(base_channels * 8, base_channels * 16)
        self.up1 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_channels * 16),
            nn.GELU(),
            nn.Linear(base_channels * 16, base_channels * 16),
        )

    def forward(self, x: torch.Tensor, diffusion_time: torch.Tensor) -> torch.Tensor:
        if diffusion_time.ndim == 1:
            diffusion_time = diffusion_time.unsqueeze(1)
        diffusion_time = diffusion_time.to(device=x.device, dtype=x.dtype)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        time_embedding = self.time_mlp(diffusion_time).unsqueeze(-1).unsqueeze(-1)
        x5 = x5 + time_embedding
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class ClinicalSplitSampler:
    def __init__(self, dataset_ids: Sequence[str], val_fraction: float, seed: int):
        self.dataset_ids = tuple(dataset_ids)
        self.datasets = {dataset_id: DATASET_REGISTRY[dataset_id] for dataset_id in self.dataset_ids}
        self.slice_cache: Dict[str, List[Path]] = {}
        self.patient_dataset: Dict[str, str] = {}

        valid_patients = []
        for dataset_id in self.dataset_ids:
            manifest_path = PRECOMPUTED_DIR / f"{dataset_id}_manifest.json"

            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Missing prepared manifest for dataset '{dataset_id}' at {manifest_path}. "
                    "Run scripts/prep_manifests.py first."
                )

            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)

            for patient_id, patient_data in manifest.items():
                slices = [Path(path_str) for path_str in patient_data.get("paths", [])]
                if not slices:
                    continue

                patient_key = f"{dataset_id}::{patient_id}"
                self.slice_cache[patient_key] = slices
                self.patient_dataset[patient_key] = dataset_id
                valid_patients.append(patient_key)

        if not valid_patients:
            raise RuntimeError(f"No CT slices found for datasets {self.dataset_ids}")

        rng = random.Random(seed)
        shuffled = list(valid_patients)
        rng.shuffle(shuffled)

        if len(shuffled) == 1:
            self.train_patients = shuffled
            self.val_patients = shuffled
        else:
            n_val = max(1, int(round(len(shuffled) * val_fraction)))
            n_val = min(n_val, len(shuffled) - 1)
            self.val_patients = shuffled[:n_val]
            self.train_patients = shuffled[n_val:]

    def sample_slice(self, split: str, rng: random.Random) -> Tuple[np.ndarray, Dict[str, int]]:
        patient_pool = self.train_patients if split == "train" else self.val_patients
        patient_key = rng.choice(patient_pool)
        dataset_id = self.patient_dataset[patient_key]
        patient_id = patient_key.split("::", 1)[1]
        slices = self.slice_cache[patient_key]
        slice_index = rng.randrange(len(slices))
        mu_img = self.datasets[dataset_id].get_processed_slice(slices[slice_index]).astype(np.float32)
        return mu_img, {
            "dataset_id": dataset_id,
            "patient_id": patient_id,
            "slice_index": slice_index,
        }

    def sample_batch_slices(self, split: str, batch_size: int, rng: random.Random) -> List[Tuple[np.ndarray, Dict[str, int]]]:
        patient_pool = list(self.train_patients if split == "train" else self.val_patients)
        if not patient_pool:
            raise RuntimeError(f"No patients available for split '{split}'")

        if batch_size <= len(patient_pool):
            selected_patients = rng.sample(patient_pool, batch_size)
        else:
            selected_patients = list(patient_pool)
            while len(selected_patients) < batch_size:
                selected_patients.append(rng.choice(patient_pool))
            rng.shuffle(selected_patients)

        batch = []
        for patient_key in selected_patients:
            dataset_id = self.patient_dataset[patient_key]
            patient_id = patient_key.split("::", 1)[1]
            slices = self.slice_cache[patient_key]
            slice_index = rng.randrange(len(slices))
            mu_img = self.datasets[dataset_id].get_processed_slice(slices[slice_index]).astype(np.float32)
            batch.append(
                (
                    mu_img,
                    {
                        "dataset_id": dataset_id,
                        "patient_id": patient_id,
                        "slice_index": slice_index,
                    },
                )
            )
        return batch


class SpectralFeatureBuilder:
    def __init__(self, n_source: int, device: torch.device, height: int = 256, width: int = 256):
        self.n_source = n_source
        self.device = device
        self.height = height
        self.width = width
        self.projector = self._build_projector()
        self.norm_map = self._build_norm_map()
        self.s, self.v = self._load_svd_weights()
        self.vt = self.v.T.contiguous()
        self.s2 = (self.s ** 2).contiguous()
        self.diag_diff = (1.0 / (self.s2 + 1e-9) - 1.0 / (self.s2.min() + 1e-9)).contiguous()
        self.null_weight = (1.0 / (self.s2.min() + 1e-9)).to(dtype=torch.float32)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing="ij",
        )
        self.coord_x = xx.to(torch.float32)
        self.coord_y = yy.to(torch.float32)
        self.fov_mask = (xx ** 2 + yy ** 2 <= 0.98).to(torch.float32)

    def _build_projector(self) -> StaticCTProjector2D:
        start_time = time.time()
        spacing = 1.6
        M = torch.tensor([[0.0, spacing], [-spacing, 0.0]], device=self.device)
        b = torch.tensor(
            [-(self.width - 1) * spacing / 2.0, (self.height - 1) * spacing / 2.0],
            device=self.device,
        )

        source_radius = 400.0
        module_radius = 366.17
        n_module = 48
        det_n_col = 48
        det_spacing = 1.0

        m_gantry = torch.eye(2, device=self.device).unsqueeze(0).repeat(self.n_source, 1, 1)
        b_gantry = torch.zeros((self.n_source, 2), device=self.device)
        active_sources = torch.eye(self.n_source, dtype=torch.bool, device=self.device)

        source_positions, module_centers, module_orientations, _ = build_uniform_static_2d_geometry(
            n_source=self.n_source,
            source_radius=source_radius,
            n_module=n_module,
            module_radius=module_radius,
            det_n_col=det_n_col,
            det_spacing=det_spacing,
            M_gantry=m_gantry,
            b_gantry=b_gantry,
            active_sources=active_sources,
        )

        source_module_mask = torch.zeros((self.n_source, n_module), dtype=torch.bool, device=self.device)
        for source_idx in range(self.n_source):
            alpha_opp = (source_idx / self.n_source + 0.5) % 1.0
            center_module = int(round(alpha_opp * n_module)) % n_module
            for offset in range(-15, 16):
                source_module_mask[source_idx, (center_module + offset) % n_module] = True

        projector = StaticCTProjector2D(
            n_row=self.height,
            n_col=self.width,
            M=M,
            b=b,
            source_positions=source_positions,
            module_centers=module_centers,
            module_orientations=module_orientations,
            det_n_col=det_n_col,
            det_spacing=det_spacing,
            source_module_mask=source_module_mask,
            M_gantry=m_gantry,
            b_gantry=b_gantry,
            active_sources=active_sources,
            backend="cuda",
            device=self.device,
        )
        return projector

    def _build_norm_map(self) -> torch.Tensor:
        start_time = time.time()
        with torch.no_grad():
            img_ones = torch.ones((self.height, self.width), device=self.device)
            norm_map = self.projector.back_project(self.projector.forward(img_ones))
        return torch.clamp(norm_map, min=1e-6)

    def _load_svd_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        start_time = time.time()
        weight_dir = WEIGHTS_DIR / f"svd_{self.n_source}"
        if not (weight_dir / "S.pt").exists() or not (weight_dir / "V.pt").exists():
            raise FileNotFoundError(f"Missing SVD weights in {weight_dir}")

        s, v, loaded_paths = load_combined_svd_weights(weight_dir, device=self.device, dtype=torch.float32)
        return s, v

    def project_signal(self, image: torch.Tensor) -> torch.Tensor:
        flat = image.reshape(-1)
        signal_flat = torch.mv(self.v, torch.mv(self.vt, flat))
        return signal_flat.reshape(self.height, self.width) * self.fov_mask

    def project_null(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.project_signal(image)) * self.fov_mask

    def project_batch_null(self, images: torch.Tensor) -> torch.Tensor:
        squeeze_channel = images.ndim == 4
        if squeeze_channel:
            images = images.squeeze(1)

        batch = images.shape[0]
        flat = images.reshape(batch, -1)
        # print(f"DEBUG: project_batch_null flat shape: {flat.shape}, v shape: {self.v.shape}")
        if flat.shape[1] == 0:
             print(f"ERROR: project_batch_null got flat.shape[1]==0! images.shape={images.shape}")
        signal_flat = torch.matmul(torch.matmul(flat, self.v), self.vt)
        null_flat = flat - signal_flat
        projected = null_flat.reshape(batch, self.height, self.width) * self.fov_mask.unsqueeze(0)

        if squeeze_channel:
            return projected.unsqueeze(1)
        return projected

    def build_measurement_components(self, target_mu: torch.Tensor, i0: float) -> Dict[str, torch.Tensor]:
        i0_tensor = torch.tensor(float(i0), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            line_integrals = self.projector.forward(target_mu)
            counts = i0_tensor * torch.exp(-line_integrals)
            counts_noisy = torch.poisson(torch.clamp(counts, min=0.0)).clamp_min(0.5)
            noisy_sinogram = -torch.log(counts_noisy / i0_tensor)
            raw_bp = self.projector.back_project(noisy_sinogram)

            x_flat = raw_bp.reshape(-1)
            vt_x = torch.mv(self.vt, x_flat)
            pinv_flat = torch.mv(self.v, vt_x / (self.s2 + 1e-9))
            filtered_flat = self.null_weight * x_flat + torch.mv(self.v, vt_x * self.diag_diff)

            pinv = pinv_flat.reshape(self.height, self.width) * self.fov_mask
            filtered = filtered_flat.reshape(self.height, self.width) * self.fov_mask
            null_component = filtered - pinv

        return {
            "pinv": pinv,
            "measurement_null": null_component,
            "full_fbp": filtered,
        }

    def build_input_channels(self, measurement_components: Dict[str, torch.Tensor]) -> torch.Tensor:
        pinv = measurement_components["pinv"]
        measurement_null = measurement_components["measurement_null"]
        full_fbp = measurement_components["full_fbp"]

        # Placeholder channels for future diffusion-state conditioning.
        # In this mode we repeat the measurement null and full FBP because the model is not diffusion-driven yet.
        diffusion_null_xt = measurement_null.clone()
        diffusion_total_xt = full_fbp.clone()

        return torch.stack(
            [
                pinv,
                measurement_null,
                full_fbp,
                diffusion_null_xt,
                diffusion_total_xt,
                self.coord_x,
                self.coord_y,
            ],
            dim=0,
        )


def predict_reconstruction(
    model: nn.Module,
    inputs: torch.Tensor,
    diffusion_time: torch.Tensor,
    builder: "SpectralFeatureBuilder",
) -> torch.Tensor:
    prediction = model(inputs, diffusion_time)
    projected_null_residual = builder.project_batch_null(prediction)
    full_fbp = inputs[:, 2:3, ...]  # full_fbp is at index 2
    return full_fbp + projected_null_residual


def predict_residual(
    model: nn.Module,
    inputs: torch.Tensor,
    diffusion_time: torch.Tensor,
    builder: "SpectralFeatureBuilder",
) -> torch.Tensor:
    prediction = model(inputs, diffusion_time)
    return builder.project_batch_null(prediction)


def exposure_mas_to_diffusion_time(exposure_mas: torch.Tensor) -> torch.Tensor:
    exposure_mas = torch.clamp(exposure_mas.to(dtype=torch.float32), min=1e-6)
    log_exposure = torch.log10(exposure_mas)
    return ((log_exposure + 2.0) / 2.0).unsqueeze(1)


def build_region_weight_map(target_mu: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    weights = torch.ones_like(target_mu)
    air_mask = target_mu < SOFT_LOWER_MU
    soft_mask = (target_mu >= SOFT_LOWER_MU) & (target_mu < SOFT_UPPER_MU)
    bone_mask = target_mu >= BONE_LOWER_MU
    weights[air_mask] = float(config.air_weight)
    weights[soft_mask] = float(config.soft_tissue_weight)
    weights[bone_mask] = float(config.bone_weight)
    return weights


def compute_losses(prediction: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sq_error = (prediction - target) ** 2
    weighted_loss = (sq_error * weight_map).sum() / torch.clamp(weight_map.sum(), min=1.0)
    mse = sq_error.mean()
    return weighted_loss, mse


def build_lr_scheduler(optimizer: AdamW, total_train_steps: int, warmup_fraction: float, min_lr_scale: float) -> LambdaLR:
    warmup_steps = max(1, int(round(total_train_steps * warmup_fraction)))
    total_train_steps = max(total_train_steps, 1)

    def lr_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return float(step_idx + 1) / float(max(warmup_steps, 1))

        if total_train_steps <= warmup_steps:
            return 1.0

        progress = float(step_idx - warmup_steps) / float(max(total_train_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_scale) + (1.0 - float(min_lr_scale)) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def maybe_load_training_state(
    dataset_id: str,
    n_source: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    reset_training: bool,
    device: torch.device,
) -> Tuple[int, float]:
    if reset_training:
        print(f"[setup] reset requested for dataset={dataset_id} n_source={n_source}; starting from scratch", flush=True)
        return 1, float("inf")

    latest_checkpoint = get_latest_checkpoint_path(dataset_id, n_source)
    if not latest_checkpoint.exists():
        return 1, float("inf")

    checkpoint = torch.load(latest_checkpoint, map_location=device)
    checkpoint_target_mode = checkpoint.get("training_target_mode", "absolute_mu")
    if checkpoint_target_mode != TARGET_MODE_NULL_RESIDUAL_MSE:
        print(
            f"[setup] checkpoint at {latest_checkpoint} uses training_target_mode={checkpoint_target_mode}; expected {TARGET_MODE_NULL_RESIDUAL_MSE}. Starting from scratch.",
            flush=True,
        )
        return 1, float("inf")

    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        print(
            f"[setup] checkpoint at {latest_checkpoint} is incompatible with the current model for dataset={dataset_id} n_source={n_source}; starting from scratch ({exc})",
            flush=True,
        )
        return 1, float("inf")

    optimizer.load_state_dict(checkpoint["optimizer_state"])

    scheduler_state = checkpoint.get("scheduler_state")
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    metrics = checkpoint.get("metrics", {})
    best_val = float(metrics.get("val_weighted_mse", float("inf")))
    print(
        f"[setup] resumed dataset={dataset_id} n_source={n_source} from epoch={start_epoch - 1} best_val={best_val:.6e}",
        flush=True,
    )
    return start_epoch, best_val


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_batch(
    sampler: ClinicalSplitSampler,
    split: str,
    builder: SpectralFeatureBuilder,
    config: TrainingConfig,
    batch_size: int,
    exposures_mas: Sequence[float],
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, float]]]:
    features = []
    targets = []
    weights = []
    diffusion_times = []
    batch_meta = []

    for sample_idx, (mu_np, sample_meta) in enumerate(sampler.sample_batch_slices(split, batch_size, rng), start=1):
        target_mu = torch.from_numpy(mu_np).to(builder.device, dtype=torch.float32)
        exposure_mas = float(rng.choice(exposures_mas))
        i0 = exposure_mas * 1e5
        measurement_components = builder.build_measurement_components(target_mu, i0=i0)
        channels = builder.build_input_channels(measurement_components)
        weight_map = build_region_weight_map(target_mu, config)

        features.append(channels)
        targets.append(target_mu.unsqueeze(0))  # target is full ground truth
        weights.append(weight_map.unsqueeze(0))
        diffusion_times.append(exposure_mas)
        batch_meta.append({
            "dataset_id": sample_meta["dataset_id"],
            "patient_id": sample_meta["patient_id"],
            "slice_index": sample_meta["slice_index"],
            "exposure_mas": exposure_mas,
            "i0": i0,
            "target_mu_std": float(target_mu.std(unbiased=False).item()),
        })

    return (
        torch.stack(features, dim=0),
        torch.stack(targets, dim=0),
        torch.stack(weights, dim=0),
        exposure_mas_to_diffusion_time(torch.tensor(diffusion_times, device=builder.device, dtype=torch.float32)),
        batch_meta,
    )


def ensure_run_layout(dataset_id: str, n_source: int) -> None:
    get_model_dir(dataset_id, n_source).mkdir(parents=True, exist_ok=True)
    get_checkpoint_dir(dataset_id, n_source).mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    dataset_id: str,
    n_source: int,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    config: TrainingConfig,
    metrics: Dict[str, float],
    checkpoint_name: str,
) -> None:
    checkpoint_path = get_model_dir(dataset_id, n_source) / checkpoint_name
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": asdict(config),
        "metrics": metrics,
        "feature_channels": CHANNEL_DESCRIPTIONS,
        "training_target_mode": TARGET_MODE_NULL_RESIDUAL_MSE,
        "region_weights": {
            "air": config.air_weight,
            "soft_tissue": config.soft_tissue_weight,
            "bone": config.bone_weight,
        },
    }
    torch.save(payload, checkpoint_path)


def append_history(dataset_id: str, n_source: int, history_entry: Dict[str, float]) -> None:
    history_path = get_model_dir(dataset_id, n_source) / "history.jsonl"
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(history_entry) + "\n")


def write_run_config(config: TrainingConfig) -> None:
    config_path = get_run_config_path(config.dataset_id, config.n_source)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **asdict(config),
                "feature_channels": CHANNEL_DESCRIPTIONS,
                "training_target_mode": TARGET_MODE_NULL_RESIDUAL_MSE,
                "model_layout": {
                    "model_dir": str(get_model_dir(config.dataset_id, config.n_source)),
                    "best_checkpoint": str(get_best_checkpoint_path(config.dataset_id, config.n_source)),
                    "latest_checkpoint": str(get_latest_checkpoint_path(config.dataset_id, config.n_source)),
                    "checkpoint_dir": str(get_checkpoint_dir(config.dataset_id, config.n_source)),
                    "history": str(get_model_dir(config.dataset_id, config.n_source) / "history.jsonl"),
                },
                "region_weights": {
                    "air": config.air_weight,
                    "soft_tissue": config.soft_tissue_weight,
                    "bone": config.bone_weight,
                },
                "training_dataset_ids": list(config.training_dataset_ids),
            },
            handle,
            indent=2,
        )


def run_phase(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: Optional[LambdaLR],
    sampler: ClinicalSplitSampler,
    builder: SpectralFeatureBuilder,
    config: TrainingConfig,
    split: str,
    epoch: int,
    rng: random.Random,
) -> Dict[str, float]:
    is_train = split == "train"
    model.train(is_train)

    steps = config.train_steps_per_epoch if is_train else config.val_steps_per_epoch
    total_weighted = 0.0
    total_mse = 0.0

    for step_idx in range(1, steps + 1):
        inputs, targets, weight_maps, diffusion_time, batch_meta = sample_batch(
            sampler=sampler,
            split=split,
            builder=builder,
            config=config,
            batch_size=config.batch_size,
            exposures_mas=config.exposures_mas,
            rng=rng,
        )

        with torch.set_grad_enabled(is_train):
            prediction_full = predict_reconstruction(model, inputs, diffusion_time, builder)
            weighted_loss_unused, mse = compute_losses(prediction_full, targets, weight_maps)
            
            # Switch to straight MSE as requested
            loss = mse

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        total_weighted += float(mse.item())  # Using MSE for both now
        total_mse += float(mse.item())
        exposures = ",".join(str(int(meta["exposure_mas"])) for meta in batch_meta)
        sample_tag = f"{batch_meta[0]['dataset_id']}:{batch_meta[0]['patient_id']}:{batch_meta[0]['slice_index']}"
        print(
            f"[{split}] dataset={config.dataset_id} n_source={config.n_source} epoch={epoch}/{config.epochs} "
            f"epoch_step={step_idx}/{steps} loss={loss.item():.6e} "
            f"mse={mse.item():.6e} lr={optimizer.param_groups[0]['lr']:.6e} exposures_mAs={exposures} sample={sample_tag}",
            flush=True,
        )

    return {
        "weighted_mse": total_weighted / max(steps, 1),
        "mse": total_mse / max(steps, 1),
        "duration_sec": 0.0,
        "last_lr": float(optimizer.param_groups[0]["lr"]),
    }


def train_single_configuration(config: TrainingConfig, reset_training: bool = False) -> None:
    print(
        f"[setup] dataset={config.dataset_id} n_source={config.n_source} device={config.device} epochs={config.epochs} train_steps={config.train_steps_per_epoch} val_steps={config.val_steps_per_epoch} batch_size={config.batch_size}",
        flush=True,
    )
    print(
        f"[setup] channels={CHANNEL_DESCRIPTIONS} region_weights=(air={config.air_weight}, soft_tissue={config.soft_tissue_weight}, bone={config.bone_weight}) exposures_mAs={list(config.exposures_mas)} warmup_fraction={config.warmup_fraction} min_lr_scale={config.min_lr_scale}",
        flush=True,
    )
    ensure_run_layout(config.dataset_id, config.n_source)
    write_run_config(config)

    device = torch.device(config.device)
    sampler = ClinicalSplitSampler(config.training_dataset_ids, val_fraction=config.val_fraction, seed=config.seed)
    builder = SpectralFeatureBuilder(config.n_source, device=device)
    model = DLRUNet(base_channels=config.base_channels).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_train_steps = config.epochs * config.train_steps_per_epoch
    scheduler = build_lr_scheduler(
        optimizer,
        total_train_steps=total_train_steps,
        warmup_fraction=config.warmup_fraction,
        min_lr_scale=config.min_lr_scale,
    )
    rng_train = random.Random(config.seed + 17)
    rng_val = random.Random(config.seed + 1017)
    start_epoch, best_val = maybe_load_training_state(
        config.dataset_id,
        config.n_source,
        model,
        optimizer,
        scheduler,
        reset_training=reset_training,
        device=device,
    )

    print(
        f"Starting DLR training: dataset={config.dataset_id}, n_source={config.n_source}, "
        f"device={config.device}, train_patients={len(sampler.train_patients)}, "
        f"val_patients={len(sampler.val_patients)}, training_datasets={list(config.training_dataset_ids)}, exposures={list(config.exposures_mas)}, rank={builder.s.numel()}",
        flush=True,
    )

    latest_checkpoint_name = "main.pt" if is_canonical_model_id(config.dataset_id) else "latest.pt"
    best_checkpoint_name = "main.pt" if is_canonical_model_id(config.dataset_id) else "best.pt"

    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = run_phase(model, optimizer, scheduler, sampler, builder, config, "train", epoch, rng_train)
        with torch.no_grad():
            val_metrics = run_phase(model, optimizer, None, sampler, builder, config, "val", epoch, rng_val)

        summary = {
            "epoch": epoch,
            "train_weighted_mse": train_metrics["weighted_mse"],
            "train_mse": train_metrics["mse"],
            "val_weighted_mse": val_metrics["weighted_mse"],
            "val_mse": val_metrics["mse"],
            "last_lr": train_metrics["last_lr"],
            "train_duration_sec": train_metrics["duration_sec"],
            "val_duration_sec": val_metrics["duration_sec"],
        }
        append_history(config.dataset_id, config.n_source, summary)
        save_checkpoint(
            config.dataset_id,
            config.n_source,
            epoch,
            model,
            optimizer,
            scheduler,
            config,
            summary,
            checkpoint_name=latest_checkpoint_name,
        )

        if epoch % config.checkpoint_every == 0:
            save_checkpoint(
                config.dataset_id,
                config.n_source,
                epoch,
                model,
                optimizer,
                scheduler,
                config,
                summary,
                checkpoint_name=f"checkpoints/epoch_{epoch:04d}.pt",
            )

        if val_metrics["weighted_mse"] < best_val:
            best_val = val_metrics["weighted_mse"]
            save_checkpoint(
                config.dataset_id,
                config.n_source,
                epoch,
                model,
                optimizer,
                scheduler,
                config,
                summary,
                checkpoint_name=best_checkpoint_name,
            )

        print(
            f"[epoch-summary] dataset={config.dataset_id} n_source={config.n_source} epoch={epoch} "
            f"train_weighted_mse={train_metrics['weighted_mse']:.6e} train_mse={train_metrics['mse']:.6e} "
            f"val_weighted_mse={val_metrics['weighted_mse']:.6e} val_mse={val_metrics['mse']:.6e} last_lr={train_metrics['last_lr']:.6e}",
            flush=True,
        )

    print(
        f"Completed DLR training: dataset={config.dataset_id}, n_source={config.n_source}, "
        f"best_checkpoint={get_best_checkpoint_path(config.dataset_id, config.n_source)}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train attenuation-domain deep learning reconstruction UNets on CUDA")
    parser.add_argument("--datasets", nargs="+", default=sorted(DATASET_REGISTRY.keys()))
    parser.add_argument("--n-sources", nargs="+", type=int, default=[80, 240])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-steps-per-epoch", type=int, default=100)
    parser.add_argument("--val-steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--exposures-mas", nargs="+", type=float, default=list(DEFAULT_EXPOSURES_MAS))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--air-weight", type=float, default=1.0)
    parser.add_argument("--soft-tissue-weight", type=float, default=10.0)
    parser.add_argument("--bone-weight", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--reset-training", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for prep_DLR.py")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError(f"prep_DLR.py is configured for CUDA execution, got {args.device}")
    if device.index is None:
        raise RuntimeError(f"prep_DLR.py requires an explicit CUDA device index, got {args.device}")
    if device.index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested device {args.device}, but only {torch.cuda.device_count()} CUDA device(s) are available")

    set_global_seed(args.seed)
    torch.cuda.set_device(device)
    print(f"Using training device: {device}", flush=True)

    invalid_datasets = [dataset_id for dataset_id in args.datasets if dataset_id not in DATASET_REGISTRY]
    if invalid_datasets:
        raise ValueError(f"Unknown datasets requested: {invalid_datasets}")

    model_dataset_ids = (CANONICAL_MODEL_NAMESPACE,) if len(args.datasets) > 1 else tuple(args.datasets)

    for dataset_id in model_dataset_ids:
        training_dataset_ids = tuple(args.datasets) if dataset_id == CANONICAL_MODEL_NAMESPACE else (dataset_id,)
        for n_source in args.n_sources:
            config = TrainingConfig(
                dataset_id=dataset_id,
                training_dataset_ids=training_dataset_ids,
                n_source=n_source,
                device=args.device,
                epochs=args.epochs,
                train_steps_per_epoch=args.train_steps_per_epoch,
                val_steps_per_epoch=args.val_steps_per_epoch,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                checkpoint_every=args.checkpoint_every,
                exposures_mas=tuple(float(value) for value in args.exposures_mas),
                val_fraction=args.val_fraction,
                seed=args.seed,
                base_channels=args.base_channels,
                air_weight=args.air_weight,
                soft_tissue_weight=args.soft_tissue_weight,
                bone_weight=args.bone_weight,
                warmup_fraction=args.warmup_fraction,
                min_lr_scale=args.min_lr_scale,
            )
            train_single_configuration(config, reset_training=args.reset_training)


if __name__ == "__main__":
    main()