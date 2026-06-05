import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from prep_DLR import (  # noqa: E402
    CHANNEL_DESCRIPTIONS,
    ClinicalSplitSampler,
    DATASET_REGISTRY,
    DLRUNet,
    MU_WATER_60KEV,
    ROOT,
    SpectralFeatureBuilder,
    exposure_mas_to_diffusion_time,
    get_best_checkpoint_path,
    get_run_config_path,
    predict_reconstruction,
)


HU_WINDOWS = {
    "head": (40, 80),
    "thorax": (-100, 1600),
    "abdomen": (50, 350),
    "pelvic": (0, 2000),
}

CANONICAL_CHECKPOINT_DATASETS = {"main", "shared"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one DLR example from a trained checkpoint")
    parser.add_argument("--datasets", nargs="+", default=sorted(DATASET_REGISTRY.keys()))
    parser.add_argument(
        "--checkpoint-dataset",
        type=str,
        default=None,
        help="Dataset ID whose checkpoint/config should be used for inference across all requested datasets",
    )
    parser.add_argument("--n-source", type=int, default=240)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val")
    parser.add_argument("--patient-id", type=str, default=None)
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--exposure-mas", nargs="+", type=float, default=[1.0, 100.0])
    return parser.parse_args()


def load_run_config(dataset_id: str, n_source: int) -> Dict:
    config_path = get_run_config_path(dataset_id, n_source)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_exposure_tag(exposure_mas: float) -> str:
    exposure_text = f"{float(exposure_mas):g}".replace(".", "p")
    return f"exposure_{exposure_text}mAs"


def mu_to_hu(image_mu: np.ndarray) -> np.ndarray:
    return image_mu * (1000.0 / MU_WATER_60KEV) - 1000.0


def summarize_error_metrics(reference: np.ndarray, estimate: np.ndarray) -> Dict[str, float]:
    error = estimate - reference
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    ref_rmse = float(np.sqrt(np.mean(reference ** 2)))
    relative_rmse = float(rmse / max(ref_rmse, 1e-12))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "relative_rmse": relative_rmse,
    }


def summarize_comparison_metrics(panels: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Dict[str, float]]]:
    spaces = {
        "range_space": ("gt_range", "fbp_range", "dlr_range"),
        "null_space": ("gt_null", "fbp_null", "dlr_null"),
        "full_space": ("gt_full", "fbp_full", "dlr_full"),
    }
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for space_name, (gt_key, fbp_key, dlr_key) in spaces.items():
        fbp_metrics = summarize_error_metrics(panels[gt_key], panels[fbp_key])
        dlr_metrics = summarize_error_metrics(panels[gt_key], panels[dlr_key])
        summary[space_name] = {
            "fbp": fbp_metrics,
            "dlr": dlr_metrics,
            "rmse_improvement_pct": {
                "value": 100.0 * (fbp_metrics["rmse"] - dlr_metrics["rmse"]) / max(fbp_metrics["rmse"], 1e-12)
            },
        }
    return summary


def select_example_slice(
    sampler: ClinicalSplitSampler,
    split: str,
    patient_id: str | None,
    slice_index: int | None,
) -> Tuple[np.ndarray, Dict[str, int | str]]:
    patient_pool = sampler.train_patients if split == "train" else sampler.val_patients
    if not patient_pool:
        raise RuntimeError(f"No patients available in split '{split}'")

    chosen_patient = patient_id or patient_pool[0]
    if chosen_patient not in sampler.slice_cache:
        raise KeyError(f"Patient '{chosen_patient}' not found in dataset split cache")

    slices = sampler.slice_cache[chosen_patient]
    dataset_id = sampler.patient_dataset[chosen_patient]
    patient_label = chosen_patient.split("::", 1)[1]
    chosen_slice = slice_index if slice_index is not None else len(slices) // 2
    if chosen_slice < 0 or chosen_slice >= len(slices):
        raise IndexError(f"Slice index {chosen_slice} out of range for patient '{chosen_patient}'")

    mu_img = sampler.datasets[dataset_id].get_processed_slice(slices[chosen_slice]).astype(np.float32)
    return mu_img, {
        "dataset_id": dataset_id,
        "patient_id": patient_label,
        "slice_index": chosen_slice,
        "slice_count": len(slices),
        "slice_path": str(slices[chosen_slice]),
    }


def save_example_figure(
    output_dir: Path,
    dataset_id: str,
    panels: Dict[str, np.ndarray],
    metadata: Dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "example.png"
    metadata_path = output_dir / "example.json"

    window_level, window_width = HU_WINDOWS.get(dataset_id, (40, 400))
    recon_vmin = float(window_level - window_width / 2.0)
    recon_vmax = float(window_level + window_width / 2.0)

    null_scale_panels = [panels["gt_null"], panels["fbp_null"], panels["dlr_null"]]
    null_std = max(float(np.std(panel)) for panel in null_scale_panels)
    null_window = max(1e-8, 2.0 * null_std)

    row_titles = ["Ground Truth", "FBP", "DLR"]
    col_titles = ["Range Space", "Null Space", "Full Space"]
    panel_order = [
        [panels["gt_range"], panels["gt_null"], panels["gt_full"]],
        [panels["fbp_range"], panels["fbp_null"], panels["fbp_full"]],
        [panels["dlr_range"], panels["dlr_null"], panels["dlr_full"]],
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13.5), facecolor="black")
    for row_idx, row_axes in enumerate(axes):
        for col_idx, axis in enumerate(row_axes):
            panel = panel_order[row_idx][col_idx]
            if col_idx == 1:
                axis.imshow(panel, cmap="gray", vmin=-null_window, vmax=null_window, interpolation="nearest")
            else:
                axis.imshow(mu_to_hu(panel), cmap="gray", vmin=recon_vmin, vmax=recon_vmax, interpolation="nearest")
            axis.set_title(f"{row_titles[row_idx]} | {col_titles[col_idx]}", color="white", fontsize=13)
            axis.axis("off")
            axis.set_facecolor("black")

    fig.suptitle(
        f"DLR Example | dataset={metadata['dataset_id']} | n_source={metadata['n_source']} | patient={metadata['patient_id']} | slice={metadata['slice_index']} | exposure_mAs={metadata['exposure_mas']}",
        color="white",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figure_path, dpi=160, facecolor="black")
    plt.close(fig)

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return figure_path


def generate_example_for_dataset(args: argparse.Namespace, dataset_id: str, exposure_mas: float, device: torch.device) -> None:
    checkpoint_dataset_id = args.checkpoint_dataset or dataset_id
    run_config = load_run_config(checkpoint_dataset_id, args.n_source)
    sampler = ClinicalSplitSampler(
        dataset_ids=[dataset_id],
        val_fraction=float(run_config["val_fraction"]),
        seed=int(run_config["seed"]),
    )
    target_mu_np, sample_meta = select_example_slice(
        sampler=sampler,
        split=args.split,
        patient_id=args.patient_id,
        slice_index=args.slice_index,
    )

    builder = SpectralFeatureBuilder(args.n_source, device=device)
    target_mu = torch.from_numpy(target_mu_np).to(device=device, dtype=torch.float32)
    measurement_components = builder.build_measurement_components(target_mu, i0=float(exposure_mas) * 1e5)
    channels = builder.build_input_channels(measurement_components).unsqueeze(0)

    checkpoint_path = get_best_checkpoint_path(checkpoint_dataset_id, args.n_source)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DLRUNet(base_channels=int(run_config["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.inference_mode():
        diffusion_time = exposure_mas_to_diffusion_time(
            torch.tensor([float(exposure_mas)], device=device, dtype=channels.dtype)
        ).to(dtype=channels.dtype)
        final_reconstruction = predict_reconstruction(model, channels, diffusion_time, builder).squeeze(0).squeeze(0)
        target_range = builder.project_signal(target_mu)
        target_null = builder.project_null(target_mu)
        fbp_range = builder.project_signal(measurement_components["full_fbp"])
        final_range = builder.project_signal(final_reconstruction)
        final_null = builder.project_null(final_reconstruction)

    panels = {
        "gt_range": target_range.detach().cpu().numpy(),
        "gt_null": target_null.detach().cpu().numpy(),
        "gt_full": target_mu.detach().cpu().numpy(),
        "fbp_range": fbp_range.detach().cpu().numpy(),
        "fbp_null": measurement_components["measurement_null"].detach().cpu().numpy(),
        "fbp_full": measurement_components["full_fbp"].detach().cpu().numpy(),
        "dlr_range": final_range.detach().cpu().numpy(),
        "dlr_null": final_null.detach().cpu().numpy(),
        "dlr_full": final_reconstruction.detach().cpu().numpy(),
    }
    comparison_metrics = summarize_comparison_metrics(panels)

    output_dir = (
        ROOT
        / "outputs"
        / "dlr_examples"
        / f"checkpoint_{checkpoint_dataset_id}"
        / dataset_id
        / f"n_source_{args.n_source}"
        / format_exposure_tag(exposure_mas)
    )
    metadata = {
        "dataset_id": dataset_id,
        "checkpoint_dataset_id": checkpoint_dataset_id,
        "n_source": args.n_source,
        "split": args.split,
        "patient_id": sample_meta["patient_id"],
        "slice_index": sample_meta["slice_index"],
        "slice_count": sample_meta["slice_count"],
        "slice_path": sample_meta["slice_path"],
        "exposure_mas": exposure_mas,
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "display_window": {
            "window_level": HU_WINDOWS.get(dataset_id, (40, 400))[0],
            "window_width": HU_WINDOWS.get(dataset_id, (40, 400))[1],
        },
        "feature_channels": CHANNEL_DESCRIPTIONS,
        "panel_layout": {
            "rows": ["Ground Truth", "FBP", "DLR"],
            "columns": ["Range Space", "Null Space", "Full Space"],
        },
        "comparison_metrics": comparison_metrics,
        "target_null_std": float(target_null.std(unbiased=False).item()),
        "null_display_window": float(2.0 * max(target_null.std(unbiased=False).item(), 1e-8)),
    }
    figure_path = save_example_figure(
        output_dir=output_dir,
        dataset_id=dataset_id,
        panels=panels,
        metadata=metadata,
    )

    print(f"Saved DLR example figure to {figure_path}", flush=True)
    print(f"Saved DLR example metadata to {output_dir / 'example.json'}", flush=True)
    for space_name, metrics in comparison_metrics.items():
        print(
            f"[{space_name}] fbp_rmse={metrics['fbp']['rmse']:.6e} dlr_rmse={metrics['dlr']['rmse']:.6e} improvement_pct={metrics['rmse_improvement_pct']['value']:.2f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for prep_DLR_example.py")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError(f"Expected CUDA device, got {args.device}")

    invalid_datasets = [dataset_id for dataset_id in args.datasets if dataset_id not in DATASET_REGISTRY]
    if invalid_datasets:
        raise ValueError(f"Unknown datasets requested: {invalid_datasets}")

    if args.checkpoint_dataset is not None and args.checkpoint_dataset not in DATASET_REGISTRY and args.checkpoint_dataset.lower() not in CANONICAL_CHECKPOINT_DATASETS:
        raise ValueError(f"Unknown checkpoint dataset requested: {args.checkpoint_dataset}")

    for exposure_mas in args.exposure_mas:
        for dataset_id in args.datasets:
            generate_example_for_dataset(args, dataset_id, float(exposure_mas), device)


if __name__ == "__main__":
    main()