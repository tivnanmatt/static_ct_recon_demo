import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "backend" / "app"))

from api.datasets import DATASET_REGISTRY  # noqa: E402
from api.deep_learning.model_registry import (  # noqa: E402
    CANONICAL_MODEL_NAMESPACE,
    MODEL_ROOT,
    get_best_checkpoint_path,
    get_model_dir,
)
from prep_DLR import (  # noqa: E402
    CHANNEL_DESCRIPTIONS,
    DLRUNet,
    build_lr_scheduler,
)


ARCHIVE_NAMESPACE = "archive"
ARCHIVE_REASON = "pre_main_cleanup"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create canonical DLR weight files and remove stale backups")
    parser.add_argument(
        "--source-dataset-240",
        type=str,
        default="thorax",
        choices=sorted(DATASET_REGISTRY.keys()),
        help="Completed 240-source dataset run whose best checkpoint becomes the canonical main model",
    )
    parser.add_argument(
        "--source-dataset-80",
        type=str,
        default=None,
        choices=sorted(DATASET_REGISTRY.keys()),
        help="Completed 80-source dataset run whose best checkpoint becomes the canonical main model",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def canonical_model_layout(n_source: int, root_prefix: str) -> Dict[str, str]:
    relative_dir = f"/backend/app/api/deep_learning/models/{CANONICAL_MODEL_NAMESPACE}/n_source_{n_source}"
    model_dir = f"{root_prefix}{relative_dir}"
    main_checkpoint = f"{model_dir}/main.pt"
    return {
        "model_dir": model_dir,
        "best_checkpoint": main_checkpoint,
        "latest_checkpoint": main_checkpoint,
        "checkpoint_dir": f"{model_dir}/checkpoints",
        "history": f"{model_dir}/history.jsonl",
        "main_checkpoint": main_checkpoint,
    }


def infer_root_prefix(source_config: Dict[str, Any]) -> str:
    model_dir = str(source_config["model_layout"]["model_dir"])
    marker = "/backend/app/api/deep_learning/models/"
    if marker in model_dir:
        return model_dir.split(marker, 1)[0]
    return str(ROOT)


def build_canonical_config(
    template_config: Dict[str, Any],
    *,
    n_source: int,
    root_prefix: str,
    source_dataset_id: str | None,
    training_status: str,
) -> Dict[str, Any]:
    payload = dict(template_config)
    payload["dataset_id"] = CANONICAL_MODEL_NAMESPACE
    payload["n_source"] = n_source
    payload["feature_channels"] = list(CHANNEL_DESCRIPTIONS)
    payload["canonical_model_namespace"] = CANONICAL_MODEL_NAMESPACE
    payload["source_dataset_id"] = source_dataset_id
    payload["training_status"] = training_status
    payload["model_layout"] = canonical_model_layout(n_source, root_prefix)
    return payload


def create_initialized_optimizer_state(model: DLRUNet, learning_rate: float, weight_decay: float) -> AdamW:
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.zero_grad(set_to_none=True)
    dummy_loss = sum(parameter.reshape(-1)[0] * 0.0 for parameter in model.parameters() if parameter.requires_grad)
    dummy_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def create_untrained_main_checkpoint(template_config: Dict[str, Any], output_path: Path) -> None:
    model = DLRUNet(
        in_channels=len(CHANNEL_DESCRIPTIONS),
        out_channels=1,
        base_channels=int(template_config["base_channels"]),
    )
    optimizer = create_initialized_optimizer_state(
        model,
        learning_rate=float(template_config["learning_rate"]),
        weight_decay=float(template_config["weight_decay"]),
    )
    total_train_steps = int(template_config["epochs"]) * int(template_config["train_steps_per_epoch"])
    scheduler = build_lr_scheduler(
        optimizer,
        total_train_steps=total_train_steps,
        warmup_fraction=float(template_config["warmup_fraction"]),
        min_lr_scale=float(template_config["min_lr_scale"]),
    )
    payload = {
        "epoch": 0,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": dict(template_config),
        "metrics": {"status": "untrained_initialized"},
        "feature_channels": list(CHANNEL_DESCRIPTIONS),
        "region_weights": dict(template_config["region_weights"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def remove_stale_backup_files() -> list[str]:
    removed_paths = []
    for backup_path in sorted(MODEL_ROOT.rglob("*.smoketest_backup.pt")):
        backup_path.unlink()
        removed_paths.append(str(backup_path))
    return removed_paths


def archive_dataset_specific_80_runs() -> list[dict[str, str]]:
    archived_entries: list[dict[str, str]] = []
    archive_root = MODEL_ROOT / ARCHIVE_NAMESPACE / ARCHIVE_REASON
    archive_root.mkdir(parents=True, exist_ok=True)

    for dataset_id in sorted(DATASET_REGISTRY.keys()):
        source_dir = get_model_dir(dataset_id, 80)
        if not source_dir.exists():
            continue

        target_dir = archive_root / dataset_id / source_dir.name
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(source_dir), str(target_dir))
        archived_entries.append({
            "dataset_id": dataset_id,
            "from": str(source_dir),
            "to": str(target_dir),
        })

    return archived_entries


def main() -> None:
    args = parse_args()

    source_dir_240 = get_model_dir(args.source_dataset_240, 240)
    source_checkpoint_240 = get_best_checkpoint_path(args.source_dataset_240, 240)
    source_config_240 = load_json(source_dir_240 / "config.json")
    root_prefix = infer_root_prefix(source_config_240)

    if not source_checkpoint_240.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {source_checkpoint_240}")

    target_dir_240 = get_model_dir(CANONICAL_MODEL_NAMESPACE, 240)
    target_dir_80 = get_model_dir(CANONICAL_MODEL_NAMESPACE, 80)
    target_dir_240.mkdir(parents=True, exist_ok=True)
    target_dir_80.mkdir(parents=True, exist_ok=True)

    canonical_config_240 = build_canonical_config(
        source_config_240,
        n_source=240,
        root_prefix=root_prefix,
        source_dataset_id=args.source_dataset_240,
        training_status="trained",
    )
    canonical_main_240 = target_dir_240 / "main.pt"
    shutil.copy2(source_checkpoint_240, canonical_main_240)
    write_json(target_dir_240 / "config.json", canonical_config_240)

    canonical_main_80 = target_dir_80 / "main.pt"
    if args.source_dataset_80 is not None:
        source_dir_80 = get_model_dir(args.source_dataset_80, 80)
        source_checkpoint_80 = get_best_checkpoint_path(args.source_dataset_80, 80)
        source_config_80 = load_json(source_dir_80 / "config.json")
        if not source_checkpoint_80.exists():
            raise FileNotFoundError(f"Missing source checkpoint: {source_checkpoint_80}")

        canonical_config_80 = build_canonical_config(
            source_config_80,
            n_source=80,
            root_prefix=root_prefix,
            source_dataset_id=args.source_dataset_80,
            training_status="trained",
        )
        shutil.copy2(source_checkpoint_80, canonical_main_80)
    else:
        canonical_config_80 = build_canonical_config(
            source_config_240,
            n_source=80,
            root_prefix=root_prefix,
            source_dataset_id=None,
            training_status="untrained_initialized",
        )
        create_untrained_main_checkpoint(canonical_config_80, canonical_main_80)

    write_json(target_dir_80 / "config.json", canonical_config_80)

    archived_80_runs = archive_dataset_specific_80_runs()
    removed_backups = remove_stale_backup_files()
    manifest = {
        "canonical_model_namespace": CANONICAL_MODEL_NAMESPACE,
        "main_models": {
            "240": {
                "checkpoint": str(canonical_main_240),
                "config": str(target_dir_240 / "config.json"),
                "source_dataset_id": args.source_dataset_240,
                "status": "trained",
            },
            "80": {
                "checkpoint": str(canonical_main_80),
                "config": str(target_dir_80 / "config.json"),
                "source_dataset_id": args.source_dataset_80,
                "status": canonical_config_80["training_status"],
            },
        },
        "archived_dataset_specific_80_runs": archived_80_runs,
        "removed_stale_backups": removed_backups,
    }
    write_json(MODEL_ROOT / CANONICAL_MODEL_NAMESPACE / "manifest.json", manifest)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()