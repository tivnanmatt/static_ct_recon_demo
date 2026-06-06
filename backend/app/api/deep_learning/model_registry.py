from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parent / "models"
CANONICAL_MODEL_NAMESPACE = "main"
CANONICAL_MODEL_ALIASES = {CANONICAL_MODEL_NAMESPACE, "shared"}


def is_canonical_model_id(dataset_id: str) -> bool:
    return dataset_id.lower() in CANONICAL_MODEL_ALIASES


def get_model_dir(dataset_id: str, n_source: int) -> Path:
    if is_canonical_model_id(dataset_id):
        return MODEL_ROOT / CANONICAL_MODEL_NAMESPACE / f"n_source_{n_source}"
    return MODEL_ROOT / dataset_id / f"n_source_{n_source}"


def get_checkpoint_dir(dataset_id: str, n_source: int) -> Path:
    return get_model_dir(dataset_id, n_source) / "checkpoints"


def get_best_checkpoint_path(dataset_id: str, n_source: int) -> Path:
    if is_canonical_model_id(dataset_id):
        return get_model_dir(dataset_id, n_source) / "main.pt"
    return get_model_dir(dataset_id, n_source) / "best.pt"


def get_latest_checkpoint_path(dataset_id: str, n_source: int) -> Path:
    if is_canonical_model_id(dataset_id):
        return get_model_dir(dataset_id, n_source) / "main.pt"
    return get_model_dir(dataset_id, n_source) / "latest.pt"


def get_run_config_path(dataset_id: str, n_source: int) -> Path:
    return get_model_dir(dataset_id, n_source) / "config.json"