import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy.io import loadmat


EXTENSION_PATTERN = re.compile(r"_extension_(\d+)\.mat$")


def extension_index_from_path(path: Path) -> int:
    match = EXTENSION_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Not an extension file: {path}")
    return int(match.group(1))


def list_extension_files(weight_dir: Path) -> List[Path]:
    return sorted(
        weight_dir.glob("*_extension_*.mat"),
        key=extension_index_from_path,
    )


def build_extension_path(weight_dir: Path, n_source: int, extension_index: int) -> Path:
    return weight_dir / f"svd_{n_source}_extension_{extension_index}.mat"


def next_extension_index(weight_dir: Path) -> int:
    files = list_extension_files(weight_dir)
    if not files:
        return 1
    return extension_index_from_path(files[-1]) + 1


def _load_base_components(weight_dir: Path, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    s_path = weight_dir / "S.pt"
    v_path = weight_dir / "V.pt"
    if not s_path.exists() or not v_path.exists():
        raise FileNotFoundError(f"Missing base SVD weights in {weight_dir}")

    s = torch.load(s_path, map_location=device).to(dtype)
    v = torch.load(v_path, map_location=device).to(dtype)
    return s, v


def _load_extension_components(ext_path: Path, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    data = loadmat(ext_path)
    if "singular_values" not in data or "eigenvectors" not in data:
        raise ValueError(f"Extension file missing keys: {ext_path}")

    s_np = np.asarray(data["singular_values"], dtype=np.float32).reshape(-1)
    v_np = np.asarray(data["eigenvectors"], dtype=np.float32)
    s = torch.from_numpy(s_np).to(device=device, dtype=dtype)
    v = torch.from_numpy(v_np).to(device=device, dtype=dtype)
    return s, v


def load_combined_svd_weights(
    weight_dir: Path,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, List[Path]]:
    s_blocks = []
    v_blocks = []
    loaded_paths: List[Path] = []

    s_base, v_base = _load_base_components(weight_dir, device=device, dtype=dtype)
    s_blocks.append(s_base)
    v_blocks.append(v_base)
    loaded_paths.extend([weight_dir / "S.pt", weight_dir / "V.pt"])

    for ext_path in list_extension_files(weight_dir):
        s_ext, v_ext = _load_extension_components(ext_path, device=device, dtype=dtype)
        s_blocks.append(s_ext)
        v_blocks.append(v_ext)
        loaded_paths.append(ext_path)

    s = torch.cat(s_blocks, dim=0)
    v = torch.cat(v_blocks, dim=1)

    sort_idx = torch.argsort(s, descending=True)
    s = s[sort_idx]
    v = v[:, sort_idx]
    return s, v, loaded_paths