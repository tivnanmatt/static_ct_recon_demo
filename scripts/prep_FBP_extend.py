import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat
from scipy.sparse.linalg import LinearOperator, eigsh

sys.path.append(str(Path(__file__).parent.parent / "backend" / "app"))

from api.simulation.svd_extensions import (  # noqa: E402
    build_extension_path,
    load_combined_svd_weights,
    next_extension_index,
)
from ct_laboratory import StaticCTProjector2D, build_uniform_static_2d_geometry  # noqa: E402


def build_projector(n_source: int, device: torch.device):
    h, w, spacing = 256, 256, 1.6

    image_M = torch.tensor([[0.0, spacing], [spacing, 0.0]], device=device)
    image_b = torch.tensor([-(w - 1) * spacing / 2.0, -(h - 1) * spacing / 2.0], device=device)

    source_radius = 400.0
    module_radius = 366.17
    n_module = 48
    det_n_col = 48
    det_spacing = 1.0

    source_module_mask = torch.zeros((n_source, n_module), dtype=torch.bool, device=device)
    for s_idx in range(n_source):
        module_center = int(round(((s_idx / n_source + 0.5) % 1.0) * n_module)) % n_module
        for offset in range(-15, 16):
            source_module_mask[s_idx, (module_center + offset) % n_module] = True

    M_gantry = torch.eye(2, device=device).unsqueeze(0).repeat(n_source, 1, 1)
    b_gantry = torch.zeros((n_source, 2), device=device)
    active_sources = torch.eye(n_source, dtype=torch.bool, device=device)

    source_positions, module_centers, module_orientations, _ = build_uniform_static_2d_geometry(
        n_source,
        source_radius,
        n_module,
        module_radius,
        det_n_col,
        det_spacing,
        M_gantry,
        b_gantry,
        active_sources,
    )

    projector = StaticCTProjector2D(
        h,
        w,
        M_gantry,
        b_gantry,
        source_positions,
        module_centers,
        module_orientations,
        det_n_col,
        det_spacing,
        source_module_mask,
        active_sources,
        image_M,
        image_b,
        backend="cuda",
        device=device,
    )

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    mask = (xx**2 + yy**2 <= 1.0).float()
    return projector, mask


def get_residual_ata_op(projector, device, existing_s, existing_v, mask=None, verbose=True):
    n_row, n_col = projector.n_row, projector.n_col
    n_pixels = n_row * n_col
    count = [0]
    start_time = [time.time()]
    s2 = existing_s**2

    def matvec(x):
        x_np = np.asarray(x)
        is_1d = x_np.ndim == 1
        x_in = x_np.reshape(-1, 1) if is_1d else x_np
        n_vecs = x_in.shape[1]

        count[0] += 1
        if verbose and (count[0] % 10 == 0 or count[0] < 5):
            print(f"  [Residual ATA] Iteration {count[0]} | n_vecs: {n_vecs} | Time: {time.time() - start_time[0]:.2f}s")

        x_torch = torch.from_numpy(x_in.T).to(device).view(n_vecs, n_row, n_col).float()

        with torch.no_grad():
            if mask is not None:
                x_torch = x_torch * mask

            y_torch = projector.forward(x_torch)
            if y_torch.dim() == 1:
                y_torch = y_torch.unsqueeze(0)

            ata_x_torch = projector.back_project(y_torch)
            if ata_x_torch.dim() == 2:
                ata_x_torch = ata_x_torch.unsqueeze(0)

            if mask is not None:
                ata_x_torch = ata_x_torch * mask

            x_flat = x_torch.view(n_vecs, n_pixels).T
            coeffs = torch.matmul(existing_v.T, x_flat)
            explained = torch.matmul(existing_v, coeffs * s2.view(-1, 1))
            residual = ata_x_torch.view(n_vecs, n_pixels).T - explained

        residual_np = residual.cpu().numpy()
        return residual_np.ravel() if is_1d else residual_np

    return LinearOperator((n_pixels, n_pixels), matvec=matvec, dtype=np.float32)


def run_extension(n_source: int, k_increment: int, device_name: str, verbose: bool = True):
    print(f"\n===== Extending SVD: {n_source} Sources, delta_k={k_increment}, device={device_name} =====")
    device = torch.device(device_name)
    weight_dir = Path(f"/app/backend/app/api/simulation/weights/svd_{n_source}")
    if not weight_dir.exists():
        raise FileNotFoundError(f"Weight directory not found: {weight_dir}")

    existing_s, existing_v, loaded_paths = load_combined_svd_weights(weight_dir, device=device, dtype=torch.float32)
    print(f"Loaded {existing_s.numel()} existing singular vectors from {len(loaded_paths) - 2} extension files")

    extension_index = next_extension_index(weight_dir)
    extension_path = build_extension_path(weight_dir, n_source, extension_index)

    projector, mask = build_projector(n_source, device=device)
    residual_op = get_residual_ata_op(
        projector,
        device=device,
        existing_s=existing_s,
        existing_v=existing_v,
        mask=mask,
        verbose=verbose,
    )

    print(f"Computing residual eigenspace for extension #{extension_index}...")
    w_vals, v_vecs = eigsh(residual_op, k=k_increment, which="LM", tol=1e-3, maxiter=5000)

    sort_idx = np.argsort(w_vals)[::-1]
    w_vals = w_vals[sort_idx]
    v_vecs = v_vecs[:, sort_idx]
    s_vals = np.sqrt(np.maximum(w_vals, 0.0)).astype(np.float32)
    v_vecs = v_vecs.astype(np.float32)

    savemat(
        extension_path,
        {
            "singular_values": s_vals,
            "eigenvectors": v_vecs,
            "extension_index": np.array([[extension_index]], dtype=np.int32),
            "n_source": np.array([[n_source]], dtype=np.int32),
            "k_increment": np.array([[k_increment]], dtype=np.int32),
            "existing_k": np.array([[existing_s.numel()]], dtype=np.int32),
        },
    )
    print(f"Saved extension #{extension_index} to {extension_path}")
    print(f"Combined rank after save: {existing_s.numel() + s_vals.size}")


def main():
    parser = argparse.ArgumentParser(description="Extend existing static-CT SVD weights by computing residual eigenvectors.")
    parser.add_argument("--n-source", type=int, nargs="+", default=[80, 240], help="Source counts to extend")
    parser.add_argument("--k-increment", type=int, default=1024, help="Number of new eigenvectors per run")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device name visible inside the runtime")
    parser.add_argument("--quiet", action="store_true", help="Reduce residual operator iteration logging")
    args = parser.parse_args()

    for n_source in args.n_source:
        run_extension(n_source, k_increment=args.k_increment, device_name=args.device, verbose=not args.quiet)


if __name__ == "__main__":
    main()