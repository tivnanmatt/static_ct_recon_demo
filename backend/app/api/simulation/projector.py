import concurrent.futures
import contextlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import base64
import os
import time
import hashlib
from pathlib import Path
from api.simulation.svd_extensions import load_combined_svd_weights
from ct_laboratory import (
    UniformStaticCTProjector2D, StaticCTProjector2D, 
    build_uniform_static_2d_geometry, standard_image_transform_2d,
    LinearGaussianLogLikelihood, MaximumAPosterioriReconstructor, TotalVariancePrior2D
)

# Monkey-patch MaximumAPosterioriReconstructor.map_step to fix the RuntimeError:
# "Trying to backward through the graph a second time" when preconditioner is not None.
def patched_map_step(self, debug=False):
    self.optimizer.zero_grad()
    if self.preconditioner is not None:
        self.volume_pred = self.preconditioner(self.volume_pred_preconditioned)
    
    log_likelihood = self.log_likelihood_fn(self.volume_pred)
    log_prior = self.log_prior_fn(self.volume_pred)
    
    # Calculate Likelihood Grad Norm
    if log_likelihood.requires_grad:
        # Crucial Fix: If preconditioner is used, log_likelihood and log_prior share the preconditioner graph.
        # Calling backward on log_likelihood frees the graph unless retain_graph=True,
        # which subsequently causes backward on log_prior to crash.
        (-1.0 * log_likelihood).backward(retain_graph=(self.preconditioner is not None and log_prior.requires_grad))
        if self.preconditioner is not None:
            grad_lik = self.volume_pred_preconditioned.grad.detach().clone()
        else:
            grad_lik = self.volume_pred.grad.detach().clone()
    else:
        if self.preconditioner is not None:
            grad_lik = torch.zeros_like(self.volume_pred_preconditioned)
        else:
            grad_lik = torch.zeros_like(self.volume_pred)

    # Calculate Prior Grad Norm (Base Score Function)
    self.optimizer.zero_grad()
    if self.preconditioner is not None:
        self.volume_pred = self.preconditioner(self.volume_pred_preconditioned)

    if log_prior.requires_grad:
        (-1.0 * log_prior).backward()
        if self.preconditioner is not None:
            grad_prior = self.volume_pred_preconditioned.grad.detach().clone()
        else:
            grad_prior = self.volume_pred.grad.detach().clone()
    else:
        if self.preconditioner is not None:
            grad_prior = torch.zeros_like(self.volume_pred_preconditioned)
        else:
            grad_prior = torch.zeros_like(self.volume_pred)

    # Restore total grad for optimizer
    if self.preconditioner is not None:
        if self.volume_pred_preconditioned.grad is None:
            self.volume_pred_preconditioned.grad = grad_lik + grad_prior
        else:
            self.volume_pred_preconditioned.grad.data = grad_lik + grad_prior
    else:
        if self.volume_pred.grad is None:
            self.volume_pred.grad = grad_lik + grad_prior
        else:
            self.volume_pred.grad.data = grad_lik + grad_prior
    
    grad_norm_lik = grad_lik.norm(2).item()
    grad_norm_prior = grad_prior.norm(2).item()

    log_posterior = log_likelihood + log_prior
    
    if self.preconditioner is not None:
        gn_total = self.volume_pred_preconditioned.grad.detach().norm(2).item()
    else:
        gn_total = self.volume_pred.grad.detach().norm(2).item()
            
    self.optimizer.step()
    if self.scheduler is not None:
        self.scheduler.step()
    
    return log_likelihood.item(), log_prior.item(), log_posterior.item(), grad_norm_lik, grad_norm_prior, gn_total

MaximumAPosterioriReconstructor.map_step = patched_map_step

from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

# Physics constants
MU_WATER_60KEV = 0.0183 # mm^-1
MU_AIR_60KEV = 0.0000
PRECOMPUTED_WEIGHTS_DIR = Path(__file__).parent / "weights"

def HU_to_atten(hu):
    """Converts Hounsfield Units to linear attenuation (mm^-1) @ 60keV."""
    return np.maximum(0, (hu + 1000.0) * MU_WATER_60KEV / 1000.0)

# --- SVD Image Filter for Reconstruction ---
class SVDImageFilter(torch.nn.Module):
    def __init__(self, s, v):
        super().__init__()
        self.register_buffer('s', s) # singular values of A
        self.register_buffer('v', v) # [N_pixels, K]
        
        # Spectral Filtering (FBP-style)
        # We apply H = V diag(1/s^2) V^T + (1/S_min^2) (I - VV^T)
        # This keeps the signal space exact while filling the null space with 1/S_min^2
        s2 = s**2
        beta = s2.min()
        
        self.register_buffer('null_weight', 1.0 / (beta + 1e-9))
        self.register_buffer('diag_diff', 1.0 / (s2 + 1e-9) - 1.0 / (beta + 1e-9))
        self.register_buffer('fov_radius_sq', torch.tensor(0.98, device=s.device, dtype=s.dtype))
        
    def forward(self, laminogram):
        # laminogram is A^T y
        h, w = laminogram.shape

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=self.s.device, dtype=laminogram.dtype),
            torch.linspace(-1, 1, w, device=self.s.device, dtype=laminogram.dtype),
            indexing="ij",
        )
        fov_mask = (xx**2 + yy**2 <= self.fov_radius_sq).to(laminogram.dtype)

        x_flat = laminogram.view(-1, 1)
        coeffs = torch.matmul(self.v.T, x_flat)
        coeffs_scaled = coeffs * self.diag_diff.view(-1, 1)
        recon_part = torch.matmul(self.v, coeffs_scaled)

        out_flat = self.null_weight * x_flat + recon_part
        out = out_flat.view(h, w)

        return out * fov_mask

class SparseEigenPreconditioner(torch.nn.Module):
    def __init__(self, s, v, eps=1e-12):
        """
        Preconditioner P for MAP optimization.
        If optimizing x = P z, we want P^2 ≈ (A^TA + beta*I)^-1.
        So P = (A^TA + beta*I)^-1/2
        """
        super().__init__()
        self.register_buffer('v', v.detach().clone())
        self.register_buffer('vt', v.T.detach().clone().contiguous())
        
        s_min = s.min()
        beta = s_min**2
        
        # Eigenvalues of P are 1/s in signal space and 1/s_min in null space
        # P = V diag(1/s) V^T + (1/s_min) (I - VV^T)
        # P = (1/s_min) I + V [1/s - 1/s_min] V^T
        
        self.register_buffer('null_weight', 1.0 / s_min)
        self.register_buffer('diag_diff', 1.0 / s - 1.0 / s_min)

        # Inverse parameters: P^-1 (z = P^-1 x preserves orientation)
        # P^-1 = V diag(s) V^T + s_min (I - VV^T)
        self.register_buffer('null_weight_inv', s_min)
        self.register_buffer('diag_diff_inv', s - s_min)

    def forward(self, z):
        # Apply P: [V @ D @ V^T + null_weight * I] z
        shape = z.shape
        z_flat = z.reshape(-1)
        
        vt_z = torch.mv(self.vt, z_flat)
        vt_z_scaled = vt_z * self.diag_diff
        v_vt_z_scaled = torch.mv(self.v, vt_z_scaled)
        
        out = v_vt_z_scaled + self.null_weight * z_flat
        return out.reshape(shape)

    def inverse(self, x):
        # Apply P^-1: [V @ D_inv @ V^T + null_weight_inv * I] x
        shape = x.shape
        x_flat = x.reshape(-1)
        
        vt_x = torch.mv(self.vt, x_flat)
        vt_x_scaled = vt_x * self.diag_diff_inv
        v_vt_x_scaled = torch.mv(self.v, vt_x_scaled)
        
        out = v_vt_x_scaled + self.null_weight_inv * x_flat
        return out.reshape(shape)
        
        out = v_vt_x_scaled + self.s_min_val * x_flat
        return out.reshape(shape)

class SimulationManager:
    def __init__(self, device=None):
        # Cache for lists of projectors: (n_source) -> list of 1-view projectors
        self.view_projectors = {}
        # Cache for all-in-one projectors: (n_source) -> 1 big projector
        self.full_projectors = {}
        
        # Cache for geometry data: (n_source) -> geometry_dict
        self.geometry_cache = {}
        # Cache for sinograms: (dataset_id, patient_id, slice_index, n_source) -> sinogram_data
        self.sinogram_cache = {}
        # Cache for shared reconstruction intermediates: (n_source, sinogram_digest) -> tensors
        self.reconstruction_cache = {}
        # Cache for normalization maps: (n_source) -> ATA1 torch.Tensor
        self.normalization_maps = {}
        # Cache for SVD filters: (n_source) -> SVDImageFilter instance
        self.svd_filters = {}
        
        # Parallel Execution
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Path for weight caching
        self.cache_dir = Path(__file__).parent / "weight_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.eigen_filter = None
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Persistent figure for faster plotting
        self.fig_geo = None
        self.ax_geo = None
        self.fig_sino = None
        self.ax_sino = None
        self.ax_prof = None
        
        # Timings
        self.timings = {} # (n_source) -> dict

    def _device_context(self):
        if self.device.type != "cuda":
            return contextlib.nullcontext()
        return torch.cuda.device(self.device)

    def _load_eigen_filter(self):
        # Deprecated: We now use SVDImageFilter initialized on-demand or cached.
        pass

    def get_sinogram(self, key):
        return self.sinogram_cache.get(key)
    
    def set_sinogram(self, key, data):
        self.sinogram_cache[key] = data

    def _build_reconstruction_cache_key(self, n_source, sinogram_np):
        sinogram_array = np.ascontiguousarray(sinogram_np, dtype=np.float32)
        digest = hashlib.sha1(sinogram_array.view(np.uint8)).hexdigest()
        return (n_source, digest)

    def _prepare_reconstruction_state(self, n_source, sinogram_np):
        with self._device_context():
            self._warmup_projectors(n_source)
            full_proj = self.full_projectors[n_source]
            geom = self.geometry_cache[n_source]
            cache_key = self._build_reconstruction_cache_key(n_source, sinogram_np)
            cached = self.reconstruction_cache.get(cache_key)
            if cached is not None:
                return cached

            sino_active = []
            for i in range(n_source):
                mask = geom["source_module_mask"][i]
                view_sino = sinogram_np[i]
                for m_idx, active in enumerate(mask):
                    if active:
                        sino_active.append(view_sino[m_idx * 48 : (m_idx + 1) * 48])
            sino_active = np.concatenate(sino_active)
            sino_torch = torch.from_numpy(sino_active).float().to(self.device)
            raw_bp = full_proj.back_project(sino_torch)
            cached = {
                "raw_bp": raw_bp,
            }
            self.reconstruction_cache[cache_key] = cached
            return cached
    
    def _warmup_projectors(self, n_source, height=256, width=256):
        """Initializes and caches both view-by-view and full projectors."""
        if n_source in self.view_projectors and n_source in self.full_projectors:
            return

        with self._device_context():
            print(f"DEBUG: Starting Projector Warmup for {n_source} sources...")
            device = self.device
            
            # 1. Setup Image Transform (row,col) -> (x,y)
            # REQUIRED FIX: Row=Y, Col=X for standard patient orientation
            # We flip mapping to vertical (Y) to corrected up-down orientation
            spacing = 1.6
            M = torch.tensor([[0.0, spacing], [-spacing, 0.0]], device=device)
            # Center the grid
            b = torch.tensor([
                -(width - 1) * spacing / 2.0, 
                (height - 1) * spacing / 2.0
            ], device=device)
            
            # 2. Geometry Parameters
            source_radius = 400.0
            module_radius = 366.17
            n_module = 48
            det_n_col = 48
            det_spacing = 1.0
            
            # Lib utilities for positions
            dummy_M = torch.eye(2, device=device).unsqueeze(0).repeat(n_source, 1, 1)
            dummy_b = torch.zeros((n_source, 2), device=device)
            dummy_active = torch.eye(n_source, dtype=torch.bool, device=device)
            
            (source_positions, module_centers, module_orientations, _) = build_uniform_static_2d_geometry(
                n_source=n_source, source_radius=source_radius, n_module=n_module, module_radius=module_radius,
                det_n_col=det_n_col, det_spacing=det_spacing, M_gantry=dummy_M, b_gantry=dummy_b, active_sources=dummy_active
            )

            source_module_mask = torch.zeros((n_source, n_module), dtype=torch.bool, device=device)
            for s in range(n_source):
                alpha_opp = (s / n_source + 0.5) % 1.0
                m_center = int(round(alpha_opp * n_module)) % n_module
                # Clinical: 31 modules (center + 15 each side)
                for offset in range(-15, 16):
                    source_module_mask[s, (m_center + offset) % n_module] = True

            self.geometry_cache[n_source] = {
                "source_positions": source_positions.cpu().numpy(),
                "module_centers": module_centers.cpu().numpy(),
                "module_orientations": module_orientations.cpu().numpy(),
                "det_n_col": det_n_col,
                "det_spacing": det_spacing,
                "module_radius": module_radius,
                "source_module_mask": source_module_mask.cpu().numpy()
            }

            # 3. Create VIEW-BY-VIEW Projectors (Cached weights)
            print(f"DEBUG: Building {n_source} View Projectors...")
            view_projs = []
            for i in range(n_source):
                mask = torch.zeros((1, n_source), dtype=torch.bool, device=device)
                mask[0, i] = True
                
                tvals_path = self.cache_dir / f"tvals_{height}_{width}_{n_source}_view_{i}.pt"
                
                p = StaticCTProjector2D(
                    n_row=height, n_col=width, M=M, b=b,
                    source_positions=source_positions, module_centers=module_centers, module_orientations=module_orientations,
                    det_n_col=det_n_col, det_spacing=det_spacing, source_module_mask=source_module_mask,
                    active_sources=mask, M_gantry=torch.eye(2, device=device).unsqueeze(0), b_gantry=torch.zeros((1, 2), device=device),
                    backend="cuda", device=device
                )
                
                if tvals_path.exists():
                    p.tvals = torch.load(tvals_path).to(device)
                else:
                    torch.save(p.tvals.cpu(), tvals_path)
                view_projs.append(p)
            self.view_projectors[n_source] = view_projs


            # 4. Create FULL ALL-AT-ONCE Projector (Cached weights)
            print(f"DEBUG: Building Full Static Projector for all {n_source} views...")
            tvals_full_path = self.cache_dir / f"tvals_{height}_{width}_{n_source}_uniform_full.pt"
            
            # Use StaticCTProjector2D directly with the Eye mask for all frames
            # This matches the structure of UniformStaticCTProjector2D but allows explicit mask
            full_p = StaticCTProjector2D(
                n_row=height, n_col=width, M=M, b=b,
                source_positions=source_positions, module_centers=module_centers, module_orientations=module_orientations,
                det_n_col=det_n_col, det_spacing=det_spacing, source_module_mask=source_module_mask,
                M_gantry=dummy_M, b_gantry=dummy_b,
                active_sources=torch.eye(n_source, dtype=torch.bool, device=device),
                backend="cuda", device=device
            )
            if tvals_full_path.exists():
                full_p.tvals = torch.load(tvals_full_path).to(device)
            else:
                torch.save(full_p.tvals.cpu(), tvals_full_path)
            self.full_projectors[n_source] = full_p

            # 5. Compute Normalization Map (A^T A 1)
            print(f"DEBUG: Computing Normalization map for {n_source} sources...")
            with torch.no_grad():
                img_ones = torch.ones((height, width), device=device)
                # A^T (A * 1)
                norm_map = full_p.back_project(full_p.forward(img_ones))
                # Avoid division by zero
                self.normalization_maps[n_source] = torch.clamp(norm_map, min=1e-6)

            # 6. Timing Test (Standard backend)
            print(f"DEBUG: Running timing tests...")
            dummy_img = torch.zeros((height, width), device=device)
            
            # Time Full
            start = time.time()
            _ = full_p.forward(dummy_img)
            full_time = (time.time() - start) * 1000.0 # ms
            
            # Time One View
            start = time.time()
            _ = view_projs[0].forward(dummy_img)
            view_one_time = (time.time() - start) * 1000.0 # ms
            
            self.timings[n_source] = {
                "full_time_ms": full_time,
                "one_view_time_ms": view_one_time,
                "total_view_time_est_ms": view_one_time * n_source
            }
            print(f"DEBUG: Full (Uniform): {full_time:.2f}ms | One View: {view_one_time:.2f}ms")

    def get_view_projector(self, n_source, view_idx):
        self._warmup_projectors(n_source)
        return self.view_projectors[n_source][view_idx]

    def project_full(self, image_np, n_source, m_as=300.0):
        """One-shot forward projection for all views."""
        with self._device_context():
            self._warmup_projectors(n_source)
            full_proj = self.full_projectors[n_source]
            
            device = self.device
            img_torch = torch.from_numpy(image_np.astype(np.float32)).to(device)
            
            with torch.no_grad():
                line_integrals = full_proj.forward(img_torch).cpu().numpy()
        
        # Poisson Noise: mAs = 100 -> 1e7 Photons (Clinical)
        i0 = float(m_as) * 1e5
            
        counts = i0 * np.exp(-line_integrals)
        counts_noisy = np.random.poisson(counts).astype(np.float32)
        counts_noisy = np.maximum(counts_noisy, 0.5)
        data = -np.log(counts_noisy / i0)
        
        # Reshape to (n_source, n_det_active)
        # We need to map this back to the "full" sinogram shape (n_source, 2304)
        n_det_active = data.size // n_source
        data = data.reshape(n_source, n_det_active)
        
        geom = self.geometry_cache[n_source]
        full_sinogram = np.zeros((n_source, 2304), dtype=np.float32)
        
        for i in range(n_source):
            mask = geom["source_module_mask"][i]
            active_modules = np.where(mask)[0]
            offset = 0
            for m in active_modules:
                start_idx = m * 48
                full_sinogram[i, start_idx : start_idx + 48] = data[i, offset : offset + 48]
                offset += 48
        return full_sinogram

    def project_view(self, image_np, n_source, view_idx, m_as=300.0):
        # image_np is mu
        with self._device_context():
            proj = self.get_view_projector(n_source, view_idx)
            geom = self.geometry_cache[n_source]
            
            device = self.device
            img_torch = torch.from_numpy(image_np.astype(np.float32)).to(device)
            
            line_integrals = proj.forward(img_torch).cpu().numpy().flatten()
        
        # Poisson Noise: mAs = 100 -> 1e7 Photons (Clinical)
        i0 = float(m_as) * 1e5
            
        counts = i0 * np.exp(-line_integrals)
        counts_noisy = np.random.poisson(counts).astype(np.float32)
        counts_noisy = np.maximum(counts_noisy, 0.5)
        data = -np.log(counts_noisy / i0)
        
        # Map to unrolled detector row (2304)
        total_dets = 48 * 48
        full_column = np.zeros(total_dets, dtype=np.float32)
        mask = geom["source_module_mask"][view_idx]
        active_modules = np.where(mask)[0]
        
        offset = 0
        for m in active_modules:
            start_idx = m * 48
            full_column[start_idx : start_idx + 48] = data[offset : offset + 48]
            offset += 48
            
        return full_column

    def reconstruct_step(self, n_source, sinogram_np, step='unfiltered', band_a=0.0, band_b=0.0, band_c=0.0, w_low=None, w_mid=None, w_high=None):
        """Debug-script-equivalent reconstruction paths for unfiltered BP, P-INV, and FBP."""
        with self._device_context():
            state = self._prepare_reconstruction_state(n_source, sinogram_np)
            raw_bp = state["raw_bp"]

            if step == 'unfiltered' or step == 'laminogram' or step == 'unfiltered-bp':
                norm_map = self.normalization_maps.get(n_source)
                if norm_map is not None:
                    recon_mu = raw_bp / norm_map
                else:
                    recon_mu = raw_bp

                print(
                    f"DEBUG: Unfiltered Step [N={n_source}] - Mu Stats: "
                    f"mean={recon_mu.mean().item():.6f}, std={recon_mu.std().item():.6f}, "
                    f"min={recon_mu.min().item():.6f}, max={recon_mu.max().item():.6f}"
                )
            else:
                filter_key = f"{step}_{n_source}"
                svd_filter = self.svd_filters.get(filter_key)
                if svd_filter is None:
                    weight_dir = PRECOMPUTED_WEIGHTS_DIR / f"svd_{n_source}"
                    if weight_dir.exists():
                        print(f"DEBUG: Loading SVD weights for N={n_source} from {weight_dir}...")
                        s, v, loaded_paths = load_combined_svd_weights(weight_dir, device=self.device)
                        print(f"DEBUG: Loaded {s.numel()} singular vectors from {len(loaded_paths) - 2} extension files")
                        svd_filter = SVDImageFilter(s, v).to(self.device)
                        if step == 'pinv':
                            with torch.no_grad():
                                svd_filter.null_weight.zero_()
                                svd_filter.diag_diff.copy_(1.0 / (s**2))
                        self.svd_filters[filter_key] = svd_filter
                    else:
                        print(f"WARNING: No SVD weights for {n_source}, returning raw back projection")
                        svd_filter = None

                if svd_filter is not None:
                    recon_mu = svd_filter(raw_bp)
                
                # Apply 2D Fourier filter if step is 'filter'
                if step == 'filter' or step == 'fbp':
                    import math
                    # Load trained optimal 2D ramp filter
                    optimal_ramp_path = PRECOMPUTED_WEIGHTS_DIR / f"svd_{n_source}" / "optimal_2d_ramp_filter.pt"
                    if optimal_ramp_path.exists():
                        H_ramp = torch.load(optimal_ramp_path, map_location=self.device)
                    else:
                        H_ramp = torch.ones((256, 256), device=self.device)
                    
                    # Compute spatial frequency bands
                    y_freq = torch.fft.fftfreq(256, d=1.0, device=self.device).view(256, 1)
                    x_freq = torch.fft.fftfreq(256, d=1.0, device=self.device).view(1, 256)
                    freq_sq = y_freq**2 + x_freq**2
                    
                    # Sigmas in pixels matching FWHMs (1.0mm, 2.0mm, 4.0mm) with pixel spacing 0.75mm
                    sigma_10 = 1.0 / (0.75 * 2.35482)
                    sigma_20 = 2.0 / (0.75 * 2.35482)
                    sigma_40 = 4.0 / (0.75 * 2.35482)
                    
                    # Gaussian LPFs
                    pi2 = 2.0 * (math.pi ** 2)
                    G10 = torch.exp(-pi2 * (sigma_10**2) * freq_sq)
                    G20 = torch.exp(-pi2 * (sigma_20**2) * freq_sq)
                    G40 = torch.exp(-pi2 * (sigma_40**2) * freq_sq)
                    
                    if w_low is not None and w_mid is not None and w_high is not None:
                        # Direct relative linear scale factors from 0.0 to 1.0 (spanning 0-100%)
                        Low_band = G20
                        Mid_band = G10 - G20
                        High_band = 1.0 - G10
                        
                        # Construct H_cutoff
                        H_cutoff = w_low * Low_band + w_mid * Mid_band + w_high * High_band
                    else:
                        # Backward compatibility foldback scale factors from dB: linear = 10^(dB/20)
                        w_a = 10.0 ** (band_a / 20.0)
                        w_b = 10.0 ** (band_b / 20.0)
                        w_c = 10.0 ** (band_c / 20.0)
                        
                        # Bands
                        # Constant Base is G40 (4.0mm FWHM LPF)
                        # Band A: 2mm to 4mm
                        band_A_filter = G20 - G40
                        # Band B: 1mm to 2mm
                        band_B_filter = G10 - G20
                        # Band C: > 1mm
                        band_C_filter = 1.0 - G10
                        
                        # Construct H_cutoff
                        H_cutoff = G40 + w_a * band_A_filter + w_b * band_B_filter + w_c * band_C_filter
                    
                    # Combined total filter
                    H_total = H_ramp * H_cutoff
                    
                    # Apply via 2D Fourier Transform
                    fbp_fft = torch.fft.fft2(recon_mu)
                    fbp_fft_filtered = fbp_fft * H_total
                    recon_mu = torch.real(torch.fft.ifft2(fbp_fft_filtered))
                    
                    # Re-apply FOV mask to avoid boundary leakage
                    yy, xx = torch.meshgrid(
                        torch.linspace(-1, 1, 256, device=self.device),
                        torch.linspace(-1, 1, 256, device=self.device),
                        indexing="ij"
                    )
                    fov_mask = (xx**2 + yy**2 <= 0.98).to(recon_mu.dtype)
                    recon_mu = recon_mu * fov_mask

                    print(
                        f"DEBUG: {step.upper()} Step [N={n_source}] - Mu Stats: "
                        f"mean={recon_mu.mean().item():.6f}, std={recon_mu.std().item():.6f}, "
                        f"min={recon_mu.min().item():.6f}, max={recon_mu.max().item():.6f}"
                    )
                else:
                    recon_mu = raw_bp
            
        recon_hu = (recon_mu.cpu().numpy() * 1000.0 / MU_WATER_60KEV) - 1000.0
        return np.clip(recon_hu, -1000, 1500)

    def run_iterative_recon(self, n_source, sinogram_np, num_iters=50, tv_weight=0.01, lr=0.1, use_precond=True, stop_requested=None):
        print(f"DEBUG [projector.py]: run_iterative_recon starting for n_source={n_source}")
        self._warmup_projectors(n_source)
        geom = self.geometry_cache[n_source]
        
        # 1. Prepare active sinogram data and ensure it matches projector output
        projector = self.full_projectors[n_source]
        
        # SVD-based reconstruction for initialization
        fbp_hu = self.reconstruct_step(n_source, sinogram_np, step='filter')
        # Map HU to mu (attenuation mm^-1)
        fbp_mu = (torch.as_tensor(fbp_hu, device=self.device, dtype=torch.float32) + 1000.0) * MU_WATER_60KEV / 1000.0
        
        with torch.no_grad():
            Ax_test = projector(fbp_mu)
            
        sino_active = []
        for i in range(n_source):
            mask = geom["source_module_mask"][i]
            view_sino = sinogram_np[i]
            for m_idx, active in enumerate(mask):
                if active:
                    sino_active.append(view_sino[m_idx*48 : (m_idx+1)*48])
        sino_active = np.concatenate(sino_active)
        
        y = torch.as_tensor(sino_active, device=self.device, dtype=torch.float32)
        if y.shape != Ax_test.shape:
            y = y.view(Ax_test.shape)

        # Likelihood
        likelihood = LinearGaussianLogLikelihood(projector, measurements=y)
        
        # Prior
        if tv_weight > 0:
            prior = TotalVariancePrior2D(regularization_weight=tv_weight)
        else:
            # Explicitly return zeros with grad for Maximum Likelihood
            prior = lambda x: torch.sum(x * 0.0)
            
        # Preconditioner
        precond = None
        inv_precond = None
        if use_precond:
            # Use 'svd_{n_source}' directory which actually contains weights
            weight_dir = PRECOMPUTED_WEIGHTS_DIR / f"svd_{n_source}"
            if (weight_dir / "S.pt").exists() and (weight_dir / "V.pt").exists():
                s, v, _ = load_combined_svd_weights(weight_dir, device=self.device, dtype=torch.float32)
                self.last_precond_s = s
                # Keep top 1000 singular values for preconditioner
                k = min(1000, s.numel())
                precond = SparseEigenPreconditioner(s[:k], v[:, :k])
                inv_precond = precond.inverse
                print(f"DEBUG: Using SparseEigenPreconditioner with S range [{s[:k].min().item():.2e}, {s[:k].max().item():.2e}]")
            else:
                print(f"WARNING: No SVD weights for preconditioner at {weight_dir}")

        # Correct lr logic: 0.1 is stable for preconditioned gradient descent
        # Pure GD calibrated at 1e-7 for 240 views
        effective_lr = lr if lr is not None else (0.1 if use_precond else 1e-7)

        reconstructor = MaximumAPosterioriReconstructor(
            log_likelihood_fn=likelihood,
            log_prior_fn=prior,
            volume_init=fbp_mu,
            preconditioner=precond,
            inv_preconditioner=inv_precond,
            lr=effective_lr
        )
        
        loss_history = []
        lik_history = []
        prior_history = []
        
        # Iteration Loop
        print(f"DEBUG [projector.py]: Entering MAP loop for {num_iters} iterations")
        for i in range(num_iters):
            if stop_requested is not None and stop_requested():
                print(f"DEBUG [projector.py]: Iterative recon cancelled at iter={i}")
                break
            ll, lp, lpost, gn_lik, gn_prior, gn_total = reconstructor.map_step()
            loss_history.append(float(lpost))
            lik_history.append(float(ll))
            prior_history.append(float(lp))
            
            if i % 10 == 0:
                print(f"DEBUG [projector.py]: Iter {i}: Lik={ll:.2e}, Prior={lp:.2e}, Post={lpost:.2e}")
            
            # Diagnostic Stats
            with torch.no_grad():
                mu = reconstructor.volume_pred
                sino_pred = projector(mu)
                rmse = torch.sqrt(torch.mean((y - sino_pred)**2)).item()
                
                # Convert to HU for display/yield
                hu = (mu.detach().cpu().numpy() * 1000.0 / MU_WATER_60KEV) - 1000.0
                
                yield {
                    "iteration": i + 1,
                    "image": np.clip(hu, -1000, 1500),
                    "log_likelihood": ll,
                    "log_prior": lp,
                    "rmse": rmse,
                    "loss_history": loss_history,
                    "likelihood_history": lik_history,
                    "prior_history": prior_history
                }

    def reconstruct_eigen_fbp(self, n_source, sinogram_np):
        # Compatibility wrapper
        return self.reconstruct_step(n_source, sinogram_np, step='filter')

    def generate_geometry_base(self, n_source, image_np, ww=None, wl=None):
        """Generates the opaque base image (patient + static sources + detectors)."""
        self._warmup_projectors(n_source)
        geom = self.geometry_cache[n_source]
        src_pos = geom["source_positions"]
        mod_centers = geom["module_centers"]
        det_radius = geom["module_radius"]

        if not hasattr(self, 'fig_base'):
            self.fig_base = Figure(figsize=(8, 8), dpi=100)
            self.fig_base.patch.set_facecolor('black')
            self.ax_base = self.fig_base.add_subplot(111)
        else:
            self.ax_base.clear()

        ax = self.ax_base
        ax.set_facecolor('black')
        ax.axis('off')
        ax.set_xlim(-450, 450)
        ax.set_ylim(-450, 450)

        # Static Sources - color: #0957BE, smaller size: 5
        ax.scatter(src_pos[:, 0], src_pos[:, 1], c='#0957BE', s=5, alpha=0.3, zorder=1)
        
        # Detectors (Inactive style)
        det_segs = []
        for i in range(len(mod_centers)):
            center = mod_centers[i]
            angle = np.arctan2(center[1], center[0])
            half_angle = (24.0 / det_radius) 
            p1 = [det_radius * np.cos(angle - half_angle), det_radius * np.sin(angle - half_angle)]
            p2 = [det_radius * np.cos(angle + half_angle), det_radius * np.sin(angle + half_angle)]
            det_segs.append([p1, p2])
        
        lc = LineCollection(det_segs, colors=['#188478']*len(det_segs), linewidths=3, zorder=3, capstyle='round', alpha=0.4)
        ax.add_collection(lc)

        # Patient Image
        extent_val = (256 * 1.6) / 2.0
        extent = [-extent_val, extent_val, -extent_val, extent_val]
        display_img = np.clip(image_np, wl-ww/2, wl+ww/2) if ww else image_np
        ax.imshow(display_img, cmap='gray', extent=extent, alpha=0.75, origin='upper', zorder=2)
        if ww: ax.images[0].set_clim(wl-ww/2, wl+ww/2)

        buf = io.BytesIO()
        self.fig_base.savefig(buf, format='png', facecolor='black', bbox_inches='tight', pad_inches=0.0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def generate_geometry_overlay(self, n_source, view_idx):
        """Generates a transparent PNG with the active source and rays."""
        self._warmup_projectors(n_source)
        geom = self.geometry_cache[n_source]
        src_pos = geom["source_positions"]
        mod_centers = geom["module_centers"]
        mask = geom["source_module_mask"][view_idx]
        det_radius = geom["module_radius"]

        # Use a persistent figure for overlay if available
        if not hasattr(self, 'fig_overlay'):
            self.fig_overlay = Figure(figsize=(8, 8), dpi=100)
            self.fig_overlay.patch.set_alpha(0)
            self.ax_overlay = self.fig_overlay.add_subplot(111)
        else:
            self.ax_overlay.clear()

        ax = self.ax_overlay
        ax.set_facecolor((0, 0, 0, 0))
        ax.axis('off')
        ax.set_xlim(-450, 450)
        ax.set_ylim(-450, 450)

        # Active Source - color: #2FB1E6, circular marker, larger size: 150
        ax.scatter([src_pos[view_idx, 0]], [src_pos[view_idx, 1]], c='#2FB1E6', marker='o', s=150, zorder=10)

        # Active Detectors
        active_segs = []
        for i in range(len(mod_centers)):
            if mask[i]:
                center = mod_centers[i]
                angle = np.arctan2(center[1], center[0])
                half_angle = (24.0 / det_radius) 
                p1 = [det_radius * np.cos(angle - half_angle), det_radius * np.sin(angle - half_angle)]
                p2 = [det_radius * np.cos(angle + half_angle), det_radius * np.sin(angle + half_angle)]
                active_segs.append([p1, p2])
        
        if active_segs:
            lc = LineCollection(active_segs, colors=['#29E1CB']*len(active_segs), linewidths=4, zorder=11, capstyle='round')
            ax.add_collection(lc)

        # Rays
        proj = self.view_projectors[n_source][view_idx]
        s_coords = proj.src.cpu().numpy()
        d_coords = proj.dst.cpu().numpy()
        # Subsample rays for performance
        indices = np.random.choice(s_coords.shape[0], min(200, s_coords.shape[0]), replace=False)
        ray_segs = np.stack([s_coords[indices], d_coords[indices]], axis=1)
        lc_rays = LineCollection(ray_segs, color='#29E1CB', alpha=0.15, linewidths=0.5, zorder=5)
        ax.add_collection(lc_rays)

        # HUD Text
        ax.text(-430, 410, f"Source: {view_idx}", color='white', fontsize=12, fontweight='bold')

        buf = io.BytesIO()
        self.fig_overlay.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0.0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def generate_geometry_frame(self, n_source, view_idx, image_np, ww=None, wl=None, m_as=300.0):
        self._warmup_projectors(n_source)
        geom = self.geometry_cache[n_source]
        src_pos = geom["source_positions"]
        mod_centers = geom["module_centers"]
        mask = geom["source_module_mask"][view_idx]
        det_radius = geom["module_radius"]
        
        # --- INITIALIZATION (Only Once) ---
        if self.fig_geo is None:
            self.fig_geo, self.ax_geo = plt.subplots(figsize=(8, 8), dpi=120)
            self.fig_geo.patch.set_facecolor('black')
            self.ax_geo.set_facecolor('black')
            self.ax_geo.axis('off')
            self.ax_geo.set_xlim(-450, 450)
            self.ax_geo.set_ylim(-450, 450)
            
            # Static Sources
            self.geo_artists = {}
            self.geo_artists['sources'] = self.ax_geo.scatter(src_pos[:, 0], src_pos[:, 1], c='#0957BE', s=5, alpha=0.3, zorder=1)
            self.geo_artists['active_source'] = self.ax_geo.scatter([0], [0], c='#2FB1E6', marker='o', s=150, zorder=10)
            
            # Detectors
            det_segs = []
            for i in range(len(mod_centers)):
                center = mod_centers[i]
                angle = np.arctan2(center[1], center[0])
                half_angle = (24.0 / det_radius) 
                p1 = [det_radius * np.cos(angle - half_angle), det_radius * np.sin(angle - half_angle)]
                p2 = [det_radius * np.cos(angle + half_angle), det_radius * np.sin(angle + half_angle)]
                det_segs.append([p1, p2])
            
            self.geo_artists['detectors'] = LineCollection(det_segs, colors=['#188478']*len(det_segs), linewidths=3, zorder=3, capstyle='round')
            self.ax_geo.add_collection(self.geo_artists['detectors'])
            
            # Rays (Init with empty collection)
            self.geo_artists['rays'] = LineCollection([], color='#29E1CB', alpha=0.08, linewidths=0.5, zorder=5)
            self.ax_geo.add_collection(self.geo_artists['rays'])
            
            # Patient Image
            extent_val = (256 * 1.6) / 2.0
            extent = [-extent_val, extent_val, -extent_val, extent_val]
            self.geo_artists['img'] = self.ax_geo.imshow(np.zeros((256, 256)), cmap='gray', extent=extent, alpha=0.75, origin='upper', zorder=2)
            
            # HUD Text elements (Simplified: only source number)
            self.geo_artists['txt_src'] = self.ax_geo.text(-430, 410, "", color='white', fontsize=12, fontweight='bold')

        # --- UPDATE (Fast Path) ---
        ax = self.ax_geo
        arts = self.geo_artists

        # Update Active Source
        arts['active_source'].set_offsets([src_pos[view_idx, 0], src_pos[view_idx, 1]])

        # Update Detectors
        det_colors = ["#29E1CB" if mask[i] else "#188478" for i in range(len(mod_centers))]
        det_alphas = [1.0 if mask[i] else 0.4 for i in range(len(mod_centers))]
        arts['detectors'].set_edgecolors(det_colors)
        arts['detectors'].set_alpha(det_alphas)

        # Update Rays
        proj = self.view_projectors[n_source][view_idx]
        s_coords = proj.src.cpu().numpy()
        d_coords = proj.dst.cpu().numpy()
        indices = np.random.choice(s_coords.shape[0], min(300, s_coords.shape[0]), replace=False)
        ray_segs = np.stack([s_coords[indices], d_coords[indices]], axis=1)
        arts['rays'].set_segments(ray_segs)

        # Update Image
        display_img = np.clip(image_np, wl-ww/2, wl+ww/2) if ww else image_np
        arts['img'].set_data(display_img)
        if ww: arts['img'].set_clim(wl-ww/2, wl+ww/2)

        # Update HUD
        arts['txt_src'].set_text(f"Source: {view_idx}")

        # FAST SAVE
        buf = io.BytesIO()
        self.fig_geo.savefig(buf, format='png', facecolor='black', bbox_inches='tight', pad_inches=0.0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def generate_sinogram_frame(self, full_sinogram, view_idx, m_as=300.0):
        n_source, n_det = full_sinogram.shape

        # --- INITIALIZATION ---
        if self.fig_sino is None:
            self.fig_sino, (self.ax_sino, self.ax_prof) = plt.subplots(2, 1, figsize=(6, 6), dpi=80, gridspec_kw={'height_ratios': [2, 1]})
            self.fig_sino.patch.set_facecolor('black')
            
            self.sino_artists = {}
            self.ax_sino.set_facecolor('black')
            self.ax_prof.set_facecolor('black')
            
            # Sinogram Image
            self.sino_artists['sino_img'] = self.ax_sino.imshow(np.zeros((n_source, n_det)), cmap='gray', vmin=0.0, vmax=8.0, aspect='auto', origin='upper')
            self.ax_sino.axis('off')
            
            # Profile Plot
            self.sino_artists['prof_line'], = self.ax_prof.plot(np.arange(n_det), np.zeros(n_det), color='white', linewidth=1.5)
            self.ax_prof.set_ylim(-0.2, 8.5)
            self.ax_prof.set_xlim(0, n_det)
            self.ax_prof.tick_params(colors='white', which='both', labelsize=7)
            
            # Titles
            self.sino_artists['txt_sino'] = self.ax_sino.set_title('', color='#52DAF2', fontsize=9)
            self.sino_artists['txt_prof'] = self.ax_prof.set_title('Line Integral Profile', color='#52DAF2', fontsize=9)

        # --- UPDATE ---
        arts = self.sino_artists
        
        # Partially fill sinogram display efficiently
        if not hasattr(self, '_display_sino_buf') or self._display_sino_buf.shape != full_sinogram.shape or view_idx == 0:
            self._display_sino_buf = np.zeros_like(full_sinogram)
        
        self._display_sino_buf[view_idx, :] = full_sinogram[view_idx, :]
        arts['sino_img'].set_data(self._display_sino_buf)
        
        # Profile line
        arts['prof_line'].set_ydata(full_sinogram[view_idx, :])
        
        # Meta
        self.ax_sino.set_title(f'Sinogram | Source {view_idx}', color='#52DAF2', fontsize=9)

        # FAST SAVE
        buf = io.BytesIO()
        self.fig_sino.savefig(buf, format='png', facecolor='black', bbox_inches='tight', pad_inches=0.1)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def generate_simulation_gif(self, dataset_id, patient_id, slice_index, n_source, image_np, full_sinogram, ww=None, wl=None, m_as=300.0):
        """Generates a side-by-side GIF."""
        output_dir = Path(__file__).parent.parent.parent / "static" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        gif_path = output_dir / f"sim_{dataset_id}_{patient_id}_{slice_index}.gif"
        
        self._warmup_projectors(n_source)
        geom = self.geometry_cache[n_source]
        
        frames = []
        step = max(1, n_source // 40)
        fig = plt.figure(figsize=(15, 7), dpi=80); fig.patch.set_facecolor('black')
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[2, 1])
        
        # Geometry info
        src_pos = geom["source_positions"]
        mod_centers = geom["module_centers"]
        det_radius = geom["module_radius"]
        
        i0 = float(m_as) * 100.0
        
        for view_idx in range(0, n_source, step):
            fig.clf()
            ax1 = fig.add_subplot(gs[:, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[1, 1])
            ax1.set_facecolor('black'); ax2.set_facecolor('black'); ax3.set_facecolor('black')
            
            # --- Geometry AX1 ---
            ax1.scatter(src_pos[:, 0], src_pos[:, 1], c='#FCC1DC', s=5, alpha=0.3)
            ax1.scatter([src_pos[view_idx, 0]], [src_pos[view_idx, 1]], c='#FD4497', marker='*', s=150, zorder=10)
            
            # Detectors
            mask = geom["source_module_mask"][view_idx]
            for i in range(len(mod_centers)):
                center = mod_centers[i]
                angle = np.arctan2(center[1], center[0])
                half_angle = (24.0 / det_radius) 
                p1 = [det_radius * np.cos(angle - half_angle), det_radius * np.sin(angle - half_angle)]
                p2 = [det_radius * np.cos(angle + half_angle), det_radius * np.sin(angle + half_angle)]
                color = "#29E1CB" if mask[i] else "#188478"
                ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=1.0 if mask[i] else 0.4, linewidth=2, zorder=3)

            # Rays (Sample 200 for GIF speed)
            proj = self.view_projectors[n_source][view_idx]
            s_coords = proj.src.cpu().numpy()
            d_coords = proj.dst.cpu().numpy()
            indices = np.random.choice(s_coords.shape[0], min(200, s_coords.shape[0]), replace=False)
            for idx in indices:
                ax1.plot([s_coords[idx, 0], d_coords[idx, 0]], [s_coords[idx, 1], d_coords[idx, 1]], color='#FD4497', alpha=0.06, linewidth=0.5, zorder=5)

            # Image
            extent_val = (256 * 1.6) / 2.0
            display_img = np.clip(image_np, wl-ww/2, wl+ww/2) if ww else image_np
            # Use origin='upper' to match recon tabs
            ax1.imshow(display_img, cmap='gray', extent=[-extent_val, extent_val, -extent_val, extent_val], origin='upper', alpha=0.75, zorder=2)
            ax1.set_xlim(-450, 450); ax1.set_ylim(-450, 450); ax1.axis('off')
            ax1.text(-430, 410, f"Source: {view_idx}", color='white', fontsize=10)
            ax1.text(-430, 380, f"I0: {i0:,.0f}", color='#52DAF2', fontsize=9)
            
            # --- Sinogram AX2 ---
            display_sino = np.zeros_like(full_sinogram)
            display_sino[:view_idx+1, :] = full_sinogram[:view_idx+1, :]
            ax2.imshow(display_sino, cmap='gray', vmin=0.0, vmax=8.0, aspect='auto', origin='upper')
            ax2.axis('off')
            
            # --- Profile AX3 ---
            ax3.plot(full_sinogram[view_idx, :], color='white', linewidth=1)
            ax3.set_ylim(-0.2, 8.5)
            ax3.tick_params(colors='white', labelsize=7)
            
            buf = io.BytesIO(); fig.savefig(buf, format='png', facecolor='black'); buf.seek(0)
            try:
                import imageio
            except ImportError as exc:
                raise RuntimeError("GIF export requires imageio to be installed in the backend environment.") from exc
            frames.append(imageio.imread(buf))
            
        plt.close(fig)
        try:
            import imageio
        except ImportError as exc:
            raise RuntimeError("GIF export requires imageio to be installed in the backend environment.") from exc
        imageio.mimsave(gif_path, frames, fps=10)
        return f"/static/outputs/{gif_path.name}"

sim_manager = SimulationManager()

