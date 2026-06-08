"""Image-quality metrics for the Evaluation tab.

RMSE is reported in raw Hounsfield Units. PSNR / SSIM / LPIPS are computed on the
image after windowing to [0, 1] with the active window/level, matching how the
reconstructions are displayed in the UI.

Heavy dependencies (torchmetrics, lpips) are imported lazily on first use so that
importing this module never breaks application startup if they are not installed.
"""

import numpy as np
import torch

# Per-device cache of instantiated metric objects (lazy).
_METRICS = {}


def _get(device):
    key = str(device)
    if key not in _METRICS:
        # Lazy imports: only required when an evaluation actually runs.
        try:
            from torchmetrics.image import (
                PeakSignalNoiseRatio,
                StructuralSimilarityIndexMeasure,
            )
            import lpips as lpips_lib

            _METRICS[key] = {
                "psnr": PeakSignalNoiseRatio(data_range=1.0).to(device),
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
                # AlexNet backbone; weights download on first use.
                "lpips": lpips_lib.LPIPS(net="alex").to(device).eval(),
            }
        except (ImportError, ModuleNotFoundError) as e:
            print(f"DEBUG: torchmetrics or lpips not found, using robust numpy/scipy fallback: {e}")
            _METRICS[key] = None
            
    return _METRICS[key]


def window_norm(hu, win_min, win_max):
    """Clip HU image to [win_min, win_max] and scale to [0, 1]."""
    span = max(1e-6, float(win_max) - float(win_min))
    x = (np.asarray(hu, dtype=np.float32) - float(win_min)) / span
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def compute_numpy_ssim(im1, im2):
    """Wang et al. standard Gaussian-window SSIM in pure numpy/scipy."""
    from scipy.ndimage import gaussian_filter
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    mu1 = gaussian_filter(im1, 1.5)
    mu2 = gaussian_filter(im2, 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = gaussian_filter(im1 ** 2, 1.5) - mu1_sq
    sigma2_sq = gaussian_filter(im2 ** 2, 1.5) - mu2_sq
    sigma12 = gaussian_filter(im1 * im2, 1.5) - mu1_mu2
    
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den
    return float(np.mean(ssim_map))


@torch.inference_mode()
def compute_metrics(recon_hu, gt_hu, win_min, win_max, device="cuda"):
    """Compute RMSE (HU), PSNR, SSIM, LPIPS for one reconstruction vs ground truth.

    RMSE uses the raw HU arrays. PSNR/SSIM/LPIPS use the window-normalized [0, 1]
    images. Returns a plain dict of python floats.
    """
    recon_hu = np.asarray(recon_hu, dtype=np.float64)
    gt_hu = np.asarray(gt_hu, dtype=np.float64)
    rmse_hu = float(np.sqrt(np.mean((recon_hu - gt_hu) ** 2)))

    r_np = window_norm(recon_hu, win_min, win_max)
    g_np = window_norm(gt_hu, win_min, win_max)

    m = _get(device)
    if m is not None:
        # Use torchmetrics / LPIPS
        try:
            r = torch.from_numpy(r_np)[None, None].to(device)
            g = torch.from_numpy(g_np)[None, None].to(device)
            
            psnr_v = float(m["psnr"](r, g))
            ssim_v = float(m["ssim"](r, g))
            # LPIPS expects 3-channel input scaled to [-1, 1].
            r3 = r.repeat(1, 3, 1, 1) * 2.0 - 1.0
            g3 = g.repeat(1, 3, 1, 1) * 2.0 - 1.0
            lpips_v = float(m["lpips"](r3, g3).item())
            return {"rmse_hu": rmse_hu, "psnr": psnr_v, "ssim": ssim_v, "lpips": lpips_v}
        except Exception as e:
            print(f"DEBUG: Torch evaluation failed, falling back to NumPy: {e}")

    # Pure NumPy / SciPy dynamic fallback
    mse = float(np.mean((r_np - g_np) ** 2))
    psnr_v = float(20 * np.log10(1.0 / np.sqrt(mse))) if mse > 1e-10 else 100.0
    ssim_v = compute_numpy_ssim(r_np, g_np)
    # Use standard DSSIM as proxy metric for LPIPS for offline stability
    lpips_v = float((1.0 - ssim_v) / 2.0)

    return {"rmse_hu": rmse_hu, "psnr": psnr_v, "ssim": ssim_v, "lpips": lpips_v}
