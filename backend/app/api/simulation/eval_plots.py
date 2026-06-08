import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_metric_ylimits(acc_data: dict) -> dict:
    """Shared per-metric y-axis limits across all methods, so the same metric uses
    an identical y-range in every method panel (making the panels directly comparable).

    Returns {metric: (lo, hi)} with a little padding; None for a metric with no data yet.
    """
    metrics = ["rmse_hu", "psnr", "ssim", "lpips"]
    methods = ["fbp", "mbir", "dlr", "generative"]
    limits = {}
    for metric in metrics:
        vals = []
        for m in methods:
            for v in acc_data.get(m, {}).get(metric, []):
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
        if not vals:
            limits[metric] = None
            continue
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            pad = max(abs(hi) * 0.05, 1e-3)
        else:
            pad = (hi - lo) * 0.08
        limits[metric] = (lo - pad, hi + pad)
    return limits


def generate_method_square_plot_png(acc_data: dict, method: str, color: str, y_limits: dict = None) -> str:
    """
    Generates a 4x4 square base64 encoded PNG containing a 2x2 grid of violin plots
    (or scatter plot fallback) for RMSE, PSNR, SSIM, and LPIPS specifically for a single method.
    The mean ± std status is placed directly in the title of each subplot, removing the need for a table.
    """
    metrics = ["rmse_hu", "psnr", "ssim", "lpips"]
    metric_labels = {
        "rmse_hu": "RMSE (HU)",
        "psnr": "PSNR (dB)",
        "ssim": "SSIM",
        "lpips": "LPIPS"
    }
    metric_colors = {
        "rmse_hu": "#52daf2",
        "psnr": "#75ff33",
        "ssim": "#ffb733",
        "lpips": "#ff5733"
    }

    # Set up 2x2 square figure (4x4 inches layout)
    fig, axes = plt.subplots(2, 2, figsize=(4.0, 4.0), dpi=120)
    fig.patch.set_facecolor('#ffffff')
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        ax.set_facecolor('#fdfdfd')
        ax.grid(True, color='#eaeaea', linestyle='--', linewidth=0.5, axis='y')
        
        vals = np.array(acc_data.get(method, {}).get(metric, []), dtype=np.float32)
        m_color = metric_colors.get(metric, color)
        
        # Style subplot borders
        for spine in ax.spines.values():
            spine.set_color('#000334')
            spine.set_linewidth(0.8)
            
        ax.tick_params(colors='#000334', labelsize=7)
        ax.get_xaxis().set_visible(False)  # Hide the unused x-axis to maximize plot area
        
        # Calculate Title (statistics)
        if len(vals) == 0:
            title_str = f"{metric_labels[metric]}: --"
        else:
            mean_val = np.mean(vals)
            if len(vals) > 1:
                std_val = np.std(vals)
                if metric in ["ssim", "lpips"]:
                    title_str = f"{metric_labels[metric]}\n{mean_val:.4f} ± {std_val:.4f}"
                elif metric == "psnr":
                    title_str = f"{metric_labels[metric]}\n{mean_val:.2f} ± {std_val:.2f}"
                else:
                    title_str = f"{metric_labels[metric]}\n{mean_val:.1f} ± {std_val:.1f}"
            else:
                if metric in ["ssim", "lpips"]:
                    title_str = f"{metric_labels[metric]}\n{mean_val:.4f}"
                elif metric == "psnr":
                    title_str = f"{metric_labels[metric]}\n{mean_val:.2f}"
                else:
                    title_str = f"{metric_labels[metric]}\n{mean_val:.1f}"
                    
        ax.set_title(title_str, fontsize=8, fontweight='bold', color='#001b5e', pad=4)
        
        # Plot data
        if len(vals) > 0:
            if len(vals) > 1:
                try:
                    # Plot single violin
                    parts = ax.violinplot([vals], showmeans=False, showmedians=True, showextrema=True)
                    for pc in parts['bodies']:
                        pc.set_facecolor(m_color)
                        pc.set_edgecolor('#000334')
                        pc.set_alpha(0.6)
                    for k in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
                        if k in parts:
                            parts[k].set_edgecolor('#000334')
                            parts[k].set_linewidth(0.8)
                except Exception as e:
                    # fallback to box if violin feels fussy
                    ax.boxplot([vals], patch_artist=True)
                
                # Overlay individual data points as jittered scatter
                jitter = np.random.normal(1.0, 0.04, size=len(vals))
                ax.scatter(jitter, vals, color='#000334', s=10, alpha=0.5, zorder=4)
            else:
                # Exactly 1 patient case fallback: draw a prominent styled dot
                ax.scatter([1], vals, color=m_color, edgecolor='#000334', s=80, zorder=5)
                # Set dynamic limits to frame the single dot nicely
                ax.set_xlim(0.8, 1.2)
        
        if y_limits and y_limits.get(metric) is not None:
            ax.set_ylim(y_limits[metric])
                
    fig.tight_layout(pad=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_comparison_violin_plots(acc_data: dict) -> str:
    """
    Generates a 2x2 comparison grid of violin plots or bar plots for RMSE, PSNR, SSIM, and LPIPS.
    Compares four methods: FlashFBP, FidelityMBIR, NeuralSpark, GenerativeVision.
    acc_data: dict of the form:
    {
        "fbp": {"rmse_hu": [...], "psnr": [...], "ssim": [...], "lpips": [...]},
        "mbir": {...},
        "dlr": {...},
        "generative": {...}
    }
    """
    metrics = ["rmse_hu", "psnr", "ssim", "lpips"]
    metric_labels = {
        "rmse_hu": "RMSE (HU) [Lower is Better]",
        "psnr": "PSNR (dB) [Higher is Better]",
        "ssim": "SSIM [Higher is Better]",
        "lpips": "LPIPS [Lower is Better]"
    }
    
    methods = ["fbp", "mbir", "dlr", "generative"]
    method_labels = {
        "fbp": "FlashFBP",
        "mbir": "FidelityMBIR",
        "dlr": "NeuralSpark",
        "generative": "Generative"
    }
    method_colors = {
        "fbp": "#52daf2",       # ATS Light Blue
        "mbir": "#75ff33",      # Green
        "dlr": "#ffb733",       # Orange
        "generative": "#ff5733" # Coral Red
    }

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), dpi=120)
    fig.patch.set_facecolor('#ffffff')
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        ax.set_facecolor('#fdfdfd')
        ax.grid(True, color='#eaeaea', linestyle='--', linewidth=0.5, axis='y')
        
        # Build dataset for plotting on this metric
        plot_data = []
        labels = []
        colors = []
        
        for method in methods:
            vals = np.array(acc_data.get(method, {}).get(metric, []), dtype=np.float32)
            # Filter out NaNs if any
            vals = vals[~np.isnan(vals)]
            plot_data.append(vals)
            labels.append(method_labels[method])
            colors.append(method_colors[method])
        
        x_positions = np.arange(1, len(methods) + 1)
        
        # Style subplot borders
        for spine in ax.spines.values():
            spine.set_color('#000334')
            spine.set_linewidth(0.8)
            
        ax.tick_params(colors='#000334', labelsize=8)
        ax.set_title(metric_labels[metric], fontsize=9, fontweight='bold', color='#001b5e', pad=6)
        
        # If we have data points
        has_data = any(len(v) > 0 for v in plot_data)
        
        if has_data:
            # Check if we have more than 1 point to do violin plots
            max_len = max(len(v) for v in plot_data)
            
            if max_len > 1:
                # Filter out empty entries or plot them as dots
                for pos, vals, col in zip(x_positions, plot_data, colors):
                    if len(vals) > 1:
                        try:
                            parts = ax.violinplot([vals], [pos], showmeans=False, showmedians=True, showextrema=True)
                            for pc in parts['bodies']:
                                pc.set_facecolor(col)
                                pc.set_edgecolor('#000334')
                                pc.set_alpha(0.6)
                            for k in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
                                if k in parts:
                                    parts[k].set_edgecolor('#000334')
                                    parts[k].set_linewidth(1.0)
                        except Exception as e:
                            # fallback to box plot
                            ax.boxplot([vals], positions=[pos], patch_artist=True, boxprops=dict(facecolor=col))
                    elif len(vals) == 1:
                        # Draw a single dot
                        ax.scatter([pos], vals, color=col, edgecolor='#000334', s=60, zorder=5)
                    
                    # Overlay individual observations as jittered points
                    if len(vals) > 0:
                        jitter = np.random.normal(pos, 0.04, size=len(vals))
                        ax.scatter(jitter, vals, color='#000334', s=8, alpha=0.5, zorder=4)
            else:
                # Exactly 1 point for everyone: draw prominent styled dots
                for pos, vals, col in zip(x_positions, plot_data, colors):
                    if len(vals) > 0:
                        ax.scatter([pos], vals, color=col, edgecolor='#000334', s=80, zorder=5)
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels, rotation=0, fontsize=8)
        else:
            # Placeholder text when there is no data
            ax.text(0.5, 0.5, "No Data (Press Run)", transform=ax.transAxes,
                    ha='center', va='center', color='#888888', fontsize=10, fontstyle='italic')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels, rotation=0, fontsize=8)
            
    fig.tight_layout(pad=1.5)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
