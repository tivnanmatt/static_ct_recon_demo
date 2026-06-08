import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_filter_graph_png(w_low: float, w_mid: float, w_high: float) -> str:
    """Generates a base64 encoded PNG 1D filter plot with white axes to show the ramp and cutoff filter."""
    # Pixel spacing in mm
    pixel_spacing = 0.75
    
    # Grid setup (Nyquist at f = 0.5 cycles/pixel)
    f_pixel = np.linspace(0.0, 0.5, 128)
    f_sq = f_pixel ** 2
    
    # Sigmas in pixels for FWHMs of 1.0mm, 2.0mm, 4.0mm
    sigma_10 = 1.0 / (pixel_spacing * 2.35482)
    sigma_20 = 2.0 / (pixel_spacing * 2.35482)
    
    pi2 = 2.0 * (np.pi ** 2)
    G10 = np.exp(-pi2 * (sigma_10**2) * f_sq)
    G20 = np.exp(-pi2 * (sigma_20**2) * f_sq)
    
    # Partition of unity bands
    Low_band = G20
    Mid_band = G10 - G20
    High_band = 1.0 - G10
    
    H_cutoff = w_low * Low_band + w_mid * Mid_band + w_high * High_band
    
    # Ideal Ramp (normalized to peak at 1.0)
    ideal_ramp = f_pixel / 0.5
    
    # Total combined filter
    total_filter = ideal_ramp * H_cutoff
    
    # Convert x-axis to cycles/mm
    f_cycles_mm = f_pixel / pixel_spacing
    
    # Wide figure with white axes background and transparent figure background
    fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=300)
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('#ffffff')

    # Customize spine, grid and axis colors to ARPA-H Primary Navy (#000334)
    for spine in ax.spines.values():
        spine.set_color('#000334')
        spine.set_linewidth(1.2)

    ax.tick_params(colors='#000334', labelsize=11)
    ax.grid(True, color='#dddddd', linestyle='-', linewidth=0.8)

    # Plot lines with thicker lines using ARPA-H and ATS brand colors
    ax.plot(f_cycles_mm, ideal_ramp, color='#8f8f8f', linestyle='--', linewidth=2.0, label='Ideal Ramp')
    ax.plot(f_cycles_mm, H_cutoff, color='#52daf2', linestyle=':', linewidth=3.0, label='Cutoff Filter')
    ax.plot(f_cycles_mm, total_filter, color='#001b5e', linestyle='-', linewidth=4.0, label='Combined Total')

    ax.set_xlim(0, 0.5 / pixel_spacing)
    ax.set_ylim(0, 1.1)

    ax.set_title('FlashFBP Filter', color='#000334', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Spatial Frequency (cycles/mm)', color='#000334', fontsize=12, fontweight='semibold')
    ax.set_ylabel('Filter Gain', color='#000334', fontsize=12, fontweight='semibold')

    # Legend placed outside the axes, to the right of the figure.
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#000334', fontsize=11,
              loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    buf = io.BytesIO()
    # bbox_inches='tight' so the externally-anchored legend is included in the saved image.
    plt.savefig(buf, format='png', transparent=True, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return base64.b64encode(buf.getvalue()).decode('utf-8')
