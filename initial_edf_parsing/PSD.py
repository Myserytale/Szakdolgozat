import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from reading_working import DATA_DIR, HYPART_DIR, load_hypart, read_edf

# 1. Load data
edf_path = DATA_DIR / "BB0008_SP2.edf"
hypart_path = HYPART_DIR / f"{edf_path.stem}.hypart"

edf_data = read_edf(edf_path, channels=["C3"])
c3_signal = edf_data["signals"]["C3"]
fs = int(edf_data["sample_rates"][0])

hypnogram, artefacts, art_channels = load_hypart(hypart_path)
c3_art_idx = art_channels.index("ArtiFFT C3-A2")

# 2. Filter for clean Stage 2 (NREM 2) samples
is_stage2 = hypnogram == 2
is_clean = artefacts[:, c3_art_idx] == 0  # 0 indicates clean/no artifact
stage2_clean = c3_signal[is_stage2 & is_clean]

# 3. Compute Power Spectral Density (PSD)
window_sec = 4
nperseg = fs * window_sec

frequencies, psd = welch(
    stage2_clean,
    fs=fs,
    window="hann",
    nperseg=nperseg,
    noverlap=nperseg // 2,
    scaling="density",
)

# 4. Plot PSD
plt.figure(figsize=(10, 5))
plt.semilogy(frequencies, psd, color="#1f77b4", linewidth=1.5)
plt.xlim(0.5, 30)
plt.xlabel("Frequency (Hz)")
plt.ylabel(r"PSD ($\mu V^2 / Hz$)")
plt.title("C3 — Clean NREM2 Power Spectral Density")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("psd_plot2.png", dpi=150)
plt.show()
