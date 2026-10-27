import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from reading_working import DATA_DIR, HYPART_DIR, load_hypart, read_edf

# 1. Settings & Channel Selection
channel = "C3"
art_channel = f"ArtiFFT {channel}-A2"

edf_path = DATA_DIR / "BB0008_SP2.edf"
hypart_path = HYPART_DIR / f"{edf_path.stem}.hypart"

# 2. Load Signal & Annotations
edf_data = read_edf(edf_path, channels=[channel])
signal = edf_data["signals"][channel]
fs = int(edf_data["sample_rates"][0])

hypnogram, artefacts, art_channels = load_hypart(hypart_path)
art_idx = art_channels.index(art_channel)

# 3. Epoch Settings
epoch_sec = 10
epoch_samples = fs * epoch_sec

welch_sec = 4
nperseg = fs * welch_sec

frequencies = np.fft.rfftfreq(nperseg, 1.0 / fs)
psds = []

# 4. Epoch Segmentation & Welch PSD
n_epochs = len(signal) // epoch_samples

for i in range(n_epochs):
    start = i * epoch_samples
    end = start + epoch_samples

    # Check for Stage 2 (NREM2) and absence of artifacts
    if not np.all(hypnogram[start:end] == 2):
        continue
    if not np.all(artefacts[start:end, art_idx] == 0):
        continue

    segment = signal[start:end]
    _, psd = welch(
        segment,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
    )
    psds.append(psd)

# 5. Plotting Spindle Peak in Sigma Band (8 - 20 Hz)
if psds:
    mean_psd = np.mean(psds, axis=0)

    # Focus on the 8-20 Hz range for sleep spindles
    mask = (frequencies >= 8) & (frequencies <= 20)
    f_sub = frequencies[mask]
    p_sub = mean_psd[mask]

    # Find peak in the spindle frequency band (11-16 Hz)
    spindle_mask = (f_sub >= 11) & (f_sub <= 16)
    peak_idx = np.argmax(p_sub[spindle_mask])
    peak_freq = float(f_sub[spindle_mask][peak_idx])
    peak_power = float(p_sub[spindle_mask][peak_idx])

    plt.figure(figsize=(10, 5))
    plt.plot(f_sub, p_sub, linewidth=2, color="#1f77b4", label=f"{channel} PSD")

    # Highlight spindle range and detected peak
    plt.axvspan(11, 16, color="orange", alpha=0.15, label="Sleep Spindle Band (11–16 Hz)")
    plt.axvline(
        peak_freq,
        color="crimson",
        linestyle="--",
        alpha=0.7,
        label=f"Spindle Peak ({peak_freq:.2f} Hz)",
    )
    plt.annotate(
        f"Spindle Peak: {peak_freq:.2f} Hz\n({peak_power:.2f} " + r"$\mu V^2 / Hz$)",
        xy=(peak_freq, peak_power),
        xytext=(peak_freq + 1.2, peak_power * 0.9),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
        fontweight="bold",
    )

    plt.xlim(8, 20)
    plt.ylim(0, np.max(p_sub) * 1.15)
    plt.xticks(np.arange(8, 21, 1))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"PSD ($\mu V^2 / Hz$)")
    plt.title(f"{channel} — Clean NREM2 Power Spectral Density (Zoomed: 8–20 Hz)")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"psd_nrem2_zoomed_{channel}.png", dpi=150)
    plt.show()
else:
    print("No clean NREM2 epochs found.")
