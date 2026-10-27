import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import welch, resample_poly

from reading_working import *

edf_path = DATA_DIR / "BB0008_SP2.edf"
hypart_path = HYPART_DIR / f"{edf_path.stem}.hypart"

with pyedflib.EdfReader(str(edf_path)) as edf:
    labels = edf.getSignalLabels()
    c3_idx = labels.index("C3")
    c3_signal = edf.readSignal(c3_idx)
    fs = int(edf.getSampleFrequency(c3_idx))

hypnogram, artefacts, art_channels = load_hypart(hypart_path)
P3_art_index = art_channels.index("ArtiFFT P3-A2")

seconds = 4
epoch_seconds = 10
epoch_samples = fs * epoch_seconds
nperseg = fs * seconds

frequencies = np.fft.rfftfreq(nperseg, 1.0 / fs)
psds = []

n_epochs = len(c3_signal) // epoch_samples

for i in range(n_epochs):
    start = i * epoch_samples
    end = start + epoch_samples

    if not np.all(hypnogram[start:end] == 2):
        continue
    if not np.all(artefacts[start:end, P3_art_index] == 0):
        continue

    segment = c3_signal[start:end]
    _, psd = welch(
        segment,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
    )

    psds.append(psd)

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
    plt.plot(f_sub, p_sub, linewidth=2, color="#1f77b4", label="C3 PSD")

    # Highlight spindle range and peak
    plt.axvspan(
        11, 16, color="orange", alpha=0.15, label="Sleep Spindle Band (11–16 Hz)"
    )
    plt.axvline(
        peak_freq,
        color="crimson",
        linestyle="--",
        alpha=0.7,
        label=f"Spindle Peak ({peak_freq:.2f} Hz)",
    )
    plt.annotate(
        f"Spindle Peak: {peak_freq:.2f} Hz\n({peak_power:.2f} signal units²/Hz)",
        xy=(peak_freq, peak_power),
        xytext=(peak_freq + 1.2, peak_power + 0.3),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
        fontweight="bold",
    )

    plt.xlim(8, 20)
    plt.ylim(0, np.max(p_sub) * 1.15)
    plt.xticks(np.arange(8, 21, 1))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (signal units²/Hz)")
    plt.title("P3 — Clean NREM2 Power Spectral Density (Zoomed: 8–20 Hz)")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig("psd_nrem2_zoomedP3-A2.png", dpi=150)
    plt.show()
else:
    print("No clean NREM2 epochs found.")
