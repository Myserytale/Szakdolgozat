import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from reading_working import load_recording

# 1. Load data and annotations (only C3 is read, instead of all 31 channels)
c3_signal, fs, hypnogram, artefacts, art_idx = load_recording("BB0008_SP2.edf", "C3")

# 2. Settings
epoch_sec = 10
epoch_len = fs * epoch_sec
nperseg = fs * 4  # 4-second Welch window gives 0.25 Hz resolution

# One-sided frequencies (matches welch output length: nperseg // 2 + 1)
frequencies = np.fft.rfftfreq(nperseg, 1.0 / fs)

stages = {
    "Stage 2 (N2)": [2],
    "Stage 3 (N3)": [3],
    "Stage 4 (N4)": [4],
}
colors = {
    "Stage 2 (N2)": "#1f77b4",
    "Stage 3 (N3)": "#2ca02c",
    "Stage 4 (N4)": "#d62728",
}

stage_psds = {s: [] for s in stages}
sample_epochs = {}

# 3. Extract clean epochs and compute PSDs
n_epochs = len(c3_signal) // epoch_len

for i in range(n_epochs):
    start = i * epoch_len
    end = start + epoch_len

    # Skip epochs with any artifacts
    if not np.all(artefacts[start:end, art_idx] <= 4):
        continue

    current_stage = hypnogram[start:end]

    for name, stage_vals in stages.items():
        if np.all(np.isin(current_stage, stage_vals)):
            segment = c3_signal[start:end]
            _, psd = welch(
                segment,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=nperseg // 2,
                scaling="density",
            )
            stage_psds[name].append(psd)

            # Keep the first clean epoch for time-domain plotting
            if name not in sample_epochs:
                sample_epochs[name] = segment

# 4. Plot PSD Comparison
plt.figure(figsize=(14, 6))

# Subplot A: Full spectrum on logarithmic scale (0.5 - 25 Hz)
plt.subplot(1, 2, 1)
for name, psd_list in stage_psds.items():
    if psd_list:
        mean_psd = np.mean(psd_list, axis=0)
        mask = (frequencies >= 0.5) & (frequencies <= 25)
        plt.semilogy(
            frequencies[mask],
            mean_psd[mask],
            label=f"{name} (n={len(psd_list)})",
            color=colors[name],
            linewidth=2,
        )

plt.axvspan(0.5, 4, color="gray", alpha=0.15, label="Delta (0.5–4 Hz)")
plt.axvspan(11, 16, color="orange", alpha=0.15, label="Sigma (11–16 Hz)")
plt.title("Full Spectrum Comparison (Log Scale)")
plt.xlabel("Frequency (Hz)")
plt.ylabel(r"PSD (Log Scale, $\mu V^2 / Hz$)")
plt.legend()
plt.grid(True, which="both", alpha=0.3)

# Subplot B: Zoomed into Sigma Band (8 - 20 Hz, Linear Scale)
plt.subplot(1, 2, 2)
for name, psd_list in stage_psds.items():
    if psd_list:
        mean_psd = np.mean(psd_list, axis=0)
        mask = (frequencies >= 8) & (frequencies <= 20)
        plt.plot(
            frequencies[mask],
            mean_psd[mask],
            label=name,
            color=colors[name],
            linewidth=2,
        )

plt.axvspan(11, 16, color="orange", alpha=0.15, label="Sigma Band (11–16 Hz)")
plt.title("Zoomed Sigma Band (Sleep Spindles)")
plt.xlabel("Frequency (Hz)")
plt.ylabel(r"PSD ($\mu V^2 / Hz$)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("psd_stage_comparison.png", dpi=150)

# 5. Plot Time-Domain Traces
if sample_epochs:
    t = np.linspace(0, epoch_sec, epoch_len)
    fig, axes = plt.subplots(
        len(sample_epochs), 1, figsize=(12, 6), sharex=True, sharey=True
    )

    if len(sample_epochs) == 1:
        axes = [axes]

    for ax, (name, seg) in zip(axes, sample_epochs.items()):
        ax.plot(t, seg, color=colors[name], linewidth=0.8)
        ax.set_title(f"Clean 10s Raw Trace — {name}")
        ax.set_ylabel(r"Amplitude ($\mu V$)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig("traces_stage_comparison.png", dpi=150)

plt.show()
