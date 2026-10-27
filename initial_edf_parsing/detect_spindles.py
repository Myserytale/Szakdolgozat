import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from reading_working import DATA_DIR, HYPART_DIR, load_hypart, read_edf

# ==============================================================================
# 1. Load Data & Annotations
# ==============================================================================
channel = "C3"
art_channel = f"ArtiFFT {channel}-A2"

edf_path = DATA_DIR / "BB0008_SP2.edf"
hypart_path = HYPART_DIR / f"{edf_path.stem}.hypart"

edf_data = read_edf(edf_path, channels=[channel])
signal = edf_data["signals"][channel]
fs = int(edf_data["sample_rates"][0])

hypnogram, artefacts, art_channels = load_hypart(hypart_path)
art_idx = art_channels.index(art_channel)

# Clean Stage 2 (NREM2) mask
is_n2 = (hypnogram == 2) & (artefacts[:, art_idx] == 0)

# ==============================================================================
# 2. Sigma Bandpass Filter (11 - 16 Hz) & Envelope Extraction
# ==============================================================================
# 4th-order zero-phase Butterworth filter
sos = butter(4, [11, 16], btype="bandpass", fs=fs, output="sos")
sigma = sosfiltfilt(sos, signal)

# Amplitude envelope via Hilbert transform
envelope = np.abs(hilbert(sigma))

# ==============================================================================
# 3. Detection Thresholding (AASM: >= 0.5s duration)
# ==============================================================================
# Threshold based on clean N2 baseline (Mean + 1.5 * Std)
mean_env = np.mean(envelope[is_n2])
std_env = np.std(envelope[is_n2])
thresh = mean_env + 1.5 * std_env

# Find segments where envelope exceeds threshold during clean N2
above_thresh = (envelope > thresh) & is_n2
diff = np.diff(above_thresh.astype(np.int8))
starts = np.where(diff == 1)[0] + 1
ends = np.where(diff == -1)[0] + 1

if above_thresh[0]:
    starts = np.insert(starts, 0, 0)
if above_thresh[-1]:
    ends = np.append(ends, len(above_thresh))

# Spindle duration limits: 0.5s to 2.5s (AASM clinical standard)
min_samples = int(0.5 * fs)
max_samples = int(2.5 * fs)

spindles = []
for s, e in zip(starts, ends):
    dur = (e - s) / fs
    if min_samples <= (e - s) <= max_samples:
        peak_amp = float(np.max(np.abs(signal[s:e])))
        spindles.append({"start": s, "end": e, "dur": dur, "peak_amp": peak_amp})

# ==============================================================================
# 4. Spindle Statistics
# ==============================================================================
n2_minutes = np.sum(is_n2) / fs / 60
density = len(spindles) / n2_minutes if n2_minutes > 0 else 0
mean_dur = np.mean([sp["dur"] for sp in spindles]) if spindles else 0
mean_amp = np.mean([sp["peak_amp"] for sp in spindles]) if spindles else 0

print(f"\n--- Spindle Detection Results ({channel}) ---")
print(f"Total N2 Sleep Analyzed : {n2_minutes:.1f} minutes")
print(f"Detected Spindles       : {len(spindles)}")
print(f"Spindle Density         : {density:.2f} spindles/minute")
print(f"Average Duration        : {mean_dur:.2f} seconds")
print(f"Average Peak Amplitude  : {mean_amp:.2f} µV")

# ==============================================================================
# 5. Visualize a Representative Detected Spindle
# ==============================================================================
if spindles:
    # Pick a clean detected spindle (e.g. the first one)
    sample_sp = spindles[0]
    center_sample = (sample_sp["start"] + sample_sp["end"]) // 2

    # Show a 5-second window centered around the spindle
    window_half = int(2.5 * fs)
    win_start = max(center_sample - window_half, 0)
    win_end = min(center_sample + window_half, len(signal))

    t = np.arange(win_start, win_end) / fs
    t_relative = t - t[0]

    # Spindle relative time
    sp_t_start = (sample_sp["start"] - win_start) / fs
    sp_t_end = (sample_sp["end"] - win_start) / fs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Subplot 1: Raw EEG trace
    ax1.plot(t_relative, signal[win_start:win_end], color="#1f77b4", linewidth=1.0, label="Raw C3")
    ax1.axvspan(sp_t_start, sp_t_end, color="orange", alpha=0.3, label="Detected Spindle")
    ax1.set_title(
        f"Detected Sleep Spindle (Duration: {sample_sp['dur']:.2f}s, Peak: {sample_sp['peak_amp']:.1f} " + r"$\mu V$)",
        fontweight="bold",
    )
    ax1.set_ylabel(r"Amplitude ($\mu V$)")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper right")

    # Subplot 2: Filtered Sigma (11-16 Hz) and Envelope
    ax2.plot(t_relative, sigma[win_start:win_end], color="#7f7f7f", linewidth=0.8, label="Sigma (11–16 Hz)")
    ax2.plot(t_relative, envelope[win_start:win_end], color="crimson", linewidth=1.5, label="Envelope")
    ax2.axhline(thresh, color="black", linestyle="--", label=f"Threshold ({thresh:.1f} " + r"$\mu V$)")
    ax2.axvspan(sp_t_start, sp_t_end, color="orange", alpha=0.3)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel(r"Amplitude ($\mu V$)")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("detected_spindle_example.png", dpi=150)
    print("Saved preview figure: detected_spindle_example.png")
    plt.show()
