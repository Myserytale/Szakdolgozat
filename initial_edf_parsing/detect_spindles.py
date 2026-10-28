import matplotlib.pyplot as plt
import numpy as np

from reading_working import DATA_DIR, HYPART_DIR, load_hypart, read_edf
from spindles import detect_custom_spindles

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
is_n2 = (hypnogram == 2) & (artefacts[:, art_idx] == 0)

# ==============================================================================
# 2. Run Spindle Detection
# ==============================================================================
spindles = detect_custom_spindles(signal=signal, fs=fs, is_n2=is_n2)

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
# 3. Visualize a Representative Detected Spindle
# ==============================================================================
if spindles:
    sample_sp = spindles[0]
    center_sample = (sample_sp["start"] + sample_sp["end"]) // 2

    # 5-second window around the spindle
    window_half = int(2.5 * fs)
    win_start = max(center_sample - window_half, 0)
    win_end = min(center_sample + window_half, len(signal))

    t = np.arange(win_start, win_end) / fs
    t_relative = t - t[0]

    sp_t_start = (sample_sp["start"] - win_start) / fs
    sp_t_end = (sample_sp["end"] - win_start) / fs

    plt.figure(figsize=(12, 4))
    plt.plot(t_relative, signal[win_start:win_end], color="#1f77b4", linewidth=1.0, label="Raw C3")
    plt.axvspan(sp_t_start, sp_t_end, color="orange", alpha=0.35, label=f"Detected Spindle ({sample_sp['dur']:.2f}s)")
    plt.title(
        f"Detected Sleep Spindle (Duration: {sample_sp['dur']:.2f}s, Peak: {sample_sp['peak_amp']:.1f} " + r"$\mu V$)",
        fontweight="bold",
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel(r"Amplitude ($\mu V$)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("detected_spindle_example.png", dpi=150)
    print("Saved preview figure: detected_spindle_example.png")
    plt.show()
