"""
Sleep Spindle Detection Comparison: Custom (SciPy) vs. YASA
===========================================================
This script benchmarks our custom Hilbert-threshold spindle detector
against YASA (Yet Another Spindle Algorithm, Raphael Vallat / UC Berkeley).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
import yasa

from reading_working import DATA_DIR, HYPART_DIR, load_hypart, read_edf

# ==============================================================================
# 1. Load Data & Annotations
# ==============================================================================
channel = "C3"
art_channel = f"ArtiFFT {channel}-A2"

edf_path = DATA_DIR / "BB0008_SP2.edf"
hypart_path = HYPART_DIR / f"{edf_path.stem}.hypart"

print(f"Loading {edf_path.name} ({channel})...")
edf_data = read_edf(edf_path, channels=[channel])
signal = edf_data["signals"][channel]
fs = int(edf_data["sample_rates"][0])

hypnogram, artefacts, art_channels = load_hypart(hypart_path)
art_idx = art_channels.index(art_channel)

# Clean Stage 2 (NREM2) mask
is_n2 = (hypnogram == 2) & (artefacts[:, art_idx] == 0)
n2_minutes = np.sum(is_n2) / fs / 60

# ==============================================================================
# 2. Detector A: Custom SciPy (Hilbert Envelope)
# ==============================================================================
print("Running Custom SciPy detector...")
sos = butter(4, [11, 16], btype="bandpass", fs=fs, output="sos")
sigma = sosfiltfilt(sos, signal)
envelope = np.abs(hilbert(sigma))

thresh = np.mean(envelope[is_n2]) + 1.5 * np.std(envelope[is_n2])

above = (envelope > thresh) & is_n2
diff = np.diff(above.astype(np.int8))
starts = np.where(diff == 1)[0] + 1
ends = np.where(diff == -1)[0] + 1
if above[0]:
    starts = np.insert(starts, 0, 0)
if above[-1]:
    ends = np.append(ends, len(above))

custom_spindles = []
for s, e in zip(starts, ends):
    dur = (e - s) / fs
    if 0.5 <= dur <= 2.5:
        custom_spindles.append({
            "start": s / fs,
            "end": e / fs,
            "dur": dur,
            "peak_amp": float(np.max(np.abs(signal[s:e]))),
        })

# ==============================================================================
# 3. Detector B: YASA (Lacourse et al., 2019)
# ==============================================================================
print("Running YASA detector...")
sp_yasa = yasa.spindles_detect(
    signal,
    fs,
    ch_names=[channel],
    hypno=hypnogram,
    include=(2,),
    freq_sp=(11, 16),
)
df_yasa = sp_yasa.summary()

# Filter YASA spindles for clean artifact epochs
yasa_spindles = []
if df_yasa is not None and len(df_yasa) > 0:
    for _, row in df_yasa.iterrows():
        s_idx = int(row["Start"] * fs)
        e_idx = int((row["Start"] + row["Duration"]) * fs)
        if np.all(artefacts[s_idx:e_idx, art_idx] == 0):
            yasa_spindles.append({
                "start": float(row["Start"]),
                "end": float(row["Start"] + row["Duration"]),
                "dur": float(row["Duration"]),
                "peak_amp": float(row["Amplitude"]),
                "freq": float(row["Frequency"]),
            })

# ==============================================================================
# 4. Overlap & Agreement Analysis
# ==============================================================================
overlap_count = 0
for c in custom_spindles:
    if any(max(c["start"], y["start"]) < min(c["end"], y["end"]) for y in yasa_spindles):
        overlap_count += 1

overlap_pct = (overlap_count / len(custom_spindles) * 100) if custom_spindles else 0

print("\n" + "=" * 55)
print(f" COMPARISON SUMMARY: Clean N2 Sleep ({n2_minutes:.1f} minutes)")
print("=" * 55)
print(f"{'Metric':<25} | {'Custom (SciPy)':<13} | {'YASA':<10}")
print("-" * 55)
print(f"{'Detected Spindles':<25} | {len(custom_spindles):<13} | {len(yasa_spindles):<10}")
print(f"{'Density (spindles/min)':<25} | {len(custom_spindles)/n2_minutes:<13.2f} | {len(yasa_spindles)/n2_minutes:<10.2f}")
print(f"{'Mean Duration (s)':<25} | {np.mean([s['dur'] for s in custom_spindles]):<13.2f} | {np.mean([s['dur'] for s in yasa_spindles]):<10.2f}")
print(f"{'Mean Peak Amplitude (µV)':<25} | {np.mean([s['peak_amp'] for s in custom_spindles]):<13.2f} | {np.mean([s['peak_amp'] for s in yasa_spindles]):<10.2f}")
print("-" * 55)
print(f"Agreement: {overlap_count} of {len(custom_spindles)} Custom spindles ({overlap_pct:.1f}%) match YASA.")
print("=" * 55)

# ==============================================================================
# 5. Visual Comparison on a Shared Spindle
# ==============================================================================
# Find a spindle detected by both
shared_sp = None
for c in custom_spindles:
    for y in yasa_spindles:
        if max(c["start"], y["start"]) < min(c["end"], y["end"]):
            shared_sp = (c, y)
            break
    if shared_sp:
        break

if shared_sp:
    c_sp, y_sp = shared_sp
    center_t = (c_sp["start"] + c_sp["end"]) / 2

    # 6-second window around the spindle
    t_start = center_t - 3.0
    t_end = center_t + 3.0
    idx_start = int(t_start * fs)
    idx_end = int(t_end * fs)

    t_rel = np.arange(idx_start, idx_end) / fs - t_start

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Subplot 1: Raw EEG with both detection overlays
    ax1.plot(t_rel, signal[idx_start:idx_end], color="#333333", linewidth=1.0, label="Raw C3 EEG")

    # Custom detection span (Orange)
    c_t0 = max(0.0, c_sp["start"] - t_start)
    c_t1 = min(6.0, c_sp["end"] - t_start)
    ax1.axvspan(c_t0, c_t1, color="orange", alpha=0.35, label=f"Custom SciPy ({c_sp['dur']:.2f}s)")

    # YASA detection span (Cyan outline)
    y_t0 = max(0.0, y_sp["start"] - t_start)
    y_t1 = min(6.0, y_sp["end"] - t_start)
    ax1.axvspan(y_t0, y_t1, facecolor="none", edgecolor="#007acc", linewidth=2.0, linestyle="--", label=f"YASA ({y_sp['dur']:.2f}s)")

    ax1.set_title(f"Sleep Spindle Detection Comparison (Time: {center_t:.1f}s)", fontsize=12, fontweight="bold")
    ax1.set_ylabel(r"Amplitude ($\mu V$)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper right", frameon=True)

    # Subplot 2: Filtered Sigma & Envelope
    ax2.plot(t_rel, sigma[idx_start:idx_end], color="#7f7f7f", linewidth=0.8, label="Filtered Sigma (11–16 Hz)")
    ax2.plot(t_rel, envelope[idx_start:idx_end], color="crimson", linewidth=1.5, label="Hilbert Envelope")
    ax2.axhline(thresh, color="black", linestyle=":", label=f"Threshold ({thresh:.1f} " + r"$\mu V$)")
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel(r"Amplitude ($\mu V$)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig("spindle_comparison_yasa.png", dpi=150)
    print("\nSaved comparison figure: spindle_comparison_yasa.png")
    plt.show()
