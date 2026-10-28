import matplotlib.pyplot as plt
import numpy as np

from reading_working import load_recording
from spindles import detect_all_spindles

# ==============================================================================
# 1. Load Data & Annotations
# ==============================================================================
edf_name = "BB0008_SP2.edf"
channel = "C3"

print(f"Loading {edf_name} ({channel})...")
signal, fs, hypnogram, artefacts, art_idx = load_recording(edf_name, channel)

# Clean Stage 2 (NREM2) mask
is_n2 = (hypnogram == 2) & (artefacts[:, art_idx] == 0)
n2_minutes = np.sum(is_n2) / fs / 60

# ==============================================================================
# 2. Run Both Spindle Detectors
# ==============================================================================
print("Running detectors...")
custom_spindles, yasa_spindles = detect_all_spindles(
    signal=signal,
    fs=fs,
    hypnogram=hypnogram,
    artefacts=artefacts,
    art_idx=art_idx,
)

# ==============================================================================
# 3. Overlap & Agreement Analysis
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
print(f"{'Mean Frequency (Hz)':<25} | {np.mean([s['freq'] for s in custom_spindles]):<13.2f} | {np.mean([s['freq'] for s in yasa_spindles]):<10.2f}")
print("-" * 55)
print(f"Agreement: {overlap_count} of {len(custom_spindles)} Custom spindles ({overlap_pct:.1f}%) match YASA.")
print("=" * 55)

# ==============================================================================
# 4. Visual Comparison on a Shared Spindle
# ==============================================================================
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
    center_t = (c_sp["start"] + c_sp["end"]) / 2 / fs

    # 6-second window around the spindle
    t_start = center_t - 3.0
    t_end = center_t + 3.0
    idx_start = int(t_start * fs)
    idx_end = int(t_end * fs)
    t_rel = np.arange(idx_start, idx_end) / fs - t_start

    plt.figure(figsize=(12, 5))
    plt.plot(t_rel, signal[idx_start:idx_end], color="#333333", linewidth=1.0, label="Raw C3 EEG")

    # Custom detection span (Orange)
    c_t0 = max(0.0, c_sp["start"] / fs - t_start)
    c_t1 = min(6.0, c_sp["end"] / fs - t_start)
    plt.axvspan(c_t0, c_t1, color="orange", alpha=0.35, label=f"Custom SciPy ({c_sp['dur']:.2f}s)")

    # YASA detection span (Cyan outline)
    y_t0 = max(0.0, y_sp["start"] / fs - t_start)
    y_t1 = min(6.0, y_sp["end"] / fs - t_start)
    plt.axvspan(y_t0, y_t1, facecolor="none", edgecolor="#007acc", linewidth=2.0, linestyle="--", label=f"YASA ({y_sp['dur']:.2f}s)")

    plt.title(f"Sleep Spindle Detection Comparison (Time: {center_t:.1f}s)", fontsize=12, fontweight="bold")
    plt.xlabel("Time (seconds)", fontsize=11)
    plt.ylabel(r"Amplitude ($\mu V$)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig("spindle_comparison_yasa.png", dpi=150)
    print("\nSaved comparison figure: spindle_comparison_yasa.png")
    plt.show()
