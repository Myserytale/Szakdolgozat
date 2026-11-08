"""
Reproduction of Figure 5a of Lázár, Dijk & Lázár (2019)
=======================================================
Infraslow oscillation of fast spindle activity (13–15 Hz), P3-A2, NREM2:
PSD of the binary spindle signal vs. the permutated reference signal,
shown for one specific recording (left) and averaged across all EDF files (right).
"""

import matplotlib.pyplot as plt
import numpy as np

from infraslow import compute_eip
from reading_working import edf_files, load_recording
from spindles import detect_yasa_spindles

# 1. Settings
single_edf = "BB0008_SP3.edf"
channel = "P3"
fast_range = (8.0, 15.0)
freq_sp = (min(11.0, fast_range[0]), max(16.0, fast_range[1]))
plot_fmin, plot_fmax = (
    0.0039,
    2.0,
)  # Wider frequency range (up to 2 Hz Nyquist at 4 Hz sampling)

# 2. Compute EIP for every EDF file in the dataset
results = {}
for edf_path in edf_files:
    name = edf_path.name
    signal, fs, hypnogram, artefacts, art_idx = load_recording(name, channel)
    is_n2 = (hypnogram == 2) & (artefacts[:, art_idx] == 0)

    spindles = detect_yasa_spindles(
        signal, fs, hypnogram, artefacts, art_idx, freq_sp=freq_sp
    )
    fast = [sp for sp in spindles if fast_range[0] <= sp["freq"] <= fast_range[1]]

    try:
        res = compute_eip(fast, is_n2, fs)
        results[name] = res
        print(f"{name}: {len(fast)} spindles | {len(res['segments'])} segments")
    except ValueError as e:
        print(f"{name}: skipped ({e})")

# 3. Single-recording and aggregate average PSDs
single_res = results[single_edf]
f = single_res["f"]

avg_psd_orig = np.mean([r["psd_orig"] for r in results.values()], axis=0)
avg_psd_perm = np.mean([r["psd_perm"] for r in results.values()], axis=0)

# 4. Plot: Left = single recording, Right = aggregate average across all EDFs
fig, (ax_single, ax_avg) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
mask = (f >= plot_fmin) & (f <= plot_fmax)

# Left: Single EDF
ax_single.loglog(
    f[mask],
    single_res["psd_orig"][mask],
    color="#1f77b4",
    linewidth=2.0,
    label="Spindle square signal",
)
ax_single.loglog(
    f[mask],
    single_res["psd_perm"][mask],
    color="black",
    linestyle="--",
    linewidth=1.8,
    label="Permutated reference signal",
)
ax_single.set_xlim(plot_fmin, plot_fmax)
ax_single.set_xlabel("Frequency (Hz)")
ax_single.set_ylabel(r"PSD ($\mathrm{units}^2 / \mathrm{Hz}$)")
ax_single.set_title(
    f"Single Recording: {single_edf} ({channel}-A2, NREM2)", fontweight="bold"
)
ax_single.grid(True, which="both", linestyle="--", alpha=0.3)
ax_single.legend(loc="lower left")

# Right: Aggregate Average across all EDFs
ax_avg.loglog(
    f[mask],
    avg_psd_orig[mask],
    color="#1f77b4",
    linewidth=2.0,
    label="Spindle square signal (mean)",
)
ax_avg.loglog(
    f[mask],
    avg_psd_perm[mask],
    color="black",
    linestyle="--",
    linewidth=1.8,
    label="Permutated reference signal (mean)",
)
ax_avg.set_xlim(plot_fmin, plot_fmax)
ax_avg.set_xlabel("Frequency (Hz)")
ax_avg.set_title(
    f"Aggregate Average ({len(results)} recordings, {channel}-A2, NREM2)",
    fontweight="bold",
)
ax_avg.grid(True, which="both", linestyle="--", alpha=0.3)
ax_avg.legend(loc="lower left")

plt.tight_layout()
plt.savefig("fig5_reproduction.png", dpi=150)
print("Saved fig5_reproduction.png")
plt.show()
