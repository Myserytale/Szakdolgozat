"""
Excess Infra Power (EIP) of all spindles on C3, NREM2
=====================================================
Same method as fsa.py (see infraslow.py), plus:
  - automatic peak ISO frequency / period
  - a 500 s time trace showing the slowly changing spindle rate
"""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import gaussian

_PKG_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_PKG_DIR / "core"), str(_PKG_DIR / "spindles")]

from infraslow import compute_eip, integrated_eip
from reading_working import load_recording
from spindles import detect_yasa_spindles

# 1. Settings
edf_name = "BB0008_SP2.edf"
channel = "C3"
fs_down = 4  # rate of the binary spindle signal

# 2. Load data & detect spindles
signal, fs, hypnogram, artefacts, art_idx = load_recording(edf_name, channel)
is_n2 = (hypnogram == 2) & (artefacts[:, art_idx] == 0)

spindles = detect_yasa_spindles(signal, fs, hypnogram, artefacts, art_idx)
print(f"{len(spindles)} clean spindles on {channel}")

# 3. Excess infra power
res = compute_eip(spindles, is_n2, fs, fs_down=fs_down)
f, eip = res["f"], res["eip"]
segments = res["segments"]
print(f"Analysed {len(segments)} segments ({sum(map(len, segments)) / fs_down / 60:.1f} min)")

# Peak in the 5–50 mHz infraslow band
f_mhz = f * 1000
iso = (f_mhz >= 5) & (f_mhz <= 50)
peak_f = f_mhz[iso][np.argmax(eip[iso])]
peak_val = eip[iso].max()
int_eip = integrated_eip(f, eip)

print(f"Peak ISO: {peak_f:.2f} mHz (period {1000 / peak_f:.1f} s) | "
      f"peak EIP {peak_val:.3f} | integrated EIP {int_eip:.2f}")

# 4. Plot
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
show = (f_mhz >= 3) & (f_mhz <= 100)

# Panel A: PSD real vs. permutated
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogy(f_mhz[show], res["psd_orig"][show], color="#1f77b4", linewidth=2.0, label="Original spindle train")
ax1.semilogy(f_mhz[show], res["psd_perm"][show], color="#7f7f7f", linestyle="--", linewidth=1.8, label="Permutated surrogate")
ax1.set_title("PSD of binary spindle signal", fontweight="bold")
ax1.set_xlabel("Frequency (mHz)")
ax1.set_ylabel(r"PSD ($\mathrm{units}^2 / \mathrm{Hz}$)")
ax1.grid(True, which="both", linestyle="--", alpha=0.3)
ax1.legend(loc="upper right")

# Panel B: EIP with peak
ax2 = fig.add_subplot(gs[0, 1])
integ = show & (f_mhz <= 35)
ax2.plot(f_mhz[show], eip[show], color="#d62728", linewidth=2.0, label="EIP")
ax2.axhline(0, color="black", linestyle=":", alpha=0.7)
ax2.fill_between(f_mhz[integ], 0, np.maximum(0, eip[integ]), color="orange", alpha=0.25,
                 label=f"Integrated EIP (0–35 mHz: {int_eip:.1f})")
ax2.plot(peak_f, peak_val, marker="o", color="darkred")
ax2.annotate(f"Peak ISO: {peak_f:.1f} mHz\n(period {1000 / peak_f:.1f} s)",
             xy=(peak_f, peak_val), xytext=(peak_f + 8, peak_val * 0.9 + 0.05),
             arrowprops=dict(arrowstyle="->", color="darkred"), fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="darkred"))
ax2.set_title(f"Excess Infra Power — {channel} NREM2", fontweight="bold")
ax2.set_xlabel("Frequency (mHz)")
ax2.set_ylabel("Excess power (relative)")
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.legend(loc="upper right")

# Panel C: 500 s of the spindle train + smoothed spindle rate
ax3 = fig.add_subplot(gs[1, :])
seg = segments[0][: 500 * fs_down]
t = np.arange(len(seg)) / fs_down
kernel = gaussian(60 * fs_down, std=10 * fs_down)  # 10 s Gaussian smoothing
onsets = np.diff(seg, prepend=0) > 0  # True at the first sample of each spindle
rate = np.convolve(onsets, kernel / kernel.sum(), mode="same") * fs_down * 60  # spindles per minute

ax3.fill_between(t, 0, seg, color="#1f77b4", alpha=0.35, step="mid", label="Spindle (1 = on)")
ax3.set_ylim(-0.1, 1.3)
ax3.set_xlabel("Time in segment (s)")
ax3.set_ylabel("Spindle state", color="#1f77b4")
ax3_r = ax3.twinx()
ax3_r.plot(t, rate, color="crimson", linewidth=2.0)
ax3_r.set_ylabel("Spindle rate (/min)", color="crimson")
ax3.set_title("500 s spindle train with smoothed spindle rate", fontweight="bold")
ax3.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("excess_infra_power.png", dpi=150)
print("Saved excess_infra_power.png")
plt.show()
