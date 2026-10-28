"""
Excess Infra Power (EIP) of Sleep Spindle Activity
==================================================
Based on:
  Lázár Z.I., Dijk D.J., Lázár A.S. (2019)
  "Infraslow oscillations in human sleep spindle activity"
  Journal of Neuroscience Methods 316: 22–34.

Method:
1. Reduce detected sleep spindles to a binary square signal (1 = Spindle, 0 = Non-spindle)
   downsampled to 4 Hz.
2. Select contiguous clean Stage 2 (N2) segments >= 256 seconds.
3. Generate surrogate reference signals by randomly shuffling spindle durations and
   inter-spindle intervals (eliminating temporal order while preserving event statistics).
4. Compute Welch PSD (256s window, 75% overlap) for Original vs. Permutated signals.
5. Calculate Excess Infra Power:
     EIP(f) = PSD_original(f) / PSD_permutated(f) - 1
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.signal.windows import gaussian

from reading_working import DATA_DIR, HYPART_DIR, load_hypart, read_edf
from spindles import detect_all_spindles

# ==============================================================================
# 1. Settings & Load Data
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
is_n2 = (hypnogram == 2) & (artefacts[:, art_idx] == 0)

# ==============================================================================
# 2. Spindle Detection
# ==============================================================================
print("Detecting sleep spindles...")
custom_sp, yasa_sp = detect_all_spindles(signal, fs, hypnogram, artefacts, art_idx)
# Using YASA detections (or custom_sp)
spindles = yasa_sp
print(f"Using {len(spindles)} spindles for Infraslow analysis.")

# ==============================================================================
# 3. Construct 4 Hz Binary Signal & Extract Contiguous N2 Segments (>= 256s)
# ==============================================================================
fs_down = 4  # Downsample binary signal to 4 Hz as in Lázár et al. (2019)
factor = fs // fs_down
n_down = len(signal) // factor

bin_signal = np.zeros(n_down, dtype=np.float32)
for sp in spindles:
    s_d = sp["start"] // factor
    e_d = sp["end"] // factor
    bin_signal[s_d:e_d] = 1.0

is_n2_down = is_n2[::factor][:n_down]

# FFT settings: 256s Hanning window with 75% overlap
nperseg = 256 * fs_down  # 1024 points
noverlap = int(nperseg * 0.75)  # 768 points

diff = np.diff(np.concatenate(([0], is_n2_down.astype(np.int8), [0])))
starts = np.where(diff == 1)[0]
ends = np.where(diff == -1)[0]


def shuffle_train(arr: np.ndarray) -> np.ndarray:
    """Randomly shuffles spindle durations and inter-spindle intervals (surrogate null)."""
    diff_arr = np.diff(np.concatenate(([0], arr, [0])))
    s = np.where(diff_arr == 1)[0]
    e = np.where(diff_arr == -1)[0]

    if len(s) <= 1:
        return arr.copy()

    sp_durs = e - s
    inter_durs = s[1:] - e[:-1]

    shuff_sp = np.random.permutation(sp_durs)
    shuff_in = np.random.permutation(inter_durs)

    reconstructed = [np.zeros(s[0], dtype=np.float32)]
    for i in range(len(shuff_sp)):
        reconstructed.append(np.ones(shuff_sp[i], dtype=np.float32))
        if i < len(shuff_in):
            reconstructed.append(np.zeros(shuff_in[i], dtype=np.float32))
    reconstructed.append(np.zeros(len(arr) - e[-1], dtype=np.float32))
    return np.concatenate(reconstructed)


# ==============================================================================
# 4. Compute Welch PSD: Original vs. Permutated Surrogates
# ==============================================================================
print(f"Computing Welch PSD on clean contiguous N2 segments (window={nperseg/fs_down:.0f}s)...")
orig_psds = []
perm_psds = []
clean_segs = []
n_shuffles = 30  # Number of random permutations per segment for baseline

for s, e in zip(starts, ends):
    if (e - s) >= nperseg and np.sum(bin_signal[s:e]) > 0:
        seg = bin_signal[s:e]
        clean_segs.append(seg)

        # Original signal PSD
        f, p_orig = welch(seg, fs=fs_down, window="hann", nperseg=nperseg, noverlap=noverlap)
        orig_psds.append(p_orig)

        # Permutated surrogate PSD (averaged across shuffles)
        shuff_ps = [
            welch(shuffle_train(seg), fs=fs_down, window="hann", nperseg=nperseg, noverlap=noverlap)[1]
            for _ in range(n_shuffles)
        ]
        perm_psds.append(np.mean(shuff_ps, axis=0))

print(f"Analyzed {len(orig_psds)} contiguous N2 segments ({sum(len(x) for x in clean_segs)/fs_down/60:.1f} mins total).")

mean_orig = np.mean(orig_psds, axis=0)
mean_perm = np.mean(perm_psds, axis=0)

# Excess Infra Power (EIP) formula
eip = (mean_orig / mean_perm) - 1.0

# Infraslow range: 3 to 100 mHz
mask_spec = (f >= 0.003) & (f <= 0.10)
f_mhz = f[mask_spec] * 1000
eip_spec = eip[mask_spec]

# Detect Peak Infraslow Frequency (between 5 and 50 mHz)
iso_band = (f_mhz >= 5.0) & (f_mhz <= 50.0)
peak_idx = np.argmax(eip_spec[iso_band])
peak_f = f_mhz[iso_band][peak_idx]
peak_val = eip_spec[iso_band][peak_idx]
period_sec = 1000.0 / peak_f

# Integrated EIP (0 - 35 mHz)
integ_band = (f_mhz >= 0.0) & (f_mhz <= 35.0)
int_eip = np.trapezoid(np.maximum(0, eip_spec[integ_band]), f_mhz[integ_band])

print("\n--- Infraslow Oscillation (ISO) Results ---")
print(f"Peak ISO Frequency : {peak_f:.2f} mHz")
print(f"Peak ISO Period    : {period_sec:.1f} seconds")
print(f"Peak Excess Power  : {peak_val:.3f}")
print(f"Integrated EIP     : {int_eip:.2f} mHz")

# ==============================================================================
# 5. Visualization (Spectral & Time-Domain)
# ==============================================================================
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

# Panel A: Power Spectral Density Comparison (Log Scale)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(f_mhz, mean_orig[mask_spec], color="#1f77b4", linewidth=2.0, label="Original Spindle Train")
ax1.plot(f_mhz, mean_perm[mask_spec], color="#7f7f7f", linestyle="--", linewidth=1.8, label="Permutated Surrogate (Null)")
ax1.set_title("Power Spectral Density of Binary Spindle Signal", fontsize=11, fontweight="bold")
ax1.set_xlabel("Frequency (mHz)", fontsize=10)
ax1.set_ylabel(r"PSD ($\mathrm{units}^2 / \mathrm{Hz}$)", fontsize=10)
ax1.set_yscale("log")
ax1.grid(True, which="both", linestyle="--", alpha=0.3)
ax1.legend(loc="upper right", frameon=True)

# Panel B: Excess Infra Power (EIP) Curve
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(f_mhz, eip_spec, color="#d62728", linewidth=2.0, label=r"$\mathrm{EIP} = \frac{\mathrm{PSD}_{\mathrm{orig}}}{\mathrm{PSD}_{\mathrm{perm}}} - 1$")
ax2.axhline(0, color="black", linestyle=":", alpha=0.7)

# Shaded integrated EIP area (0 - 35 mHz)
ax2.fill_between(
    f_mhz[integ_band],
    0,
    np.maximum(0, eip_spec[integ_band]),
    color="orange",
    alpha=0.25,
    label=f"Integrated EIP (0–35 mHz: {int_eip:.1f})",
)

# Annotate peak
ax2.plot(peak_f, peak_val, marker="o", markersize=6, color="darkred")
ax2.annotate(
    f"Peak ISO: {peak_f:.1f} mHz\n(Period: {period_sec:.1f}s)",
    xy=(peak_f, peak_val),
    xytext=(peak_f + 8, peak_val * 0.9 + 0.05),
    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="darkred", alpha=0.9),
)

ax2.set_title(f"Excess Infra Power (EIP) — {channel} NREM2", fontsize=11, fontweight="bold")
ax2.set_xlabel("Frequency (mHz)", fontsize=10)
ax2.set_ylabel("Excess Power (relative)", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.legend(loc="upper right", frameon=True)

# Panel C: Representative 500s Spindle Train with Smoothed Infraslow Rate
ax3 = fig.add_subplot(gs[1, :])
sample_seg = clean_segs[0][:500 * fs_down]  # 500 seconds
t_seg = np.arange(len(sample_seg)) / fs_down

# Gaussian smoothing window (10s std) to show the slow undulating rate
w = gaussian(20 * fs_down * 3, std=10 * fs_down)
w /= np.sum(w)
smooth_rate = np.convolve(sample_seg, w, mode="same") * 60  # Rate in spindles/min

ax3.fill_between(t_seg, 0, sample_seg, color="#1f77b4", alpha=0.35, step="mid", label="Spindle Event (1 = ON, 0 = OFF)")
ax3_twin = ax3.twinx()
ax3_twin.plot(t_seg, smooth_rate, color="crimson", linewidth=2.0, label="Infra-slow Spindle Rate (spindles/min)")
ax3_twin.set_ylabel("Spindle Rate (/min)", color="crimson", fontsize=10)
ax3_twin.tick_params(axis="y", labelcolor="crimson")

ax3.set_title("Representative 500s Spindle Train showing Infraslow Undulation (~20–50s Period)", fontsize=11, fontweight="bold")
ax3.set_xlabel("Time in Segment (seconds)", fontsize=10)
ax3.set_ylabel("Spindle Binary State", color="#1f77b4", fontsize=10)
ax3.set_ylim(-0.1, 1.3)
ax3.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("excess_infra_power.png", dpi=150)
print("\nSaved figure: excess_infra_power.png")

if __name__ == "__main__":
    plt.show()
