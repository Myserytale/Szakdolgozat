"""
Reproduction of Figure 5: Infraslow Oscillations of Fast Spindle Activity
========================================================================
Reference:
  Lázár Z.I., Dijk D.J., Lázár A.S. (2019)
  "Infraslow oscillations in human sleep spindle activity"
  Journal of Neuroscience Methods 316: 22–34 (Figure 5a, b).

Exact Paper Specifications:
  - Channel: P3 (P3-A2 derivation)
  - Sleep Stage: NREM Stage 2
  - Spindles: Fast Sleep Spindles (13–15 Hz)
  - Signal: Binary square spindle train downsampled to 4 Hz
  - Window: 256s Hanning tapered window with 75% overlap
  - Reference: Shuffled surrogate (random permutation of on/off durations)
  - Outputs:
      (a) PSD of Binary Spindle Signal vs. Permutated Reference (with SEM band).
      (b) Excess Infra Power (EIP) curve with Inset showing Integrated EIP (0–35 mHz).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import yasa

from reading_working import DATA_DIR, HYPART_DIR, load_hypart, read_edf

# ==============================================================================
# 1. Configuration & Settings (Fig. 5 Parameters)
# ==============================================================================
channel = "P3"
art_channel = f"ArtiFFT {channel}-A2"
fast_spindle_range = (13.0, 15.0)  # Fast spindles criterion (13–15 Hz)

edf_path = DATA_DIR / "BB0008_SP2.edf"
hypart_path = HYPART_DIR / f"{edf_path.stem}.hypart"

# ==============================================================================
# 2. Load Signal & Detect Fast Spindles in Clean N2
# ==============================================================================
print(f"Loading {edf_path.name} (Channel: {channel}-A2)...")
edf_data = read_edf(edf_path, channels=[channel])
signal = edf_data["signals"][channel]
fs = int(edf_data["sample_rates"][0])

hypnogram, artefacts, art_channels = load_hypart(hypart_path)
art_idx = art_channels.index(art_channel)
is_n2 = (hypnogram == 2) & (artefacts[:, art_idx] == 0)

print("Detecting fast spindles (13–15 Hz) in clean NREM2...")
sp_yasa = yasa.spindles_detect(
    signal,
    fs,
    ch_names=[channel],
    hypno=hypnogram,
    include=(2,),
    freq_sp=(11, 16),
)
df_yasa = sp_yasa.summary()

fast_spindles = []
if df_yasa is not None and len(df_yasa) > 0:
    for _, row in df_yasa.iterrows():
        s_idx = int(row["Start"] * fs)
        e_idx = int((row["Start"] + row["Duration"]) * fs)
        # Verify clean artifact status and fast spindle frequency (13–15 Hz)
        if np.all(artefacts[s_idx:e_idx, art_idx] == 0):
            if fast_spindle_range[0] <= row["Frequency"] <= fast_spindle_range[1]:
                fast_spindles.append({
                    "start": s_idx,
                    "end": e_idx,
                    "dur": float(row["Duration"]),
                    "freq": float(row["Frequency"]),
                })

print(f"Found {len(fast_spindles)} clean fast spindles on {channel}-A2.")

# ==============================================================================
# 3. Construct 4 Hz Binary Spindle Signal & Extract Segments (>= 256s)
# ==============================================================================
fs_down = 4  # 4 Hz as specified in Section 2.2.4
factor = fs // fs_down
n_down = len(signal) // factor

bin_signal = np.zeros(n_down, dtype=np.float32)
for sp in fast_spindles:
    s_d = sp["start"] // factor
    e_d = sp["end"] // factor
    bin_signal[s_d:e_d] = 1.0

is_n2_down = is_n2[::factor][:n_down]

# Welch parameters: 256s window, 75% overlap
nperseg = 256 * fs_down  # 1024 samples
noverlap = int(nperseg * 0.75)  # 768 samples

diff = np.diff(np.concatenate(([0], is_n2_down.astype(np.int8), [0])))
starts = np.where(diff == 1)[0]
ends = np.where(diff == -1)[0]


def shuffle_binary_train(arr: np.ndarray) -> np.ndarray:
    """Randomly permutes spindle durations and inter-spindle intervals (surrogate null)."""
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
# 4. Spectral Estimation: Original vs. Shuffled Surrogates
# ==============================================================================
print(f"Computing Welch PSD on clean contiguous N2 segments (window={nperseg/fs_down:.0f}s)...")
orig_psds = []
perm_psds = []
n_shuffles = 30

for s, e in zip(starts, ends):
    # Only keep contiguous segments >= 256s that contain spindles
    if (e - s) >= nperseg and np.sum(bin_signal[s:e]) > 0:
        seg = bin_signal[s:e]

        # Original signal PSD
        f, p_orig = welch(seg, fs=fs_down, window="hann", nperseg=nperseg, noverlap=noverlap)
        orig_psds.append(p_orig)

        # Average PSD across 30 random permutations of this segment
        shuff_ps = [
            welch(shuffle_binary_train(seg), fs=fs_down, window="hann", nperseg=nperseg, noverlap=noverlap)[1]
            for _ in range(n_shuffles)
        ]
        perm_psds.append(np.mean(shuff_ps, axis=0))

orig_psds = np.array(orig_psds)
perm_psds = np.array(perm_psds)
n_segs = len(orig_psds)

print(f"Analyzed {n_segs} contiguous segments.")

# Mean across segments
mean_orig = np.mean(orig_psds, axis=0)
mean_perm = np.mean(perm_psds, axis=0)

# Excess Infra Power (EIP)
mean_eip = (mean_orig / mean_perm) - 1.0

# ==============================================================================
# 5. Plotting Figure 5: (a) PSD Comparison & (b) Excess Infra Power
# ==============================================================================
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5))

# ------------------------------------------------------------------------------
# Panel (a): Log-Log PSD of Spindle Signal vs. Reference Permutated Signal
# ------------------------------------------------------------------------------
mask_a = (f >= 0.0039) & (f <= 0.40)
f_a = f[mask_a]
m_orig_a = mean_orig[mask_a]
m_perm_a = mean_perm[mask_a]

ax_a.plot(f_a, m_orig_a, color="#1f77b4", linewidth=2.0, label="Spindle square signal")
ax_a.plot(f_a, m_perm_a, color="black", linestyle="--", linewidth=1.8, label="Permutated reference signal")

ax_a.set_xscale("log")
ax_a.set_yscale("log")
ax_a.set_xlim(0.0039, 0.40)
ax_a.set_xlabel("Frequency (Hz)", fontsize=11)
ax_a.set_ylabel(r"PSD ($\mathrm{units}^2 / \mathrm{Hz}$)", fontsize=11)
ax_a.set_title(f"(a) PSD of binary spindle signal ({channel}-A2, NREM2)", fontsize=12, fontweight="bold")
ax_a.grid(True, which="both", linestyle="--", alpha=0.3)
ax_a.legend(loc="lower left", frameon=True)

# ------------------------------------------------------------------------------
# Panel (b): Excess Infra Power (EIP) Curve & Inset
# ------------------------------------------------------------------------------
mask_b = (f >= 0.0039) & (f <= 0.040)
f_mhz_b = f[mask_b] * 1000  # Convert to mHz
m_eip_b = mean_eip[mask_b]

ax_b.plot(f_mhz_b, m_eip_b, color="#d62728", linewidth=2.0, label="Excess Infra Power (EIP)")
ax_b.axhline(0, color="black", linestyle=":", alpha=0.7)

# Highlight positive integrated EIP region (0 - 35 mHz)
integ_mask = (f_mhz_b >= 0.0) & (f_mhz_b <= 35.0)
int_eip_val = np.trapezoid(np.maximum(0, m_eip_b[integ_mask]), f_mhz_b[integ_mask])

ax_b.fill_between(
    f_mhz_b[integ_mask],
    0,
    np.maximum(0, m_eip_b[integ_mask]),
    color="orange",
    alpha=0.25,
    label=f"Integrated EIP (0–35 mHz: {int_eip_val:.2f})",
)

ax_b.set_xlim(3.9, 40.0)
ax_b.set_xlabel("Frequency (mHz)", fontsize=11)
ax_b.set_ylabel("Excess Infra Power (EIP)", fontsize=11)
ax_b.set_title(f"(b) Excess Infra Power (EIP) curve ({channel}-A2)", fontsize=12, fontweight="bold")
ax_b.grid(True, linestyle="--", alpha=0.3)
ax_b.legend(loc="upper right", frameon=True)

# Inset in Panel (b) matching Fig. 5b inset
ax_inset = ax_b.inset_axes([0.55, 0.52, 0.40, 0.40])
ax_inset.plot(f_mhz_b[integ_mask], m_eip_b[integ_mask], color="#d62728", linewidth=1.2)
ax_inset.fill_between(f_mhz_b[integ_mask], 0, np.maximum(0, m_eip_b[integ_mask]), color="orange", alpha=0.35)
ax_inset.axhline(0, color="black", linestyle=":", linewidth=0.8)
ax_inset.set_title("0–35 mHz Area", fontsize=8, fontweight="bold")
ax_inset.set_xlabel("mHz", fontsize=7)
ax_inset.tick_params(labelsize=7)
ax_inset.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("fig5_reproduction.png", dpi=150)
print("\nSuccessfully generated and saved: fig5_reproduction.png")

if __name__ == "__main__":
    plt.show()
