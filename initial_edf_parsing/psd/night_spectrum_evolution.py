"""
Full-Night EEG Spectral Evolution & Slow-Wave / Spindle Relationship
====================================================================
Computes 30-second epoch Welch PSDs across the Lights-Off -> Lights-On sleep window
and visualizes:
1. Hypnogram (Wake, REM, N1, N2, N3, N4).
2. Full-night spectrogram (0.5-20 Hz, dB scale), interpolating only short (<=2 epoch)
   artifact gaps and masking extended Wake/movement blocks in neutral dark gray.
3. Slow Waves (0.75-4 Hz) vs. Sigma Spindles (11-16 Hz) across the night.
4. 1st-half vs. 2nd-half clean NREM2 PSD comparison (0.5-25 Hz log & 8-20 Hz zoomed Sigma).
"""

from datetime import datetime
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

_CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

from reading_working import DATA_DIR, PROJECT_ROOT, load_recording

# ==============================================================================
# 1. Settings & Load Data (Cropped to Lights-Off -> Lights-On)
# ==============================================================================
edf_name = "BB0008_SP3.edf"
channel = "C3"
epoch_sec = 30
welch_sec = 4
max_interp_gap_epochs = 2


def _parse_dt(date_str: str, time_str: str) -> datetime:
    text = f"{date_str.strip()} {time_str.strip()}"
    for fmt in ("%d-%b-%y %H:%M:%S", "%d-%b-%y %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date/time format: {text!r}")


def get_sleep_window_sec(bed_wake_csv: Path, stem: str) -> tuple[float, float]:
    """Returns (lights_off_sec, lights_on_sec) relative to EDF start."""
    subj, night = stem.split("_SP")
    for line in bed_wake_csv.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(";")
        if len(parts) >= 8 and parts[0].strip() == subj and parts[1].strip() == night:
            t0 = _parse_dt(parts[2], parts[3])
            toff = _parse_dt(parts[4], parts[5])
            ton = _parse_dt(parts[6], parts[7])
            return (toff - t0).total_seconds(), (ton - t0).total_seconds()
    raise ValueError(f"{stem} not found in {bed_wake_csv}")


def interpolate_short_gaps(
    arr_2d: np.ndarray, valid_mask: np.ndarray, max_gap: int = 2
) -> np.ndarray:
    """Linearly interpolates interior invalid runs of length <= `max_gap` along axis 0."""
    out = arr_2d.copy()
    n = len(valid_mask)
    diff = np.diff(np.concatenate(([0], (~valid_mask).astype(np.int8), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        if s > 0 and e < n and (e - s) <= max_gap:
            left = out[s - 1]
            right = out[e]
            gap = e - s
            for k in range(gap):
                alpha = (k + 1) / (gap + 1)
                out[s + k] = (1.0 - alpha) * left + alpha * right
    return out


def rolling_nanmean(arr: np.ndarray, win: int = 5) -> np.ndarray:
    """Centered rolling mean ignoring NaN values."""
    half = win // 2
    out = np.full_like(arr, np.nan)
    for idx in range(len(arr)):
        chunk = arr[max(0, idx - half) : min(len(arr), idx + half + 1)]
        if np.any(np.isfinite(chunk)):
            out[idx] = np.nanmean(chunk)
    return out


print(f"Loading {edf_name} ({channel})...")
signal, fs, hypnogram, artefacts, art_idx = load_recording(edf_name, channel)
art_mask = artefacts[:, art_idx] != 0

off_s, on_s = get_sleep_window_sec(
    DATA_DIR / "Bed-Wake_times_25MAY2010_final.csv", Path(edf_name).stem
)
first_sample = max(0, int(round(off_s * fs)))
last_sample = min(len(signal), int(round(on_s * fs)))

signal = signal[first_sample:last_sample]
hypnogram = hypnogram[first_sample:last_sample]
art_mask = art_mask[first_sample:last_sample]

epoch_len = fs * epoch_sec
n_epochs = len(signal) // epoch_len
nperseg = fs * welch_sec
step = nperseg // 2  # 50% overlap between 4s sub-windows
freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)

# ==============================================================================
# 2. Epoch-Wise Welch PSD Across the Sleep Window
# ==============================================================================
Sxx_spec = np.full((n_epochs, len(freqs)), np.nan, dtype=np.float64)
Sxx_clean = np.full((n_epochs, len(freqs)), np.nan, dtype=np.float64)
hyp_epochs = hypnogram[: n_epochs * epoch_len : epoch_len].astype(float)

for ep in range(n_epochs):
    s = ep * epoch_len
    e = s + epoch_len

    # Average clean 4s sub-windows within this 30s epoch
    sub_psds = []
    for sub_s in range(s, e - nperseg + 1, step):
        sub_e = sub_s + nperseg
        if np.any(art_mask[sub_s:sub_e]):
            continue
        _, p_sub = welch(
            signal[sub_s:sub_e],
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=0,
            scaling="density",
        )
        sub_psds.append(p_sub)

    if sub_psds:
        Sxx_spec[ep] = np.mean(sub_psds, axis=0)

    if not np.any(art_mask[s:e]):
        _, Sxx_clean[ep] = welch(
            signal[s:e],
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=step,
            scaling="density",
        )

t_hours = np.arange(n_epochs) * epoch_sec / 3600.0
valid_spec = ~np.isnan(Sxx_spec[:, 0])
valid_clean = ~np.isnan(Sxx_clean[:, 0])

if not np.any(valid_spec):
    raise RuntimeError("No clean 4s sub-windows found in the sleep window.")

Sxx_short_interp = interpolate_short_gaps(
    Sxx_spec, valid_spec, max_gap=max_interp_gap_epochs
)
interp_valid = ~np.isnan(Sxx_short_interp[:, 0])
n_short_filled = int(np.sum(interp_valid) - np.sum(valid_spec))
n_masked_long = int(np.sum(~interp_valid))
print(
    f"Sleep window ({off_s:.0f}-{on_s:.0f}s): {int(np.sum(valid_spec))}/{n_epochs} epochs "
    f"with clean sub-windows, {n_short_filled} short-gap epochs interpolated, "
    f"{n_masked_long} extended artifact epochs masked; strictly clean epochs: {int(np.sum(valid_clean))}."
)

# Slow-wave and Sigma/Spindle band powers across the night
slow_wave_mask = (freqs >= 0.75) & (freqs <= 4.0)
sigma_mask = (freqs >= 11.0) & (freqs <= 16.0)
slow_wave_power = np.full(n_epochs, np.nan, dtype=np.float64)
sigma_power = np.full(n_epochs, np.nan, dtype=np.float64)
slow_wave_power[interp_valid] = np.trapezoid(
    Sxx_short_interp[interp_valid][:, slow_wave_mask], freqs[slow_wave_mask], axis=1
)
sigma_power[interp_valid] = np.trapezoid(
    Sxx_short_interp[interp_valid][:, sigma_mask], freqs[sigma_mask], axis=1
)

slow_wave_smooth = rolling_nanmean(slow_wave_power, win=5)
sigma_smooth = rolling_nanmean(sigma_power, win=5)

# Clean NREM2 PSD comparison: 1st half vs. 2nd half of the sleep window
mid_ep = n_epochs // 2
mid_hour = t_hours[mid_ep]
n2_first = (hyp_epochs[:mid_ep] == 2) & valid_clean[:mid_ep]
n2_second = (hyp_epochs[mid_ep:] == 2) & valid_clean[mid_ep:]

n_n2_first = int(np.sum(n2_first))
n_n2_second = int(np.sum(n2_second))
if n_n2_first == 0 or n_n2_second == 0:
    raise RuntimeError(
        f"Need clean NREM2 epochs in both halves (1st={n_n2_first}, 2nd={n_n2_second})."
    )

psd_1st = np.mean(Sxx_clean[:mid_ep][n2_first], axis=0)
psd_2nd = np.mean(Sxx_clean[mid_ep:][n2_second], axis=0)

sigma_f = freqs[sigma_mask]
peak_f_1st = float(sigma_f[np.argmax(psd_1st[sigma_mask])])
peak_p_1st = float(np.max(psd_1st[sigma_mask]))
peak_f_2nd = float(sigma_f[np.argmax(psd_2nd[sigma_mask])])
peak_p_2nd = float(np.max(psd_2nd[sigma_mask]))

print(
    f"1st half clean NREM2 (n={n_n2_first}): Sigma peak = {peak_f_1st:.2f} Hz "
    f"({peak_p_1st:.2f} uV^2/Hz)"
)
print(
    f"2nd half clean NREM2 (n={n_n2_second}): Sigma peak = {peak_f_2nd:.2f} Hz "
    f"({peak_p_2nd:.2f} uV^2/Hz)"
)

# ==============================================================================
# 3. 4-Row Visualization
# ==============================================================================
fig = plt.figure(figsize=(14, 11.5))
gs = fig.add_gridspec(
    4, 2, height_ratios=[0.85, 1.75, 1.30, 1.55], hspace=0.40, wspace=0.22
)

# Row 1: Hypnogram (Guildford codes: 5=Wake, 0=REM, 1=N1, 2=N2, 3=N3, 4=N4)
ax_hyp = fig.add_subplot(gs[0, :])
stage_to_y = {5: 0.0, 0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}
hyp_plot = np.full(len(hyp_epochs), np.nan, dtype=float)
for code, y_val in stage_to_y.items():
    hyp_plot[hyp_epochs == code] = y_val
ax_hyp.step(t_hours, hyp_plot, where="post", color="#222222", linewidth=1.2, zorder=2)

ax_hyp.axvline(mid_hour, color="#555555", linestyle="--", linewidth=1.1, alpha=0.8)
ax_hyp.set_yticks([0, 1, 2, 3, 4, 5])
ax_hyp.set_yticklabels(["W", "REM", "N1", "N2", "N3", "N4"], fontsize=8.5)
ax_hyp.set_ylim(-0.4, 5.4)
ax_hyp.invert_yaxis()
ax_hyp.set_xlim(0, t_hours[-1])
ax_hyp.set_title(
    f"{edf_name} ({channel}) — Full-Night Spectral Evolution (Lights-Off to Lights-On)",
    fontsize=12,
    fontweight="bold",
)
ax_hyp.set_ylabel("Stage")
ax_hyp.grid(True, linestyle="--", alpha=0.3)

# Row 2: Clean Spectrogram (0.5-20 Hz, dB; extended artifact blocks masked dark gray)
ax_spec = fig.add_subplot(gs[1, :], sharex=ax_hyp)
spec_mask = (freqs >= 0.5) & (freqs <= 20.0)
spec_freqs = freqs[spec_mask]
Sxx_db = np.full((n_epochs, int(np.sum(spec_mask))), np.nan, dtype=np.float64)
with np.errstate(divide="ignore", invalid="ignore"):
    Sxx_db[interp_valid] = 10.0 * np.log10(
        np.maximum(Sxx_short_interp[interp_valid][:, spec_mask], 1e-12)
    )

vmin, vmax = np.nanpercentile(Sxx_db, [5, 98])
cmap = plt.get_cmap("turbo").copy()
cmap.set_bad("#1e1e24")

pcm = ax_spec.pcolormesh(
    t_hours,
    spec_freqs,
    np.ma.masked_invalid(Sxx_db.T),
    shading="nearest",
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
)
ax_spec.axhline(11.0, color="white", linestyle=":", linewidth=1.0, alpha=0.85)
ax_spec.axhline(16.0, color="white", linestyle=":", linewidth=1.0, alpha=0.85)
ax_spec.axvline(mid_hour, color="white", linestyle="--", linewidth=1.1, alpha=0.85)
ax_spec.set_ylabel("Frequency (Hz)")
ax_spec.set_title(
    f"Full-Night Spectrogram (0.5–20 Hz, Sigma 11–16 Hz dotted | dark gray = extended artifact >{max_interp_gap_epochs * epoch_sec}s)",
    fontsize=10.5,
    fontweight="bold",
)
cax = ax_spec.inset_axes([1.01, 0.0, 0.014, 1.0])
cbar = fig.colorbar(pcm, cax=cax)
cbar.set_label(r"PSD (dB $\mu V^2 / Hz$)", fontsize=8.5)

# Row 3: Slow Waves (0.75-4 Hz) vs. Sigma Spindles (11-16 Hz)
ax_sw = fig.add_subplot(gs[2, :], sharex=ax_hyp)
ax_sig = ax_sw.twinx()

l1 = ax_sw.plot(
    t_hours,
    slow_wave_smooth,
    color="#1f77b4",
    linewidth=1.6,
    label="Slow Waves (0.75–4 Hz, 2.5m avg)",
)
l2 = ax_sig.plot(
    t_hours,
    sigma_smooth,
    color="#ff7f0e",
    linewidth=1.6,
    label="Sigma Spindles (11–16 Hz, 2.5m avg)",
)
ax_sw.axvline(mid_hour, color="#555555", linestyle="--", linewidth=1.1, alpha=0.8)
ax_sw.set_ylabel(r"Slow Waves 0.75–4 Hz ($\mu V^2$)", color="#1f77b4")
ax_sig.set_ylabel(r"Sigma / Spindles 11–16 Hz ($\mu V^2$)", color="#ff7f0e")
ax_sw.tick_params(axis="y", colors="#1f77b4")
ax_sig.tick_params(axis="y", colors="#ff7f0e")
ax_sw.set_xlabel("Time from Lights-Off (hours)")
ax_sw.set_title(
    "Slow Waves (0.75–4 Hz) vs. Sigma Spindles (11–16 Hz) Across the Night",
    fontsize=10.5,
    fontweight="bold",
)
ax_sw.grid(True, linestyle="--", alpha=0.3)
lines = l1 + l2
labels = [ln.get_label() for ln in lines]
ax_sw.legend(lines, labels, loc="upper right", fontsize=8.5, ncol=2)

# Row 4a: 1st-Half vs. 2nd-Half Clean NREM2 PSD (0.5-25 Hz, Log Scale)
ax_log = fig.add_subplot(gs[3, 0])
log_mask = (freqs >= 0.5) & (freqs <= 25.0)
ax_log.semilogy(
    freqs[log_mask],
    psd_1st[log_mask],
    color="#1f77b4",
    linewidth=2.0,
    label=f"1st Half NREM2 (n={n_n2_first} ep)",
)
ax_log.semilogy(
    freqs[log_mask],
    psd_2nd[log_mask],
    color="#d62728",
    linewidth=2.0,
    label=f"2nd Half NREM2 (n={n_n2_second} ep)",
)
ax_log.axvspan(0.75, 4.0, color="#1f77b4", alpha=0.10, label="Slow Waves (0.75–4 Hz)")
ax_log.axvspan(
    11.0, 16.0, color="orange", alpha=0.15, label="Sigma / Spindles (11–16 Hz)"
)
ax_log.set_xlim(0.5, 25.0)
ax_log.set_xlabel("Frequency (Hz)")
ax_log.set_ylabel(r"PSD (Log, $\mu V^2 / Hz$)")
ax_log.set_title(
    "Clean NREM2: 1st Half vs. 2nd Half (0.5–25 Hz)", fontsize=10.5, fontweight="bold"
)
ax_log.grid(True, which="both", linestyle="--", alpha=0.3)
ax_log.legend(loc="upper right", fontsize=8)

# Row 4b: 1st-Half vs. 2nd-Half Clean NREM2 Zoomed Sigma (8-20 Hz, Linear Scale)
ax_zoom = fig.add_subplot(gs[3, 1])
zoom_mask = (freqs >= 8.0) & (freqs <= 20.0)
ax_zoom.plot(
    freqs[zoom_mask],
    psd_1st[zoom_mask],
    color="#1f77b4",
    linewidth=2.2,
    label=f"1st Half (peak {peak_f_1st:.2f} Hz, {peak_p_1st:.2f})",
)
ax_zoom.plot(
    freqs[zoom_mask],
    psd_2nd[zoom_mask],
    color="#d62728",
    linewidth=2.2,
    label=f"2nd Half (peak {peak_f_2nd:.2f} Hz, {peak_p_2nd:.2f})",
)
ax_zoom.axvspan(11.0, 16.0, color="orange", alpha=0.15, label="Sigma Band (11–16 Hz)")
ax_zoom.axvline(peak_f_1st, color="#1f77b4", linestyle="--", linewidth=1.1, alpha=0.75)
ax_zoom.axvline(peak_f_2nd, color="#d62728", linestyle="--", linewidth=1.1, alpha=0.75)
ax_zoom.set_xlim(8.0, 20.0)
ax_zoom.set_xticks(np.arange(8, 21, 1))
ax_zoom.set_xlabel("Frequency (Hz)")
ax_zoom.set_ylabel(r"PSD ($\mu V^2 / Hz$)")
ax_zoom.set_title(
    "Clean NREM2 Zoomed Sigma Band (8–20 Hz)", fontsize=10.5, fontweight="bold"
)
ax_zoom.grid(True, linestyle="--", alpha=0.3)
ax_zoom.legend(loc="upper right", fontsize=8)

out_path = PROJECT_ROOT / "outputs" / "figures" / "night_spectrum_evolution.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.subplots_adjust(left=0.07, right=0.92, top=0.95, bottom=0.06)
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    plt.show()
