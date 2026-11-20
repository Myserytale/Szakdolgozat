"""
CSV-Backed Full-Night Sleep Spindle & Stage Evolution
=====================================================
Works purely from pre-extracted CSV files (no raw EDF reading required):
  - Guildford/Bed-Wake_times_25MAY2010_final.csv (Lights-Off / Lights-On windows)
  - outputs/ecg_csv/ecg_analysis_hypno_only_0b_kubios_like.csv (sleep-stage segments)
  - outputs/spindles_csv/{yasa,bridged,custom}/*.csv (detected sleep spindles)

Generates a 4-row figure:
  1. Hypnogram (W, REM, N1, N2, N3, N4) over the Lights-Off -> Lights-On window.
  2. CSV Spindle Time-Frequency & Amplitude Map (11-16 Hz), free of raw EEG artifacts.
  3. Deep Slow-Wave Sleep (N3/N4 %) vs. CSV Spindle Rate (YASA, Bridged, Custom).
  4. 1st-Half vs. 2nd-Half Sigma Spindle Spectral Profile (11-16 Hz) for BB0008_SP2
     and averaged across all 14 recordings.
"""

from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BED_WAKE_CSV = PROJECT_ROOT / "Guildford" / "Bed-Wake_times_25MAY2010_final.csv"
ECG_HYP_CSV = (
    PROJECT_ROOT / "outputs" / "ecg_csv" / "ecg_analysis_hypno_only_0b_kubios_like.csv"
)
SPINDLES_CSV_DIR = PROJECT_ROOT / "outputs" / "spindles_csv"
OUT_FIG_PATH = (
    PROJECT_ROOT / "outputs" / "figures" / "night_spectrum_evolution_csv.png"
)

TARGET_SUBJECT = "BB0008"
TARGET_SP = 2
TARGET_STEM = f"{TARGET_SUBJECT}_SP{TARGET_SP}"


def _parse_dt(date_str: str, time_str: str) -> datetime:
    text = f"{date_str.strip()} {time_str.strip()}"
    for fmt in ("%d-%b-%y %H:%M:%S", "%d-%b-%y %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date/time format: {text!r}")


def load_bed_wake_windows(csv_path: Path) -> dict[str, tuple[float, float]]:
    """Parses Lights-Off and Lights-On offsets (seconds from recording start) per recording."""
    windows: dict[str, tuple[float, float]] = {}
    for line in csv_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(";")
        if len(parts) < 8:
            continue
        subj = parts[0].strip()
        night = parts[1].strip()
        t_start = _parse_dt(parts[2], parts[3])
        t_off = _parse_dt(parts[4], parts[5])
        t_on = _parse_dt(parts[6], parts[7])
        windows[f"{subj}_SP{night}"] = (
            (t_off - t_start).total_seconds(),
            (t_on - t_start).total_seconds(),
        )
    return windows


def load_hypnogram_from_ecg_csv(
    ecg_csv_path: Path, subject: str, sp: int
) -> np.ndarray:
    """Reconstructs the 1 Hz Lights-Off -> Lights-On hypnogram from the ECG summary CSV."""
    df = pd.read_csv(ecg_csv_path, usecols=["subject", "sp", "hypno", "length"])
    sub = df[(df["subject"] == subject) & (df["sp"] == sp)]
    if sub.empty:
        raise ValueError(f"No hypnogram rows for {subject} SP{sp} in {ecg_csv_path}")
    return np.repeat(
        sub["hypno"].to_numpy(dtype=np.int16), sub["length"].to_numpy(dtype=int)
    )


def load_spindle_df(csv_path: Path, off_s: float, on_s: float) -> pd.DataFrame:
    """Loads a spindle CSV and filters events to the [off_s, on_s] sleep window."""
    df = pd.read_csv(csv_path)
    mid_raw = 0.5 * (df["start_s"].to_numpy(float) + df["end_s"].to_numpy(float))
    mask = (mid_raw >= off_s) & (mid_raw <= on_s)
    sub = df.loc[mask].copy()
    sub["mid_rel_s"] = mid_raw[mask] - off_s
    sub["t_hour"] = sub["mid_rel_s"] / 3600.0
    return sub


def rolling_mean(arr: np.ndarray, win: int = 5) -> np.ndarray:
    """Centered rolling mean over `win` samples."""
    half = win // 2
    return np.array(
        [
            np.mean(arr[max(0, i - half) : min(len(arr), i + half + 1)])
            for i in range(len(arr))
        ],
        dtype=np.float64,
    )


def compute_rolling_spindle_rate(
    df: pd.DataFrame, n_epochs: int, epoch_sec: float = 30.0, smooth_win: int = 5
) -> np.ndarray:
    """Computes smoothed spindle rate (spindles/min) across 30s epochs."""
    counts = np.zeros(n_epochs, dtype=np.float64)
    if not df.empty:
        ep_idx = np.floor(df["mid_rel_s"].to_numpy(float) / epoch_sec).astype(int)
        ep_idx = ep_idx[(ep_idx >= 0) & (ep_idx < n_epochs)]
        np.add.at(counts, ep_idx, 1.0)
    rate_per_min = counts * (60.0 / epoch_sec)
    return rolling_mean(rate_per_min, win=smooth_win)


def compute_weighted_freq_profile(
    df: pd.DataFrame, f_grid: np.ndarray, duration_hours: float
) -> np.ndarray:
    """Computes amplitude*duration-weighted spindle density across frequency (uV*s / h / Hz)."""
    if len(df) < 2 or duration_hours <= 0:
        return np.zeros_like(f_grid)
    freqs_hz = df["frequency_Hz"].to_numpy(float)
    weights = df["amplitude_uV"].to_numpy(float) * df["length_s"].to_numpy(float)
    valid = np.isfinite(freqs_hz) & np.isfinite(weights) & (weights > 0)
    if np.sum(valid) < 2:
        return np.zeros_like(f_grid)
    kde = gaussian_kde(freqs_hz[valid], weights=weights[valid], bw_method=0.28)
    return kde(f_grid) * (float(np.sum(weights[valid])) / duration_hours)


def main() -> None:
    # 1. Load metadata & target recording CSVs
    windows = load_bed_wake_windows(BED_WAKE_CSV)
    if TARGET_STEM not in windows:
        raise RuntimeError(f"{TARGET_STEM} missing from {BED_WAKE_CSV}")
    off_s, on_s = windows[TARGET_STEM]
    sleep_dur_s = on_s - off_s
    sleep_dur_h = sleep_dur_s / 3600.0
    mid_hour = sleep_dur_h / 2.0

    hyp_1hz = load_hypnogram_from_ecg_csv(ECG_HYP_CSV, TARGET_SUBJECT, TARGET_SP)
    epoch_sec = 30
    n_epochs = len(hyp_1hz) // epoch_sec
    t_hours = np.arange(n_epochs) * epoch_sec / 3600.0
    hyp_epochs = hyp_1hz[: n_epochs * epoch_sec : epoch_sec]

    sp_dfs = {
        det: load_spindle_df(
            SPINDLES_CSV_DIR / det / f"{TARGET_STEM}.csv", off_s, on_s
        )
        for det in ("yasa", "bridged", "custom")
    }
    print(
        f"{TARGET_STEM}: sleep window={sleep_dur_h:.2f}h ({n_epochs} ep) | "
        f"CSV spindles: YASA={len(sp_dfs['yasa'])}, Bridged={len(sp_dfs['bridged'])}, Custom={len(sp_dfs['custom'])}"
    )

    # 2. Compute SWS fraction & spindle rates
    sws_1hz = np.isin(hyp_1hz[: n_epochs * epoch_sec], [3, 4]).astype(float)
    sws_ep_pct = sws_1hz.reshape(n_epochs, epoch_sec).mean(axis=1) * 100.0
    sws_smooth_pct = rolling_mean(sws_ep_pct, win=5)

    rates = {
        det: compute_rolling_spindle_rate(df, n_epochs, epoch_sec=epoch_sec, smooth_win=5)
        for det, df in sp_dfs.items()
    }

    # 3. 1st-half vs. 2nd-half Sigma spindle spectral profile (target & all 14 recordings)
    f_grid = np.linspace(11.0, 16.0, 201)
    yasa_target = sp_dfs["yasa"]
    half_h = sleep_dur_h / 2.0
    prof_1st = compute_weighted_freq_profile(
        yasa_target[yasa_target["t_hour"] < half_h], f_grid, half_h
    )
    prof_2nd = compute_weighted_freq_profile(
        yasa_target[yasa_target["t_hour"] >= half_h], f_grid, half_h
    )
    peak_f_1st = float(f_grid[np.argmax(prof_1st)])
    peak_f_2nd = float(f_grid[np.argmax(prof_2nd)])

    all_yasa_csvs = sorted((SPINDLES_CSV_DIR / "yasa").glob("*.csv"))
    all_prof_1st, all_prof_2nd = [], []
    for csv_path in all_yasa_csvs:
        stem = csv_path.stem
        if stem not in windows:
            continue
        w_off, w_on = windows[stem]
        w_half_h = (w_on - w_off) / 7200.0
        df_rec = load_spindle_df(csv_path, w_off, w_on)
        all_prof_1st.append(
            compute_weighted_freq_profile(
                df_rec[df_rec["t_hour"] < w_half_h], f_grid, w_half_h
            )
        )
        all_prof_2nd.append(
            compute_weighted_freq_profile(
                df_rec[df_rec["t_hour"] >= w_half_h], f_grid, w_half_h
            )
        )

    arr_1st = np.vstack(all_prof_1st)
    arr_2nd = np.vstack(all_prof_2nd)
    mean_1st = np.mean(arr_1st, axis=0)
    mean_2nd = np.mean(arr_2nd, axis=0)
    sem_1st = np.std(arr_1st, axis=0, ddof=1) / np.sqrt(len(arr_1st))
    sem_2nd = np.std(arr_2nd, axis=0, ddof=1) / np.sqrt(len(arr_2nd))
    all_peak_1st = float(f_grid[np.argmax(mean_1st)])
    all_peak_2nd = float(f_grid[np.argmax(mean_2nd)])

    print(
        f"{TARGET_STEM} YASA Sigma peak: 1st={peak_f_1st:.2f} Hz, 2nd={peak_f_2nd:.2f} Hz | "
        f"All {len(arr_1st)} recordings: 1st={all_peak_1st:.2f} Hz, 2nd={all_peak_2nd:.2f} Hz"
    )

    # 4. Build 4-row figure
    fig = plt.figure(figsize=(14, 11.5))
    gs = fig.add_gridspec(
        4, 2, height_ratios=[0.85, 1.75, 1.30, 1.55], hspace=0.42, wspace=0.22
    )

    # Row 1: Hypnogram (Guildford codes: 5=Wake, 0=REM, 1=N1, 2=N2, 3=N3, 4=N4)
    ax_hyp = fig.add_subplot(gs[0, :])
    stage_to_y = {5: 0.0, 0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}
    hyp_plot = np.full(len(hyp_epochs), np.nan, dtype=float)
    for code, y_val in stage_to_y.items():
        hyp_plot[hyp_epochs == code] = y_val
    ax_hyp.step(t_hours, hyp_plot, where="post", color="#222222", linewidth=1.2)
    ax_hyp.axvline(mid_hour, color="#555555", linestyle="--", linewidth=1.1, alpha=0.8)
    ax_hyp.set_yticks([0, 1, 2, 3, 4, 5])
    ax_hyp.set_yticklabels(["W", "REM", "N1", "N2", "N3", "N4"], fontsize=8.5)
    ax_hyp.set_ylim(-0.4, 5.4)
    ax_hyp.invert_yaxis()
    ax_hyp.set_xlim(0, sleep_dur_h)
    ax_hyp.set_ylabel("Stage")
    ax_hyp.set_title(
        f"{TARGET_STEM} (C3) — CSV-Backed Full-Night Spindle & Stage Evolution",
        fontsize=12,
        fontweight="bold",
    )
    ax_hyp.grid(True, linestyle="--", alpha=0.3)

    # Row 2: CSV Spindle Time-Frequency & Amplitude Map (11-16 Hz)
    ax_map = fig.add_subplot(gs[1, :], sharex=ax_hyp)
    t_edges = np.linspace(0.0, sleep_dur_h, 281)  # ~2-min bins
    f_edges = np.linspace(10.75, 16.25, 23)       # 0.25 Hz bins
    grid_vals = np.zeros((len(t_edges) - 1, len(f_edges) - 1), dtype=np.float64)

    t_vals = yasa_target["t_hour"].to_numpy(float)
    f_vals = yasa_target["frequency_Hz"].to_numpy(float)
    amp_vals = yasa_target["amplitude_uV"].to_numpy(float)
    dur_vals = yasa_target["length_s"].to_numpy(float)

    t_bin = np.clip(np.digitize(t_vals, t_edges) - 1, 0, len(t_edges) - 2)
    f_bin = np.clip(np.digitize(f_vals, f_edges) - 1, 0, len(f_edges) - 2)
    np.add.at(grid_vals, (t_bin, f_bin), amp_vals * dur_vals)
    grid_smooth = gaussian_filter(grid_vals, sigma=(1.1, 0.8))

    ax_map.pcolormesh(
        t_edges,
        f_edges,
        grid_smooth.T,
        shading="flat",
        cmap="PuBu",
        alpha=0.85,
    )
    vmin_amp, vmax_amp = np.percentile(amp_vals, [5, 95])
    sc = ax_map.scatter(
        t_vals,
        f_vals,
        c=amp_vals,
        s=14 + 12 * np.clip(dur_vals, 0.4, 2.5),
        cmap="plasma",
        vmin=vmin_amp,
        vmax=vmax_amp,
        edgecolors="black",
        linewidths=0.35,
        alpha=0.9,
        zorder=3,
    )
    ax_map.axvline(mid_hour, color="#444444", linestyle="--", linewidth=1.1, alpha=0.8)
    ax_map.set_ylim(10.75, 16.25)
    ax_map.set_ylabel("Spindle Freq (Hz)")
    ax_map.set_title(
        f"CSV Spindle Time–Frequency & Amplitude Map (11–16 Hz, n={len(yasa_target)} YASA spindles)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_map.grid(True, linestyle="--", alpha=0.3)
    cax = ax_map.inset_axes([1.01, 0.0, 0.014, 1.0])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Amplitude (µV)", fontsize=8.5)

    # Row 3: Deep Slow-Wave Sleep (N3/N4 %) vs. CSV Spindle Rate
    ax_sws = fig.add_subplot(gs[2, :], sharex=ax_hyp)
    ax_rate = ax_sws.twinx()

    l_sws = ax_sws.fill_between(
        t_hours,
        0,
        sws_smooth_pct,
        color="#1f77b4",
        alpha=0.22,
        label="Slow-Wave Sleep N3/N4 (%)",
    )
    ax_sws.plot(t_hours, sws_smooth_pct, color="#1f77b4", linewidth=1.1, alpha=0.75)

    l_yasa = ax_rate.plot(
        t_hours, rates["yasa"], color="#d62728", linewidth=1.6, label="YASA Rate (/min)"
    )
    l_br = ax_rate.plot(
        t_hours, rates["bridged"], color="#2ca02c", linewidth=1.3, alpha=0.85, label="Bridged Rate (/min)"
    )
    l_cu = ax_rate.plot(
        t_hours, rates["custom"], color="#ff7f0e", linewidth=1.2, alpha=0.85, label="Custom Rate (/min)"
    )
    ax_sws.axvline(mid_hour, color="#555555", linestyle="--", linewidth=1.1, alpha=0.8)
    ax_sws.set_ylim(0, 108)
    ax_sws.set_ylabel("Slow-Wave Sleep N3/N4 (%)", color="#1f77b4")
    ax_rate.set_ylabel("Spindle Rate (events / min)", color="#222222")
    ax_sws.tick_params(axis="y", colors="#1f77b4")
    ax_sws.set_xlabel("Time from Lights-Off (hours)")
    ax_sws.set_title(
        "Slow-Wave Sleep (N3/N4) vs. Sigma Spindle Rate (2.5-min rolling window)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_sws.grid(True, linestyle="--", alpha=0.3)
    handles = [l_sws] + l_yasa + l_br + l_cu
    labels = [h.get_label() for h in handles]
    ax_sws.legend(handles, labels, loc="upper right", fontsize=8.2, ncol=4)

    # Row 4a: Target recording 1st vs. 2nd half Sigma profile
    ax_p1 = fig.add_subplot(gs[3, 0])
    ax_p1.plot(
        f_grid,
        prof_1st,
        color="#1f77b4",
        linewidth=2.2,
        label=f"1st Half (peak {peak_f_1st:.2f} Hz)",
    )
    ax_p1.plot(
        f_grid,
        prof_2nd,
        color="#d62728",
        linewidth=2.2,
        label=f"2nd Half (peak {peak_f_2nd:.2f} Hz)",
    )
    ax_p1.axvline(peak_f_1st, color="#1f77b4", linestyle="--", linewidth=1.1, alpha=0.75)
    ax_p1.axvline(peak_f_2nd, color="#d62728", linestyle="--", linewidth=1.1, alpha=0.75)
    ax_p1.set_xlim(11.0, 16.0)
    ax_p1.set_xlabel("Spindle Frequency (Hz)")
    ax_p1.set_ylabel(r"Weighted Density ($\mu V\cdot s\ /\ h\ /\ Hz$)")
    ax_p1.set_title(
        f"{TARGET_STEM}: 1st Half vs. 2nd Half Sigma Spindles",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_p1.grid(True, linestyle="--", alpha=0.3)
    ax_p1.legend(loc="upper left", fontsize=8.5)

    # Row 4b: All 14 recordings 1st vs. 2nd half Sigma profile
    ax_p2 = fig.add_subplot(gs[3, 1])
    ax_p2.plot(
        f_grid,
        mean_1st,
        color="#1f77b4",
        linewidth=2.2,
        label=f"1st Half Mean (peak {all_peak_1st:.2f} Hz)",
    )
    ax_p2.fill_between(
        f_grid, mean_1st - sem_1st, mean_1st + sem_1st, color="#1f77b4", alpha=0.15
    )
    ax_p2.plot(
        f_grid,
        mean_2nd,
        color="#d62728",
        linewidth=2.2,
        label=f"2nd Half Mean (peak {all_peak_2nd:.2f} Hz)",
    )
    ax_p2.fill_between(
        f_grid, mean_2nd - sem_2nd, mean_2nd + sem_2nd, color="#d62728", alpha=0.15
    )
    ax_p2.axvline(all_peak_1st, color="#1f77b4", linestyle="--", linewidth=1.1, alpha=0.75)
    ax_p2.axvline(all_peak_2nd, color="#d62728", linestyle="--", linewidth=1.1, alpha=0.75)
    ax_p2.set_xlim(11.0, 16.0)
    ax_p2.set_xlabel("Spindle Frequency (Hz)")
    ax_p2.set_ylabel(r"Weighted Density ($\mu V\cdot s\ /\ h\ /\ Hz$)")
    ax_p2.set_title(
        f"All {len(arr_1st)} Recordings (YASA CSV): 1st Half vs. 2nd Half (±SEM)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_p2.grid(True, linestyle="--", alpha=0.3)
    ax_p2.legend(loc="upper left", fontsize=8.5)

    OUT_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.92, top=0.95, bottom=0.06)
    fig.savefig(OUT_FIG_PATH, dpi=150)
    print(f"Saved figure to {OUT_FIG_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
