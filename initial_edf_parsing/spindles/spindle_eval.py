"""
Modular Sleep Spindle Comparison Engine
=======================================
Evaluates Custom (SciPy Hilbert) vs. YASA spindle detection across five modules:
1. Sample-level (Point-by-Point) Binary Classification
2. Event-level (Spindle-by-Spindle) Matching & IoU (Warby et al., 2014 / Lacourse et al., 2019)
3. Boundary & Temporal Precision (Onset, Offset, Center-to-Center, Duration Error)
4. Morphological & Spectral Feature Agreement (Pearson r, Bias, Limits of Agreement)
5. Parameter Sensitivity / Threshold Sweep (PR Curve & F1 Optimization)
6. Spindle Incidence Across Sleep Stages (Wake, N1, N2, N3, REM)
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import pearsonr
import yasa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from spindles import (
    create_spindle_mask,
    detect_custom_spindles,
    detect_custom_spindles_bridged,
    detect_yasa_spindles,
)


# ==============================================================================
# Module 1: Sample-Level (By-Sample) Comparison
# ==============================================================================
def compute_sample_metrics(
    custom_spindles: list[dict],
    yasa_spindles: list[dict],
    valid_mask: np.ndarray,
    fs: int,
) -> dict:
    """Computes sample-by-sample classification metrics restricted to valid (clean N2) samples.

    Treats YASA as the reference standard and Custom as the predicted binary mask.
    """
    n_samples = len(valid_mask)
    c_mask = create_spindle_mask(n_samples, custom_spindles, as_int=False)[valid_mask]
    y_mask = create_spindle_mask(n_samples, yasa_spindles, as_int=False)[valid_mask]

    tp = int(np.sum(c_mask & y_mask))
    fp = int(np.sum(c_mask & ~y_mask))
    fn = int(np.sum(~c_mask & y_mask))
    tn = int(np.sum(~c_mask & ~y_mask))
    total = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou_global = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # Cohen's Kappa (accounts for ~97% non-spindle background imbalance)
    p_o = (tp + tn) / total if total > 0 else 0.0
    p_yes = ((tp + fp) / total) * ((tp + fn) / total) if total > 0 else 0.0
    p_no = ((tn + fn) / total) * ((tn + fp) / total) if total > 0 else 0.0
    p_e = p_yes + p_no
    kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) > 0 else 0.0

    # Matthews Correlation Coefficient (MCC)
    denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    return {
        "tp_sec": round(tp / fs, 2),
        "fp_sec": round(fp / fs, 2),
        "fn_sec": round(fn / fs, 2),
        "tn_sec": round(tn / fs, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "iou_global": round(iou_global, 4),
        "cohen_kappa": round(kappa, 4),
        "mcc": round(mcc, 4),
    }


# ==============================================================================
# Module 2: Event-Level Matching (IoU & 1-to-1 Greedy Matching)
# ==============================================================================
def compute_event_iou(sp1: dict, sp2: dict) -> float:
    """Computes Intersection over Union (Jaccard index) between two spindle intervals."""
    inter = max(0, min(sp1["end"], sp2["end"]) - max(sp1["start"], sp2["start"]))
    if inter == 0:
        return 0.0
    union = max(sp1["end"], sp2["end"]) - min(sp1["start"], sp2["start"])
    return float(inter / union)


def match_spindle_events(
    custom_spindles: list[dict],
    yasa_spindles: list[dict],
    iou_thresh: float = 0.2,
) -> dict:
    """Performs 1-to-1 greedy bipartite matching sorted by descending IoU (Lacourse et al., 2019).

    Also tracks split/merged events (where one spindle in detector A overlaps multiple in B).
    """
    candidates = []
    any_overlap_custom = set()
    any_overlap_yasa = set()

    for i, c in enumerate(custom_spindles):
        for j, y in enumerate(yasa_spindles):
            iou = compute_event_iou(c, y)
            if iou > 0:
                any_overlap_custom.add(i)
                any_overlap_yasa.add(j)
                if iou >= iou_thresh:
                    candidates.append((iou, i, j))

    # Sort descending by IoU for greedy 1-to-1 assignment
    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_c = set()
    matched_y = set()
    matches = []

    for iou, i, j in candidates:
        if i not in matched_c and j not in matched_y:
            matched_c.add(i)
            matched_y.add(j)
            matches.append(
                {
                    "custom_idx": i,
                    "yasa_idx": j,
                    "iou": round(iou, 4),
                    "custom": custom_spindles[i],
                    "yasa": yasa_spindles[j],
                }
            )

    tp = len(matches)
    fp = len(custom_spindles) - tp
    fn = len(yasa_spindles) - tp

    precision = tp / len(custom_spindles) if custom_spindles else 0.0
    recall = tp / len(yasa_spindles) if yasa_spindles else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    ious = [m["iou"] for m in matches]

    return {
        "iou_threshold": iou_thresh,
        "n_custom": len(custom_spindles),
        "n_yasa": len(yasa_spindles),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "median_iou": round(float(np.median(ious)), 4) if ious else 0.0,
        "any_overlap_custom_pct": round(len(any_overlap_custom) / len(custom_spindles) * 100, 2)
        if custom_spindles
        else 0.0,
        "matches": matches,
        "unmatched_custom": [custom_spindles[i] for i in range(len(custom_spindles)) if i not in matched_c],
        "unmatched_yasa": [yasa_spindles[j] for j in range(len(yasa_spindles)) if j not in matched_y],
    }


# ==============================================================================
# Module 3: Boundary & Temporal Alignment Analysis
# ==============================================================================
def compute_boundary_metrics(matches: list[dict], fs: int) -> dict:
    """Computes onset, offset, center, and duration differences (Custom - YASA) in milliseconds."""
    if not matches:
        return {}

    onset_diff_ms = []
    offset_diff_ms = []
    center_diff_ms = []
    dur_diff_ms = []

    for m in matches:
        c, y = m["custom"], m["yasa"]
        onset_diff_ms.append((c["start"] - y["start"]) / fs * 1000.0)
        offset_diff_ms.append((c["end"] - y["end"]) / fs * 1000.0)
        c_mid = 0.5 * (c["start"] + c["end"])
        y_mid = 0.5 * (y["start"] + y["end"])
        center_diff_ms.append((c_mid - y_mid) / fs * 1000.0)
        dur_diff_ms.append((c["dur"] - y["dur"]) * 1000.0)

    def _stats(arr):
        a = np.asarray(arr, dtype=float)
        return {
            "mean_ms": round(float(np.mean(a)), 1),
            "std_ms": round(float(np.std(a)), 1),
            "median_ms": round(float(np.median(a)), 1),
            "mae_ms": round(float(np.mean(np.abs(a))), 1),
            "values": [round(float(x), 1) for x in a],
        }

    return {
        "onset_error": _stats(onset_diff_ms),
        "offset_error": _stats(offset_diff_ms),
        "center_error": _stats(center_diff_ms),
        "duration_error": _stats(dur_diff_ms),
    }


# ==============================================================================
# Module 4: Morphological & Spectral Feature Agreement
# ==============================================================================
def compute_feature_agreement(matches: list[dict]) -> dict:
    """Compares duration (s), peak amplitude (uV), and frequency (Hz) on matched spindles."""
    if len(matches) < 2:
        return {}

    features = ["dur", "peak_amp", "freq"]
    out = {}
    for feat in features:
        c_vals = np.array([m["custom"][feat] for m in matches], dtype=float)
        y_vals = np.array([m["yasa"][feat] for m in matches], dtype=float)
        diff = c_vals - y_vals
        r_val, p_val = pearsonr(c_vals, y_vals)
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff))
        out[feat] = {
            "custom_mean": round(float(np.mean(c_vals)), 3),
            "custom_std": round(float(np.std(c_vals)), 3),
            "yasa_mean": round(float(np.mean(y_vals)), 3),
            "yasa_std": round(float(np.std(y_vals)), 3),
            "bias": round(mean_diff, 3),
            "mae": round(float(np.mean(np.abs(diff))), 3),
            "loa_lower": round(mean_diff - 1.96 * std_diff, 3),
            "loa_upper": round(mean_diff + 1.96 * std_diff, 3),
            "pearson_r": round(float(r_val), 4),
            "p_value": float(p_val),
            "custom_values": [round(float(x), 3) for x in c_vals],
            "yasa_values": [round(float(x), 3) for x in y_vals],
        }
    return out


# ==============================================================================
# Module 5: Parameter Sensitivity Curves (Threshold & IoU Sweeps)
# ==============================================================================
def run_parameter_sweeps(
    signal: np.ndarray,
    fs: int,
    is_n2: np.ndarray,
    yasa_spindles: list[dict],
    custom_default_spindles: list[dict],
    bridged_default_spindles: list[dict] | None = None,
    freq_range=(11, 16),
    duration=(0.5, 2.5),
    max_dip_sec=0.2,
    min_part_sec=0.05,
    peak_offset_std=0.5,
) -> dict:
    """Runs fast sweeps over:
    1. Detector `std_thresh` (from 0.75 to 3.0 SD) -> Precision-Recall & F1 curve for Custom & Bridged.
    2. Event matching `iou_thresh` (from 0.05 to 0.85) -> F1 stability curve.
    """
    # Precompute sigma & envelope once for fast threshold sweeping
    sos = butter(4, freq_range, btype="bandpass", fs=fs, output="sos")
    sigma = sosfiltfilt(sos, signal)
    envelope = np.abs(hilbert(sigma))
    env_mean = float(np.mean(envelope[is_n2]))
    env_std = float(np.std(envelope[is_n2]))

    min_samples = int(duration[0] * fs)
    max_samples = int(duration[1] * fs)
    min_part_samples = int(min_part_sec * fs)
    max_dip_samples = int(max_dip_sec * fs)

    std_grid = np.round(np.arange(0.75, 3.05, 0.25), 2)
    threshold_sweep = []

    for std_val in std_grid:
        thresh = env_mean + std_val * env_std
        high_thresh = env_mean + (std_val + peak_offset_std) * env_std
        above = (envelope > thresh) & is_n2
        diff = np.diff(above.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if above[0]:
            starts = np.insert(starts, 0, 0)
        if above[-1]:
            ends = np.append(ends, len(above))

        sp_list = [
            {"start": int(s), "end": int(e), "dur": float((e - s) / fs)}
            for s, e in zip(starts, ends)
            if min_samples <= (e - s) <= max_samples
        ]
        ev = match_spindle_events(sp_list, yasa_spindles, iou_thresh=0.2)

        # Bridged segments at this std_val
        raw_segs = [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= min_part_samples]
        merged_segs: list[tuple[int, int]] = []
        for s, e in raw_segs:
            if merged_segs:
                prev_s, prev_e = merged_segs[-1]
                if (s - prev_e) <= max_dip_samples and np.all(is_n2[prev_e:s]):
                    comb_peak = max(float(np.max(envelope[prev_s:prev_e])), float(np.max(envelope[s:e])))
                    if comb_peak >= high_thresh:
                        merged_segs[-1] = (prev_s, e)
                        continue
            merged_segs.append((s, e))

        b_list = [
            {"start": int(s), "end": int(e), "dur": float((e - s) / fs)}
            for s, e in merged_segs
            if min_samples <= (e - s) <= max_samples
        ]
        ev_b = match_spindle_events(b_list, yasa_spindles, iou_thresh=0.2)

        threshold_sweep.append(
            {
                "std_thresh": float(std_val),
                "uV_thresh": round(thresh, 2),
                "uV_high_thresh": round(high_thresh, 2),
                "count": len(sp_list),
                "precision": ev["precision"],
                "recall": ev["recall"],
                "f1": ev["f1"],
                "mean_iou": ev["mean_iou"],
                "bridged_count": len(b_list),
                "bridged_precision": ev_b["precision"],
                "bridged_recall": ev_b["recall"],
                "bridged_f1": ev_b["f1"],
                "bridged_mean_iou": ev_b["mean_iou"],
            }
        )

    iou_grid = np.round(np.arange(0.05, 0.90, 0.05), 2)
    iou_sweep = []
    for iou_t in iou_grid:
        ev = match_spindle_events(custom_default_spindles, yasa_spindles, iou_thresh=float(iou_t))
        row = {
            "iou_thresh": float(iou_t),
            "tp": ev["tp"],
            "precision": ev["precision"],
            "recall": ev["recall"],
            "f1": ev["f1"],
        }
        if bridged_default_spindles is not None:
            ev_b = match_spindle_events(bridged_default_spindles, yasa_spindles, iou_thresh=float(iou_t))
            row.update(
                {
                    "bridged_tp": ev_b["tp"],
                    "bridged_precision": ev_b["precision"],
                    "bridged_recall": ev_b["recall"],
                    "bridged_f1": ev_b["f1"],
                }
            )
        iou_sweep.append(row)

    return {
        "threshold_sweep": threshold_sweep,
        "iou_sweep": iou_sweep,
    }


# ==============================================================================
# Module 6: Spindle Incidence Across Sleep Stages
# ==============================================================================
# Guildford hypnogram codes (see pyeeg/datatools/io/special/guildford.py):
# 0: REM, 1: NREM1, 2: NREM2, 3: NREM3, 4: NREM4, 5: WAKE, 9: Unscored
STAGE_GROUPS = [
    ("N1", (1,)),
    ("N2", (2,)),
    ("N3", (3,)),
    ("N4", (4,)),
    ("REM", (0,)),
]


def compute_stage_comparison(
    signal: np.ndarray,
    fs: int,
    hypnogram: np.ndarray,
    artefacts: np.ndarray,
    art_idx: int,
    std_thresh: float = 1.5,
    iou_thresh: float = 0.2,
) -> dict:
    """Compares Custom, Bridged, and YASA spindle incidence across sleep stages (N1, N2, N3, N4, REM)."""
    clean = artefacts[:, art_idx] == 0
    is_n2 = clean & (hypnogram == 2)
    all_codes = tuple(c for _, codes in STAGE_GROUPS for c in codes)
    valid = clean & np.isin(hypnogram, all_codes)

    custom = detect_custom_spindles(signal, fs, is_n2, std_thresh=std_thresh, detect_mask=valid)
    bridged = detect_custom_spindles_bridged(
        signal, fs, is_n2, std_thresh=std_thresh, peak_std_thresh=std_thresh + 0.5, detect_mask=valid
    )
    sos = butter(4, (11, 16), btype="bandpass", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, signal)))
    env_mean = float(np.mean(env[is_n2]))
    env_std = float(np.std(env[is_n2]))
    custom_thresh_uv = env_mean + std_thresh * env_std
    custom_high_thresh_uv = env_mean + (std_thresh + 0.5) * env_std
    del env

    # Calibrate YASA's RMS threshold to the N2 baseline (same as Custom) so N2 counts match
    import mne
    from yasa.others import moving_transform, trimbothstd

    sig_sigma = mne.filter.filter_data(
        signal.astype(np.float64), fs, 11, 16, l_trans_bandwidth=1.5, h_trans_bandwidth=1.5, method="fir", verbose=0
    )
    _, mrms = moving_transform(x=sig_sigma, sf=fs, window=0.3, step=0.1, method="rms", interp=True)
    t_n2 = mrms[hypnogram == 2].mean() + 1.5 * trimbothstd(mrms[hypnogram == 2], cut=0.10)
    mask_all = np.isin(hypnogram, all_codes)
    rms_param = float((t_n2 - mrms[mask_all].mean()) / trimbothstd(mrms[mask_all], cut=0.10))
    del sig_sigma, mrms

    sp_obj = yasa.spindles_detect(
        signal,
        fs,
        ch_names=["EEG"],
        hypno=hypnogram,
        include=all_codes,
        freq_sp=(11, 16),
        thresh={"rel_pow": 0.2, "corr": 0.65, "rms": rms_param},
    )
    df = sp_obj.summary()
    yasa_sp = []
    if df is not None and len(df) > 0:
        for _, row in df.iterrows():
            s_idx = int(row["Start"] * fs)
            e_idx = int((row["Start"] + row["Duration"]) * fs)
            if np.all(artefacts[s_idx:e_idx, art_idx] == 0):
                yasa_sp.append(
                    {
                        "start": s_idx,
                        "end": e_idx,
                        "dur": float(row["Duration"]),
                        "peak_amp": float(row["Amplitude"]),
                        "freq": float(row["Frequency"]),
                    }
                )

    ev = match_spindle_events(custom, yasa_sp, iou_thresh=iou_thresh)
    matched_c = {m["custom_idx"] for m in ev["matches"]}
    matched_y = {m["yasa_idx"] for m in ev["matches"]}

    ev_b = match_spindle_events(bridged, yasa_sp, iou_thresh=iou_thresh)
    matched_b = {m["custom_idx"] for m in ev_b["matches"]}
    matched_y_b = {m["yasa_idx"] for m in ev_b["matches"]}

    def _mid_stage(sp: dict) -> int:
        return int(hypnogram[(sp["start"] + sp["end"]) // 2])

    c_stage = np.array([_mid_stage(sp) for sp in custom], dtype=int)
    b_stage = np.array([_mid_stage(sp) for sp in bridged], dtype=int)
    y_stage = np.array([_mid_stage(sp) for sp in yasa_sp], dtype=int)

    def _feat_mean(spindles: list[dict], idx: np.ndarray, key: str):
        vals = [spindles[i][key] for i in idx]
        return round(float(np.mean(vals)), 3) if vals else None

    stages = []
    for name, codes in STAGE_GROUPS:
        minutes = float(np.sum(clean & np.isin(hypnogram, codes)) / fs / 60.0)
        c_idx = np.where(np.isin(c_stage, codes))[0]
        b_idx = np.where(np.isin(b_stage, codes))[0]
        y_idx = np.where(np.isin(y_stage, codes))[0]
        n_c, n_b, n_y = len(c_idx), len(b_idx), len(y_idx)
        c_matched = sum(1 for i in c_idx if i in matched_c)
        y_matched = sum(1 for j in y_idx if j in matched_y)
        b_matched = sum(1 for i in b_idx if i in matched_b)
        yb_matched = sum(1 for j in y_idx if j in matched_y_b)
        stages.append(
            {
                "stage": name,
                "clean_minutes": round(minutes, 1),
                "n_custom": n_c,
                "n_bridged": n_b,
                "n_yasa": n_y,
                "custom_density": round(n_c / minutes, 2) if minutes > 0 else 0.0,
                "bridged_density": round(n_b / minutes, 2) if minutes > 0 else 0.0,
                "yasa_density": round(n_y / minutes, 2) if minutes > 0 else 0.0,
                "n_both": y_matched,
                "n_both_bridged": yb_matched,
                "precision": round(c_matched / n_c, 4) if n_c else None,
                "recall": round(y_matched / n_y, 4) if n_y else None,
                "bridged_precision": round(b_matched / n_b, 4) if n_b else None,
                "bridged_recall": round(yb_matched / n_y, 4) if n_y else None,
                "custom_freq": _feat_mean(custom, c_idx, "freq"),
                "bridged_freq": _feat_mean(bridged, b_idx, "freq"),
                "yasa_freq": _feat_mean(yasa_sp, y_idx, "freq"),
                "custom_dur": _feat_mean(custom, c_idx, "dur"),
                "bridged_dur": _feat_mean(bridged, b_idx, "dur"),
                "yasa_dur": _feat_mean(yasa_sp, y_idx, "dur"),
                "custom_amp": _feat_mean(custom, c_idx, "peak_amp"),
                "bridged_amp": _feat_mean(bridged, b_idx, "peak_amp"),
                "yasa_amp": _feat_mean(yasa_sp, y_idx, "peak_amp"),
            }
        )

    return {
        "baseline": "clean N2 baseline for Custom/Bridged; clean sleep (N1–N4, REM) for YASA",
        "custom_thresh_uv": round(custom_thresh_uv, 2),
        "custom_high_thresh_uv": round(custom_high_thresh_uv, 2),
        "n_custom": len(custom),
        "n_bridged": len(bridged),
        "n_yasa": len(yasa_sp),
        "stages": stages,
    }


# ==============================================================================
# Helper: Extract Representative Waveform Windows for Interactive Web App
# ==============================================================================
def extract_example_windows(
    signal: np.ndarray,
    fs: int,
    is_n2: np.ndarray,
    event_results: dict,
    freq_range=(11, 16),
    std_thresh=1.5,
) -> dict:
    """Extracts 5-second signal + intermediate detection feature windows for:
    - True Positive (High IoU agreement)
    - Boundary Difference (Custom wider/narrower than YASA)
    - Custom-Only (False Positive relative to YASA)
    - YASA-Only (False Negative relative to YASA)
    """
    sos = butter(4, freq_range, btype="bandpass", fs=fs, output="sos")
    sigma = sosfiltfilt(sos, signal)
    envelope = np.abs(hilbert(sigma))
    custom_thresh = float(np.mean(envelope[is_n2]) + std_thresh * np.std(envelope[is_n2]))

    # Compute YASA's 3 decision streams on the signal for transparent visual explanation:
    # 1. Moving RMS (0.3s window, 0.1s step interpolated) -> here exact 300ms sliding RMS
    win_rms = int(0.3 * fs)
    sigma_sq = sigma ** 2
    kernel = np.ones(win_rms) / win_rms
    moving_rms = np.sqrt(np.convolve(sigma_sq, kernel, mode="same"))
    yasa_rms_thresh = float(np.mean(moving_rms[is_n2]) + 1.5 * np.std(moving_rms[is_n2]))

    def _pack_window(center_sample: int, title: str, subtitle: str) -> dict:
        half_win = int(2.5 * fs)  # 5-second total window
        s_idx = max(0, center_sample - half_win)
        e_idx = min(len(signal), center_sample + half_win)
        t_sec = np.round((np.arange(s_idx, e_idx) - s_idx) / fs, 4).tolist()

        c_spans = [
            [round((sp["start"] - s_idx) / fs, 3), round((sp["end"] - s_idx) / fs, 3), sp["freq"], sp["dur"]]
            for m in event_results["matches"]
            for sp in [m["custom"]]
            if sp["end"] > s_idx and sp["start"] < e_idx
        ] + [
            [round((sp["start"] - s_idx) / fs, 3), round((sp["end"] - s_idx) / fs, 3), sp["freq"], sp["dur"]]
            for sp in event_results["unmatched_custom"]
            if sp["end"] > s_idx and sp["start"] < e_idx
        ]

        y_spans = [
            [round((sp["start"] - s_idx) / fs, 3), round((sp["end"] - s_idx) / fs, 3), sp["freq"], sp["dur"]]
            for m in event_results["matches"]
            for sp in [m["yasa"]]
            if sp["end"] > s_idx and sp["start"] < e_idx
        ] + [
            [round((sp["start"] - s_idx) / fs, 3), round((sp["end"] - s_idx) / fs, 3), sp["freq"], sp["dur"]]
            for sp in event_results["unmatched_yasa"]
            if sp["end"] > s_idx and sp["start"] < e_idx
        ]

        return {
            "title": title,
            "subtitle": subtitle,
            "abs_time_sec": round(s_idx / fs, 1),
            "t": t_sec,
            "raw": np.round(signal[s_idx:e_idx], 2).tolist(),
            "sigma": np.round(sigma[s_idx:e_idx], 2).tolist(),
            "envelope": np.round(envelope[s_idx:e_idx], 2).tolist(),
            "moving_rms": np.round(moving_rms[s_idx:e_idx], 2).tolist(),
            "custom_thresh": round(custom_thresh, 2),
            "yasa_rms_thresh": round(yasa_rms_thresh, 2),
            "custom_spans": c_spans,
            "yasa_spans": y_spans,
        }

    examples = {}
    matches = event_results["matches"]
    if matches:
        # Best agreement (highest IoU)
        best_m = max(matches, key=lambda m: m["iou"])
        mid = (best_m["custom"]["start"] + best_m["custom"]["end"]) // 2
        examples["high_agreement"] = _pack_window(
            mid,
            f"True Positive (High Agreement, IoU = {best_m['iou']:.2f})",
            "Both detectors capture the spindle with nearly identical onset and offset.",
        )
        # Boundary discrepancy (moderate IoU ~0.35-0.55)
        mod_matches = [m for m in matches if 0.3 <= m["iou"] <= 0.6]
        boundary_m = mod_matches[0] if mod_matches else min(matches, key=lambda m: m["iou"])
        mid_b = (boundary_m["custom"]["start"] + boundary_m["custom"]["end"]) // 2
        examples["boundary_diff"] = _pack_window(
            mid_b,
            f"Boundary Discrepancy (Matched, IoU = {boundary_m['iou']:.2f})",
            "YASA's soft-thresholding extends or trims boundaries compared to hard Hilbert thresholding.",
        )

    if event_results["unmatched_custom"]:
        fp_sp = event_results["unmatched_custom"][len(event_results["unmatched_custom"]) // 2]
        mid_fp = (fp_sp["start"] + fp_sp["end"]) // 2
        examples["custom_only"] = _pack_window(
            mid_fp,
            "Custom-Only Detection (Unmatched by YASA)",
            "Crosses the 1.5 SD Hilbert envelope threshold for >0.5s, but fails YASA's multi-feature (rel_pow / corr) criteria.",
        )

    if event_results["unmatched_yasa"]:
        fn_sp = event_results["unmatched_yasa"][len(event_results["unmatched_yasa"]) // 2]
        mid_fn = (fn_sp["start"] + fn_sp["end"]) // 2
        examples["yasa_only"] = _pack_window(
            mid_fn,
            "YASA-Only Detection (Missed by Custom)",
            "Strong relative sigma power and covariance detected by YASA, or Custom stay above hard threshold was <0.5s.",
        )

    return examples


# ==============================================================================
# Full Pipeline Runner
# ==============================================================================
def run_full_modular_comparison(
    edf_name: str = "BB0008_SP2.edf",
    channel: str = "C3",
    iou_thresh: float = 0.2,
    std_thresh: float = 1.5,
) -> dict:
    """Runs the complete benchmark (Custom & Bridged vs. YASA) on a recording and returns a structured dict."""
    from reading_working import load_recording

    signal, fs, hypnogram, artefacts, art_idx = load_recording(edf_name, channel)
    is_n2 = (hypnogram == 2) & (artefacts[:, art_idx] == 0)
    n2_minutes = float(np.sum(is_n2) / fs / 60.0)

    custom_spindles = detect_custom_spindles(signal, fs, is_n2, std_thresh=std_thresh)
    bridged_spindles = detect_custom_spindles_bridged(
        signal, fs, is_n2, std_thresh=std_thresh, peak_std_thresh=std_thresh + 0.5
    )
    yasa_spindles = detect_yasa_spindles(signal, fs, hypnogram, artefacts, art_idx)

    sample_metrics = compute_sample_metrics(custom_spindles, yasa_spindles, is_n2, fs)
    bridged_sample_metrics = compute_sample_metrics(bridged_spindles, yasa_spindles, is_n2, fs)

    event_metrics = match_spindle_events(custom_spindles, yasa_spindles, iou_thresh=iou_thresh)
    bridged_event_metrics = match_spindle_events(bridged_spindles, yasa_spindles, iou_thresh=iou_thresh)

    boundary_metrics = compute_boundary_metrics(event_metrics["matches"], fs)
    bridged_boundary_metrics = compute_boundary_metrics(bridged_event_metrics["matches"], fs)

    feature_metrics = compute_feature_agreement(event_metrics["matches"])
    bridged_feature_metrics = compute_feature_agreement(bridged_event_metrics["matches"])

    sweeps = run_parameter_sweeps(
        signal, fs, is_n2, yasa_spindles, custom_spindles, bridged_default_spindles=bridged_spindles
    )
    examples = extract_example_windows(signal, fs, is_n2, event_metrics, std_thresh=std_thresh)
    stage_metrics = compute_stage_comparison(
        signal, fs, hypnogram, artefacts, art_idx, std_thresh=std_thresh, iou_thresh=iou_thresh
    )

    def _compact_ev(ev_dict: dict) -> dict:
        out = {k: v for k, v in ev_dict.items() if k not in ("matches", "unmatched_custom", "unmatched_yasa")}
        out["ious"] = [m["iou"] for m in ev_dict["matches"]]
        return out

    return {
        "metadata": {
            "edf_name": edf_name,
            "channel": channel,
            "fs": fs,
            "clean_n2_minutes": round(n2_minutes, 2),
            "custom_density_per_min": round(len(custom_spindles) / n2_minutes, 2) if n2_minutes > 0 else 0.0,
            "bridged_density_per_min": round(len(bridged_spindles) / n2_minutes, 2) if n2_minutes > 0 else 0.0,
            "yasa_density_per_min": round(len(yasa_spindles) / n2_minutes, 2) if n2_minutes > 0 else 0.0,
        },
        "sample_level": sample_metrics,
        "bridged_sample_level": bridged_sample_metrics,
        "event_level": _compact_ev(event_metrics),
        "bridged_event_level": _compact_ev(bridged_event_metrics),
        "boundary_level": boundary_metrics,
        "bridged_boundary_level": bridged_boundary_metrics,
        "feature_level": feature_metrics,
        "bridged_feature_level": bridged_feature_metrics,
        "sweeps": sweeps,
        "examples": examples,
        "stage_level": stage_metrics,
    }


if __name__ == "__main__":
    print("Running full modular comparison on BB0008_SP2.edf (C3)...")
    report = run_full_modular_comparison("BB0008_SP2.edf", "C3")

    m = report["metadata"]
    s = report["sample_level"]
    sb = report["bridged_sample_level"]
    e = report["event_level"]
    eb = report["bridged_event_level"]
    b = report["boundary_level"]
    bb = report["bridged_boundary_level"]
    f = report["feature_level"]
    fb = report["bridged_feature_level"]

    print("\n" + "=" * 78)
    print(f" MODULAR SPINDLE BENCHMARK: {m['edf_name']} ({m['channel']}) | Clean N2: {m['clean_n2_minutes']} min")
    print("=" * 78)
    print(f" [1] SAMPLE-LEVEL (Point-by-Point on Clean N2 vs. YASA)")
    print(f"     Custom:  Prec: {s['precision']*100:5.1f}% | Rec: {s['recall']*100:5.1f}% | F1: {s['f1']:.3f} | κ: {s['cohen_kappa']:.3f} | MCC: {s['mcc']:.3f}")
    print(f"     Bridged: Prec: {sb['precision']*100:5.1f}% | Rec: {sb['recall']*100:5.1f}% | F1: {sb['f1']:.3f} | κ: {sb['cohen_kappa']:.3f} | MCC: {sb['mcc']:.3f}")
    print("-" * 78)
    print(f" [2] EVENT-LEVEL (1-to-1 Greedy Matching @ IoU >= {e['iou_threshold']})")
    print(f"     Counts:  Custom={e['n_custom']} ({m['custom_density_per_min']}/min) | Bridged={eb['n_custom']} ({m['bridged_density_per_min']}/min) | YASA={e['n_yasa']} ({m['yasa_density_per_min']}/min)")
    print(f"     Custom:  TP={e['tp']:3d} | FP={e['fp']:3d} | FN={e['fn']:3d} | Prec={e['precision']*100:5.1f}% | Rec={e['recall']*100:5.1f}% | F1={e['f1']:.3f} | IoU={e['mean_iou']:.3f}")
    print(f"     Bridged: TP={eb['tp']:3d} | FP={eb['fp']:3d} | FN={eb['fn']:3d} | Prec={eb['precision']*100:5.1f}% | Rec={eb['recall']*100:5.1f}% | F1={eb['f1']:.3f} | IoU={eb['mean_iou']:.3f}")
    print("-" * 78)
    if b and bb:
        print(f" [3] BOUNDARY PRECISION (Method - YASA on Matched Spindles)")
        print(f"     Custom  Onset: {b['onset_error']['mean_ms']:+6.1f} ± {b['onset_error']['std_ms']:.1f} ms | Offset: {b['offset_error']['mean_ms']:+6.1f} ± {b['offset_error']['std_ms']:.1f} ms | Dur: {b['duration_error']['mean_ms']:+6.1f} ms")
        print(f"     Bridged Onset: {bb['onset_error']['mean_ms']:+6.1f} ± {bb['onset_error']['std_ms']:.1f} ms | Offset: {bb['offset_error']['mean_ms']:+6.1f} ± {bb['offset_error']['std_ms']:.1f} ms | Dur: {bb['duration_error']['mean_ms']:+6.1f} ms")
        print("-" * 78)
    st = report["stage_level"]
    print(f" [6] SPINDLES PER SLEEP STAGE (Custom = {st['custom_thresh_uv']} µV, Peak = {st['custom_high_thresh_uv']} µV)")
    for row in st["stages"]:
        print(
            f"     {row['stage']:<5} {row['clean_minutes']:6.1f} min | "
            f"Custom {row['n_custom']:4d} ({row['custom_density']:.2f}/min) | "
            f"Bridged {row['n_bridged']:4d} ({row['bridged_density']:.2f}/min) | "
            f"YASA {row['n_yasa']:4d} ({row['yasa_density']:.2f}/min)"
        )
    print("=" * 78)

    reports_dir = Path(__file__).resolve().parents[2] / "outputs" / "reports"
    out_path = (reports_dir / "spindle_comparison_report.json") if reports_dir.exists() else (Path(__file__).resolve().parent / "spindle_comparison_report.json")
    with open(out_path, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"Saved detailed benchmark data to {out_path}")
