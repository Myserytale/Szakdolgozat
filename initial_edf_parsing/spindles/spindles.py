"""
Sleep Spindle Detection & CSV I/O Utilities
===========================================
Contains clean implementations of:
1. detect_custom_spindles: Fast SciPy bandpass (11-16 Hz) + Hilbert envelope.
2. detect_custom_spindles_bridged: Dip-tolerant Hilbert detector that bridges short dips
   below threshold when surrounding segments peak well above threshold.
3. detect_yasa_spindles: Standard YASA detector (Lacourse et al., 2019).
4. detect_all_spindles: Runs Custom and YASA detectors and returns clean N2 events.
5. save_spindles_csv / load_spindles_csv / load_recording_spindles_csv: Standardized
   spindle CSV serialization and loading organized by detector method.
"""

import csv
from pathlib import Path
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
import yasa

SPINDLES_CSV_DIR = (
    Path(__file__).resolve().parents[2] / "outputs" / "spindles_csv"
    if (Path(__file__).resolve().parents[2] / "outputs" / "spindles_csv").exists()
    else Path(__file__).resolve().parent / "spindles_csv"
)

STAGE_NAMES = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "N4",
    5: "REM",
    9: "Unscored",
}

CSV_COLUMNS = [
    "start_s",
    "end_s",
    "length_s",
    "sleep_stage",
    "amplitude_uV",
    "frequency_Hz",
    "n_channels",
]


def detect_custom_spindles(
    signal: np.ndarray,
    fs: int,
    is_n2: np.ndarray,
    freq_range=(11, 16),
    std_thresh=1.5,
    duration=(0.5, 2.5),
    detect_mask: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    envelope: np.ndarray | None = None,
) -> list[dict]:
    """Detects sleep spindles using 11-16 Hz bandpass filtering and Hilbert envelope thresholding.

    `is_n2` defines the baseline used for the threshold. `detect_mask` (optional) defines
    where spindles may be detected; it defaults to `is_n2`.
    """
    if detect_mask is None:
        detect_mask = is_n2

    if sigma is None:
        # 4th-order zero-phase bandpass filter
        sos = butter(4, freq_range, btype="bandpass", fs=fs, output="sos")
        sigma = sosfiltfilt(sos, signal)
    if envelope is None:
        envelope = np.abs(hilbert(sigma))

    # Threshold based on clean N2 baseline
    thresh = np.mean(envelope[is_n2]) + std_thresh * np.std(envelope[is_n2])

    above = (envelope > thresh) & detect_mask
    diff = np.diff(above.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, len(above))

    min_samples = int(duration[0] * fs)
    max_samples = int(duration[1] * fs)

    # Zero-padded FFT size (4 s -> 0.25 Hz frequency resolution)
    nfft = 4 * fs
    fft_freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    f_mask = (fft_freqs >= freq_range[0]) & (fft_freqs <= freq_range[1])

    spindles = []
    for s, e in zip(starts, ends):
        dur = (e - s) / fs
        if min_samples <= (e - s) <= max_samples:
            seg = sigma[s:e] * np.hanning(e - s)
            psd = np.abs(np.fft.rfft(seg, n=nfft)) ** 2
            peak_freq = float(fft_freqs[f_mask][np.argmax(psd[f_mask])])

            spindles.append(
                {
                    "start": int(s),
                    "end": int(e),
                    "dur": float(dur),
                    "peak_amp": float(np.max(np.abs(signal[s:e]))),
                    "freq": peak_freq,
                }
            )
    return spindles


def detect_custom_spindles_bridged(
    signal: np.ndarray,
    fs: int,
    is_n2: np.ndarray,
    freq_range=(11, 16),
    std_thresh=1.5,
    peak_std_thresh=2.0,
    max_dip_sec=0.2,
    min_part_sec=0.05,
    duration=(0.5, 2.5),
    detect_mask: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    envelope: np.ndarray | None = None,
) -> list[dict]:
    """Detects sleep spindles while ignoring brief dips below the threshold if the surrounding
    spindle activity peaks well above threshold.

    Parameters
    ----------
    signal : np.ndarray
        1D EEG signal array.
    fs : int
        Sampling rate in Hz.
    is_n2 : np.ndarray
        Boolean mask for clean N2 baseline calculation.
    freq_range : tuple, optional
        Spindle bandpass range in Hz, default (11, 16).
    std_thresh : float, optional
        Base envelope threshold in SD above clean N2 mean, default 1.5.
    peak_std_thresh : float, optional
        Higher "well above threshold" requirement in SD above clean N2 mean
        for bridging across a dip, default 2.0.
    max_dip_sec : float, optional
        Maximum duration (in seconds) of a dip below `std_thresh` that can be bridged, default 0.2.
    min_part_sec : float, optional
        Minimum duration (in seconds) of an above-threshold sub-segment before merging, default 0.05.
    duration : tuple, optional
        Valid final spindle duration range (min_s, max_s), default (0.5, 2.5).
    detect_mask : np.ndarray | None, optional
        Boolean mask where spindles may be detected; defaults to `is_n2`.
    """
    if detect_mask is None:
        detect_mask = is_n2

    if sigma is None:
        # 4th-order zero-phase bandpass filter
        sos = butter(4, freq_range, btype="bandpass", fs=fs, output="sos")
        sigma = sosfiltfilt(sos, signal)
    if envelope is None:
        envelope = np.abs(hilbert(sigma))

    # Base threshold and higher "well above" peak threshold based on clean N2 baseline
    env_mean = float(np.mean(envelope[is_n2]))
    env_std = float(np.std(envelope[is_n2]))
    thresh = env_mean + std_thresh * env_std
    high_thresh = env_mean + peak_std_thresh * env_std

    above = (envelope > thresh) & detect_mask
    diff = np.diff(above.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, len(above))

    min_part_samples = int(min_part_sec * fs)
    max_dip_samples = int(max_dip_sec * fs)
    min_samples = int(duration[0] * fs)
    max_samples = int(duration[1] * fs)

    # Filter out single-sample glitches before gap-merging
    raw_segments = [
        (int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= min_part_samples
    ]

    # Merge adjacent segments separated by a brief dip if their combined peak is well above threshold
    merged_segments: list[tuple[int, int]] = []
    for s, e in raw_segments:
        if merged_segments:
            prev_s, prev_e = merged_segments[-1]
            dip_len = s - prev_e
            if dip_len <= max_dip_samples and np.all(detect_mask[prev_e:s]):
                combined_peak = max(
                    float(np.max(envelope[prev_s:prev_e])),
                    float(np.max(envelope[s:e])),
                )
                if combined_peak >= high_thresh:
                    merged_segments[-1] = (prev_s, e)
                    continue
        merged_segments.append((s, e))

    # Zero-padded FFT size (4 s -> 0.25 Hz frequency resolution)
    nfft = 4 * fs
    fft_freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    f_mask = (fft_freqs >= freq_range[0]) & (fft_freqs <= freq_range[1])

    spindles = []
    for s, e in merged_segments:
        dur = (e - s) / fs
        if min_samples <= (e - s) <= max_samples:
            seg = sigma[s:e] * np.hanning(e - s)
            psd = np.abs(np.fft.rfft(seg, n=nfft)) ** 2
            peak_freq = float(fft_freqs[f_mask][np.argmax(psd[f_mask])])

            spindles.append(
                {
                    "start": int(s),
                    "end": int(e),
                    "dur": float(dur),
                    "peak_amp": float(np.max(np.abs(signal[s:e]))),
                    "freq": peak_freq,
                }
            )
    return spindles


def detect_yasa_spindles(
    signal: np.ndarray,
    fs: int,
    hypnogram: np.ndarray,
    artefacts: np.ndarray,
    art_idx: int,
    stage_code: int | tuple[int, ...] = 2,
    freq_sp=(12, 15),
) -> list[dict]:
    """Detects sleep spindles using YASA (Lacourse et al., 2019) and filters for clean epochs."""
    include = (stage_code,) if isinstance(stage_code, int) else tuple(stage_code)
    sp = yasa.spindles_detect(
        signal,
        fs,
        ch_names=["EEG"],
        hypno=hypnogram,
        include=include,
        freq_sp=freq_sp,
    )
    df = sp.summary()

    spindles = []
    if df is not None and len(df) > 0:
        for _, row in df.iterrows():
            s_idx = int(row["Start"] * fs)
            e_idx = int((row["Start"] + row["Duration"]) * fs)
            if np.all(artefacts[s_idx:e_idx, art_idx] == 0):
                spindles.append(
                    {
                        "start": s_idx,
                        "end": e_idx,
                        "dur": float(row["Duration"]),
                        "peak_amp": float(row["Amplitude"]),
                        "freq": float(row["Frequency"]),
                    }
                )
    return spindles


def detect_yasa_spindles_multichannel(
    signals: dict[str, np.ndarray],
    fs: int,
    hypnogram: np.ndarray,
    artefacts: np.ndarray,
    art_idx_map: dict[str, int],
    stage_code: int | tuple[int, ...] = 2,
    freq_sp=(12, 15),
) -> dict[str, list[dict]]:
    """Detects sleep spindles across multiple EEG channels simultaneously using YASA."""
    ch_names = list(signals.keys())
    include = (stage_code,) if isinstance(stage_code, int) else tuple(stage_code)
    data_2d = np.vstack([signals[ch] for ch in ch_names])

    sp = yasa.spindles_detect(
        data_2d,
        fs,
        ch_names=ch_names,
        hypno=hypnogram,
        include=include,
        freq_sp=freq_sp,
    )
    df = sp.summary() if sp is not None else None

    by_channel: dict[str, list[dict]] = {ch: [] for ch in ch_names}
    if df is not None and len(df) > 0:
        for _, row in df.iterrows():
            ch = str(row["Channel"])
            if ch not in by_channel:
                continue
            s_idx = int(row["Start"] * fs)
            e_idx = int((row["Start"] + row["Duration"]) * fs)
            art_idx = art_idx_map[ch]
            if np.all(artefacts[s_idx:e_idx, art_idx] == 0):
                by_channel[ch].append(
                    {
                        "start": s_idx,
                        "end": e_idx,
                        "dur": float(row["Duration"]),
                        "peak_amp": float(row["Amplitude"]),
                        "freq": float(row["Frequency"]),
                    }
                )
    return by_channel


def detect_all_spindles(
    signal: np.ndarray,
    fs: int,
    hypnogram: np.ndarray,
    artefacts: np.ndarray,
    art_idx: int,
) -> tuple[list[dict], list[dict]]:
    """Runs both Custom and YASA detectors on clean N2 sleep."""
    is_n2 = (hypnogram == 2) & (artefacts[:, art_idx] == 0)
    custom = detect_custom_spindles(signal, fs, is_n2)
    yasa_sp = detect_yasa_spindles(signal, fs, hypnogram, artefacts, art_idx)
    return custom, yasa_sp


def create_spindle_mask(
    n_samples: int,
    spindles: list[dict],
    as_int: bool = False,
) -> np.ndarray:
    """Constructs a 1D boolean (or 0/1 integer) array where 1 (True) indicates a spindle

    and 0 (False) indicates non-spindle background.

    Parameters
    ----------
    n_samples : int
        Total length of the recording (e.g. len(signal)).
    spindles : list[dict]
        List of spindle dictionaries containing 'start' and 'end' sample indices.
    as_int : bool, optional
        If True, returns an int8 array (0 and 1). Default is False (boolean True/False).

    Returns
    -------
    np.ndarray
        1D binary mask of shape (n_samples,).
    """
    mask = np.zeros(n_samples, dtype=np.int8 if as_int else bool)
    for sp in spindles:
        mask[sp["start"] : sp["end"]] = 1 if as_int else True
    return mask


# ==============================================================================
# Multi-Channel Coincidence & Overlap Filtering
# ==============================================================================
def compute_coincidence_mask(
    spindles_by_channel: dict[str, list[dict]],
    n_samples: int,
) -> np.ndarray:
    """Sums binary spindle masks across all channels -> shape (n_samples,) with values 0..N."""
    coincidence_count = np.zeros(n_samples, dtype=np.int8)
    for sp_list in spindles_by_channel.values():
        coincidence_count += create_spindle_mask(n_samples, sp_list, as_int=True)
    return coincidence_count


def annotate_multichannel_coincidence(
    target_spindles: list[dict],
    spindles_by_channel: dict[str, list[dict]],
    n_samples: int,
) -> list[dict]:
    """Annotates each spindle in `target_spindles` with `n_channels` (peak simultaneous active channels)."""
    coincidence_count = compute_coincidence_mask(spindles_by_channel, n_samples)
    annotated = []
    for sp in target_spindles:
        s, e = sp["start"], sp["end"]
        n_ch = int(np.max(coincidence_count[s:e])) if e > s else 1
        annotated.append({**sp, "n_channels": max(1, n_ch)})
    return annotated


def filter_spindles_min_coincidence(
    target_spindles: list[dict],
    spindles_by_channel: dict[str, list[dict]] | None = None,
    n_samples: int | None = None,
    min_channels: int = 5,
) -> list[dict]:
    """Keeps only spindles where at least `min_channels` EEG channels are simultaneously active.

    If `target_spindles` already carry `"n_channels"` (e.g. loaded from CSV), filters directly
    without needing `spindles_by_channel` or `n_samples`.
    """
    if spindles_by_channel is not None and n_samples is not None:
        target_spindles = annotate_multichannel_coincidence(
            target_spindles, spindles_by_channel, n_samples
        )
    return [sp for sp in target_spindles if int(sp.get("n_channels", 1)) >= min_channels]


def filter_csv_spindles_by_overlap(
    target_spindles: list[dict],
    spindles_by_group: dict[str, list[dict]],
    min_count: int = 5,
    min_overlap_s: float = 0.1,
    fs: int = 256,
) -> list[dict]:
    """Keeps spindles that overlap (by at least `min_overlap_s`) in >= `min_count` channels/files."""
    min_ov_samples = int(min_overlap_s * fs)
    filtered = []

    for sp in target_spindles:
        s, e = sp["start"], sp["end"]
        count = sum(
            1
            for sp_list in spindles_by_group.values()
            if any(
                min(e, other["end"]) - max(s, other["start"]) >= min_ov_samples
                for other in sp_list
            )
        )
        if count >= min_count:
            filtered.append({**sp, "n_channels": count})

    return filtered


# ==============================================================================
# CSV Serialization & Loading Utilities
# ==============================================================================
def get_spindle_csv_path(
    edf_name: str,
    detector: str = "yasa",
    base_dir: Path = SPINDLES_CSV_DIR,
) -> Path:
    """Returns the canonical CSV path `spindles_csv/<detector>/<recording_stem>.csv`."""
    stem = Path(edf_name).stem
    return base_dir / detector / f"{stem}.csv"


def save_spindles_csv(
    csv_path: Path | str,
    spindles: list[dict],
    fs: int,
    hypnogram: np.ndarray,
) -> Path:
    """Writes detected spindles to a standardized CSV file."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for sp in spindles:
            stage_code = int(hypnogram[sp["start"]])
            writer.writerow(
                [
                    f"{sp['start'] / fs:.2f}",
                    f"{sp['end'] / fs:.2f}",
                    f"{sp['dur']:.2f}",
                    STAGE_NAMES.get(stage_code, str(stage_code)),
                    f"{sp['peak_amp']:.2f}",
                    f"{sp['freq']:.2f}",
                    int(sp.get("n_channels", 1)),
                ]
            )
    return csv_path


def load_spindles_csv(
    csv_path: Path | str,
    fs: int,
    allowed_stages: set[str] | tuple[str, ...] | None = None,
    min_channels: int = 1,
) -> list[dict]:
    """Loads spindles from a CSV file and converts timestamps to sample indices at rate `fs`."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    stage_filter = set(allowed_stages) if allowed_stages is not None else None
    spindles: list[dict] = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage_str = row.get("sleep_stage", "").strip()
            if stage_filter is not None and stage_str not in stage_filter:
                continue
            n_ch_raw = row.get("n_channels")
            n_channels = int(n_ch_raw) if (n_ch_raw is not None and n_ch_raw != "") else 1
            if n_channels < min_channels:
                continue

            start_s = float(row["start_s"])
            end_s = float(row["end_s"])
            dur_s = float(row.get("length_s", end_s - start_s))
            amp_uv = float(row.get("amplitude_uV", 0.0))
            freq_hz = float(row.get("frequency_Hz", 0.0))

            spindles.append(
                {
                    "start": int(round(start_s * fs)),
                    "end": int(round(end_s * fs)),
                    "dur": dur_s,
                    "peak_amp": amp_uv,
                    "freq": freq_hz,
                    "stage": stage_str,
                    "n_channels": n_channels,
                }
            )
    return spindles


def load_recording_spindles_csv(
    edf_name: str,
    fs: int,
    allowed_stages: set[str] | tuple[str, ...] | None = None,
    min_channels: int = 1,
    base_dir: Path = SPINDLES_CSV_DIR,
) -> dict[str, list[dict]]:
    """Loads pre-exported CSV spindles for ('custom', 'bridged', 'yasa') for a recording.

    Falls back to legacy flat `spindles_csv/<stem>.csv` for YASA if needed.
    """
    stem = Path(edf_name).stem
    result: dict[str, list[dict]] = {}
    for det in ("custom", "bridged", "yasa"):
        p = get_spindle_csv_path(edf_name, det, base_dir=base_dir)
        if not p.exists() and det == "yasa":
            legacy_p = base_dir / f"{stem}.csv"
            if legacy_p.exists():
                p = legacy_p
        result[det] = load_spindles_csv(
            p, fs, allowed_stages=allowed_stages, min_channels=min_channels
        )
    return result
