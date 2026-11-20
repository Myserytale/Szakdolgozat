from pathlib import Path
import numpy as np
import pyedflib
from scipy.signal import butter, find_peaks, sosfiltfilt

# Base directories (resolves relative to Szakdolg root or default path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Guildford"
HYPART_DIR = DATA_DIR / "hypart"

edf_files = sorted(DATA_DIR.glob("*.edf"))

ECG_CHANNEL = "ECG1 ECG2"

# All 12 scalp EEG derivations present across the Guildford dataset
EEG_CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "T3",
    "T4",
    "P3",
    "P4",
    "O1",
    "O2",
]


def decode_rle(values: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """Expands run-length encoded pairs (value, duration) into a flat array."""
    return np.repeat(values, durations)


def read_edf(file_path: Path | str, channels: list[str] | None = None) -> dict:
    """Reads an EDF file and returns signals, sampling rates, and channel labels.

    Parameters
    ----------
    file_path : Path | str
        Path to the .edf file.
    channels : list[str] | None, optional
        Specific channel labels to read (e.g. ['C3', 'A2']).
        If None, reads all available channels.

    Returns
    -------
    dict with keys:
        - 'filename': str
        - 'labels': list of loaded channel labels
        - 'sample_rates': list of sampling rates (one per label)
        - 'signals': dict mapping label -> 1D numpy array
        - 'fs': dict mapping label -> sample rate (convenient lookup)
    """
    file_path = Path(file_path)

    with pyedflib.EdfReader(str(file_path)) as edf:
        all_labels = edf.getSignalLabels()
        target_labels = channels if channels is not None else all_labels

        signals = {}
        sample_rates = []
        loaded_labels = []

        for label in target_labels:
            if label not in all_labels:
                raise ValueError(
                    f"Channel '{label}' not found in {file_path.name}. "
                    f"Available channels: {all_labels}"
                )
            idx = all_labels.index(label)
            signals[label] = edf.readSignal(idx)
            fs = edf.getSampleFrequency(idx)
            sample_rates.append(fs)
            loaded_labels.append(label)

    return {
        "filename": file_path.name,
        "labels": loaded_labels,
        "sample_rates": sample_rates,
        "signals": signals,
        "fs": dict(zip(loaded_labels, sample_rates)),
    }


def load_hypart(file_path: Path | str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Loads hypnogram and artifact matrix from a .hypart file using vectorized RLE decoding.

    Parameters
    ----------
    file_path : Path | str
        Path to the .hypart file.

    Returns
    -------
    hypnogram : np.ndarray (shape: [n_samples,], dtype: int8)
        Sleep stage per sample (0: Wake, 1: N1, 2: N2, 3: N3, 4: N4, 5: REM).
    artefacts : np.ndarray (shape: [n_samples, n_art_channels], dtype: int8)
        Artifact flags matrix (0 = clean).
    art_channels : list[str]
        Names of the artifact channels corresponding to columns in `artefacts`.
    """
    file_path = Path(file_path)

    with open(file_path, "r") as f:
        lines = f.readlines()

    # 1. Decode Hypnogram
    raw_hyp = lines[0].strip().split(",")
    dur_col_hyp = np.array(raw_hyp[1:], dtype=np.int32).reshape(-1, 2)
    hypnogram = decode_rle(dur_col_hyp[:, 0], dur_col_hyp[:, 1]).astype(np.int8)

    # 2. Decode Artifact channels
    art_channels = []
    art_list = []

    for line in lines[1:]:
        parts = line.strip().split(",")
        art_channels.append(parts[0])
        enart = np.array(parts[1:], dtype=np.int32).reshape(-1, 2)
        art_list.append(decode_rle(enart[:, 0], enart[:, 1]).astype(np.int8))

    artefacts = np.column_stack(art_list)
    return hypnogram, artefacts, art_channels


def load_recording(edf_name: str, channel: str):
    """Loads one EEG channel together with its hypnogram and artifact flags.

    Parameters
    ----------
    edf_name : str
        File name inside DATA_DIR, e.g. "BB0008_SP2.edf".
    channel : str
        EEG channel label, e.g. "C3" or "P3".

    Returns
    -------
    signal : np.ndarray
        The EEG signal of the channel.
    fs : int
        Sampling rate in Hz.
    hypnogram : np.ndarray
        Sleep stage per sample.
    artefacts : np.ndarray
        Artifact flag matrix (0 = clean).
    art_idx : int
        Column of `artefacts` belonging to this channel (e.g. "ArtiFFT C3-A2").
    """
    edf_path = DATA_DIR / edf_name
    hypart_path = HYPART_DIR / f"{edf_path.stem}.hypart"

    edf_data = read_edf(edf_path, channels=[channel])
    signal = edf_data["signals"][channel]
    fs = int(edf_data["sample_rates"][0])

    hypnogram, artefacts, art_channels = load_hypart(hypart_path)

    # Left channels are referenced to A2, right channels to A1 (e.g. "ArtiFFT C4-A1")
    matches = [i for i, name in enumerate(art_channels) if name.startswith(f"ArtiFFT {channel}-")]
    if not matches:
        raise ValueError(f"No artifact channel for '{channel}'. Available: {art_channels}")
    art_idx = matches[0]

    return signal, fs, hypnogram, artefacts, art_idx


def load_multichannel_recording(
    edf_name: str, channels: list[str] | tuple[str, ...] | None = None
) -> tuple[dict[str, np.ndarray], int, np.ndarray, np.ndarray, dict[str, int]]:
    """Loads multiple EEG channels (defaults to all 12 scalp channels) with hypnogram and artifact indices.

    Parameters
    ----------
    edf_name : str
        File name inside DATA_DIR, e.g. "BB0008_SP3.edf".
    channels : list[str] | tuple[str, ...] | None, optional
        EEG channel labels to load. Defaults to `EEG_CHANNELS` (all 12 scalp channels).

    Returns
    -------
    signals : dict[str, np.ndarray]
        Mapping of channel label -> 1D signal array.
    fs : int
        Sampling rate in Hz.
    hypnogram : np.ndarray
        Sleep stage per sample.
    artefacts : np.ndarray
        Artifact flag matrix (0 = clean).
    art_idx_map : dict[str, int]
        Mapping of channel label -> column index in `artefacts`.
    """
    target_channels = list(channels) if channels is not None else list(EEG_CHANNELS)
    edf_path = DATA_DIR / edf_name
    hypart_path = HYPART_DIR / f"{edf_path.stem}.hypart"

    edf_data = read_edf(edf_path, channels=target_channels)
    signals = edf_data["signals"]
    fs = int(edf_data["sample_rates"][0])

    hypnogram, artefacts, art_channels = load_hypart(hypart_path)

    art_idx_map: dict[str, int] = {}
    for ch in target_channels:
        matches = [
            i for i, name in enumerate(art_channels) if name.startswith(f"ArtiFFT {ch}-")
        ]
        if not matches:
            raise ValueError(f"No artifact channel for '{ch}'. Available: {art_channels}")
        art_idx_map[ch] = matches[0]

    return signals, fs, hypnogram, artefacts, art_idx_map


def detect_ecg_rpeaks(
    ecg_raw: np.ndarray, fs: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cleans an ECG trace and detects R-peaks using bandpass filtering and peak snapping.

    Parameters
    ----------
    ecg_raw : np.ndarray
        Raw 1D ECG signal array (mV).
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    ecg_clean : np.ndarray
        0.5-40 Hz zero-phase bandpass-filtered ECG signal (mV).
    rpeaks : np.ndarray
        Sorted unique integer sample indices of detected R-peaks.
    polarity : float
        Detected dominant QRS polarity (+1.0 or -1.0).
    """
    ecg_arr = np.asarray(ecg_raw, dtype=np.float64)
    sos_clean = butter(3, (0.5, 40.0), btype="bandpass", fs=fs, output="sos")
    ecg_clean = sosfiltfilt(sos_clean, ecg_arr)

    sos_qrs = butter(3, (5.0, 25.0), btype="bandpass", fs=fs, output="sos")
    qrs = sosfiltfilt(sos_qrs, ecg_arr)

    polarity = 1.0 if np.percentile(qrs, 99.0) >= np.percentile(-qrs, 99.0) else -1.0
    oriented_qrs = polarity * qrs
    prominence = float(np.percentile(oriented_qrs, 99.0) * 0.45)

    if prominence <= 0.0:
        return ecg_clean, np.array([], dtype=int), polarity

    peaks, _ = find_peaks(oriented_qrs, distance=int(0.35 * fs), prominence=prominence)
    if len(peaks) == 0:
        return ecg_clean, np.array([], dtype=int), polarity

    # Snap candidate peaks to the true R-wave apex on the 0.5-40 Hz cleaned waveform
    half_win = int(0.04 * fs)
    oriented_clean = polarity * ecg_clean
    n_samples = len(ecg_clean)
    snapped = np.empty_like(peaks)
    for i, peak in enumerate(peaks):
        s = max(0, int(peak) - half_win)
        e = min(n_samples, int(peak) + half_win + 1)
        snapped[i] = s + int(np.argmax(oriented_clean[s:e]))

    rpeaks = np.unique(snapped).astype(int)
    return ecg_clean, rpeaks, polarity


def load_ecg(edf_name: str, channel: str = ECG_CHANNEL) -> dict:
    """Loads the ECG channel from an EDF recording and extracts cleaned signal and R-peaks.

    Parameters
    ----------
    edf_name : str
        File name inside DATA_DIR, e.g. "BB0008_SP3.edf".
    channel : str, optional
        ECG channel label in the EDF file. Defaults to `ECG_CHANNEL` ("ECG1 ECG2").

    Returns
    -------
    dict with keys:
        - 'ecg_raw': 1D numpy array (mV)
        - 'ecg_clean': 1D numpy array (0.5-40 Hz filtered, mV)
        - 'fs': int sampling rate in Hz
        - 'rpeaks': 1D integer numpy array of R-peak sample indices
        - 'polarity': float (+1.0 or -1.0)
        - 'y_lim_mv': float recommended symmetric y-axis limit in mV
    """
    edf_path = DATA_DIR / edf_name
    edf_data = read_edf(edf_path, channels=[channel])
    ecg_raw = np.asarray(edf_data["signals"][channel], dtype=np.float64)
    fs = int(edf_data["sample_rates"][0])

    ecg_clean, rpeaks, polarity = detect_ecg_rpeaks(ecg_raw, fs)
    p998 = float(np.percentile(np.abs(ecg_clean), 99.8)) if len(ecg_clean) > 0 else 0.5
    y_lim_mv = float(np.clip(np.ceil(p998 * 1.35 * 20.0) / 20.0, 0.20, 2.0))

    return {
        "ecg_raw": ecg_raw,
        "ecg_clean": ecg_clean,
        "fs": fs,
        "rpeaks": rpeaks,
        "polarity": polarity,
        "y_lim_mv": y_lim_mv,
    }


def compute_epoch_ecg_stats(
    rpeaks: np.ndarray, start: int, end: int, fs: int
) -> dict:
    """Computes beat count, heart rate, and basic RR statistics for a sample interval.

    Parameters
    ----------
    rpeaks : np.ndarray
        Sorted integer array of R-peak sample indices for the recording.
    start : int
        Epoch start sample index (inclusive).
    end : int
        Epoch end sample index (exclusive).
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    dict with keys 'rpeaks', 'n_beats', 'hr_bpm', 'mean_rr_ms', 'sdnn_ms'.
    """
    ep_peaks = rpeaks[(rpeaks >= start) & (rpeaks < end)]
    n_beats = int(len(ep_peaks))
    if n_beats < 2:
        return {
            "rpeaks": ep_peaks,
            "n_beats": n_beats,
            "hr_bpm": float("nan"),
            "mean_rr_ms": float("nan"),
            "sdnn_ms": float("nan"),
        }

    rr_ms = np.diff(ep_peaks) / float(fs) * 1000.0
    valid_rr = rr_ms[(rr_ms >= 300.0) & (rr_ms <= 2000.0)]
    if len(valid_rr) == 0:
        return {
            "rpeaks": ep_peaks,
            "n_beats": n_beats,
            "hr_bpm": float("nan"),
            "mean_rr_ms": float("nan"),
            "sdnn_ms": float("nan"),
        }

    hr_bpm = float(np.mean(60000.0 / valid_rr))
    mean_rr_ms = float(np.mean(valid_rr))
    sdnn_ms = float(np.std(valid_rr, ddof=1)) if len(valid_rr) >= 2 else float("nan")
    return {
        "rpeaks": ep_peaks,
        "n_beats": n_beats,
        "hr_bpm": hr_bpm,
        "mean_rr_ms": mean_rr_ms,
        "sdnn_ms": sdnn_ms,
    }
