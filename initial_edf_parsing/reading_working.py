from pathlib import Path
import numpy as np
import pyedflib

# Base directories (resolves relative to Szakdolg root or default path)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Guildford"
HYPART_DIR = DATA_DIR / "hypart"

edf_files = sorted(DATA_DIR.glob("*.edf"))


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
