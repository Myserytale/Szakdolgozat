from pathlib import Path

import numpy as np
import pyedflib

DATA_DIR = Path("/home/lev/UNI_1/UNI.4/Szakdolg/Guildford/")
HYPART_DIR = DATA_DIR / "hypart"

edf_files = sorted(DATA_DIR.glob("*.edf"))


def read_edf(file_path: Path):
    with pyedflib.EdfReader(str(file_path)) as edf:
        labels = edf.getSignalLabels()
        fs = [edf.getSampleFrequency(i) for i in range(len(labels))]
        n_samples = [edf.getNSamples()[i] for i in range(len(labels))]

        signals = {label: edf.readSignal(i) for i, label in enumerate(labels)}
    return {
        "filename": file_path.name,
        "labels": labels,
        "sample_rates": fs,
        "signals": signals,
    }


def load_hypart(file_path: Path | str):
    """
    hypnogram vector and artifact matrix
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    raw_hyp = lines[0].strip().split(",")
    # enhyp- duration col
    dur_col_hyp = np.array(raw_hyp[1:], dtype=np.int32).reshape(-1, 2)
    # cumhyp- cumulative sum after sleep stages
    cumhyp = np.cumsum(dur_col_hyp[:, 1])
    # cumsum lenght array
    hypnogram = np.zeros(cumhyp[-1], dtype=np.int8)

    hypnogram[0 : cumhyp[0]] = dur_col_hyp[0, 0]

    for i in range(len(cumhyp) - 1):
        hypnogram[cumhyp[i] : cumhyp[i + 1]] = dur_col_hyp[i + 1, 0]

    channels = []
    art_channels = []
    for line in lines[1:]:
        parts = line.strip().split(",")
        channels.append(parts[0])
        enart = np.array(parts[1:], dtype=np.int32).reshape(-1, 2)
        cumart = np.cumsum(enart[:, 1])
        art = np.zeros(cumart[-1], dtype=np.int8)
        art[0 : cumart[0]] = enart[0, 0]
        for i in range(len(cumart) - 1):
            art[cumart[i] : cumart[i + 1]] = enart[i + 1, 0]
        art_channels.append(art)

    artefacts = np.array(art_channels).T  # Shape: (n_samples, n_channels)
    return hypnogram, artefacts, channels


def decode_rle(values: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """Expands run-length encoded pairs (value, duration) into a flat array."""
    return np.repeat(values, durations)
