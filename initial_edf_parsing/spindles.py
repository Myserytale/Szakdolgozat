"""
Sleep Spindle Detection Utilities
=================================
Contains clean implementations of:
1. detect_custom_spindles: Fast SciPy bandpass (11-16 Hz) + Hilbert envelope.
2. detect_yasa_spindles: Standard YASA detector (Lacourse et al., 2019).
3. detect_all_spindles: Runs both detectors and returns clean N2 events.
"""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
import yasa


def detect_custom_spindles(
    signal: np.ndarray,
    fs: int,
    is_n2: np.ndarray,
    freq_range=(11, 16),
    std_thresh=1.5,
    duration=(0.5, 2.5),
) -> list[dict]:
    """Detects sleep spindles using 11-16 Hz bandpass filtering and Hilbert envelope thresholding."""
    # 4th-order zero-phase bandpass filter
    sos = butter(4, freq_range, btype="bandpass", fs=fs, output="sos")
    sigma = sosfiltfilt(sos, signal)
    envelope = np.abs(hilbert(sigma))

    # Threshold based on clean N2 baseline
    thresh = np.mean(envelope[is_n2]) + std_thresh * np.std(envelope[is_n2])

    above = (envelope > thresh) & is_n2
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


def detect_yasa_spindles(
    signal: np.ndarray,
    fs: int,
    hypnogram: np.ndarray,
    artefacts: np.ndarray,
    art_idx: int,
    stage_code: int | tuple[int, ...] = 2,
    freq_sp=(11, 16),
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
