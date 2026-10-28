"""
Excess Infra Power (EIP) of sleep spindle activity
==================================================
Method from Lázár Z.I., Dijk D.J., Lázár A.S. (2019),
"Infraslow oscillations in human sleep spindle activity",
J. Neurosci. Methods 316: 22–34, Section 2.2.4.

1. Spindles -> binary square signal (1 = spindle, 0 = none), downsampled to 4 Hz.
2. Keep contiguous clean segments of at least 256 s.
3. Welch PSD (256 s Hann window, 75% overlap) of the real signal and of
   surrogates where spindle durations and gaps are randomly shuffled.
4. EIP(f) = PSD_original(f) / PSD_permutated(f) - 1
"""

import numpy as np
from scipy.signal import welch


def make_binary_signal(spindles: list[dict], n_samples: int, fs: int, fs_down: int = 4) -> np.ndarray:
    """Builds the 1/0 spindle signal at the downsampled rate `fs_down`."""
    factor = fs // fs_down
    binary = np.zeros(n_samples // factor, dtype=np.float32)
    for sp in spindles:
        binary[sp["start"] // factor : sp["end"] // factor] = 1.0
    return binary


def contiguous_segments(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Returns (start, end) index pairs of True-runs in `mask` that are at least `min_len` long."""
    diff = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


def shuffle_train(binary: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate: randomly reorders spindle durations and the gaps between them.

    Keeps the number of spindles, their durations and the gap lengths,
    but destroys their order in time.
    """
    diff = np.diff(np.concatenate(([0], binary, [0])))
    s = np.where(diff == 1)[0]
    e = np.where(diff == -1)[0]
    if len(s) <= 1:
        return binary.copy()

    spindle_durs = rng.permutation(e - s)
    gap_durs = rng.permutation(s[1:] - e[:-1])

    pieces = [np.zeros(s[0], dtype=np.float32)]  # leading silence stays in place
    for i, dur in enumerate(spindle_durs):
        pieces.append(np.ones(dur, dtype=np.float32))
        if i < len(gap_durs):
            pieces.append(np.zeros(gap_durs[i], dtype=np.float32))
    pieces.append(np.zeros(len(binary) - e[-1], dtype=np.float32))  # trailing silence
    return np.concatenate(pieces)


def compute_eip(
    spindles: list[dict],
    clean_mask: np.ndarray,
    fs: int,
    fs_down: int = 4,
    window_sec: int = 256,
    n_shuffles: int = 30,
    seed: int = 0,
) -> dict:
    """Computes the PSD of the real and shuffled spindle signals and the EIP.

    Parameters
    ----------
    spindles : list[dict]
        Spindles with 'start' and 'end' sample indices (at rate `fs`).
    clean_mask : np.ndarray
        Boolean array (length = signal length), True where data may be used
        (e.g. clean NREM2).
    fs : int
        Original sampling rate.

    Returns
    -------
    dict with keys 'f', 'psd_orig', 'psd_perm', 'eip', 'segments'
    (segments = list of binary signal pieces that were analysed).
    """
    rng = np.random.default_rng(seed)  # fixed seed -> same figure every run

    binary = make_binary_signal(spindles, len(clean_mask), fs, fs_down)
    mask_down = clean_mask[:: fs // fs_down][: len(binary)]

    nperseg = window_sec * fs_down
    noverlap = int(nperseg * 0.75)

    def psd(x):
        return welch(x, fs=fs_down, window="hann", nperseg=nperseg, noverlap=noverlap)

    orig_psds, perm_psds, segments = [], [], []
    for s, e in contiguous_segments(mask_down, nperseg):
        seg = binary[s:e]
        if seg.sum() == 0:  # no spindles -> nothing to analyse
            continue
        f, p_orig = psd(seg)
        p_perm = np.mean([psd(shuffle_train(seg, rng))[1] for _ in range(n_shuffles)], axis=0)
        orig_psds.append(p_orig)
        perm_psds.append(p_perm)
        segments.append(seg)

    if not segments:
        raise ValueError(f"No clean segment of at least {window_sec} s contains spindles.")

    psd_orig = np.mean(orig_psds, axis=0)
    psd_perm = np.mean(perm_psds, axis=0)
    return {
        "f": f,
        "psd_orig": psd_orig,
        "psd_perm": psd_perm,
        "eip": psd_orig / psd_perm - 1.0,
        "segments": segments,
    }


def integrated_eip(f: np.ndarray, eip: np.ndarray, f_max: float = 0.035) -> float:
    """Area of the positive EIP between 0 and `f_max` Hz, in mHz units (inset of Fig. 5b).

    The f = 0 bin is skipped: Welch removes the mean, so the DC power is ~0 in both spectra.
    """
    band = (f > 0) & (f <= f_max)
    return float(np.trapezoid(np.maximum(0, eip[band]), f[band] * 1000))
