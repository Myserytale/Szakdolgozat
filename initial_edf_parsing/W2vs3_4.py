"""
Sleep Stage Spectral Comparison (NREM 2 vs. N3 vs. N4)
======================================================
This script compares the electroencephalogram (EEG) Power Spectral Density (PSD)
across NREM sleep stages (N2, N3, and N4) using the C3 derivation (C3-A2 reference).

Key Objectives:
1. Load EDF sleep data and matching .hypart artifact annotations.
2. Segment the signal into 10-second artifact-free epochs for each sleep stage.
3. Compute the PSD using Welch's method (Hann window, 50% overlap).
4. Visualize:
   - Full spectrum (0.5–25 Hz, log scale) highlighting Slow Waves / Delta (0.5–4 Hz).
   - Zoomed Sigma band (8–20 Hz, linear scale) highlighting Sleep Spindles (11–16 Hz).
   - Time-domain raw traces showing stage-specific waveform morphology.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from reading_working import DATA_DIR, HYPART_DIR, load_hypart, read_edf

# ==============================================================================
# CONFIGURATION & PARAMETERS
# ==============================================================================
EDF_FILENAME = "BB0008_SP2.edf"
TARGET_CHANNEL = "C3"
ARTIFACT_CHANNEL = "ArtiFFT C3-A2"

# Epoch & Spectral Analysis Settings
EPOCH_SEC = 10  # Duration of each analysis epoch in seconds
WELCH_WINDOW_SEC = 4  # Window length for Welch's method (4s gives 0.25 Hz resolution)
WELCH_OVERLAP_RATIO = 0.5  # 50% overlap between Welch windows

# Sleep Stages to analyze:
# Stage 2 (N2), Stage 3 (N3 / SWS), Stage 4 (N4 / Deep SWS)
STAGE_CONFIG = {
    "Stage 2 (N2)": {"codes": [2], "color": "#1f77b4"},
    "Stage 3 (N3)": {"codes": [3], "color": "#2ca02c"},
    "Stage 4 (N4)": {"codes": [4], "color": "#d62728"},
}

# Frequency bands of interest (in Hz)
FREQ_BANDS = {
    "Delta (0.5–4 Hz)": {"range": (0.5, 4.0), "color": "gray", "alpha": 0.15},
    "Sigma / Spindles (11–16 Hz)": {
        "range": (11.0, 16.0),
        "color": "orange",
        "alpha": 0.20,
    },
}


# ==============================================================================
# DATA EXTRACTION & PREPROCESSING
# ==============================================================================
def load_recording(edf_name: str, channel: str, art_channel: str):
    """Loads EEG signal, sampling rate, hypnogram, and artifact indicators."""
    edf_path = DATA_DIR / edf_name
    hypart_path = HYPART_DIR / f"{edf_path.stem}.hypart"

    print(f"Loading EDF: {edf_path.name}")
    edf_data = read_edf(edf_path)
    labels = edf_data["labels"]

    if channel not in labels:
        raise ValueError(f"Channel '{channel}' not found in EDF signals: {labels}")

    chan_idx = labels.index(channel)
    signal = edf_data["signals"][channel]
    fs = int(edf_data["sample_rates"][chan_idx])

    print(f"Loading annotations: {hypart_path.name}")
    hypnogram, artefacts, art_channels = load_hypart(hypart_path)

    if art_channel not in art_channels:
        raise ValueError(
            f"Artifact channel '{art_channel}' not found. Available: {art_channels}"
        )
    art_idx = art_channels.index(art_channel)

    return signal, fs, hypnogram, artefacts, art_idx


def extract_clean_stage_segments(
    signal: np.ndarray,
    hypnogram: np.ndarray,
    artefacts: np.ndarray,
    art_idx: int,
    fs: int,
    epoch_sec: int,
    stage_config: dict,
):
    """
    Slices the continuous signal into non-overlapping epochs, filters out
    epochs with artifacts, and groups them by sleep stage.
    """
    epoch_samples = fs * epoch_sec
    total_epochs = len(signal) // epoch_samples

    stage_segments = {name: [] for name in stage_config}
    sample_traces = {}

    for i in range(total_epochs):
        start = i * epoch_samples
        end = start + epoch_samples

        # Discard epochs containing any artifacts
        if not np.all(artefacts[start:end, art_idx] == 0):
            continue

        epoch_stage = hypnogram[start:end]

        for name, conf in stage_config.items():
            if np.all(np.isin(epoch_stage, conf["codes"])):
                segment = signal[start:end]
                stage_segments[name].append(segment)

                # Keep the first clean segment for time-domain display
                if name not in sample_traces:
                    sample_traces[name] = segment

    return stage_segments, sample_traces


# ==============================================================================
# SPECTRAL ANALYSIS
# ==============================================================================
def compute_psds(stage_segments: dict, fs: int, nperseg: int, noverlap: int):
    """
    Computes Welch's Power Spectral Density for all segments across stages.
    Returns the one-sided frequency array (0 to Nyquist) and a dict of PSDs.
    """
    frequencies = None
    stage_psds = {name: [] for name in stage_segments}

    for name, segments in stage_segments.items():
        for seg in segments:
            f, psd = welch(
                seg,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="density",
            )
            stage_psds[name].append(psd)
            if frequencies is None:
                frequencies = f

    # Fallback if no segments were found
    if frequencies is None:
        frequencies = np.fft.rfftfreq(nperseg, 1.0 / fs)

    return frequencies, stage_psds


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================
def plot_psd_comparison(
    frequencies: np.ndarray,
    stage_psds: dict,
    stage_config: dict,
    output_path: Path | str = "psd_stage_comparison.png",
):
    """
    Plots two complementary PSD views:
    1. Full spectrum (0.5 - 25 Hz) on logarithmic scale.
    2. Zoomed Sigma band (8 - 20 Hz) on linear scale for sleep spindle inspection.
    """
    fig, (ax_full, ax_sigma) = plt.subplots(1, 2, figsize=(15, 6))

    mean_psds = {}
    for name, psd_list in stage_psds.items():
        if psd_list:
            mean_psds[name] = np.mean(psd_list, axis=0)

    # --------------------------------------------------------------------------
    # Subplot 1: Full Spectrum (Log Scale)
    # --------------------------------------------------------------------------
    mask_full = (frequencies >= 0.5) & (frequencies <= 25.0)
    freqs_full = frequencies[mask_full]

    for name, mean_psd in mean_psds.items():
        count = len(stage_psds[name])
        ax_full.semilogy(
            freqs_full,
            mean_psd[mask_full],
            label=f"{name} (n={count} epochs)",
            color=stage_config[name]["color"],
            linewidth=2.0,
        )

    # Shaded physiological frequency bands
    for band_name, band_info in FREQ_BANDS.items():
        low, high = band_info["range"]
        ax_full.axvspan(
            low,
            high,
            color=band_info["color"],
            alpha=band_info["alpha"],
            label=band_name,
        )

    ax_full.set_title(
        "Full Spectrum Comparison (0.5 – 25 Hz)", fontsize=13, fontweight="bold"
    )
    ax_full.set_xlabel("Frequency (Hz)", fontsize=11)
    ax_full.set_ylabel(r"PSD (Log Scale, $\mu\mathrm{V}^2 / \mathrm{Hz}$)", fontsize=11)
    ax_full.set_xlim(0.5, 25.0)
    ax_full.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_full.legend(loc="upper right", frameon=True)

    # --------------------------------------------------------------------------
    # Subplot 2: Zoomed Sigma Band (Sleep Spindles: 8 - 20 Hz)
    # --------------------------------------------------------------------------
    mask_sigma = (frequencies >= 8.0) & (frequencies <= 20.0)
    freqs_sigma = frequencies[mask_sigma]

    spindle_mask = (freqs_sigma >= 11.0) & (freqs_sigma <= 16.0)

    for name, mean_psd in mean_psds.items():
        p_sub = mean_psd[mask_sigma]
        color = stage_config[name]["color"]
        ax_sigma.plot(
            freqs_sigma,
            p_sub,
            label=name,
            color=color,
            linewidth=2.0,
        )

        # Annotate peak spindle frequency in Stage 2
        if "Stage 2" in name and np.any(spindle_mask):
            peak_idx = np.argmax(p_sub[spindle_mask])
            peak_f = freqs_sigma[spindle_mask][peak_idx]
            peak_p = p_sub[spindle_mask][peak_idx]

            ax_sigma.plot(peak_f, peak_p, marker="o", markersize=6, color="crimson")
            ax_sigma.annotate(
                f"N2 Spindle Peak: {peak_f:.2f} Hz\n({peak_p:.2f} "
                + r"$\mu\mathrm{V}^2/\mathrm{Hz}$)",
                xy=(peak_f, peak_p),
                xytext=(peak_f + 1.0, peak_p * 1.05),
                arrowprops=dict(
                    arrowstyle="->",
                    color="crimson",
                    lw=1.5,
                ),
                fontweight="bold",
                fontsize=9.5,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="crimson",
                    alpha=0.8,
                ),
            )

    ax_sigma.axvspan(
        11.0,
        16.0,
        color="orange",
        alpha=0.20,
        label="Spindle Band (11–16 Hz)",
    )
    ax_sigma.set_title(
        "Zoomed Sigma Band (Sleep Spindles: 8 – 20 Hz)", fontsize=13, fontweight="bold"
    )
    ax_sigma.set_xlabel("Frequency (Hz)", fontsize=11)
    ax_sigma.set_ylabel(
        r"PSD (Linear Scale, $\mu\mathrm{V}^2 / \mathrm{Hz}$)", fontsize=11
    )
    ax_sigma.set_xlim(8.0, 20.0)
    ax_sigma.grid(True, linestyle="--", alpha=0.3)
    ax_sigma.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved PSD comparison figure: {output_path}")


def plot_raw_traces(
    sample_traces: dict,
    stage_config: dict,
    fs: int,
    epoch_sec: int,
    output_path: Path | str = "traces_stage_comparison.png",
):
    """
    Plots representative clean 10-second raw EEG time traces for each stage,
    sharing time and amplitude axes for clear visual comparison.
    """
    if not sample_traces:
        print("No sample traces available to plot.")
        return

    n_stages = len(sample_traces)
    t = np.linspace(0, epoch_sec, fs * epoch_sec, endpoint=False)

    fig, axes = plt.subplots(
        n_stages, 1, figsize=(13, 2.5 * n_stages), sharex=True, sharey=True
    )
    if n_stages == 1:
        axes = [axes]

    for ax, (name, seg) in zip(axes, sample_traces.items()):
        color = stage_config.get(name, {}).get("color", "#333333")
        ax.plot(t, seg, color=color, linewidth=0.9)
        ax.set_title(
            f"Clean {epoch_sec}s Raw Trace — {name} (C3-A2)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel(r"Amplitude ($\mu\mathrm{V}$)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved raw traces figure: {output_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    # 1. Load data
    signal, fs, hypnogram, artefacts, art_idx = load_recording(
        EDF_FILENAME, TARGET_CHANNEL, ARTIFACT_CHANNEL
    )
    print(
        f"Sampling frequency: {fs} Hz | Total signal duration: {len(signal) / fs / 3600:.2f} hours"
    )

    # 2. Extract clean epochs per sleep stage
    print(
        f"\nExtracting clean {EPOCH_SEC}s epochs (rejection criterion: {ARTIFACT_CHANNEL} == 0)..."
    )
    stage_segments, sample_traces = extract_clean_stage_segments(
        signal=signal,
        hypnogram=hypnogram,
        artefacts=artefacts,
        art_idx=art_idx,
        fs=fs,
        epoch_sec=EPOCH_SEC,
        stage_config=STAGE_CONFIG,
    )

    for name, segs in stage_segments.items():
        print(
            f"  • {name:15}: {len(segs):4d} clean epochs ({len(segs) * EPOCH_SEC / 60:.1f} minutes)"
        )

    # 3. Compute Welch PSD
    nperseg = fs * WELCH_WINDOW_SEC
    noverlap = int(nperseg * WELCH_OVERLAP_RATIO)
    print(
        f"\nComputing Welch PSD (window={WELCH_WINDOW_SEC}s, nperseg={nperseg}, overlap={noverlap})..."
    )
    frequencies, stage_psds = compute_psds(
        stage_segments=stage_segments,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    print(
        f"Frequency resolution: {frequencies[1] - frequencies[0]:.2f} Hz | Max frequency: {frequencies[-1]:.1f} Hz"
    )

    # 4. Generate & Save Figures
    print("\nGenerating figures...")
    plot_psd_comparison(
        frequencies, stage_psds, STAGE_CONFIG, "psd_stage_comparison.png"
    )
    plot_raw_traces(
        sample_traces, STAGE_CONFIG, fs, EPOCH_SEC, "traces_stage_comparison.png"
    )

    print("\nProcessing complete! Showing plots...")
    plt.show()


if __name__ == "__main__":
    main()
