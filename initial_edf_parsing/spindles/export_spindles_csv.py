"""
Export detected sleep spindles to organized CSV files
=====================================================
Directory structure:
  spindles_csv/
    ├── custom/   (BB0008_SP2.csv ... BB0009_SP8.csv)
    ├── bridged/  (BB0008_SP2.csv ... BB0009_SP8.csv)
    └── yasa/     (BB0008_SP2.csv ... BB0009_SP8.csv)

Columns: start_s, end_s, length_s, sleep_stage, amplitude_uV, frequency_Hz, n_channels
"""

import argparse
import csv
from pathlib import Path
import sys
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from reading_working import EEG_CHANNELS, edf_files, load_multichannel_recording, load_recording
from spindles import (
    SPINDLES_CSV_DIR,
    annotate_multichannel_coincidence,
    detect_custom_spindles,
    detect_custom_spindles_bridged,
    detect_yasa_spindles,
    detect_yasa_spindles_multichannel,
    filter_spindles_min_coincidence,
    get_spindle_csv_path,
    save_spindles_csv,
)

ALL_DETECTORS = ("custom", "bridged", "yasa")


def _csv_has_n_channels(csv_path: Path) -> bool:
    """Checks whether an existing CSV file already includes the `n_channels` column."""
    if not csv_path.exists():
        return False
    with open(csv_path, "r", newline="") as f:
        header = next(csv.reader(f), [])
    return "n_channels" in header


def export_spindles_for_recording(
    edf_name: str,
    channel: str = "C3",
    detectors: tuple[str, ...] = ALL_DETECTORS,
    stages: tuple[int, ...] = (2, 3, 4),
    multichannel: bool = True,
    min_channels: int = 1,
    overwrite: bool = True,
) -> dict[str, Path]:
    """Runs specified spindle detectors on one recording and saves organized CSV files.

    When `multichannel=True` (default), detects across all 12 scalp EEG channels
    (`EEG_CHANNELS`) to populate `n_channels` (1..12) for each spindle on `channel`.
    """
    normalized_dets = tuple("bridged" if d == "custom_bridged" else d for d in detectors)
    to_run = [
        det
        for det in normalized_dets
        if overwrite
        or not get_spindle_csv_path(edf_name, det).exists()
        or (multichannel and not _csv_has_n_channels(get_spindle_csv_path(edf_name, det)))
    ]
    if not to_run:
        return {det: get_spindle_csv_path(edf_name, det) for det in normalized_dets}

    saved_paths: dict[str, Path] = {}

    if multichannel:
        ch_list = list(EEG_CHANNELS)
        if channel not in ch_list:
            ch_list.append(channel)
        signals, fs, hypnogram, artefacts, art_idx_map = load_multichannel_recording(
            edf_name, channels=ch_list
        )
        n_samples = len(hypnogram)
        stage_mask = np.isin(hypnogram, stages)
        n2_mask = hypnogram == 2

        need_custom = "custom" in to_run
        need_bridged = "bridged" in to_run

        if need_custom or need_bridged:
            sos = butter(4, (11, 16), btype="bandpass", fs=fs, output="sos")
            custom_by_ch: dict[str, list[dict]] = {}
            bridged_by_ch: dict[str, list[dict]] = {}

            for ch in ch_list:
                sig_ch = signals[ch]
                clean_ch = artefacts[:, art_idx_map[ch]] == 0
                is_clean_n2 = n2_mask & clean_ch
                is_clean_stage = stage_mask & clean_ch

                sigma_ch = sosfiltfilt(sos, sig_ch)
                env_ch = np.abs(hilbert(sigma_ch))

                if need_custom:
                    custom_by_ch[ch] = detect_custom_spindles(
                        sig_ch,
                        fs,
                        is_clean_n2,
                        detect_mask=is_clean_stage,
                        sigma=sigma_ch,
                        envelope=env_ch,
                    )
                if need_bridged:
                    bridged_by_ch[ch] = detect_custom_spindles_bridged(
                        sig_ch,
                        fs,
                        is_clean_n2,
                        detect_mask=is_clean_stage,
                        sigma=sigma_ch,
                        envelope=env_ch,
                    )

        for det in to_run:
            if det == "custom":
                spindles = annotate_multichannel_coincidence(
                    custom_by_ch[channel], custom_by_ch, n_samples
                )
            elif det == "bridged":
                spindles = annotate_multichannel_coincidence(
                    bridged_by_ch[channel], bridged_by_ch, n_samples
                )
            elif det == "yasa":
                yasa_by_ch = detect_yasa_spindles_multichannel(
                    signals, fs, hypnogram, artefacts, art_idx_map, stage_code=stages
                )
                spindles = annotate_multichannel_coincidence(
                    yasa_by_ch[channel], yasa_by_ch, n_samples
                )
            else:
                raise ValueError(f"Unknown detector '{det}'. Expected one of {ALL_DETECTORS}.")

            if min_channels > 1:
                spindles = filter_spindles_min_coincidence(spindles, min_channels=min_channels)

            out_csv = get_spindle_csv_path(edf_name, det)
            save_spindles_csv(out_csv, spindles, fs, hypnogram)
            saved_paths[det] = out_csv
            n_ge5 = sum(1 for sp in spindles if sp.get("n_channels", 1) >= 5)
            rel_path = out_csv.relative_to(SPINDLES_CSV_DIR.parent)
            print(
                f"Saved {len(spindles):4d} spindles ({n_ge5:3d} in >=5 ch) [{det:7s}] "
                f"from {edf_name} ({channel}) -> {rel_path}"
            )
        return saved_paths

    # Single-channel fallback when multichannel=False
    signal, fs, hypnogram, artefacts, art_idx = load_recording(edf_name, channel)
    clean = artefacts[:, art_idx] == 0
    is_clean_n2 = (hypnogram == 2) & clean
    is_clean_stage = np.isin(hypnogram, stages) & clean

    for det in to_run:
        if det == "yasa":
            spindles = detect_yasa_spindles(
                signal, fs, hypnogram, artefacts, art_idx, stage_code=stages
            )
        elif det == "bridged":
            spindles = detect_custom_spindles_bridged(
                signal, fs, is_clean_n2, detect_mask=is_clean_stage
            )
        elif det == "custom":
            spindles = detect_custom_spindles(
                signal, fs, is_clean_n2, detect_mask=is_clean_stage
            )
        else:
            raise ValueError(f"Unknown detector '{det}'. Expected one of {ALL_DETECTORS}.")

        out_csv = get_spindle_csv_path(edf_name, det)
        save_spindles_csv(out_csv, spindles, fs, hypnogram)
        saved_paths[det] = out_csv
        rel_path = out_csv.relative_to(SPINDLES_CSV_DIR.parent)
        print(f"Saved {len(spindles):4d} spindles ({det:7s}) from {edf_name} ({channel}) -> {rel_path}")

    return saved_paths


def export_all_recordings(
    channel: str = "C3",
    detectors: tuple[str, ...] = ALL_DETECTORS,
    stages: tuple[int, ...] = (2, 3, 4),
    multichannel: bool = True,
    min_channels: int = 1,
    overwrite: bool = True,
) -> None:
    """Exports spindle CSVs across all recordings in DATA_DIR."""
    for edf_path in edf_files:
        export_spindles_for_recording(
            edf_path.name,
            channel=channel,
            detectors=detectors,
            stages=stages,
            multichannel=multichannel,
            min_channels=min_channels,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export sleep spindles to organized CSV files.")
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=list(ALL_DETECTORS),
        choices=["custom", "bridged", "yasa"],
        help="Detectors to export (default: custom bridged yasa).",
    )
    parser.add_argument("--channel", default="C3", help="EEG channel name (default: C3).")
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="Sleep stage codes to include (default: 2 3 4 for N2-N4).",
    )
    parser.add_argument(
        "--min-channels",
        type=int,
        default=1,
        help="Only export spindles simultaneously present in at least this many EEG channels (default: 1, all exported with n_channels column).",
    )
    parser.add_argument(
        "--single-channel",
        action="store_true",
        help="Skip 12-channel coincidence calculation and only run on --channel.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip CSV files that already exist on disk (unless missing n_channels).",
    )
    args = parser.parse_args()

    export_all_recordings(
        channel=args.channel,
        detectors=tuple(args.detectors),
        stages=tuple(args.stages),
        multichannel=not args.single_channel,
        min_channels=args.min_channels,
        overwrite=not args.no_overwrite,
    )

