"""
Export detected sleep spindles to a CSV file
============================================
Columns: start_s, end_s, length_s, sleep_stage, amplitude_uV, frequency_Hz
"""

import csv

from reading_working import load_recording
from spindles import detect_custom_spindles, detect_yasa_spindles

# 1. Settings
edf_name = "BB0008_SP2.edf"
channel = "C3"
detector = "yasa"  # "yasa" or "custom"
stages = (2, 3, 4)  # NREM sleep stages (2=N2, 3=N3, 4=N4)
out_csv = "spindles.csv"

stage_names = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "N4",
    5: "REM",
    9: "Unscored",
}

# 2. Load data
signal, fs, hypnogram, artefacts, art_idx = load_recording(edf_name, channel)

# 3. Detect spindles
if detector == "yasa":
    spindles = detect_yasa_spindles(
        signal, fs, hypnogram, artefacts, art_idx, stage_code=stages
    )
else:
    is_clean_stage = (
        sum(hypnogram == s for s in stages).astype(bool)
        & (artefacts[:, art_idx] == 0)
    )
    spindles = detect_custom_spindles(signal, fs, is_clean_stage)

# 4. Write CSV
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["start_s", "end_s", "length_s", "sleep_stage", "amplitude_uV", "frequency_Hz"]
    )
    for sp in spindles:
        stage_code = int(hypnogram[sp["start"]])
        writer.writerow(
            [
                f"{sp['start'] / fs:.2f}",
                f"{sp['end'] / fs:.2f}",
                f"{sp['dur']:.2f}",
                stage_names.get(stage_code, str(stage_code)),
                f"{sp['peak_amp']:.2f}",
                f"{sp['freq']:.2f}",
            ]
        )

print(f"Saved {len(spindles)} spindles ({detector}) from {edf_name} ({channel}) to {out_csv}")
