"""
ECG analysis on hypnogram-defined segments with configurable period splitting and selectable NeuroKit profiles
=====================================================================

Pipeline
--------
1. Read the original EDF.
2. Extract ECG and hypnogram.
3. Truncate both to the sleep interval defined by SLEEPTIMEFILE.
4. Optionally split the truncated interval into configured periods.
5. Within each period, split the data into contiguous homogeneous hypnogram segments.
6. For each segment, compute the same ECG quality, heart-rate, time-domain HRV,
   optional frequency-domain HRV, and optional nonlinear HRV measures as in
   ecg_analysis1.py.
7. Write one CSV per input EDF into OUTFOLDER.

Important difference from ecg_analysis1.py
------------------------------------------
This script does NOT split the data into adjacent 20-minute periods before
hypnogram segmentation. Therefore, a homogeneous hypnogram run that crosses a
20-minute boundary remains one segment.

Column convention
-----------------
The output columns are kept compatible with ecg_analysis1.py. Since there are no
20-minute periods here, the column `period` is always 0. The column `start` is
the segment start time in seconds measured from the beginning of the truncated
lights-off/lights-on interval, matching export4.py's segmentation convention.
The column `length` is the segment duration in seconds.
"""

import os
import argparse
import json
import subprocess
import warnings
import numpy as np
import pandas as pd
import pyeeg
import neurokit2 as nk

### CONFIGURATION #############################################################
# All analysis configuration is loaded from a JSON file supplied on the command line.

def load_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)

    # JSON object keys are strings; hypnogram stage codes are integers.
    config["HYPNO_STAGE_MAP"] = {
        int(old_stage): int(new_stage)
        for old_stage, new_stage in config.get("HYPNO_STAGE_MAP", {}).items()
    }

    # Expand environment variables (e.g. $GUILDFORD) in configured paths.
    config["SLEEPTIMEFILE"] = os.path.expandvars(config.get("SLEEPTIMEFILE", ""))
    for profile in config["ANALYSIS_PROFILES"].values():
        profile["OUTFOLDER"] = os.path.expandvars(profile["OUTFOLDER"])

    return config


def apply_config(config):
    global SLEEPTIMEFILE, REMOVESTARTUPWAKE, FS
    global ACTIVE_PROFILE, ANALYSIS_PROFILES, PROFILE, OUTFOLDER
    global MIN_SECONDS_FOR_ANALYSIS, QUALITY_BAD_THRESHOLD
    global FILTER_MIN_LENGTH_SEC, FILTER_MIN_QUALITY_MEAN
    global FILTER_MAX_ARTEFACT_FRACTION, FILTER_MIN_VALID_BEATS_FRACTION
    global FILTER_MIN_BEATS_KEPT
    global COMPUTE_FREQUENCY_DOMAIN_HRV, COMPUTE_NONLINEAR_HRV
    global MIN_BEATS_FOR_BASIC_RATE, MIN_BEATS_FOR_TIME_HRV
    global MIN_BEATS_FOR_ADVANCED_HRV, MIN_LENGTH_SEC_FOR_ADVANCED_HRV
    global HYPNO_STAGE_MAP, ECG_QUALITY_METHOD
    global EDF_EEG_CHANNELS, ECG_COLUMN_INDEX

    SLEEPTIMEFILE = config["SLEEPTIMEFILE"]
    REMOVESTARTUPWAKE = config["REMOVESTARTUPWAKE"]
    FS = float(config["FS"])

    ACTIVE_PROFILE = config["ACTIVE_PROFILE"]
    ANALYSIS_PROFILES = config["ANALYSIS_PROFILES"]
    PROFILE = ANALYSIS_PROFILES[ACTIVE_PROFILE]
    OUTFOLDER = PROFILE["OUTFOLDER"]

    MIN_SECONDS_FOR_ANALYSIS = config["MIN_SECONDS_FOR_ANALYSIS"]
    QUALITY_BAD_THRESHOLD = config["QUALITY_BAD_THRESHOLD"]

    FILTER_MIN_LENGTH_SEC = config["FILTER_MIN_LENGTH_SEC"]
    FILTER_MIN_QUALITY_MEAN = config["FILTER_MIN_QUALITY_MEAN"]
    FILTER_MAX_ARTEFACT_FRACTION = config["FILTER_MAX_ARTEFACT_FRACTION"]
    FILTER_MIN_VALID_BEATS_FRACTION = config["FILTER_MIN_VALID_BEATS_FRACTION"]
    FILTER_MIN_BEATS_KEPT = config["FILTER_MIN_BEATS_KEPT"]

    COMPUTE_FREQUENCY_DOMAIN_HRV = config["COMPUTE_FREQUENCY_DOMAIN_HRV"]
    COMPUTE_NONLINEAR_HRV = config["COMPUTE_NONLINEAR_HRV"]

    MIN_BEATS_FOR_BASIC_RATE = config["MIN_BEATS_FOR_BASIC_RATE"]
    MIN_BEATS_FOR_TIME_HRV = config["MIN_BEATS_FOR_TIME_HRV"]
    MIN_BEATS_FOR_ADVANCED_HRV = config["MIN_BEATS_FOR_ADVANCED_HRV"]
    MIN_LENGTH_SEC_FOR_ADVANCED_HRV = config["MIN_LENGTH_SEC_FOR_ADVANCED_HRV"]

    HYPNO_STAGE_MAP = config["HYPNO_STAGE_MAP"]
    ECG_QUALITY_METHOD = config["ECG_QUALITY_METHOD"]
    EDF_EEG_CHANNELS = config["EDF_EEG_CHANNELS"]
    ECG_COLUMN_INDEX = int(config["ECG_COLUMN_INDEX"])

###############################################################################

def preprocess_hypnogram(hyp):
    """
    Apply the configured sleep-stage mapping to the hypnogram.

    HYPNO_STAGE_MAP defines replacements as:
        original_stage -> new_stage

    Examples:
        {4: 3}       : merge stages 3 and 4 into stage 3
        {3: 2, 4: 2} : merge stages 2, 3 and 4 into stage 2
        {}           : no stage merging
    """
    hyp = np.asarray(hyp, dtype=np.int16).copy()

    for old_stage, new_stage in HYPNO_STAGE_MAP.items():
        hyp[hyp == old_stage] = new_stage

    return hyp

def is_filter_active(value):
    return value is not None and value > 0


def rle_segments(x):
    x = np.asarray(x)
    if len(x) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=x.dtype)

    change_idx = np.where(np.diff(x) != 0)[0] + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(x)]))   # exclusive
    values = x[starts]
    return starts, ends, values


def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
            if len(x) == 0:
                return np.nan
            x = np.asarray(x).ravel()[0]
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def get_first_existing(dct, keys):
    for key in keys:
        if key in dct:
            return dct[key]
    return np.nan


def extract_corrected_rpeaks(candidate):
    """Return a 1D integer array from different NeuroKit peak-correction outputs."""
    if isinstance(candidate, dict):
        for key in ["ECG_R_Peaks", "R_Peaks", "Peaks", "peaks"]:
            if key in candidate:
                return np.asarray(candidate[key], dtype=int)
        return np.array([], dtype=int)

    return np.asarray(candidate, dtype=int).ravel()


def correct_rpeaks_kubios_like(rpeaks, fs):
    """Apply NeuroKit peak correction if enabled by the active profile.

    The function is intentionally defensive because NeuroKit versions differ
    slightly in the exact return format of signal_fixpeaks(). If correction
    fails for a segment, the raw R-peaks are retained and the error text is
    returned instead of failing the whole segment.
    """
    rpeaks = np.asarray(rpeaks, dtype=int).ravel()

    if not PROFILE.get("USE_SIGNAL_FIXPEAKS", False) or len(rpeaks) < 3:
        return rpeaks, ""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = nk.signal_fixpeaks(
                rpeaks,
                sampling_rate=fs,
                iterative=PROFILE.get("SIGNAL_FIXPEAKS_ITERATIVE", True),
                method=PROFILE.get("SIGNAL_FIXPEAKS_METHOD", "Kubios")
            )

        # Common NeuroKit return: artifacts, peaks_corrected
        if isinstance(output, tuple) and len(output) >= 2:
            corrected = extract_corrected_rpeaks(output[1])
        else:
            corrected = extract_corrected_rpeaks(output)

        if len(corrected) >= 2:
            return corrected, ""
        return rpeaks, "signal_fixpeaks_returned_too_few_peaks"

    except Exception as e:
        return rpeaks, f"signal_fixpeaks_failed: {e}"


def empty_result_row(period_idx, hyp_value, start_sec, length_sec):
    return {
        'period': int(period_idx),
        'hypno': int(hyp_value),
        'start': int(start_sec),
        'length': int(length_sec),

        'quality_mean': np.nan,
        'quality_std': np.nan,
        'quality_min': np.nan,
        'quality_max': np.nan,
        'artefact_fraction': np.nan,
        'valid_beats_fraction': np.nan,

        'n_samples': np.nan,
        'n_beats_raw_detected': np.nan,
        'n_beats_corrected': np.nan,
        'n_beats_changed_estimate': np.nan,
        'correction_fraction': np.nan,
        'n_beats_detected': np.nan,
        'n_beats_kept': np.nan,

        'mean_hr_bpm': np.nan,
        'median_hr_bpm': np.nan,
        'mean_rr_ms': np.nan,
        'median_rr_ms': np.nan,

        'sdnn_ms': np.nan,
        'rmssd_ms': np.nan,
        'pnn50_percent': np.nan,

        'lf_power': np.nan,
        'hf_power': np.nan,
        'lf_hf_ratio': np.nan,

        'poincare_sd1': np.nan,
        'poincare_sd2': np.nan,
        'sample_entropy': np.nan,

        'segment_ok': 0,
        'usable_segment': 0,
        'failed_filters': '',
        'error': ''
    }


def evaluate_usability(result):
    failed = []

    if int(result.get('segment_ok', 0)) != 1:
        failed.append('segment_ok')

    if is_filter_active(FILTER_MIN_LENGTH_SEC):
        val = safe_float(result.get('length'))
        if np.isnan(val) or val < FILTER_MIN_LENGTH_SEC:
            failed.append('min_length')

    if is_filter_active(FILTER_MIN_QUALITY_MEAN):
        val = safe_float(result.get('quality_mean'))
        if np.isnan(val) or val < FILTER_MIN_QUALITY_MEAN:
            failed.append('min_quality_mean')

    if is_filter_active(FILTER_MAX_ARTEFACT_FRACTION):
        val = safe_float(result.get('artefact_fraction'))
        if np.isnan(val) or val > FILTER_MAX_ARTEFACT_FRACTION:
            failed.append('max_artefact_fraction')

    if is_filter_active(FILTER_MIN_VALID_BEATS_FRACTION):
        val = safe_float(result.get('valid_beats_fraction'))
        if np.isnan(val) or val < FILTER_MIN_VALID_BEATS_FRACTION:
            failed.append('min_valid_beats_fraction')

    if is_filter_active(FILTER_MIN_BEATS_KEPT):
        val = safe_float(result.get('n_beats_kept'))
        if np.isnan(val) or val < FILTER_MIN_BEATS_KEPT:
            failed.append('min_beats_kept')

    result['failed_filters'] = ';'.join(failed)
    result['usable_segment'] = 0 if len(failed) > 0 else 1
    return result


def analyze_ecg_segment(ecg_seg, fs, period_idx, hyp_value, start_sec, length_sec):
    result = empty_result_row(period_idx, hyp_value, start_sec, length_sec)
    result['n_samples'] = int(len(ecg_seg))

    seg_len_sec_exact = len(ecg_seg) / fs

    if seg_len_sec_exact < MIN_SECONDS_FOR_ANALYSIS:
        result['error'] = f'too_short_for_analysis(<{MIN_SECONDS_FOR_ANALYSIS}s)'
        return evaluate_usability(result)

    try:
        ecg_seg = np.asarray(ecg_seg, dtype=np.float64)

        # Quality estimation
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                quality = nk.ecg_quality(ecg_seg, sampling_rate=fs, method=ECG_QUALITY_METHOD)
            quality = np.asarray(quality, dtype=np.float64)

            if quality.size > 0:
                result['quality_mean'] = safe_float(np.nanmean(quality))
                result['quality_std'] = safe_float(np.nanstd(quality))
                result['quality_min'] = safe_float(np.nanmin(quality))
                result['quality_max'] = safe_float(np.nanmax(quality))

                if is_filter_active(QUALITY_BAD_THRESHOLD):
                    result['artefact_fraction'] = safe_float(np.mean(quality < QUALITY_BAD_THRESHOLD))
        except Exception:
            pass

        # Cleaning + R peak detection
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ecg_clean = nk.ecg_clean(
                    ecg_seg,
                    sampling_rate=fs,
                    method=PROFILE.get("ECG_CLEAN_METHOD", "neurokit")
                )

                try:
                    _, info = nk.ecg_peaks(
                        ecg_clean,
                        sampling_rate=fs,
                        method=PROFILE.get("ECG_PEAKS_METHOD", "neurokit"),
                        correct_artifacts=PROFILE.get("CORRECT_ARTIFACTS_IN_ECG_PEAKS", False)
                    )
                except TypeError:
                    # Older NeuroKit versions may not support all keyword arguments.
                    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)

            rpeaks_raw = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
            rpeaks, correction_error = correct_rpeaks_kubios_like(rpeaks_raw, fs)

            result['n_beats_raw_detected'] = int(len(rpeaks_raw))
            result['n_beats_corrected'] = int(len(rpeaks))
            result['n_beats_changed_estimate'] = int(abs(len(rpeaks_raw) - len(rpeaks)))
            if len(rpeaks_raw) > 0:
                result['correction_fraction'] = safe_float(
                    abs(len(rpeaks_raw) - len(rpeaks)) / len(rpeaks_raw)
                )
            if correction_error:
                result['error'] = correction_error
        except Exception as e:
            result['error'] = f'rpeak_detection_failed: {e}'
            return evaluate_usability(result)

        result['n_beats_detected'] = int(len(rpeaks))

        if len(rpeaks) < MIN_BEATS_FOR_BASIC_RATE:
            result['n_beats_kept'] = int(len(rpeaks))
            result['valid_beats_fraction'] = np.nan
            result['error'] = 'too_few_rpeaks'
            result['segment_ok'] = 1
            return evaluate_usability(result)

        rr_ms = np.diff(rpeaks) / fs * 1000.0

        # Plausibility filtering on RR
        keep_mask = (
            np.isfinite(rr_ms)
            & (rr_ms >= PROFILE.get("RR_MIN_MS", 300.0))
            & (rr_ms <= PROFILE.get("RR_MAX_MS", 2000.0))
        )
        rr_kept = rr_ms[keep_mask]

        if len(rr_kept) > 0:
            result['n_beats_kept'] = int(len(rr_kept) + 1)
        else:
            result['n_beats_kept'] = 0

        if result['n_beats_detected'] > 0:
            result['valid_beats_fraction'] = safe_float(
                result['n_beats_kept'] / result['n_beats_detected']
            )

        # Basic rate / RR stats
        if len(rr_kept) > 0:
            hr_bpm = 60000.0 / rr_kept
            result['mean_hr_bpm'] = safe_float(np.nanmean(hr_bpm))
            result['median_hr_bpm'] = safe_float(np.nanmedian(hr_bpm))
            result['mean_rr_ms'] = safe_float(np.nanmean(rr_kept))
            result['median_rr_ms'] = safe_float(np.nanmedian(rr_kept))

        # Time-domain HRV from kept RR only
        if len(rr_kept) >= 2:
            result['sdnn_ms'] = safe_float(np.nanstd(rr_kept, ddof=1))

            diff_rr = np.diff(rr_kept)
            if len(diff_rr) > 0:
                result['rmssd_ms'] = safe_float(np.sqrt(np.nanmean(diff_rr ** 2)))
                result['pnn50_percent'] = safe_float(100.0 * np.mean(np.abs(diff_rr) > 50.0))

        # Advanced HRV only if long enough and enough beats
        advanced_hrv_allowed = (
            result['n_beats_kept'] is not np.nan
            and safe_float(result['n_beats_kept']) >= MIN_BEATS_FOR_ADVANCED_HRV
            and seg_len_sec_exact >= MIN_LENGTH_SEC_FOR_ADVANCED_HRV
        )

        if advanced_hrv_allowed and (COMPUTE_FREQUENCY_DOMAIN_HRV or COMPUTE_NONLINEAR_HRV):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hrv = nk.hrv(rpeaks, sampling_rate=fs, show=False)

                if isinstance(hrv, pd.DataFrame) and len(hrv) > 0:
                    hrv_row = hrv.iloc[0].to_dict()

                    # Fill time-domain values from NK only if still missing
                    if np.isnan(result['sdnn_ms']):
                        result['sdnn_ms'] = safe_float(get_first_existing(hrv_row, ['HRV_SDNN']))
                    if np.isnan(result['rmssd_ms']):
                        result['rmssd_ms'] = safe_float(get_first_existing(hrv_row, ['HRV_RMSSD']))
                    if np.isnan(result['pnn50_percent']):
                        result['pnn50_percent'] = safe_float(get_first_existing(hrv_row, ['HRV_pNN50']))

                    if COMPUTE_NONLINEAR_HRV:
                        result['poincare_sd1'] = safe_float(get_first_existing(hrv_row, ['HRV_SD1']))
                        result['poincare_sd2'] = safe_float(get_first_existing(hrv_row, ['HRV_SD2']))
                        result['sample_entropy'] = safe_float(get_first_existing(hrv_row, ['HRV_SampEn']))

                if COMPUTE_FREQUENCY_DOMAIN_HRV:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            hrv_freq = nk.hrv_frequency(
                                rpeaks,
                                sampling_rate=fs,
                                psd_method=PROFILE.get("HRV_FREQUENCY_PSD_METHOD", "welch"),
                                interpolation_rate=PROFILE.get("HRV_FREQUENCY_INTERPOLATION_RATE", 4),
                                show=False
                            )

                        if isinstance(hrv_freq, pd.DataFrame) and len(hrv_freq) > 0:
                            hrv_freq_row = hrv_freq.iloc[0].to_dict()
                            result['lf_power'] = safe_float(get_first_existing(hrv_freq_row, ['HRV_LF']))
                            result['hf_power'] = safe_float(get_first_existing(hrv_freq_row, ['HRV_HF']))
                            result['lf_hf_ratio'] = safe_float(get_first_existing(hrv_freq_row, ['HRV_LFHF']))
                    except Exception:
                        pass

            except Exception:
                pass

        result['segment_ok'] = 1
        return evaluate_usability(result)

    except Exception as e:
        result['error'] = str(e)
        return evaluate_usability(result)



def parse_args():
    parser = argparse.ArgumentParser(
        description="ECG analysis on hypnogram-defined segments using a JSON configuration file."
    )
    parser.add_argument("input_edf", help="Input EDF file")
    parser.add_argument("config_json", help="JSON configuration file")
    return parser.parse_args()


def segment_and_analyze(ecg, hyp, fs, fs_int, splitting):
    """Split the truncated sleep interval into periods, then segment each period by hypnogram.

    splitting must be one of:
        {}                              -> no splitting (one period)
        {"by": "length", "param": S}  -> periods of S seconds
        {"by": "number", "param": N}  -> N periods of equal length

    Splitting is applied to the already sleep-time-truncated ECG and hypnogram.
    """
    rows = []
    n_samples = len(ecg)

    if not splitting:
        period_bounds = [(0, n_samples)]
        print("No period splitting: using the full truncated interval as one period.")
    else:
        split_by = splitting.get("by")
        param = splitting.get("param")

        if split_by == "length":
            period_length_sec = float(param)
            if period_length_sec <= 0:
                raise ValueError("splitting.param must be > 0 for splitting by length.")

            period_samples = int(round(period_length_sec * fs))
            if period_samples <= 0:
                raise ValueError("Configured splitting length is shorter than one sample.")

            period_bounds = [
                (p_start, min(p_start + period_samples, n_samples))
                for p_start in range(0, n_samples, period_samples)
            ]
            print(f"Period splitting by length: {period_length_sec:g} seconds.")

        elif split_by == "number":
            n_periods = int(param)
            if n_periods <= 0:
                raise ValueError("splitting.param must be a positive integer for splitting by number.")
            if n_periods > n_samples:
                raise ValueError("Number of periods cannot exceed the number of samples.")

            # Integer boundaries cover the complete truncated interval exactly once.
            bounds = np.linspace(0, n_samples, n_periods + 1, dtype=int)
            period_bounds = list(zip(bounds[:-1], bounds[1:]))
            print(f"Period splitting by number: {n_periods} equal-length periods.")

        else:
            raise ValueError("splitting.by must be either 'length' or 'number'.")

    print("Number of periods:", len(period_bounds))

    for period_idx, (p_start, p_end) in enumerate(period_bounds):
        ecg_period = ecg[p_start:p_end]
        hyp_period = hyp[p_start:p_end]

        starts, ends, values = rle_segments(hyp_period)
        print(f"Period {period_idx:02d}: found {len(starts)} homogeneous hypnogram segments.")

        for start, end, hyp_value in zip(starts, ends, values):
            if end <= start:
                continue

            ecg_seg = ecg_period[start:end]
            if len(ecg_seg) == 0:
                continue

            start_sec_within_period = int(start // fs_int)
            length_sec = int(len(ecg_seg) // fs_int)

            row = analyze_ecg_segment(
                ecg_seg=ecg_seg,
                fs=fs,
                period_idx=period_idx,
                hyp_value=hyp_value,
                start_sec=start_sec_within_period,
                length_sec=length_sec
            )
            rows.append(row)

    return rows


def main():
    args = parse_args()
    print(args)

    config = load_config(args.config_json)
    apply_config(config)

    os.makedirs(OUTFOLDER, exist_ok=True)

    filename = args.input_edf
    print("Active analysis profile:", ACTIVE_PROFILE)
    print("Output folder:", OUTFOLDER)
    print("Input EDF:", filename)
    print("Sleep-time file:", SLEEPTIMEFILE if SLEEPTIMEFILE else "<disabled>")
    print("Period splitting:", config.get("splitting", {}))

    fileroot = os.path.splitext(os.path.split(filename)[1])[0]

    eeg, art, hyp, fs, ch = pyeeg.datatools.io.special.guildford.read_file(
        filename,
        eeg=EDF_EEG_CHANNELS,
        artefact=False,
        ecg=True,
        hypno=True
    )

    infile = os.path.splitext(os.path.split(filename)[1])[0]
    print("Infile:", infile)
    print("Channels returned:", ch)
    print("Sampling frequency raw:", fs)

    fs = float(np.asarray(fs).ravel()[0])
    fs_int = int(round(fs))
    print("Sampling frequency used:", fs)

    # Assumption:
    # eeg[:, 0] = C3
    # eeg[:, 1] = ECG
    ecg = np.asarray(eeg[:, ECG_COLUMN_INDEX], dtype=np.float64)
    hyp = np.asarray(hyp, dtype=np.int16)

    hyp = preprocess_hypnogram(hyp)

    # Sleep-time truncation
    if SLEEPTIMEFILE:
        sleeptimefile = os.path.expandvars(SLEEPTIMEFILE)
        tstart, tout, ton = pyeeg.datatools.io.special.guildford.sleeptimeinfo(infile, sleeptimefile)
        first_index, last_index = pyeeg.sleep.tools.find_lights_out_on_limits(
            tstart,
            tout,
            ton,
            FS,
            remove_startup_wake=REMOVESTARTUPWAKE,
            hyp=None,
            wake_code=None
        )
        print('...from', (tout - tstart), 'to', (ton - tstart))
    else:
        first_index, last_index = 0, len(ecg)

    if first_index < 0 or last_index > len(ecg) or first_index >= last_index:
        print("Sleeptimeinfo error: (", first_index, ",", last_index, ") not consistent with length", len(ecg))
        first_index, last_index = 0, len(ecg)

    ecg = ecg[first_index:last_index]
    hyp = hyp[first_index:last_index]

    if len(ecg) != len(hyp):
        print("Length mismatch after truncation:", len(ecg), len(hyp))
        return 1

    print("Truncated length in samples:", len(ecg))

    rows = segment_and_analyze(
        ecg=ecg,
        hyp=hyp,
        fs=fs,
        fs_int=fs_int,
        splitting=config.get("splitting", {})
    )

    df = pd.DataFrame(rows, columns=[
        'period',
        'hypno',
        'start',
        'length',

        'quality_mean',
        'quality_std',
        'quality_min',
        'quality_max',
        'artefact_fraction',
        'valid_beats_fraction',

        'n_samples',
        'n_beats_raw_detected',
        'n_beats_corrected',
        'n_beats_changed_estimate',
        'correction_fraction',
        'n_beats_detected',
        'n_beats_kept',

        'mean_hr_bpm',
        'median_hr_bpm',
        'mean_rr_ms',
        'median_rr_ms',
        'sdnn_ms',
        'rmssd_ms',
        'pnn50_percent',

        'lf_power',
        'hf_power',
        'lf_hf_ratio',

        'poincare_sd1',
        'poincare_sd2',
        'sample_entropy',

        'segment_ok',
        'usable_segment',
        'failed_filters',
        'error'
    ])

    outfile_csv = os.path.join(OUTFOLDER, f"{fileroot}.csv")
    df.to_csv(outfile_csv, index=False)

    print("Saved CSV:", outfile_csv)
    print("Rows written:", len(df))
    print("Usable rows:", int(df['usable_segment'].sum()) if len(df) > 0 else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
