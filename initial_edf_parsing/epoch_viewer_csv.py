"""
CSV-Backed Sleep Epoch & Spindle Viewer
=======================================
Interactive PSG epoch viewer that loads pre-exported sleep spindle detections
directly from `spindles_csv/{custom,bridged,yasa}/<recording>.csv` for instant
startup and fast switching across recordings.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from export_spindles_csv import export_spindles_for_recording
from reading_working import edf_files, load_recording
from spindles import (
    SPINDLES_CSV_DIR,
    filter_spindles_min_coincidence,
    load_recording_spindles_csv,
)

# ==============================================================================
# 1. Settings & Initial Recording Load
# ==============================================================================
edf_names = [p.name for p in edf_files]
current_rec_idx = (
    edf_names.index("BB0008_SP3.edf") if "BB0008_SP3.edf" in edf_names else 0
)
channel = "C3"
epoch_sec = 30  # 30-second clinical PSG epoch
allowed_stages = None  # None = all stages in CSV (N2, N3, N4), or {"N2"} for N2-only
min_channels_threshold = 5  # Coincidence requirement across the 12 scalp EEG channels

stage_names = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "N4", 5: "REM", 9: "Unscored"}
stage_colors = {
    0: "#ff7f0e",
    1: "#bcbd22",
    2: "#1f77b4",
    3: "#2ca02c",
    4: "#d62728",
    5: "#9467bd",
    9: "#7f7f7f",
}

current_epoch = 0
y_lim = 100.0  # Fixed amplitude scale (µV), adjustable with Up/Down keys
show_custom = True
show_bridged = True
show_yasa = True
show_artifacts = True
filter_min_channels = True  # Toggleable with button ">=5 Ch: ON/OFF" or key "5"

# State variables populated by load_current_recording()
edf_name = ""
signal = np.array([])
fs = 256
hypnogram = np.array([])
art_mask = np.array([], dtype=bool)
sigma = np.array([])
envelope = np.array([])
sigma_thresh = 0.0
sigma_high_thresh = 0.0
epoch_len = 7680
n_epochs = 1
t = np.array([])
art_epochs = np.array([], dtype=bool)
n_art_epochs = 0
art_pct = 0.0
art_intervals: list[tuple[int, int]] = []
custom_spindles: list[dict] = []
bridged_spindles: list[dict] = []
yasa_spindles: list[dict] = []


def load_current_recording():
    """Loads EEG, hypnogram, artifacts, and CSV spindles for `edf_names[current_rec_idx]`."""
    global edf_name, signal, fs, hypnogram, art_mask, sigma, envelope
    global sigma_thresh, sigma_high_thresh, epoch_len, n_epochs, t
    global art_epochs, n_art_epochs, art_pct, art_intervals
    global custom_spindles, bridged_spindles, yasa_spindles, current_epoch

    edf_name = edf_names[current_rec_idx]
    print(f"Loading recording {edf_name} ({channel}) & spindle CSVs...")
    signal, fs, hypnogram, artefacts, art_idx = load_recording(edf_name, channel)
    art_mask = artefacts[:, art_idx] != 0

    is_n2 = (hypnogram == 2) & (~art_mask)
    sos = butter(4, (11, 16), btype="bandpass", fs=fs, output="sos")
    sigma = sosfiltfilt(sos, signal)
    envelope = np.abs(hilbert(sigma))
    env_mean = float(np.mean(envelope[is_n2]))
    env_std = float(np.std(envelope[is_n2]))
    sigma_thresh = env_mean + 1.5 * env_std
    sigma_high_thresh = env_mean + 2.0 * env_std

    epoch_len = fs * epoch_sec
    n_epochs = len(signal) // epoch_len
    t = np.linspace(0, epoch_sec, epoch_len, endpoint=False)
    current_epoch = int(np.clip(current_epoch, 0, n_epochs - 1))

    ep_art_reshaped = art_mask[: n_epochs * epoch_len].reshape(n_epochs, epoch_len)
    art_epochs = np.any(ep_art_reshaped, axis=1)
    n_art_epochs = int(np.sum(art_epochs))
    art_pct = (n_art_epochs / n_epochs * 100) if n_epochs > 0 else 0.0

    diff = np.diff(np.concatenate(([0], art_epochs.astype(np.int8), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    art_intervals = [(int(s), int(e - s)) for s, e in zip(starts, ends)]

    # Ensure CSVs exist for this recording, then load from CSV
    export_spindles_for_recording(edf_name, channel=channel, overwrite=False)
    csv_sp = load_recording_spindles_csv(
        edf_name, fs, allowed_stages=allowed_stages, base_dir=SPINDLES_CSV_DIR
    )
    custom_spindles = csv_sp["custom"]
    bridged_spindles = csv_sp["bridged"]
    yasa_spindles = csv_sp["yasa"]
    c_ge5 = len(filter_spindles_min_coincidence(custom_spindles, min_channels=min_channels_threshold))
    b_ge5 = len(filter_spindles_min_coincidence(bridged_spindles, min_channels=min_channels_threshold))
    y_ge5 = len(filter_spindles_min_coincidence(yasa_spindles, min_channels=min_channels_threshold))
    print(
        f"Loaded from CSV ({edf_name}): "
        f"Custom={len(custom_spindles)} (>={min_channels_threshold}ch: {c_ge5}) | "
        f"Bridged={len(bridged_spindles)} (>={min_channels_threshold}ch: {b_ge5}) | "
        f"YASA={len(yasa_spindles)} (>={min_channels_threshold}ch: {y_ge5}) spindles."
    )


def get_active_spindles(spindles: list[dict]) -> list[dict]:
    """Filters `spindles` by `min_channels_threshold` when `filter_min_channels` is enabled."""
    if not filter_min_channels:
        return spindles
    return filter_spindles_min_coincidence(spindles, min_channels=min_channels_threshold)


load_current_recording()

# 2. Figure Layout: Hypnogram (top), Raw EEG (middle), Sigma 11-16 Hz (bottom)
fig, (ax_hyp, ax_eeg, ax_sig) = plt.subplots(
    3, 1, figsize=(13, 7.5), gridspec_kw={"height_ratios": [1, 2.5, 1.5]}
)
plt.subplots_adjust(bottom=0.20, hspace=0.32)

hyp_cursor = None
eeg_line = None
sig_line = None
env_line = None
base_thresh_line = None
peak_thresh_line = None
art_span_col = None
art_bar_col = None
highlights = []


def update_hypnogram_title():
    c_act = len(get_active_spindles(custom_spindles))
    b_act = len(get_active_spindles(bridged_spindles))
    y_act = len(get_active_spindles(yasa_spindles))
    mode_tag = f">={min_channels_threshold} Ch" if filter_min_channels else "All 1+ Ch"
    ax_hyp.set_title(
        f"Full-Night Hypnogram — {edf_name} [CSV Spindles ({mode_tag}): "
        f"C={c_act}/{len(custom_spindles)}, B={b_act}/{len(bridged_spindles)}, Y={y_act}/{len(yasa_spindles)}]",
        fontsize=9.5,
        fontweight="bold",
    )


def rebuild_recording_axes():
    """Draws the full-night hypnogram and initializes trace lines for the loaded recording."""
    global hyp_cursor, eeg_line, sig_line, env_line, base_thresh_line, peak_thresh_line
    global art_span_col, art_bar_col

    ax_hyp.clear()
    ax_eeg.clear()
    ax_sig.clear()

    hyp_epochs = hypnogram[: n_epochs * epoch_len : epoch_len].astype(float)
    hyp_epochs[hyp_epochs == 9] = np.nan

    art_span_col = ax_hyp.broken_barh(
        art_intervals,
        (-1.5, 7.0),
        facecolors="#ff4d4d",
        edgecolors="none",
        alpha=0.16,
        zorder=0,
    )
    art_bar_col = ax_hyp.broken_barh(
        art_intervals,
        (-1.35, 0.7),
        facecolors="#d62728",
        edgecolors="none",
        alpha=0.85,
        zorder=1,
        label=f"Artifact ({n_art_epochs}/{n_epochs} ep, {art_pct:.1f}%)",
    )
    art_span_col.set_visible(show_artifacts)
    art_bar_col.set_visible(show_artifacts)

    ax_hyp.step(
        np.arange(n_epochs),
        hyp_epochs,
        where="post",
        color="#222222",
        linewidth=1.0,
        zorder=2,
        label="Sleep Stage",
    )
    ax_hyp.set_yticks([-1, 0, 1, 2, 3, 4, 5])
    ax_hyp.set_yticklabels(["Art", "W", "N1", "N2", "N3", "N4", "REM"], fontsize=8)
    ax_hyp.set_ylim(-1.5, 5.5)
    ax_hyp.invert_yaxis()
    ax_hyp.set_xlim(0, n_epochs)
    update_hypnogram_title()
    ax_hyp.grid(True, linestyle="--", alpha=0.3)
    ax_hyp.legend(loc="upper right", fontsize=8, framealpha=0.9)
    hyp_cursor = ax_hyp.axvline(current_epoch, color="#0066cc", linewidth=2.0, zorder=5)

    # Middle: Raw EEG trace
    (eeg_line,) = ax_eeg.plot(
        t, signal[:epoch_len], linewidth=0.8, label=f"{channel} EEG"
    )
    ax_eeg.set_xlim(0, epoch_sec)
    ax_eeg.set_ylabel("EEG (µV)")
    ax_eeg.grid(True, linestyle="--", alpha=0.35)

    # Bottom: Sigma (11-16 Hz) + Envelope + Thresholds
    (sig_line,) = ax_sig.plot(
        t, sigma[:epoch_len], color="#555555", linewidth=0.7, label="Sigma (11–16 Hz)"
    )
    (env_line,) = ax_sig.plot(
        t,
        envelope[:epoch_len],
        color="#d62728",
        linewidth=1.2,
        label="Hilbert Envelope",
    )
    base_thresh_line = ax_sig.axhline(
        sigma_thresh,
        color="#d62728",
        linestyle="--",
        linewidth=1.0,
        label=f"Base 1.5 SD ({sigma_thresh:.1f} µV)",
    )
    peak_thresh_line = ax_sig.axhline(
        sigma_high_thresh,
        color="#2ca02c",
        linestyle=":",
        linewidth=1.1,
        label=f"Peak 2.0 SD ({sigma_high_thresh:.1f} µV)",
    )
    ax_sig.set_xlim(0, epoch_sec)
    ax_sig.set_ylim(-35, 35)
    ax_sig.set_xlabel(
        "Time within epoch (s)   [Keys: ◀/▶ epoch | S/W spindle | E/Q artifact | [/] rec | ↑/↓ zoom µV | 1/2/3/4/5 toggle]"
    )
    ax_sig.set_ylabel("Sigma (µV)")
    ax_sig.grid(True, linestyle="--", alpha=0.35)
    ax_sig.legend(loc="upper right", fontsize=8, ncol=4)


def draw_spindles(spindles, start, end, color, edge, y_pos, tag):
    count = 0
    for sp in get_active_spindles(spindles):
        if sp["end"] > start and sp["start"] < end:
            count += 1
            s_t = max(0.0, (sp["start"] - start) / fs)
            e_t = min(float(epoch_sec), (sp["end"] - start) / fs)
            n_ch = int(sp.get("n_channels", 1))
            for ax in (ax_eeg, ax_sig):
                highlights.append(
                    ax.axvspan(
                        s_t,
                        e_t,
                        facecolor=color,
                        edgecolor=edge,
                        alpha=0.35,
                        label=f"{tag} (CSV)" if (ax is ax_eeg and count == 1) else "",
                    )
                )
            highlights.append(
                ax_eeg.text(
                    (s_t + e_t) / 2,
                    y_pos,
                    f"{tag} ({n_ch}ch)\n{sp['dur']:.2f}s | {sp['freq']:.1f}Hz",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor=edge,
                        alpha=0.9,
                    ),
                )
            )
    return count


def update_view():
    while highlights:
        artist = highlights.pop()
        try:
            artist.remove()
        except ValueError:
            pass

    start = current_epoch * epoch_len
    end = start + epoch_len
    stage = int(hypnogram[start])
    sec = current_epoch * epoch_sec
    time_str = f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"

    hyp_cursor.set_xdata([current_epoch, current_epoch])
    eeg_line.set_ydata(signal[start:end])
    eeg_line.set_color(stage_colors.get(stage, "#1f77b4"))
    ax_eeg.set_ylim(-y_lim, y_lim)

    sig_line.set_ydata(sigma[start:end])
    env_line.set_ydata(envelope[start:end])

    # Shade exact artifact intervals within the epoch in red
    art_ep = art_mask[start:end]
    if show_artifacts and np.any(art_ep):
        diff = np.diff(np.concatenate(([0], art_ep.astype(np.int8), [0])))
        for a_s, a_e in zip(np.where(diff == 1)[0], np.where(diff == -1)[0]):
            highlights.append(
                ax_eeg.axvspan(
                    a_s / fs, a_e / fs, color="red", alpha=0.15, label="Artifact"
                )
            )

    c_cnt = (
        draw_spindles(
            custom_spindles, start, end, "orange", "orange", y_lim * 0.82, "Custom"
        )
        if show_custom
        else 0
    )
    b_cnt = (
        draw_spindles(
            bridged_spindles, start, end, "#c8f7c5", "#2ca02c", y_lim * 0.56, "Bridged"
        )
        if show_bridged
        else 0
    )
    y_cnt = (
        draw_spindles(
            yasa_spindles, start, end, "#d0edff", "#007acc", y_lim * 0.30, "YASA"
        )
        if show_yasa
        else 0
    )

    if np.any(art_ep):
        art_dur = float(np.sum(art_ep) / fs)
        art_status = f"ARTIFACT ({art_dur:.1f}s)"
    else:
        art_status = "CLEAN"

    ch_tag = f">={min_channels_threshold}ch" if filter_min_channels else "1+ch"
    ax_eeg.set_title(
        f"[{edf_name}] Epoch {current_epoch + 1}/{n_epochs} | {time_str} | {stage_names.get(stage, stage)} "
        f"| [{art_status}] | CSV ({ch_tag}) — Custom: {c_cnt}  Bridged: {b_cnt}  YASA: {y_cnt} | Scale: ±{int(y_lim)} µV",
        fontsize=10.5,
        fontweight="bold",
    )

    handles, labels = ax_eeg.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_eeg.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    fig.canvas.draw_idle()


# ==============================================================================
# 3. Navigation & Callbacks
# ==============================================================================
def step_epoch(delta):
    global current_epoch
    current_epoch = int(np.clip(current_epoch + delta, 0, n_epochs - 1))
    update_view()


def step_recording(delta):
    global current_rec_idx, current_epoch
    current_rec_idx = (current_rec_idx + delta) % len(edf_names)
    current_epoch = 0
    highlights.clear()
    load_current_recording()
    rebuild_recording_axes()
    update_view()


def jump_spindle(direction=1):
    global current_epoch
    c_act = get_active_spindles(custom_spindles)
    b_act = get_active_spindles(bridged_spindles)
    y_act = get_active_spindles(yasa_spindles)
    active = (
        (c_act if show_custom else [])
        + (b_act if show_bridged else [])
        + (y_act if show_yasa else [])
    )
    if not active:
        active = c_act + b_act + y_act
    epochs = (
        range(current_epoch + 1, n_epochs)
        if direction > 0
        else range(current_epoch - 1, -1, -1)
    )
    for ep in epochs:
        s, e = ep * epoch_len, (ep + 1) * epoch_len
        if any(sp["end"] > s and sp["start"] < e for sp in active):
            current_epoch = ep
            update_view()
            return


def jump_artifact(direction=1):
    global current_epoch
    epochs = (
        range(current_epoch + 1, n_epochs)
        if direction > 0
        else range(current_epoch - 1, -1, -1)
    )
    for ep in epochs:
        if art_epochs[ep]:
            current_epoch = ep
            update_view()
            return


def toggle_custom(event=None):
    global show_custom
    show_custom = not show_custom
    btn_custom.label.set_text(f"Custom: {'ON' if show_custom else 'OFF'}")
    update_view()


def toggle_bridged(event=None):
    global show_bridged
    show_bridged = not show_bridged
    btn_bridged.label.set_text(f"Bridged: {'ON' if show_bridged else 'OFF'}")
    update_view()


def toggle_yasa(event=None):
    global show_yasa
    show_yasa = not show_yasa
    btn_yasa.label.set_text(f"YASA: {'ON' if show_yasa else 'OFF'}")
    update_view()


def toggle_artifacts(event=None):
    global show_artifacts
    show_artifacts = not show_artifacts
    art_span_col.set_visible(show_artifacts)
    art_bar_col.set_visible(show_artifacts)
    btn_art.label.set_text(f"Artifacts: {'ON' if show_artifacts else 'OFF'}")
    btn_art.color = "#ffcccc" if show_artifacts else "#e0e0e0"
    update_view()


def toggle_min_channels(event=None):
    global filter_min_channels
    filter_min_channels = not filter_min_channels
    btn_min_ch.label.set_text(
        f">={min_channels_threshold} Ch: {'ON' if filter_min_channels else 'OFF'}"
    )
    btn_min_ch.color = "#e6ccff" if filter_min_channels else "#e0e0e0"
    update_hypnogram_title()
    update_view()


def on_click(event):
    global current_epoch
    if event.inaxes is ax_hyp and event.xdata is not None:
        current_epoch = int(np.clip(event.xdata, 0, n_epochs - 1))
        update_view()


def on_key(event):
    global y_lim
    if event.key in ("right", "d"):
        step_epoch(1)
    elif event.key in ("left", "a"):
        step_epoch(-1)
    elif event.key in ("s", "S"):
        jump_spindle(1)
    elif event.key in ("w", "W"):
        jump_spindle(-1)
    elif event.key in ("e", "E"):
        jump_artifact(1)
    elif event.key in ("q", "Q"):
        jump_artifact(-1)
    elif event.key == "]":
        step_recording(1)
    elif event.key == "[":
        step_recording(-1)
    elif event.key == "up":
        y_lim = max(25.0, y_lim - 25.0)
        update_view()
    elif event.key == "down":
        y_lim = min(400.0, y_lim + 25.0)
        update_view()
    elif event.key == "1":
        toggle_custom()
    elif event.key == "2":
        toggle_bridged()
    elif event.key == "3":
        toggle_yasa()
    elif event.key == "4":
        toggle_artifacts()
    elif event.key == "5":
        toggle_min_channels()


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)

# ==============================================================================
# 4. Buttons (Single Clean Row)
# ==============================================================================
btn_prev_rec = Button(plt.axes([0.015, 0.04, 0.065, 0.055]), "◀◀ Rec", color="#e2e8f0")
btn_next_rec = Button(plt.axes([0.085, 0.04, 0.065, 0.055]), "Rec ▶▶", color="#e2e8f0")
btn_prev = Button(plt.axes([0.155, 0.04, 0.06, 0.055]), "◀ Ep")
btn_next = Button(plt.axes([0.22, 0.04, 0.06, 0.055]), "Ep ▶")
btn_pspn = Button(plt.axes([0.285, 0.04, 0.09, 0.055]), "◀ Prev Spn", color="#ffebcc")
btn_nspn = Button(plt.axes([0.38, 0.04, 0.09, 0.055]), "Next Spn ▶", color="#ffebcc")
btn_custom = Button(plt.axes([0.475, 0.04, 0.09, 0.055]), "Custom: ON", color="#ffe0b2")
btn_bridged = Button(
    plt.axes([0.57, 0.04, 0.09, 0.055]), "Bridged: ON", color="#c8f7c5"
)
btn_yasa = Button(plt.axes([0.665, 0.04, 0.085, 0.055]), "YASA: ON", color="#d0edff")
btn_art = Button(plt.axes([0.755, 0.04, 0.11, 0.055]), "Artifacts: ON", color="#ffcccc")
btn_min_ch = Button(
    plt.axes([0.87, 0.04, 0.115, 0.055]),
    f">={min_channels_threshold} Ch: {'ON' if filter_min_channels else 'OFF'}",
    color="#e6ccff" if filter_min_channels else "#e0e0e0",
)

btn_prev_rec.on_clicked(lambda _: step_recording(-1))
btn_next_rec.on_clicked(lambda _: step_recording(1))
btn_prev.on_clicked(lambda _: step_epoch(-1))
btn_next.on_clicked(lambda _: step_epoch(1))
btn_pspn.on_clicked(lambda _: jump_spindle(-1))
btn_nspn.on_clicked(lambda _: jump_spindle(1))
btn_custom.on_clicked(toggle_custom)
btn_bridged.on_clicked(toggle_bridged)
btn_yasa.on_clicked(toggle_yasa)
btn_art.on_clicked(toggle_artifacts)
btn_min_ch.on_clicked(toggle_min_channels)

if __name__ == "__main__":
    rebuild_recording_axes()
    update_view()
    plt.show()
