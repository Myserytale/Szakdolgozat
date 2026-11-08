import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from reading_working import load_recording
from spindles import detect_all_spindles, detect_custom_spindles_bridged

# ==============================================================================
# 1. Settings & Load Data
# ==============================================================================
edf_name = "BB0008_SP3.edf"
channel = "C3"
epoch_sec = 30  # 30-second clinical PSG epoch

print(f"Loading {edf_name} ({channel})...")
signal, fs, hypnogram, artefacts, art_idx = load_recording(edf_name, channel)
art_mask = artefacts[:, art_idx] != 0

# Precompute 11-16 Hz sigma band + Hilbert envelope + thresholds
is_n2 = (hypnogram == 2) & (~art_mask)
sos = butter(4, (11, 16), btype="bandpass", fs=fs, output="sos")
sigma = sosfiltfilt(sos, signal)
envelope = np.abs(hilbert(sigma))
env_mean = float(np.mean(envelope[is_n2]))
env_std = float(np.std(envelope[is_n2]))
sigma_thresh = env_mean + 1.5 * env_std
sigma_high_thresh = env_mean + 2.0 * env_std

print("Detecting sleep spindles (Custom + Bridged + YASA)...")
custom_spindles, yasa_spindles = detect_all_spindles(
    signal, fs, hypnogram, artefacts, art_idx
)
bridged_spindles = detect_custom_spindles_bridged(signal, fs, is_n2)
print(
    f"Detected: Custom={len(custom_spindles)} | Bridged={len(bridged_spindles)} | YASA={len(yasa_spindles)} spindles."
)

# ==============================================================================
# 2. Epoch & Stage Setup
# ==============================================================================
epoch_len = fs * epoch_sec
n_epochs = len(signal) // epoch_len
t = np.linspace(0, epoch_sec, epoch_len, endpoint=False)

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

# ==============================================================================
# 3. Figure Layout: Hypnogram (top), Raw EEG (middle), Sigma 11-16 Hz (bottom)
# ==============================================================================
fig, (ax_hyp, ax_eeg, ax_sig) = plt.subplots(
    3, 1, figsize=(13, 7.5), gridspec_kw={"height_ratios": [1, 3, 1.5]}
)
plt.subplots_adjust(bottom=0.20, hspace=0.32)

# Top: Full-night hypnogram (click anywhere to jump to that epoch)
hyp_epochs = hypnogram[: n_epochs * epoch_len : epoch_len].astype(float)
hyp_epochs[hyp_epochs == 9] = np.nan
ax_hyp.step(
    np.arange(n_epochs), hyp_epochs, where="post", color="#333333", linewidth=1.0
)
ax_hyp.set_yticks([0, 1, 2, 3, 4, 5])
ax_hyp.set_yticklabels(["W", "N1", "N2", "N3", "N4", "REM"], fontsize=8)
ax_hyp.invert_yaxis()
ax_hyp.set_xlim(0, n_epochs)
ax_hyp.set_title(
    "Full-Night Hypnogram (click anywhere to jump to epoch)",
    fontsize=9.5,
    fontweight="bold",
)
ax_hyp.grid(True, linestyle="--", alpha=0.3)
hyp_cursor = ax_hyp.axvline(0, color="red", linewidth=1.8)

# Middle: Raw EEG trace
(eeg_line,) = ax_eeg.plot(t, signal[:epoch_len], linewidth=0.8, label=f"{channel} EEG")
ax_eeg.set_xlim(0, epoch_sec)
ax_eeg.set_ylabel("EEG (µV)")
ax_eeg.grid(True, linestyle="--", alpha=0.35)

# Bottom: Sigma (11-16 Hz) + Envelope + Thresholds
(sig_line,) = ax_sig.plot(
    t, sigma[:epoch_len], color="#555555", linewidth=0.7, label="Sigma (11–16 Hz)"
)
(env_line,) = ax_sig.plot(
    t, envelope[:epoch_len], color="#d62728", linewidth=1.2, label="Hilbert Envelope"
)
ax_sig.axhline(
    sigma_thresh,
    color="#d62728",
    linestyle="--",
    linewidth=1.0,
    label=f"Base 1.5 SD ({sigma_thresh:.1f} µV)",
)
ax_sig.axhline(
    sigma_high_thresh,
    color="#2ca02c",
    linestyle=":",
    linewidth=1.1,
    label=f"Peak 2.0 SD ({sigma_high_thresh:.1f} µV)",
)
ax_sig.set_xlim(0, epoch_sec)
ax_sig.set_ylim(-35, 35)
ax_sig.set_xlabel(
    "Time within epoch (s)   [Keys: ◀/▶ epoch | S/W next/prev spindle | ↑/↓ zoom µV | 1/2/3 toggle]"
)
ax_sig.set_ylabel("Sigma (µV)")
ax_sig.grid(True, linestyle="--", alpha=0.35)
ax_sig.legend(loc="upper right", fontsize=8, ncol=4)

highlights = []


def draw_spindles(spindles, start, end, color, edge, y_pos, tag):
    count = 0
    for sp in spindles:
        if sp["end"] > start and sp["start"] < end:
            count += 1
            s_t = max(0.0, (sp["start"] - start) / fs)
            e_t = min(float(epoch_sec), (sp["end"] - start) / fs)
            for ax in (ax_eeg, ax_sig):
                highlights.append(
                    ax.axvspan(
                        s_t,
                        e_t,
                        facecolor=color,
                        edgecolor=edge,
                        alpha=0.35,
                        label=f"{tag} Spindle" if (ax is ax_eeg and count == 1) else "",
                    )
                )
            highlights.append(
                ax_eeg.text(
                    (s_t + e_t) / 2,
                    y_pos,
                    f"{tag}\n{sp['dur']:.2f}s | {sp['freq']:.1f}Hz",
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
        highlights.pop().remove()

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
    if np.any(art_ep):
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

    art_status = "ARTIFACT" if np.any(art_ep) else "CLEAN"
    ax_eeg.set_title(
        f"Epoch {current_epoch + 1}/{n_epochs} | {time_str} | {stage_names.get(stage, stage)} "
        f"| [{art_status}] | Custom: {c_cnt}  Bridged: {b_cnt}  YASA: {y_cnt} | Scale: ±{int(y_lim)} µV",
        fontsize=11,
        fontweight="bold",
    )

    handles, labels = ax_eeg.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_eeg.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    fig.canvas.draw_idle()


# ==============================================================================
# 4. Navigation & Callbacks
# ==============================================================================
def step_epoch(delta):
    global current_epoch
    current_epoch = int(np.clip(current_epoch + delta, 0, n_epochs - 1))
    update_view()


def jump_spindle(direction=1):
    global current_epoch
    active = (
        (custom_spindles if show_custom else [])
        + (bridged_spindles if show_bridged else [])
        + (yasa_spindles if show_yasa else [])
    )
    if not active:
        active = custom_spindles + bridged_spindles + yasa_spindles
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


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)

# ==============================================================================
# 5. Buttons (Single Clean Row)
# ==============================================================================
btn_prev = Button(plt.axes([0.04, 0.04, 0.10, 0.055]), "◀ Prev")
btn_next = Button(plt.axes([0.15, 0.04, 0.10, 0.055]), "Next ▶")
btn_pspn = Button(
    plt.axes([0.27, 0.04, 0.13, 0.055]), "◀ Prev Spindle", color="#ffebcc"
)
btn_nspn = Button(
    plt.axes([0.41, 0.04, 0.13, 0.055]), "Next Spindle ▶", color="#ffebcc"
)
btn_custom = Button(plt.axes([0.56, 0.04, 0.12, 0.055]), "Custom: ON", color="#ffe0b2")
btn_bridged = Button(
    plt.axes([0.69, 0.04, 0.12, 0.055]), "Bridged: ON", color="#c8f7c5"
)
btn_yasa = Button(plt.axes([0.82, 0.04, 0.12, 0.055]), "YASA: ON", color="#d0edff")

btn_prev.on_clicked(lambda _: step_epoch(-1))
btn_next.on_clicked(lambda _: step_epoch(1))
btn_pspn.on_clicked(lambda _: jump_spindle(-1))
btn_nspn.on_clicked(lambda _: jump_spindle(1))
btn_custom.on_clicked(toggle_custom)
btn_bridged.on_clicked(toggle_bridged)
btn_yasa.on_clicked(toggle_yasa)

if __name__ == "__main__":
    update_view()
    plt.show()
