import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

from reading_working import load_recording
from spindles import detect_all_spindles

# ==============================================================================
# 1. Settings & Load Data
# ==============================================================================
edf_name = "BB0008_SP2.edf"
channel = "C3"
epoch_sec = 30  # Standard clinical PSG epoch duration (30 seconds)

print(f"Loading {edf_name} ({channel})...")
signal, fs, hypnogram, artefacts, art_idx = load_recording(edf_name, channel)

# Detect spindles (Custom SciPy + YASA)
print("Detecting sleep spindles (Custom + YASA)...")
custom_spindles, yasa_spindles = detect_all_spindles(
    signal=signal,
    fs=fs,
    hypnogram=hypnogram,
    artefacts=artefacts,
    art_idx=art_idx,
)
all_spindles = custom_spindles + yasa_spindles
print(f"Detected: Custom={len(custom_spindles)} | YASA={len(yasa_spindles)} spindles.")

# Toggle states
show_custom = True
show_yasa = True

# ==============================================================================
# 2. Epoch Setup & Stage Mappings
# ==============================================================================
epoch_len = fs * epoch_sec
n_epochs = len(signal) // epoch_len
t = np.linspace(0, epoch_sec, epoch_len, endpoint=False)

stage_names = {
    0: "Wake",
    1: "Stage 1 (N1)",
    2: "Stage 2 (N2)",
    3: "Stage 3 (N3)",
    4: "Stage 4 (N4)",
    5: "REM Sleep",
    9: "Unscored",
}
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

# ==============================================================================
# 3. Figure & View Update
# ==============================================================================
fig, ax = plt.subplots(figsize=(13, 6.2))
plt.subplots_adjust(bottom=0.24)  # Leave room for 2 rows of buttons

(line,) = ax.plot(t, signal[:epoch_len], color="#1f77b4", linewidth=0.8, label="C3 EEG")
ax.set_xlim(0, epoch_sec)
ax.set_xlabel("Time within epoch (seconds)", fontsize=11)
ax.set_ylabel(r"Amplitude ($\mu V$)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.35)

active_highlights = []


def clear_highlights():
    while active_highlights:
        item = active_highlights.pop()
        item.remove()


def update_view():
    """Redraws the EEG trace and overlays toggled spindle detections."""
    clear_highlights()

    start = current_epoch * epoch_len
    end = start + epoch_len
    seg = signal[start:end]

    # Stage & Artifact info
    stage_code = int(hypnogram[start])
    stage_name = stage_names.get(stage_code, f"Stage {stage_code}")
    color = stage_colors.get(stage_code, "#1f77b4")
    is_clean = np.all(artefacts[start:end, art_idx] == 0)
    art_status = "CLEAN" if is_clean else "ARTIFACT"

    # Elapsed time string
    sec_elapsed = current_epoch * epoch_sec
    time_str = f"{sec_elapsed // 3600:02d}:{(sec_elapsed % 3600) // 60:02d}:{sec_elapsed % 60:02d}"

    line.set_ydata(seg)
    line.set_color(color)

    max_amp = max(float(np.max(np.abs(seg))) * 1.25, 75.0)
    ax.set_ylim(-max_amp, max_amp)

    # 1. Overlay Custom Spindles (Orange)
    c_count = 0
    if show_custom:
        for sp in custom_spindles:
            if sp["end"] > start and sp["start"] < end:
                c_count += 1
                s_t = max(0.0, (sp["start"] - start) / fs)
                e_t = min(float(epoch_sec), (sp["end"] - start) / fs)

                span = ax.axvspan(s_t, e_t, color="orange", alpha=0.35, label="Custom Spindle" if c_count == 1 else "")
                active_highlights.append(span)

                badge = ax.text(
                    (s_t + e_t) / 2, max_amp * 0.82, f"Custom\n{sp['dur']:.2f}s | {sp['freq']:.1f}Hz",
                    ha="center", va="center", fontsize=8, fontweight="bold", color="#b35900",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="orange", alpha=0.9),
                )
                active_highlights.append(badge)

    # 2. Overlay YASA Spindles (Cyan)
    y_count = 0
    if show_yasa:
        for sp in yasa_spindles:
            if sp["end"] > start and sp["start"] < end:
                y_count += 1
                s_t = max(0.0, (sp["start"] - start) / fs)
                e_t = min(float(epoch_sec), (sp["end"] - start) / fs)

                span = ax.axvspan(
                    s_t, e_t, facecolor="#d0edff", edgecolor="#007acc", linestyle="--", linewidth=1.8, alpha=0.40,
                    label="YASA Spindle" if y_count == 1 else "",
                )
                active_highlights.append(span)

                badge = ax.text(
                    (s_t + e_t) / 2, max_amp * 0.60, f"YASA\n{sp['dur']:.2f}s | {sp['freq']:.1f}Hz",
                    ha="center", va="center", fontsize=8, fontweight="bold", color="#005580",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#007acc", alpha=0.9),
                )
                active_highlights.append(badge)

    # Title & Legend
    parts = []
    if show_custom:
        parts.append(f"Custom ({c_count})")
    if show_yasa:
        parts.append(f"YASA ({y_count})")
    spindle_str = f"Spindles: {', '.join(parts)}" if parts else "Spindles: (All OFF)"

    ax.set_title(
        f"Epoch {current_epoch + 1}/{n_epochs}  |  Time: {time_str}  |  {stage_name}  |  [{art_status}]  |  ★ {spindle_str}",
        fontsize=11.5, fontweight="bold",
    )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=True)
    fig.canvas.draw_idle()


# ==============================================================================
# 4. Navigation & Toggle Callbacks
# ==============================================================================
def next_epoch(event=None):
    global current_epoch
    if current_epoch < n_epochs - 1:
        current_epoch += 1
        update_view()


def prev_epoch(event=None):
    global current_epoch
    if current_epoch > 0:
        current_epoch -= 1
        update_view()


def next_spindle(event=None):
    """Jumps directly to the next epoch containing ANY active spindle."""
    global current_epoch
    active = []
    if show_custom:
        active.extend(custom_spindles)
    if show_yasa:
        active.extend(yasa_spindles)
    if not active:
        active = all_spindles

    for ep in range(current_epoch + 1, n_epochs):
        s = ep * epoch_len
        e = s + epoch_len
        if any(sp["end"] > s and sp["start"] < e for sp in active):
            current_epoch = ep
            update_view()
            return
    print("No more spindles found ahead.")


def next_clean(event=None):
    """Jumps to the next clean (artifact-free) epoch."""
    global current_epoch
    for ep in range(current_epoch + 1, n_epochs):
        s = ep * epoch_len
        e = s + epoch_len
        if np.all(artefacts[s:e, art_idx] == 0):
            current_epoch = ep
            update_view()
            return
    print("No more clean epochs found ahead.")


def toggle_custom(event=None):
    global show_custom
    show_custom = not show_custom
    btn_custom.label.set_text(f"Custom: {'ON' if show_custom else 'OFF'}")
    btn_custom.ax.set_facecolor("#ffe0b2" if show_custom else "#e0e0e0")
    update_view()


def toggle_yasa(event=None):
    global show_yasa
    show_yasa = not show_yasa
    btn_yasa.label.set_text(f"YASA: {'ON' if show_yasa else 'OFF'}")
    btn_yasa.ax.set_facecolor("#d0edff" if show_yasa else "#e0e0e0")
    update_view()


def on_key(event):
    if event.key in ["right", "d"]:
        next_epoch()
    elif event.key in ["left", "a"]:
        prev_epoch()
    elif event.key in ["s", "S"]:
        next_spindle()
    elif event.key in ["c", "C"]:
        next_clean()
    elif event.key in ["1", "u", "U"]:
        toggle_custom()
    elif event.key in ["2", "y", "Y"]:
        toggle_yasa()


fig.canvas.mpl_connect("key_press_event", on_key)

# ------------------------------------------------------------------------------
# 5. Buttons Setup (Two Rows)
# ------------------------------------------------------------------------------
# Row 1: Navigation Buttons
ax_prev = plt.axes([0.10, 0.12, 0.16, 0.065])
ax_next = plt.axes([0.28, 0.12, 0.16, 0.065])
ax_clean = plt.axes([0.48, 0.12, 0.18, 0.065])
ax_spind = plt.axes([0.68, 0.12, 0.22, 0.065])

btn_prev = Button(ax_prev, "◀ Previous")
btn_next = Button(ax_next, "Next ▶")
btn_clean = Button(ax_clean, "Next Clean ◯")
btn_spind = Button(ax_spind, "Next Spindle ★", color="#ffebcc", hovercolor="#ffd699")

btn_prev.on_clicked(prev_epoch)
btn_next.on_clicked(next_epoch)
btn_clean.on_clicked(next_clean)
btn_spind.on_clicked(next_spindle)

# Row 2: Spindle Visibility Toggles
ax_custom = plt.axes([0.28, 0.035, 0.21, 0.055])
ax_yasa = plt.axes([0.51, 0.035, 0.21, 0.055])

btn_custom = Button(ax_custom, "Custom: ON", color="#ffe0b2", hovercolor="#ffcc80")
btn_yasa = Button(ax_yasa, "YASA: ON", color="#d0edff", hovercolor="#b3e0ff")

btn_custom.on_clicked(toggle_custom)
btn_yasa.on_clicked(toggle_yasa)

# Start viewer
if __name__ == "__main__":
    update_view()
    plt.show()
