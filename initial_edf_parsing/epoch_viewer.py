import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
import yasa

from reading_working import DATA_DIR, HYPART_DIR, load_hypart, read_edf

# ==============================================================================
# 1. Settings & Load Data
# ==============================================================================
channel = "C3"
art_channel = f"ArtiFFT {channel}-A2"
epoch_sec = 30  # Standard clinical PSG epoch duration (30 seconds)

edf_path = DATA_DIR / "BB0008_SP2.edf"
hypart_path = HYPART_DIR / f"{edf_path.stem}.hypart"

print(f"Loading {edf_path.name} ({channel})...")
edf_data = read_edf(edf_path, channels=[channel])
signal = edf_data["signals"][channel]
fs = int(edf_data["sample_rates"][0])

hypnogram, artefacts, art_channels = load_hypart(hypart_path)
art_idx = art_channels.index(art_channel)

# Clean Stage 2 (NREM2) mask
is_n2 = (hypnogram == 2) & (artefacts[:, art_idx] == 0)

# ==============================================================================
# 2. Spindle Detection: Custom SciPy & YASA
# ==============================================================================
print("Running Custom SciPy detector...")
sos = butter(4, [11, 16], btype="bandpass", fs=fs, output="sos")
sigma = sosfiltfilt(sos, signal)
envelope = np.abs(hilbert(sigma))

thresh = np.mean(envelope[is_n2]) + 1.5 * np.std(envelope[is_n2])
above = (envelope > thresh) & is_n2
diff = np.diff(above.astype(np.int8))
starts = np.where(diff == 1)[0] + 1
ends = np.where(diff == -1)[0] + 1

if above[0]:
    starts = np.insert(starts, 0, 0)
if above[-1]:
    ends = np.append(ends, len(above))

min_samples = int(0.5 * fs)
max_samples = int(2.5 * fs)

custom_spindles = []
for s, e in zip(starts, ends):
    dur = (e - s) / fs
    if min_samples <= (e - s) <= max_samples:
        custom_spindles.append({"start": s, "end": e, "dur": dur})

print("Running YASA detector...")
sp_yasa = yasa.spindles_detect(
    signal,
    fs,
    ch_names=[channel],
    hypno=hypnogram,
    include=(2,),
    freq_sp=(11, 16),
)
df_yasa = sp_yasa.summary()

yasa_spindles = []
if df_yasa is not None and len(df_yasa) > 0:
    for _, row in df_yasa.iterrows():
        s_idx = int(row["Start"] * fs)
        e_idx = int((row["Start"] + row["Duration"]) * fs)
        if np.all(artefacts[s_idx:e_idx, art_idx] == 0):
            yasa_spindles.append({
                "start": s_idx,
                "end": e_idx,
                "dur": float(row["Duration"]),
            })

print(f"Detected: Custom={len(custom_spindles)} | YASA={len(yasa_spindles)} spindles in clean N2.")

# Combined list for navigation
all_spindles = custom_spindles + yasa_spindles

# Toggle states for spindle overlays
show_custom = True
show_yasa = True

# ==============================================================================
# 3. Epoch Setup & Stage Mappings
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
    0: "#ff7f0e",  # Orange (Wake)
    1: "#bcbd22",  # Yellow-Green (N1)
    2: "#1f77b4",  # Blue (N2)
    3: "#2ca02c",  # Green (N3)
    4: "#d62728",  # Red (N4)
    5: "#9467bd",  # Purple (REM)
    9: "#7f7f7f",  # Gray (Unscored)
}

current_epoch = 0

# ==============================================================================
# 4. Interactive Plot Setup
# ==============================================================================
fig, ax = plt.subplots(figsize=(13, 6.2))
plt.subplots_adjust(bottom=0.24)  # Room for navigation & toggle button rows

(line,) = ax.plot(t, signal[:epoch_len], color="#1f77b4", linewidth=0.8, label="C3 EEG")
ax.set_xlim(0, epoch_sec)
ax.set_xlabel("Time within epoch (seconds)", fontsize=11)
ax.set_ylabel(r"Amplitude ($\mu V$)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.35)

# Container for dynamically drawn spindle spans and badges
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

    # Time in recording (HH:MM:SS)
    sec_elapsed = current_epoch * epoch_sec
    time_str = f"{sec_elapsed // 3600:02d}:{(sec_elapsed % 3600) // 60:02d}:{sec_elapsed % 60:02d}"

    # Update line data & color
    line.set_ydata(seg)
    line.set_color(color)

    # Dynamic y-limits with headroom
    max_amp = max(float(np.max(np.abs(seg))) * 1.25, 75.0)
    ax.set_ylim(-max_amp, max_amp)

    # 1. Overlay Custom SciPy Spindles (Orange fill) if toggled ON
    c_count = 0
    if show_custom:
        for sp in custom_spindles:
            if sp["end"] > start and sp["start"] < end:
                c_count += 1
                sp_t0 = max(0.0, (sp["start"] - start) / fs)
                sp_t1 = min(float(epoch_sec), (sp["end"] - start) / fs)

                span = ax.axvspan(
                    sp_t0,
                    sp_t1,
                    color="orange",
                    alpha=0.35,
                    label="Custom Spindle" if c_count == 1 else "",
                )
                active_highlights.append(span)

                txt = ax.text(
                    (sp_t0 + sp_t1) / 2,
                    max_amp * 0.82,
                    f"Custom\n{sp['dur']:.2f}s",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#b35900",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="orange", alpha=0.9),
                )
                active_highlights.append(txt)

    # 2. Overlay YASA Spindles (Dashed Cyan border / light fill) if toggled ON
    y_count = 0
    if show_yasa:
        for sp in yasa_spindles:
            if sp["end"] > start and sp["start"] < end:
                y_count += 1
                sp_t0 = max(0.0, (sp["start"] - start) / fs)
                sp_t1 = min(float(epoch_sec), (sp["end"] - start) / fs)

                span = ax.axvspan(
                    sp_t0,
                    sp_t1,
                    facecolor="#d0edff",
                    edgecolor="#007acc",
                    linestyle="--",
                    linewidth=1.8,
                    alpha=0.40,
                    label="YASA Spindle" if y_count == 1 else "",
                )
                active_highlights.append(span)

                txt = ax.text(
                    (sp_t0 + sp_t1) / 2,
                    max_amp * 0.60,
                    f"YASA\n{sp['dur']:.2f}s",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#005580",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#007acc", alpha=0.9),
                )
                active_highlights.append(txt)

    # Build title string according to active toggles
    parts = []
    if show_custom:
        parts.append(f"Custom ({c_count})")
    if show_yasa:
        parts.append(f"YASA ({y_count})")

    spindle_str = f"Spindles: {', '.join(parts)}" if parts else "Spindles: (All OFF)"

    ax.set_title(
        f"Epoch {current_epoch + 1}/{n_epochs}  |  Time: {time_str}  |  {stage_name}  |  [{art_status}]  |  ★ {spindle_str}",
        fontsize=11.5,
        fontweight="bold",
    )

    # Dynamic deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=True)

    fig.canvas.draw_idle()


# ==============================================================================
# 5. Navigation Controls & Toggle Callbacks
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
    """Toggles visibility of Custom SciPy spindles."""
    global show_custom
    show_custom = not show_custom
    btn_custom.label.set_text(f"Custom: {'ON' if show_custom else 'OFF'}")
    btn_custom.ax.set_facecolor("#ffe0b2" if show_custom else "#e0e0e0")
    btn_custom.color = "#ffe0b2" if show_custom else "#e0e0e0"
    update_view()


def toggle_yasa(event=None):
    """Toggles visibility of YASA spindles."""
    global show_yasa
    show_yasa = not show_yasa
    btn_yasa.label.set_text(f"YASA: {'ON' if show_yasa else 'OFF'}")
    btn_yasa.ax.set_facecolor("#d0edff" if show_yasa else "#e0e0e0")
    btn_yasa.color = "#d0edff" if show_yasa else "#e0e0e0"
    update_view()


def on_key(event):
    """Keyboard shortcuts."""
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
# Buttons Setup (Two Rows)
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
