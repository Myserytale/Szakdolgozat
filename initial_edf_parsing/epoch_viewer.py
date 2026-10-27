import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

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
# 2. Spindle Detection (Sigma 11-16 Hz & Hilbert Envelope)
# ==============================================================================
print("Running spindle detection...")
# 4th-order zero-phase bandpass filter (11 - 16 Hz)
sos = butter(4, [11, 16], btype="bandpass", fs=fs, output="sos")
sigma = sosfiltfilt(sos, signal)
envelope = np.abs(hilbert(sigma))

# Baseline threshold during clean N2 (Mean + 1.5 * Std)
thresh = np.mean(envelope[is_n2]) + 1.5 * np.std(envelope[is_n2])

# Find supra-threshold intervals
above = (envelope > thresh) & is_n2
diff = np.diff(above.astype(np.int8))
starts = np.where(diff == 1)[0] + 1
ends = np.where(diff == -1)[0] + 1

if above[0]:
    starts = np.insert(starts, 0, 0)
if above[-1]:
    ends = np.append(ends, len(above))

# Filter by duration: 0.5s to 2.5s (AASM standard)
min_samples = int(0.5 * fs)
max_samples = int(2.5 * fs)

spindles = []
for s, e in zip(starts, ends):
    dur = (e - s) / fs
    if min_samples <= (e - s) <= max_samples:
        peak_amp = float(np.max(np.abs(signal[s:e])))
        spindles.append({"start": s, "end": e, "dur": dur, "peak_amp": peak_amp})

print(f"Detected {len(spindles)} sleep spindles in clean N2.")

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
fig, ax = plt.subplots(figsize=(13, 6))
plt.subplots_adjust(bottom=0.20)  # Reserve room for buttons

(line,) = ax.plot(t, signal[:epoch_len], color="#1f77b4", linewidth=0.8)
ax.set_xlim(0, epoch_sec)
ax.set_xlabel("Time within epoch (seconds)", fontsize=11)
ax.set_ylabel(r"Amplitude ($\mu V$)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.35)

# Container for spindle highlight spans and annotations
active_highlights = []


def clear_highlights():
    while active_highlights:
        item = active_highlights.pop()
        item.remove()


def update_view():
    """Redraws the EEG trace and highlights spindles in the current epoch."""
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

    # Highlight spindles overlapping this epoch
    epoch_spindles = 0
    for sp in spindles:
        if sp["end"] > start and sp["start"] < end:
            epoch_spindles += 1
            sp_t0 = max(0.0, (sp["start"] - start) / fs)
            sp_t1 = min(float(epoch_sec), (sp["end"] - start) / fs)

            span = ax.axvspan(sp_t0, sp_t1, color="orange", alpha=0.35)
            active_highlights.append(span)

            # Text label above the spindle
            txt = ax.text(
                (sp_t0 + sp_t1) / 2,
                max_amp * 0.82,
                f"Spindle\n{sp['dur']:.2f}s",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#b35900",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="orange", alpha=0.85),
            )
            active_highlights.append(txt)

    spindle_str = f"Spindles: {epoch_spindles}" if epoch_spindles > 0 else "No spindles"

    ax.set_title(
        f"Epoch {current_epoch + 1}/{n_epochs}  |  Time: {time_str}  |  {stage_name}  |  [{art_status}]  |  ★ {spindle_str}",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.canvas.draw_idle()


# ==============================================================================
# 5. Navigation Controls & Callbacks
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
    """Jumps directly to the next epoch that contains a detected sleep spindle."""
    global current_epoch
    for ep in range(current_epoch + 1, n_epochs):
        s = ep * epoch_len
        e = s + epoch_len
        if any(sp["end"] > s and sp["start"] < e for sp in spindles):
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


fig.canvas.mpl_connect("key_press_event", on_key)

# Add Navigation Buttons
ax_prev = plt.axes([0.10, 0.05, 0.16, 0.07])
ax_next = plt.axes([0.28, 0.05, 0.16, 0.07])
ax_clean = plt.axes([0.48, 0.05, 0.18, 0.07])
ax_spind = plt.axes([0.68, 0.05, 0.22, 0.07])

btn_prev = Button(ax_prev, "◀ Previous")
btn_next = Button(ax_next, "Next ▶")
btn_clean = Button(ax_clean, "Next Clean ◯")
btn_spind = Button(ax_spind, "Next Spindle ★", color="#ffebcc", hovercolor="#ffd699")

btn_prev.on_clicked(prev_epoch)
btn_next.on_clicked(next_epoch)
btn_clean.on_clicked(next_clean)
btn_spind.on_clicked(next_spindle)

# Start viewer
if __name__ == "__main__":
    update_view()
    plt.show()
