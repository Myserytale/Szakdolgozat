import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import welch, resample_poly

from reading_working import *

edf_path = DATA_DIR / "BB0008_SP2.edf"
hypart_path = HYPART_DIR / f"{edf_path.stem}.hypart"

with pyedflib.EdfReader(str(edf_path)) as edf:
    labels = edf.getSignalLabels()
    c3_idx = labels.index("C3")
    c3_signal = edf.readSignal(c3_idx)
    fs = edf.getSampleFrequency(c3_idx)

hypnogram, artefacts, art_channels = load_hypart(hypart_path)

seconds = 4

nperseg = fs * seconds

C3_art_idx = art_channels.index("ArtiFFT C3-A2")

is_stage2 = hypnogram == 2
is_clean = artefacts[:, C3_art_idx] == 0

good_data = is_clean & is_stage2
data_notbest = is_stage2

stage2_clean = c3_signal[good_data]


frequencies, psd = welch(
    stage2_clean,
    fs=fs,
    window="hann",
    nperseg=nperseg,
    noverlap=nperseg // 2,
    scaling="density",
)

plt.semilogy(frequencies, psd)
plt.xlim(0, 30)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (signal units²/Hz)")
plt.title("C3 — Whole-recording PSD")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("psd_plot2.png", dpi=150)
plt.show()
