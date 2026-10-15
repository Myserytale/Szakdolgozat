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

frequencies, psd = welch(
    c3_signal,
    fs=fs,
    window="hann",
    nperseg=nperseg,
    noverlap=nperseg // 2,
    scaling="density",
)

plt.figure(figsize=(10, 5))
plt.semilogy(frequencies, psd)
plt.xlim(0, 30)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (signal units²/Hz)")
plt.title("C3 — Whole-recording PSD")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.savefig("psd_plot.png", dpi=150)
