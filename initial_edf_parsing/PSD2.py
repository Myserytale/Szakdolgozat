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
