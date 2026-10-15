import numpy as np
import matplotlib.pyplot as plp

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


def inspect_array(name, data):
    arr = np.asarray(data)

    print(f"\n{name}")
    print("Type:", type(data))
    print("Shape:", arr.shape)
    print("Dtype:", arr.dtype)
    print("First values:", arr.flat[:20])

    if arr.size and arr.size < 1_000_000:
        values, counts = np.unique(arr, return_counts=True)
        print("Unique values:", list(zip(values[:30], counts[:30])))


inspect_array("Hypnogram", hypnogram)
inspect_array("Artefacts", artefacts)
inspect_array("Artifact channels", art_channels)

fs = 64
seconds = 4
