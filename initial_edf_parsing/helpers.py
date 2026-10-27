import pyedflib
from reading_working import DATA_DIR

# Inspect all EDF files and list their channels & sampling rates
for path in sorted(DATA_DIR.glob("*.edf")):
    print(f"\n{path.name}")
    with pyedflib.EdfReader(str(path)) as edf:
        for i, label in enumerate(edf.getSignalLabels()):
            print(f"{label:15} {edf.getSampleFrequency(i):.1f} Hz")
