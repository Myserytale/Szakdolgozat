from pathlib import Path
import pyedflib

directory = Path("/home/lev/UNI_1/UNI.4/Szakdolg/Guildford")

for path in directory.glob("*.edf"):
    print(f"\n{path.name}")

    with pyedflib.EdfReader(str(path)) as edf:
        for i, label in enumerate(edf.getSignalLabels()):
            print(f"{label:15} {edf.getSampleFrequency(i)} Hz")
