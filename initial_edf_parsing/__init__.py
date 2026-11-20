from pathlib import Path
import sys

_PKG_DIR = Path(__file__).resolve().parent
for _sub in ("core", "spindles", "infraslow", "psd", "viewers"):
    _p = str(_PKG_DIR / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
