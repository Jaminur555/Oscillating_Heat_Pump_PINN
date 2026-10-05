"""Run configuration: vessel-driver variant + per-driver results/figures directories.

Select the driver with OHP_TV (outer | mean4 | inner); all outputs land in runs/<driver>/
so variants never overwrite each other (v1 bug: OHP_TV=mean4 wrote into results/).
Run the ohp.* modules from the repository root:  python -m ohp.evaluate
"""
import os

DRIVER = os.environ.get("OHP_TV", "outer")          # headline driver: outer vessel TCs
RUNS_DIR = os.path.join("runs", DRIVER)             # relative to CWD (repo root)
FIG_DIR = os.path.join(RUNS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
