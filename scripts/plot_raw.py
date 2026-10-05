import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from ohp.data import load_all, FLUIDS, RUNS
from ohp.config import RUNS_DIR as RES, FIG_DIR as FIG
D = load_all()
fig, ax = plt.subplots(3, 3, figsize=(15, 10), sharey="row")
for j, r in enumerate(RUNS):
    for fl, c in zip(FLUIDS, ["C0", "C1", "C2"]):
        d = D[(r, fl)]
        ax[0, j].plot(d["t"], d["Te"], c, label=f"{fl} Te")
        ax[1, j].plot(d["t"], d["Tv"] - d["Te"], c, label=f"{fl}")
        ax[2, j].plot(d["t"], d["Tc"], c, label=f"{fl} Tc")
    ax[0, j].plot(d["t"], d["Tv"], "k--", label="Tv (vessel)")
    ax[0, j].set_title(f"Run {r}"); ax[0, j].legend(fontsize=7)
    ax[2, j].set_xlabel("time [s] (5 s/sample)")
ax[0, 0].set_ylabel("Temperature [°C]"); ax[1, 0].set_ylabel("Tv − Te [K]"); ax[2, 0].set_ylabel("Tc [°C]")
plt.tight_layout(); plt.savefig(FIG + "/raw_overview.png", dpi=110)
print("saved")
