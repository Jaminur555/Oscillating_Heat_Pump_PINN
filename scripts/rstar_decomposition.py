"""Review round 2, item 5: decompose the R* ordering into inlet-coupling (a_f) and pipe (kappa_f) shares.

ln(R*_i/R*_j) = ln(a_i/a_j) - ln(kappa_i/kappa_j).  Both terms are ratios of *fitted* rates;
the sloppy direction rescales (a_f, kappa_f) for ALL fluids jointly (fig5), which cancels in
per-fluid ratios, so the shares are invariant to the identified sloppiness.  Report from the
final (all-runs) ODE-shoot and PINN-const fits.
Out: runs/outer/rstar_decomposition.log (console).
Run from repo root:  KMP_DUPLICATE_LIB_OK=TRUE python scripts/rstar_decomposition.py
"""
import pickle, numpy as np
from ohp.data import FLUIDS

rt = pickle.load(open("runs/outer/loro_torch.pkl", "rb"))
pairs = [("EOHP", "MOHP"), ("MOHP", "WOHP"), ("EOHP", "WOHP")]

for name in ["ODE-shoot", "PINN-const"]:
    key = name if name == "ODE-shoot" else "PINN-const|seed0"
    phys = rt[None][key]["phys"]
    a = np.exp(np.asarray(phys["loga"])); k = np.exp(np.asarray(phys["logK"]))
    print(f"=== {name} (final fit) ===")
    print("fluid | a_f [1/s] | kappa_f [1/s] | R*")
    for f, ai, ki in zip(FLUIDS, a, k):
        print(f"{f:5s} | {ai:.5f} | {ki:.6f} | {ai/ki:.2f}")
    for i, j in pairs:
        la, lk = np.log(a[FLUIDS.index(i)] / a[FLUIDS.index(j)]), np.log(k[FLUIDS.index(i)] / k[FLUIDS.index(j)])
        tot = la - lk
        print(f"ln R*_{i}/R*_{j} = {tot:.3f}  =  ln a-ratio {la:+.3f}  -  ln kappa-ratio {lk:+.3f}"
              f"   -> shares: a {100*la/tot:.0f}% / kappa {100*(-lk)/tot:.0f}%")
    print()
