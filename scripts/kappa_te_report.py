import pickle, numpy as np

t = pickle.load(open("runs/outer/loro_torch.pkl", "rb"))
FL = ["EOHP", "MOHP", "WOHP"]

def agg(fold, name, seeds):
    vals = [np.mean([t[fold][f"{name}|seed{s}"]["met"][(fold, f)]["rmse_Te"] for f in FL]) for s in seeds]
    return np.mean(vals), np.std(vals)

print("fold | PINN-const (s0-2) | PINN-kTe (s0-2) | delta")
for fold in [1, 2, 3]:
    mc = agg(fold, "PINN-const", [0, 1, 2])
    mk = agg(fold, "PINN-kTe", [0, 1, 2])
    print(f"{fold}    {mc[0]:.3f}±{mc[1]:.3f}        {mk[0]:.3f}±{mk[1]:.3f}      {mk[0]-mc[0]:+.3f}")

print("\nfinal fit (fold None), per seed:")
for s in [0, 1, 2]:
    su = t[None][f"PINN-kTe|seed{s}"]["summary"]
    print(f" s{s}: k0={np.round(su['K'],5)} beta={np.round(su['beta'],4)} R*(50C)={np.round(su['Rstar'],2)}")
mc = t[None]["PINN-const|seed0"]["summary"]
print("const ref: K=", np.round(mc["K"], 5), " R*=", np.round(mc["Rstar"], 2))
