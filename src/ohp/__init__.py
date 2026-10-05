"""Physics-informed grey-box digital twin of oscillating heat pipes (EOHP / MOHP / WOHP).

Library modules: data (loading), pinn (two-node model + inverse PINN, PyTorch),
baselines (ODE-shoot / MLP / GP / SINDy), gru (GRU baseline), metrics (raw-data metrics),
ode_twin (ONNX export of the identified ODE + replay + sweep), config (driver variant).

Executable pipelines and analyses live in scripts/ (run from the repo root):
evaluate, summarize, make_report, rerun_baselines, plot_raw, tv_sensitivity,
derived_checks, gru_baseline, scenario_case, ensemble_bands, adiabatic_validation,
kappa_te_*, make_paper_figures.
"""
