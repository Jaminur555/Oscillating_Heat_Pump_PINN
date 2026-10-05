# ohp-twin — Physics-Informed Grey-Box Model and ONNX Simulator of an Oscillating Heat Pipe

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Can a simple two-equation physical model, tuned by machine learning, predict how an
oscillating heat pipe's evaporator responds to heating? This repository answers yes —
and shows it beats black-box ML when extrapolating to heating levels never seen in
training.

**What it does:**
- Identifies a lumped two-node thermal model of a helically coiled oscillating heat
  pipe (HCOHP) from public transient thermocouple data, by a physics-informed neural
  network (PINN) and by classical ODE shooting.
- Validates it leave-one-run-out (including a true extrapolation fold) against
  IC-fair black-box baselines (MLP, GP, SINDy, GRU).
- Exports the identified model as a stand-alone ONNX simulator.
- Illustrates model-implied trade-offs between setpoint tracking and ripple rejection
  along the dimensionless coupling ratio R\* = a_f/κ_f (explicitly *not* fluid
  recommendations).

Manuscript: `paper/paper.tex`. Beginner's walkthrough of the whole project:
`PROJECT_GUIDE.md`.

## Headline results

Held-out evaporator RMSE (K), mean over fluids, scored against raw data:

| held-out run | Trivial | MLP | GP | SINDy | Ridge-lin. | Lin-SS | GRU | One-node | ODE-shoot | PINN-const |
|---|---|---|---|---|---|---|---|---|---|---|
| Run 1 (down extrap.) | 1.87 | 3.93 | 0.77 | 0.65 | 2.24 | 0.28 | 0.77±0.09 | 0.39 | 0.41 | 0.45±0.01 |
| Run 2 (interp.) | 0.82 | 5.98 | 0.71 | 1.19 | 5.78 | 0.30 | 0.69±0.18 | 0.24 | 0.26 | 0.35±0.01 |
| Run 3 (up extrap.) | 2.84 | 23.54 | 25.80 | 86.49 | 2.66 | **0.68** | 4.99±0.34 | 0.71 | 0.73 | 0.85±0.01 |

Lin-SS = unconstrained per-fluid linear state-space (3×8 free params, output-error
shooting). It matches or beats the physical model everywhere — the linear structure,
not the physical parameterisation, carries extrapolation.

## Key findings

1. **Structure beats optimiser.** Every linear two-state form fitted by shooting
   (physical ODE, one-node, free linear-SS) extrapolates at 0.3–0.7 K; nonlinear
   black boxes fail (5–87 K); a trivial gap baseline stays within 3 K — read all
   claims against that anchor. The GRU is the best black box (it exploits temporal
   structure) but still ~6× worse on extrapolation. The physical model is kept for
   parsimony (9 vs 24 params) and the interpretable R\*.
2. **R\* = a/κ is the reportable per-fluid output.** The rates themselves are sloppy;
   their ratio is not: invariant to the sampling interval, robust to
   seed/method/driver choice, and it orders the fluids identically everywhere
   (EOHP ≈ 12.6–13.5 > MOHP ≈ 7.1–7.3 > WOHP ≈ 5.8–6.0). Its ~48% ethanol drift
   matches the dry-out signature in the source data; not attributed to pipe vs
   contact resistance.
3. **κ(Te) — a negative result, kept.** Per-run temperature dependence is
   identifiable (β̂_E ≈ +0.03, β̂_W ≤ −0.02, β̂_M ≈ 0 — the initial "not
   identifiable" verdict was an optimiser artefact) but does **not** transfer across
   heating levels (8/9 LORO folds unchanged or worse), so the deployed model keeps κ
   constant. Per-fluid condenser d proved unnecessary.
4. **A validation channel that was never fitted.** The adiabatic temperature,
   excluded from all fitting, is reproduced by the two-node model to < 0.5 K held-out
   for EOHP/WOHP (MOHP Run 3 is the identified hard case).
5. **Scenario twins.** EOHP's tracking advantage is robust (100% of rate-rescaled
   ×/÷2 and jackknife refit variants); WOHP's ripple edge holds at the identified
   rates but sits inside the rate-identifiability band (`scenario_bands.log`).
6. **Honest uncertainty.** Seed ensembles quantify parameter uncertainty only
   (2 sd ≈ 0.01–0.05 K vs residuals 0.25–0.89 K) — not full predictive bands;
   conformal prediction is future work.

## Repository layout

```
src/ohp/        library: data, pinn (PINN + physics), baselines (ODE-shoot/MLP/GP/SINDy),
                gru, metrics, ode_twin (ONNX export + replay + sweep), config
scripts/        executables (run from repo root):
                evaluate            LORO pipeline -> runs/<driver>/loro_torch.pkl
                summarize, make_report, rerun_baselines, plot_raw, tv_sensitivity,
                derived_checks      reports, figures, diagnostics
                gru_baseline        GRU LORO evaluation
                scenario_case       duty-cycle working-fluid case study
                scenario_bands      rate-rescale + jackknife bands on scenario metrics
                ensemble_bands      10-seed parameter/predictive bands
                adiabatic_validation, kappa_te_diagnostic (superseded), kappa_te_report,
                kappa_te_profile    beta grid profile: kappa(Te) identifiable per run?
                rstar_decomposition a_f vs kappa_f shares of the R* ordering
                review_baselines    trivial / ridge / one-node / free linear-SS
                make_paper_figures  publish set -> paper/figures/
tests/          unit tests (synthetic; no data needed): rollout parity, DT_S invariance,
                metrics, review-baseline fitters, jackknife formulas
paper/          LaTeX manuscript (sections/) + figures/
docs/           source papers (dataset + descriptor + li2019)
data_raw/       place the Mendeley Excel files here (not redistributed; see below)
runs/<driver>/  results: pickles, CSV tables, logs, figures (driver = OHP_TV variant)
```

## Install & reproduce

Requires Python ≥ 3.10. `pip install -e .` (numpy, scipy, pandas, openpyxl,
scikit-learn, matplotlib, torch, onnx, onnxruntime; pysindy for the SINDy baseline).

1. Download the dataset (Mendeley Data, DOI 10.17632/wnf5jwzp3c.3, CC BY 4.0) and
   place the `HCOHP Primary Data - Run {1,2,3}.xlsx` files in `data_raw/`.
2. From the repo root:
   ```
   python scripts/evaluate.py           # LORO fits (hours, 1 CPU, resumable)
   python scripts/gru_baseline.py       # GRU baseline (~2 h)
   python scripts/review_baselines.py   # trivial/ridge/one-node/linear-SS (~10 min)
   python scripts/kappa_te_profile.py   # beta(Te) grid profiles + LORO transfer (~40 min)
   python scripts/make_report.py        # figures, R* tables, ONNX models, sweep
   python scripts/scenario_case.py      # model-implied scenarios (~1 min)
   python scripts/scenario_bands.py     # rate-rescale/jackknife bands on scenarios (~10 min)
   python scripts/ensemble_bands.py     # seed bands
   python scripts/bootstrap_ci.py       # jackknife CIs for ODE-shoot (~5 min)
   python scripts/summarize_all.py      # ONE-pipeline tables (metrics_long, summary)
   python scripts/make_paper_figures.py # paper figure set
   ```
   Outputs land in `runs/<driver>/` (`OHP_TV=outer|mean4|inner`, default outer; each
   driver gets its own folder). Tests: `python -m unittest discover -s tests`.

## Method in one paragraph

The measured vessel temperature T_v(t) drives two lumped nodes,
`dTe/dt = a_f (Tv − Te) − κ_f (Te − Tc)` and `dTc/dt = r κ_f (Te − Tc) − d (Tc − Tcool)`,
per fluid f. The PINN trains a trial network on the data plus this ODE residual
(autodiff in time) with the physical parameters learned jointly; ODE shooting fits the
same system by least squares. Rollouts from the measured initial state are scored
against the raw records. The dimensionless ratio R\* = a_f/κ_f is the reportable
per-fluid output: the rates themselves are sloppy, their ratio is not.

## Data provenance (confirmed 2026-10)

Dataset: "Dataset on Helically Coiled Oscillating Heat Pipe (HCOHP)", S. K. Yeboah &
J. Darkwa, DOI 10.17632/wnf5jwzp3c.3 (CC BY 4.0). Descriptor: Data in Brief 33 (2020)
106505. Parent study: Yeboah & Darkwa, Int. J. Thermal Sciences 131 (2018),
10.1016/j.ijthermalsci.2018.02.014. Confirmed: EOHP/MOHP/WOHP = ethanol/methanol/DI
water; copper vessel L = 0.30 m (ID/OD 7.8/8.0 cm); helical coil 2 mm ID / 1 mm wall,
coil dia 8 cm, 10 turns; sections 0.19/0.20/0.19 m; 5 s sampling (Yokogawa logger,
Omega K-type TCs).

Known data caveats (examined, documented in `runs/outer/`):
- Vessel inner/outer TC columns identical across fluid sheets within a run (shared
  bath measurement — basis of the equal-coupling assumption).
- Derived vessel-flux columns reproduce the cylindrical wall conductance
  (29 714 W/K) but are TC-resolution-limited.
- The derived-sheets flux constant h = 2100 W/m²K could not be traced to any source.
- MOHP Run 3 contains a flow-state change (~2000 s) and a hot-slug transient
  (~4150 s), left in (exclusion windows change that fold's RMSE only marginally).

## Limitations

One geometry, three fluids, three runs, a single extrapolation fold; slow ramps
(oscillation statistics unresolved at 5 s); equal vessel–evaporator coupling assumed;
absolute fluxes not identifiable. No claims about hotspot-flux or geometry
optimisation.

## Citing

See `CITATION.cff`. Please also cite the source dataset (DOI 10.17632/wnf5jwzp3c.3).
License: MIT (code). The dataset is CC BY 4.0, copyright its authors.
