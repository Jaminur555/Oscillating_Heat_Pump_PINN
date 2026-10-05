# OHP PROJECT — COMPLETE BEGINNER'S WALKTHROUGH

Written 2026-10-05, after review round 2 (commit 2be2359).
Read top to bottom the first time; afterwards use it as a reference (section 7 is a glossary).

================================================================
PART 1 — WHAT THIS PROJECT IS ABOUT
================================================================

1.1 THE PHYSICAL DEVICE
-----------------------
An "oscillating heat pipe" (OHP) is a simple-looking cooling device: a long pipe bent
into many turns, part-filled with a working fluid (in our data: ethanol "EOHP",
methanol "MOHP", water "WOHP"). One end (the EVAPORATOR) is heated, the other end (the
CONDENSER) is cooled. Vapour bubbles grow and collapse, which makes the fluid slosh
back and forth and carries heat from hot end to cold end very effectively.

The experimental dataset (downloaded from Mendeley Data, free, CC BY licence) comes
from one apparatus: a helically coiled OHP ("HCOHP") driven by a heated VESSEL
(a water bath whose temperature we can read). For each fluid there were 3 heating
runs, where the vessel settles at roughly 58, 70 and 93 degrees C. So we have
3 runs x 3 fluids = 9 temperature records ("series").

Each series contains thermocouple (digital thermometer) readings, every 5 seconds:
    Tv = vessel temperature      (the "input": how hard we drive the system)
    Te = evaporator temperature  (the "output" we want to predict)
    Ta = adiabatic temperature   (middle section; NEVER used in fitting)
    Tc = condenser temperature   (second output, moves only a little)

1.2 THE GOAL
------------
Question: can we learn a small mathematical model from these 9 series that
(1) predicts Te of a run it was NOT trained on, even a hotter/colder run
    ("extrapolation"), and
(2) gives an interpretable number that ranks the three fluids?

Answer learned in this project: YES, IF the model has the right *structure*
(a 2-node energy balance). Fancy black-box machine learning fails at (1);
a plain linear model with the right structure succeeds.

1.3 THE MODEL (the heart of everything — understand this first)
---------------------------------------------------------------
Two "lumped nodes" = two temperatures that represent whole sections:

    dTe/dt = a_f * (Tv - Te)  -  kappa_f * (Te - Tc)        (evaporator node)
    dTc/dt = r * kappa_f * (Te - Tc)  -  d * (Tc - Tcool)   (condenser node)

Read them as word-equations:
    "Te rises towards the vessel temperature, at rate a"        (heat comes IN)
    "and Te drops towards Tc, at rate kappa"                    (heat flows through the pipe)
    "Tc rises when heat arrives from Te (scaled by r)"
    "and Tc falls towards the coolant temperature Tcool, at rate d"   (heat leaves)

The five kinds of parameter (all fitted from data):
    a_f     [1/s]  vessel->evaporator coupling, ONE PER FLUID (f = E, M, W)
    kappa_f [1/s]  pipe conductance (heat transfer through the OHP), ONE PER FLUID
    r       [-]    capacity ratio of the two nodes, SHARED by all fluids
    d       [1/s]  condenser heat-loss rate, SHARED
    Tcool   [degC] effective coolant temperature, SHARED
That is 3 + 3 + 1 + 1 + 1 = 9 numbers total — deliberately tiny.

THE KEY OUTPUT, R*:
    R* = a_f / kappa_f  =  "inlet resistance / pipe resistance" (dimensionless)
Why it matters: in steady state the ODE algebra collapses to
    R* = (Te - Tc) / (Tv - Te)
i.e. you can compute R* directly from plateau data without any fitting. The fitted
model reproduces those ratios — a strong consistency check. R* ranks the fluids
EOHP > MOHP > WOHP in every single fit we ever ran.

1.4 THE CENTRAL FINDINGS (as of round 2)
----------------------------------------
F1. Accuracy: held-out Te error is 0.24–0.46 K (interpolating) and 0.68–0.85 K
    (extrapolating in heating level). For scale: the signal rises ~60 K, and a
    trivial "Te = Tv - constant" baseline already gets within ~3 K.
F2. WHAT carries the accuracy is the *linear two-state structure fitted by
    shooting* — NOT the physics parameterisation, NOT the neural network.
    Proof: an unconstrained per-fluid linear model (no physics at all, 24 free
    numbers) scores 0.28/0.30/0.68 K — equal or better than the 9-parameter
    physical model in every fold. All nonlinear black boxes fail (5–87 K).
F3. R* is robust (sampling rate, seeds, methods, jackknife) and its ethanol drift
    (~48% across heating levels) matches the ethanol dry-out reported in the
    source study. Decomposition: about 65% of the E-vs-W R* gap comes from the
    inlet coupling a_f, 35% from the pipe kappa_f.
F4. kappa(Te) — temperature-dependent pipe conductance — IS identifiable within a
    single run, but the fitted exponent does NOT transfer to other heating levels
    (8 of 9 leave-one-out folds get worse or unchanged). So the deployed model
    keeps kappa constant; the temperature dependence is a heating-level effect.
F5. The exported ONNX "twin" (section 4.5) replays held-out runs and runs
    what-if scenarios. Tracking metrics robustly favour EOHP; the "WOHP rejects
    ripple better" edge is real at the fitted rates but lies inside the
    rate-uncertainty band (not credibly separable).

================================================================
PART 2 — JARGON, IN ONE PARAGRAPH EACH
================================================================

* GREY-BOX MODEL: between "white box" (derive everything from first principles —
  impossible here, two-phase flow) and "black box" (a neural net that maps inputs
  to outputs with no insight). We impose the physically-mandatory skeleton (two
  energy balances) but LEARN its few coefficients from data.

* FITTING / IDENTIFICATION: finding the 9 parameter values that make the model's
  simulated temperatures match the measured ones as closely as possible.

* ODE SHOOTING ("ODE-shoot" in the code): the classical way. Guess parameters,
  numerically integrate the ODE from the measured starting temperature, compare
  the simulated curve with the measured curve, let scipy's `least_squares`
  adjust the parameters to reduce the mismatch, repeat. "Output-error" fitting.

* PINN (physics-informed neural network): the modern way. Train a small neural
  network U(t) to output (Te, Tc); the loss = (data mismatch) + (how badly the
  ODE is violated by U, computed with automatic differentiation). The physical
  parameters are trained jointly. FINDING: for this small linear system it works
  exactly as well as shooting — no better. That honest null result is F2.

* LORO = leave-one-run-out validation. Train on 2 runs (6 series), predict the
  3rd (3 series). Because run plateaus are ~58/70/93 degC:
    hold out Run 2 (70)  -> INTERPOLATION   (trained on 58 and 93)
    hold out Run 1 (58)  -> DOWNWARD extrapolation
    hold out Run 3 (93)  -> UPWARD extrapolation (the hard, interesting fold)

* RMSE: root-mean-square error in kelvin between prediction and raw measurement.
  All headline numbers in the paper are RMSE of Te against the RAW data.

* BASELINES: reference models that show what the score MEANS.
    Trivial      Te = Tv - c_f  (one number per fluid: the mean gap)
    MLP          standard feed-forward neural network
    GP           Gaussian process regressor
    SINDy        sparse-symbolic regression (tries to discover the equation)
    GRU          recurrent network (has memory, made for sequences)
    Ridge-linear linear state-space fitted on finite-difference derivatives
    One-node     our ODE with the condenser node deleted (ablation)
    Linear-SS    unconstrained per-fluid linear state-space (NEW in round 2)
                 dX/dt = A*X + g*Tv + c, all 8 entries of (A, g, c) free.

* IDENTIFIABILITY / SLOPPINESS: a parameter is identifiable if the data pin it
  down. We discovered (a, kappa) jointly are NOT — rescale BOTH by 2x and the fit
  barely changes (their RATIO is what the data constrain). Such systems are
  called "sloppy". Consequence: report R*, not the raw rates; and absolute-rate
  dependent quantities (scenario timings) carry a x2 band.

* JACKKNIFE: uncertainty estimate by re-fitting with one data series removed at
  a time (9 refits) and watching how the parameters move. Cheap, honest.

* ONNX: a portable file format for small computation graphs. We export the fitted
  right-hand-side of the ODE so anyone can run the model without Python/torch.

* kappa(Te): the variant model kappa_f(Te) = k0_f * exp(beta_f * (Te - 50)) —
  pipe conductance that grows/shrinks with temperature. See F4.

================================================================
PART 3 — REPOSITORY MAP
================================================================

    src/ohp/           THE LIBRARY (import, don't run directly)
        config.py      where outputs go (runs/<driver>/), driver selection
        data.py        Excel -> clean arrays (smoothing, 5 s axis, raw copy)
        metrics.py     RMSE etc. against the RAW full-rate data
        pinn.py        the physics + the PINN trainer + the simulator
        baselines.py   ODE-shoot fitter + MLP/GP/SINDy baselines
        gru.py         GRU baseline model
        ode_twin.py    ONNX export + streaming replay + scenario sweep

    scripts/           EXECUTABLES (run from repo root, in roughly this order)
        evaluate.py            the MAIN experiment: all LORO fits -> loro_torch.pkl
        rerun_baselines.py     re-fits MLP/GP with fair features -> loro.pkl
        gru_baseline.py        GRU LORO -> keys "GRU|seedN" in loro_torch.pkl
        review_baselines.py    Trivial / Ridge / One-node / Linear-SS
        kappa_te_profile.py    beta-grid profiles + transfer test (round 2)
        kappa_te_report.py     prints PINN-const vs PINN-kTe comparison
        kappa_te_diagnostic.py OLD, superseded (its beta fit stalled — kept for record)
        adiabatic_validation.py Ta ~ w1*Te + w2*Tc checks (parts A–D)
        bootstrap_ci.py        jackknife over the 9 series (+ t-intervals)
        scenario_case.py       the 15 duty-cycle scenarios through 4 twins
        scenario_bands.py      rate-rescale x/div-2 + jackknife bands (round 2)
        ensemble_bands.py      seed-spread bands (what they do/don't measure)
        rstar_decomposition.py 65/35 a-vs-kappa split of R* (round 2)
        make_report.py         internal report: figures, R* tables, ONNX, sweep
        summarize_all.py       ONE pipeline -> the two summary CSV tables
        make_paper_figures.py  paper/figures/fig1..fig9 from the pickles
        plot_raw.py            quick-look plots of the raw data
        tv_sensitivity.py      re-run with a different vessel-driver choice
        derived_checks.py      checks against the dataset's derived Excel sheets

    tests/             unit tests, synthetic data only (run without the dataset)
        test_rollout_parity.py, test_invariants.py, test_metrics.py,
        test_review_tools.py   (NEW: baseline fitters + jackknife formulas)

    paper/             the manuscript (LaTeX)
        paper.tex + sections/*.tex + figures/fig1..fig9

    runs/outer/        ALL RESULTS (driver = "outer" vessel TCs): pickles,
                       CSV tables, .log files (provenance for every claim),
                       ONNX twins. *.pkl and *.onnx are gitignored (big/regenerable).

    data_raw/          the 3 Mendeley Excel files (NOT in git; download manually)

================================================================
PART 4 — FILE BY FILE, WITH CODE SNAPSHOTS
================================================================
Format of each entry:
    WHAT  — the purpose in one or two sentences
    HOW   — the steps the code performs
    SNAP  — a short real excerpt + plain-language explanation
    OUT   — what it writes, and which paper claim it supports

----------------------------------------------------------------
4.1  src/ohp/config.py
----------------------------------------------------------------
WHAT: Chooses which vessel thermocouples drive the model ("driver") and where all
      outputs are written.
HOW: reads environment variable OHP_TV (default "outer"), builds the output path
     runs/<driver>/ and creates it.
SNAP:
    DRIVER = os.environ.get("OHP_TV", "outer")
    RUNS_DIR = os.path.join("runs", DRIVER)
    -> every result file lives under runs/outer/ (or runs/mean4/ ...), so
       different driver choices can never overwrite each other.
OUT: nothing itself; every other module imports RUNS_DIR from here.

----------------------------------------------------------------
4.2  src/ohp/data.py
----------------------------------------------------------------
WHAT: Turns the three Excel files into 9 clean series dictionaries.
HOW: for each (run, fluid):
       1. read sheet "<FLUID> R<run>" (header row 5),
       2. average the right thermocouple groups (3 evaporator, 2 adiabatic,
          3 condenser, 2-of-4 vessel depending on driver),
       3. build t = index * 5.0 seconds,
       4. keep BOTH a smoothed/subsampled version (for fitting) and the raw
          full-rate copy (for honest scoring).
SNAP:
    out["Te"] = df[ev].mean(axis=1).values.astype(float)   # mean of 3 TCs
    out["t"]  = np.arange(len(out["Te"])) * DT_S           # 5 s per sample
    -> a "series" is just a dict of numpy arrays
       {"t", "Tv", "Te", "Ta", "Tc", "Te_sd", "raw": {...}, "Tv_dot"}
OUT: `D = load_all()` — the input to everything else. DT_S = 5.0 is confirmed
     from the dataset article (Yokogawa logger).

----------------------------------------------------------------
4.3  src/ohp/metrics.py
----------------------------------------------------------------
WHAT: The scoring rules — one function, deliberately tiny.
HOW: interpolate the model's (5 s) prediction onto the raw (1 s) time axis,
     then compute RMSE / end-error / max-error of Te and Tc.
SNAP:
    te_i = np.interp(raw["t"], d["t"], te)
    e = te_i - raw["Te"]
    return dict(rmse_Te=float(np.sqrt(np.mean(e ** 2))), ...)
    -> "metrics_raw" = scored against raw data (older, optimistic scoring against
       the smoothed data was retired in 2026-10).
OUT: the rmse_Te numbers that fill Table 1 of the paper.

----------------------------------------------------------------
4.4  src/ohp/pinn.py   (THE CORE FILE — read this twice)
----------------------------------------------------------------
WHAT: Defines the physics, trains the PINN, and simulates the fitted ODE.
HOW the physics is expressed (kappa as a function, both variants):
SNAP 1 — the right-hand side, vectorised over many points at once:
    def rhs(phys, fl_oh, Te, Tc, Tv, state_dependent, kappa_te=False):
        a   = torch.exp(fl_oh @ phys["loga"])       # per-fluid coupling
        kap = kappa_fn(phys, fl_oh, Te, Tc, ...)    # per-fluid conductance
        q   = kap * (Te - Tc)                       # pipe heat flux term
        dTe = a * (Tv - Te) - q                     # <- the evaporator ODE
        dTc = torch.exp(phys["logr"]) * q - torch.exp(d_rate) * (Tc - phys["Tcool"])
        return dTe, dTc
    -> parameters are STORED as logarithms (loga, logK, ...) so optimisation
       can never push a rate negative: exp() of any real number is positive.
    -> fl_oh is a one-hot vector like [1,0,0] selecting the fluid.

SNAP 2 — the two kappa variants of interest:
    logk = fl_oh @ phys["logK"]                     # constant kappa (default)
    if kappa_te:
        logk = logk + (fl_oh @ phys["beta"]) * (Te - 50.0)
    -> kappa_te makes log(kappa) a straight line in temperature:
       kappa_f(Te) = k0_f * exp(beta_f * (Te - 50)).
       beta is the number the whole round-2 kappa story is about (see 4.10).

SNAP 3 — the training loss (what a PINN actually minimises):
    pred = U(netp, oh_net)                          # network's (Te, Tc) curves
    l_data = mean(((pred - measured)/sd) ** 2)      # look like the data
    dTe, dTc = rhs(phys, ..., y0c, y1c, tc_tv, ...) # ODE at network outputs
    R0 = (dy0 - dTe) / se                           # d(network)/dt via autograd
    loss = l_data + w_phys * l_phys                 # + obey the physics
    -> the network's time-derivative is taken by AUTOMATIC DIFFERENTIATION
       (torch.autograd.grad), compared against the ODE's right-hand side.
       The 9 physical parameters sit in the same optimiser as the network
       weights (Adam, cosine-decayed learning rate, 6000 steps).
    -> a data-only warm-up (first 25% of steps) stabilises the start.

SNAP 4 — how ANY fitted model is later evaluated (no neural network involved):
    def rollout(phys, fl_idx, Tv_half, y0, h, ...):    # plain RK4 integrator
        k1 = f(y, v0); k2 = f(y + h/2*k1, vm); k3 = ...; k4 = ...
        y  = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
    -> simulate_series() builds a smooth spline of the measured Tv, rolls the
       identified ODE from the MEASURED initial temperature, and returns
       (Te_sim, Tc_sim) on the data time axis. Identical protocol for every
       method — that is what makes the comparison fair.
OUT: train_pinn() -> fitted "phys" dict; simulate_series() -> predictions.
     Used by scripts/evaluate.py for the "PINN-*" and (via phys) all rollouts.

----------------------------------------------------------------
4.5  src/ohp/baselines.py
----------------------------------------------------------------
WHAT: The classical fitter (ODE-shoot) and three black-box baselines.
SNAP — ODE shooting in 12 lines:
    def resid(lp):                       # lp = log-parameters
        for (k, d) in series:
            y = _roll(P, d, FLUIDS.index(k[1]))   # integrate the ODE
            out += [y[0] - d["Te"], 4*(y[1] - d["Tc"])]  # weighted residuals
    s = least_squares(resid, np.log(p0), diff_step=1e-3, max_nfev=100)
    -> "shooting" = integrate from the measured start, compare curves, let
       least-squares adjust. Tc residuals are weighted x4 (Tc moves little;
       without a weight it would be ignored).
    -> diff_step=1e-3: the finite-difference step size for the parameter
       gradient — the DEFAULT step (~1e-8) is smaller than integrator noise
       and the optimiser stalls. Remember this lesson, see 4.10.
Also here: fit_mlp, fit_gp (sklearn), fit_sindy (pysindy). All receive the
same features including the measured initial condition ("IC-fair").
OUT: fit_ode_shoot is the "ODE-shoot" row of Table 1 and the deployed model.

----------------------------------------------------------------
4.6  src/ohp/gru.py
----------------------------------------------------------------
WHAT: The recurrent black-box baseline (best black box on extrapolation).
HOW: a GRU cell (a mini network with memory) consumes
     [t, Tv, dTv/dt, fluid-onehot, Te_prev, Tc_prev] each step and predicts
     the NEXT (Te, Tc); trained "teacher-forced" (true previous state fed in),
     tested autoregressively (its own previous output fed back) from the
     measured initial state. ~12 min per fit on CPU.
OUT: 0.77/0.69/4.99 K — 6x worse than structured fits on the extrapolation
     fold, but far better than MLP/GP/SINDy (23–87 K).

----------------------------------------------------------------
4.7  src/ohp/ode_twin.py
----------------------------------------------------------------
WHAT: Exports the FITTED ODE (never the neural network) as a portable ONNX
      graph, and runs it: streaming replay + scenario sweeps.
SNAP — what an ONNX "twin" is:
    export_onnx(phys, state_dependent, path)   # writes a tiny .onnx file of
    # MatMul/Exp/Add nodes computing: kappa, q, dTe, dTc, R* = a/kappa
    -> runs anywhere onnxruntime runs (no Python, no torch, no repo).
SNAP — streaming replay (how a real deployment would use it):
    for i in range(n - 1):                   # every second:
        k1..k4 = f(y, Tv_stream[i..i+1])     # 4 ONNX calls = one RK4 step
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)  # advance the state
    -> the twin only ever SEES the vessel temperature stream; Te/Tc come out.
OUT: runs/outer/ode_twin_fold{1,2,3,None}.onnx — used by scenario scripts,
     fig6 (replay), fig7 (case study).

----------------------------------------------------------------
4.8  scripts/evaluate.py   (THE MAIN EXPERIMENT)
----------------------------------------------------------------
WHAT: Runs the whole LORO benchmark and saves everything, resumable.
HOW: for each fold (hold out run 1, 2, 3, or None = fit on everything):
       train ODE-shoot  -> predict held-out -> score  -> store
       train PINN-const x10 seeds, PINN-state x10 seeds
       train PINN-kTe x3 seeds (the kappa(Te) variant)
       train PINN-const-df x3 seeds (per-fluid d ablation)
     Everything goes into runs/outer/loro_torch.pkl as nested dicts:
       res[fold]["PINN-const|seed0"]["met"][(run, fluid)]["rmse_Te"]
SNAP — the resumable-run pattern used by all heavy scripts:
    def run(name, fn):
        if name in R: return          # already done on an earlier invocation
        R[name] = fn(); save()        # else compute, then persist immediately
    -> long experiments can be interrupted and restarted safely.
OUT: loro_torch.pkl — the backbone of Table 1, fig2, fig3, fig4.

----------------------------------------------------------------
4.9  scripts/review_baselines.py
----------------------------------------------------------------
WHAT: The four "reference" models from the review: Trivial, Ridge-linear,
      One-node, Linear-SS. Refactored (round 2) so the fitters are importable
      and unit-testable; the LORO loop runs only under `if __name__ == "__main__":`.
SNAP — Linear-SS, the model that changed the paper's framing:
    A = p[:4].reshape(2, 2); g = p[4:6]; c = p[6:8]      # 8 free numbers/fluid
    f = lambda t, y: np.clip(A @ y, -1e4, 1e4) + g * Tv(t) + c
    y = solve_ivp(f, ..., t_eval=d["t"]).y               # output-error shooting
    -> identical protocol to the physical model (same optimiser, same
       integration), but ZERO physics content: A, g, c are just numbers.
       Score 0.28/0.30/0.68 K => physics parameterisation not needed for
       accuracy (F2). The np.clip keeps diverging fits numerically finite.
OUT: runs/outer/review_baselines.pkl + review_baselines.log; rows of Table 1.

----------------------------------------------------------------
4.10 scripts/kappa_te_profile.py   (ROUND 2, ITEM 1 — the retraction)
----------------------------------------------------------------
WHAT: Proves F4 with a grid search that cannot stall.
BACKGROUND: the OLD script (kappa_te_diagnostic.py) fitted beta by
      least-squares starting from beta=0 — but least_squares estimates a
      gradient by nudging each parameter by a step RELATIVE to its size;
      at exactly 0 the nudge is ~0, the gradient looked flat, beta never
      moved, and we wrongly concluded "not identifiable".
HOW (fix): never optimise beta. Put beta on a GRID of fixed values
      (-0.02 ... +0.04), and at every grid point re-fit the other 5
      parameters by shooting; the RMSE-vs-beta curve IS the answer.
SNAP:
    for b in BETA_GRID:                 # fixed beta...
        p, rm = fit([d], b, warm)       # ...others refit; warm start
        res[b] = (p, rm[0])
    best = min(res, key=lambda b: res[b][1])
    -> warm start: each grid point starts from the previous solution —
       much faster and smoother profiles.
    PART A: each run alone -> beta IS visible within a run
            (ethanol 67–71% RMSE improvement at beta ~ +0.03,
             water 45–48% at beta ~ -0.02, methanol none)
    PART B: fit (params+beta) on 2 runs, SIMULATE the held-out run:
            applying the training beta is worse or equal in 8/9 folds
            -> within-run yes, transfer NO. Also a diverged-rollout guard:
SNAP (robustness trick used everywhere integrators can blow up):
    if not s.success or s.y.shape[1] != len(d["t"]):
        y = np.full((2, len(d["t"])), 1e3)     # huge-but-finite fake curve
    -> the optimiser treats it as "very bad" and steps away, instead of
       crashing on a broadcast error.
OUT: runs/outer/kappa_te_profile.{csv,log}; the rewritten "Negative and
     corrected results" subsection of the paper.

----------------------------------------------------------------
4.11 scripts/adiabatic_validation.py
----------------------------------------------------------------
WHAT: The free consistency check on channels never used in fitting.
HOW: fit Ta ~ w1*Te + w2*Tc (2 weights) in-sample (A), leave-one-run-out (B),
     through the model's own predicted Te/Tc — the "twin chain" (C), and the
     1-parameter version w2 = 1-w1 (D, added round 2).
SNAP (D — closed form, no loop over w1 needed):
    X = concatenate(d["Te"] - d["Tc"] for training runs)   # Ta-Tc = w1*(Te-Tc)
    w1 = X @ y / (X @ X)                                   # 1-D least squares
    -> EOHP/WOHP held-out error 0.17–0.58 K; conclusion: the check is weak
       (any mix "between the nodes" passes) but free, and it validates that
       the two fitted states are mutually consistent with an unused sensor.
OUT: adiabatic_validation.log; fig8 + the "Adiabatic-channel" subsection.

----------------------------------------------------------------
4.12 scripts/bootstrap_ci.py
----------------------------------------------------------------
WHAT: Jackknife uncertainty for the shooting model's parameters and R*.
HOW: fit on all 9 series (the "full fit"), then re-fit 9 times leaving one
     series out; the SPREAD of the leave-one-out fits estimates the sd.
SNAP:
    jk_sd = np.sqrt((n - 1) / n * ((T[1:] - T[1:].mean(0)) ** 2).sum(0))
    -> textbook delete-one jackknife sd (n = 9). R* rows are kept in LOG
       scale (values like 2.61; multiplicative error = exp(sd)).
    Round 2 added: 95% t-intervals with 8 degrees of freedom
       (t = 2.306), and jackknife_fits.csv = all 10 parameter vectors
       (feeds scenario_bands.py).
OUT: jackknife.csv/.log — "ordering robust at 5.4 and 8.2 sigma"; the R*
     interval row of Table 2.

----------------------------------------------------------------
4.13 scripts/scenario_case.py  +  scripts/scenario_bands.py
----------------------------------------------------------------
WHAT (case): the 3-families-x-3-setpoints duty-cycle study through the 4
      ONNX twins: A ramp+hold (3 rates), B thermostat on/off, C 60 s slug.
      Metrics: deviation from setpoint, vessel-evaporator lag, settling time,
      ripple (peak-to-peak), worst deviation.
SNAP (a waveform is just an analytic array — no data involved):
    def cyclic(Tset, n_cyc=4, t_on=900.0, t_off=900.0, ...):
        segs.append(v_hi - (v_hi - v) * np.exp(-tt / tau_on))   # heat up
        segs.append(T_off + (v - T_off) * np.exp(-tt / tau_off))# cool down
    -> the twin then REPLAYS each waveform; everything stays inside the
       fitted envelope (Tv in [20, 95] degC, timescales 150–1800 s).
WHAT (bands, round 2): how much do those scenario numbers depend on the
      sloppy ABSOLUTE rates? Re-runs all scenarios with (a,kappa) jointly
      rescaled x0.5/x1/x2 and through the 9 jackknife refits.
SNAP:
    p["loga"] = np.asarray(p["loga"]) + np.log(s)   # rescale a and kappa
    p["logK"] = np.asarray(p["logK"]) + np.log(s)   # R* unchanged!
    -> findings F5: lag/deviation are rate-INVARIANT (they are quasi-static:
       Tv - Te = R*(Te - Tc) at steady state), ripple is not; EOHP-WOHP
       ripple ordering flips inside the band (60/40 share).
OUT: scenario_case.{csv,log}, scenario_bands.{csv,log}; fig7 + the
     "Exported model and scenario" subsection.

----------------------------------------------------------------
4.14 scripts/rstar_decomposition.py   (ROUND 2, ITEM 5)
----------------------------------------------------------------
WHAT: splits the R* ordering into inlet-coupling vs pipe shares.
SNAP (the whole idea in two lines):
    la = np.log(a_E / a_W); lk = np.log(kappa_E / kappa_W)
    tot = la - lk        # = ln(R*_E / R*_W);  shares: 100*la/tot etc.
    -> ODE fit: 0.831 = 0.538 - (-0.293)  =>  65% a_f / 35% kappa_f.
       Because the shares use per-fluid RATIOS, the global rescaling
       sloppiness cancels — the split is robust to it.
OUT: rstar_decomposition.log; quoted in "Identified parameters and R*".

----------------------------------------------------------------
4.15 scripts/ensemble_bands.py
----------------------------------------------------------------
WHAT: what the 10 PINN seeds' spread does and does not measure.
HOW: seed-mean +/- 2sd bands on held-out rollouts; empirical coverage of the
     raw data by those bands; scenario-metric CIs across seeds.
OUT: ensemble_bands.{csv,log}; fig9 + "What the seed ensemble does not
     measure" (seed spread = parameter uncertainty only, 0.01–0.05 K vs
     actual residuals 0.25–0.89 K).

----------------------------------------------------------------
4.16 reporting chain: make_report.py -> summarize_all.py -> make_paper_figures.py
----------------------------------------------------------------
make_report.py     : regenerates internal figures, R* tables, the ONNX twins
                     and the sweep from the pickles (no fitting).
summarize_all.py   : THE one pipeline for the summary tables — merges
                     loro.pkl (MLP/GP/SINDy) + loro_torch.pkl (ODE/PINN/GRU)
                     + review_baselines.pkl and rewrites
                     loro_metrics_long.csv + loro_summary.csv, so tables can
                     never drift from the pickles of record.
SNAP:
    RES = {ho: {**lj.get(ho, {}), **rt.get(ho, {}), **rb.get(ho, {})} ...}
    -> dict-merge of the three result files, fold by fold.
make_paper_figures.py: rebuilds fig1..fig9 (PDF vector + 300 dpi PNG) from
                     the same pickles, with the house style (white
                     background, Okabe-Ito colours, legends outside, 8 pt
                     serif). fig3 plots every method of Table 1.

----------------------------------------------------------------
4.17 smaller / auxiliary scripts
----------------------------------------------------------------
rerun_baselines.py : re-fit MLP/GP with IC-fair features -> loro.pkl.
plot_raw.py        : quick 3x3 look at the raw series.
tv_sensitivity.py  : redo ODE-shoot LORO with a different vessel driver
                     (outer / mean4 / inner) — R* values move ~12%, ranking
                     and conclusions never change.
derived_checks.py  : cross-checks against the dataset's own "derived" Excel
                     sheets (wall conduction, Run-3 transients, R_eff drift).
kappa_te_report.py : prints the PINN-const vs PINN-kTe table from the pkl.
kappa_te_diagnostic.py: SUPERSEDED — kept only as the record of the bug
                     described in 4.10 (its section B plateau check is fine).

----------------------------------------------------------------
4.18 tests/
----------------------------------------------------------------
All synthetic — they run in CI without the dataset. `path_bootstrap.py`
puts src/ on sys.path; run from repo root:
    python -m unittest discover -s tests
    test_rollout_parity.py  torch RK4 rollout tracks scipy LSODA (< 5e-4 K)
    test_invariants.py      DT_S scaling: rates scale, R* and trajectories don't
    test_metrics.py         metrics_raw definitions
    test_review_tools.py    (round 2) fitters recover KNOWN synthetic systems
                            (trivial gap; Linear-SS from exact linear data;
                            one-node from exact one-node data) + jackknife
                            formula identities (sd of mean, pseudo-values).
SNAP (the strongest test — a fitter must find a system we planted):
    def test_linss_recovers_linear_system(self):
        tr = [(("1", "EOHP"), synth_linear(1)), ...]  # exact linear series
        pred = fit_linss(tr)
        rmse = ...                                    # rollout on held-out
        self.assertLess(rmse, 0.05)                   # must re-identify it
    -> if the optimiser ever stalls the way 4.10 describes, this fails.

================================================================
PART 5 — THE PAPER, SECTION BY SECTION (paper/sections/*.tex)
================================================================
    abstract.tex          the whole story in 1 paragraph (read it first!)
    introduction.tex      motivation + the 6-item contribution list
    data.tex              the dataset, channels, 5 s sampling, LORO folds
    methods.tex           the ODE, PINN, shooting, all baselines, protocol
    results.tex           Tables 1–2, figs 2–9, every number with its
                          runs/outer/*.log provenance path next to it
    discussion.tex        what it means + the honest limitations list
    related_work.tex      13 refs, positioned against our findings
    conclusion.tex        summary
    data_availability.tex dataset DOI + Zenodo placeholder
Compile:  cd paper && pdflatex paper.tex   (twice for cross-references).
Numbers flow:  scripts/*.py -> runs/outer/*.pkl|.csv|.log -> figs/tables in
results.tex. Never type a number into the .tex that you cannot trace to a
log file — that discipline is what got this repo through review.

================================================================
PART 6 — HOW TO RUN EVERYTHING (fresh machine, from repo root)
================================================================
    # 0) one-time: pip install -e .   and put the 3 Excel files in data_raw/
    # 1) heavy fits (hours; resumable — rerun skips what exists):
    python scripts/evaluate.py            # ODE + PINN, 4 folds x seeds
    python scripts/gru_baseline.py        # GRU (~2 h)
    python scripts/rerun_baselines.py     # MLP/GP
    python scripts/review_baselines.py    # Trivial/Ridge/One-node/Linear-SS
    python scripts/kappa_te_profile.py    # ~40 min
    # 2) uncertainty / checks (minutes):
    python scripts/bootstrap_ci.py        # ~15 min
    python scripts/adiabatic_validation.py
    python scripts/ensemble_bands.py
    python scripts/scenario_case.py
    python scripts/scenario_bands.py      # ~10 min
    python scripts/rstar_decomposition.py
    # 3) outputs:
    python scripts/make_report.py
    python scripts/summarize_all.py
    python scripts/make_paper_figures.py
    # 4) tests + paper:
    python -m unittest discover -s tests
    cd paper && pdflatex paper.tex && pdflatex paper.tex
On Windows use:  set KMP_DUPLICATE_LIB_OK=TRUE  and the anaconda python.

================================================================
PART 7 — GLOSSARY (alphabetical)
================================================================
adiabatic  : unheated/uncooled middle pipe section (Ta sensors live here)
beta       : exponent of kappa(Te) = k0*exp(beta*(Te-50)); see F4
driver     : the vessel-temperature signal that drives the model (Tv)
dry-out    : fluid shortage at the evaporator at high power; hits ethanol first
fold       : one LORO split; "fold 3" = hold out run 3 (upward extrapolation)
IC-fair    : baselines also receive the measured initial condition
jackknife  : uncertainty from leave-one-series-out refits (9 here)
kappa      : pipe conductance per evaporator heat capacity [1/s]
LORO       : leave-one-run-out validation
LSODA      : the stiff-capable ODE integrator used inside all fits
ONNX       : portable computation-graph format; the exported twin
PINN       : physics-informed neural network
residual   : (prediction - measurement); shooting minimises their squares
R*         : = a_f/kappa_f = (Te-Tc)/(Tv-Te); THE per-fluid output
RMSE       : root-mean-square error [K]; headline metric
rollout    : simulate the fitted ODE forward from a measured initial state
sloppy     : direction in parameter space the data barely constrain
seed       : random-number-generator start; 10 seeds test optimiser stability
shooting   : classical fit-by-simulation (integrate, compare, adjust)
Tc/Te/Tv   : condenser / evaporator / vessel temperatures
twin       : the exported ONNX model of the fitted ODE
warm start : initialise an optimiser from a nearby earlier solution

================================================================
PART 8 — THE STORY IN FIVE SENTENCES (for when you forget everything else)
================================================================
1. Nine public temperature series (3 fluids x 3 heating runs) are modelled by a
   two-node energy balance whose 9 parameters are learned from data.
2. The model predicts unseen runs at sub-kelvin accuracy, including
   extrapolation to unseen heating levels, where every black-box baseline
   fails badly.
3. But careful baselines showed the credit belongs to the linear two-state
   structure and honest fitting — a physics-free linear twin does just as
   well; the physical form is kept for parsimony and interpretability.
4. The interpretable output is R* = a/kappa: robust, plateau-recoverable, and
   its ethanol drift mirrors the ethanol dry-out reported in the source data.
5. Everything else in the repo is machinery to make those three sentences
   bullet-proof: uncertainty bands, transfer tests, held-out channels,
   exported simulators, and a log-file trail for every number.
