"""Digital twin of the identified OHP model -- the fitted LUMPED ODE, not the PINN network.

What is exported: the learned right-hand side f(Te, Tc, Tv; fluid) -> (dTe/dt, dTc/dt) and R* = a_f/kappa_f,
i.e. the physical parameters identified by the PINN / ODE-shooting fits. The PINN trial network U(t, series)
is a fitting device only and is NOT part of the twin (for the state-dependent variant, the kappa(.) MLP is).

export_onnx   : writes that right-hand side as a stand-alone ONNX graph (MatMul/Tanh/Exp only
                -> runs anywhere onnxruntime runs).
replay        : streaming loop -- every second read Tv, advance the state with RK4 (4 ONNX calls), emit Te, Tc, R*.
sweep         : runs standardised transient heating scenarios through the twin for all fluids / all fitted parameter sets.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

FLUIDS = ["EOHP", "MOHP", "WOHP"]


def export_onnx(phys, state_dependent, path):
    f32 = lambda x: np.asarray(x, np.float32)
    inits, nodes = [], []
    def C(name, arr): inits.append(numpy_helper.from_array(f32(arr), name)); return name
    def N(op, ins, out, **kw): nodes.append(helper.make_node(op, ins, [out], **kw)); return out

    # selector vectors for the 3-vector state (Te, Tc, Tv)
    C("sel_Te_Tc", [[1.0], [-1.0], [0.0]]); C("sel_Tv_Te", [[-1.0], [0.0], [1.0]]); C("sel_Tc", [[0.0], [1.0], [0.0]])
    C("loga", np.asarray(phys["loga"]).reshape(3, 1)); C("logK", np.asarray(phys["logK"]).reshape(3, 1))
    C("r", np.exp(float(phys["logr"]))); C("d", np.exp(float(phys["logd"]))); C("Tcool", float(phys["Tcool"]))

    N("MatMul", ["fluid", "loga"], "loga_f"); N("Exp", ["loga_f"], "a")
    N("MatMul", ["fluid", "logK"], "logk0")
    if state_dependent:
        M = np.zeros((3, 2), np.float32); M[0, 0] = 1 / 30.0; M[0, 1] = 1 / 30.0; M[1, 1] = -1 / 30.0
        C("featM", M); C("featB", [[-50.0 / 30.0, -30.0 / 30.0]])
        N("MatMul", ["state", "featM"], "feat_a"); N("Add", ["feat_a", "featB"], "feat")
        N("Concat", ["fluid", "feat"], "x0", axis=1)
        x = "x0"
        for i, (W, b) in enumerate(phys["g"]):
            C(f"gW{i}", W); C(f"gb{i}", b)
            N("MatMul", [x, f"gW{i}"], f"gz{i}"); N("Add", [f"gz{i}", f"gb{i}"], f"gh{i}")
            x = N("Tanh", [f"gh{i}"], f"gt{i}") if i < len(phys["g"]) - 1 else f"gh{i}"
        N("Add", ["logk0", x], "logk")
    else:
        N("Identity", ["logk0"], "logk")
    N("Exp", ["logk"], "kappa")
    N("MatMul", ["state", "sel_Te_Tc"], "dTeTc"); N("Mul", ["kappa", "dTeTc"], "q")
    N("MatMul", ["state", "sel_Tv_Te"], "dTvTe"); N("Mul", ["a", "dTvTe"], "inflow")
    N("Sub", ["inflow", "q"], "dTe")
    N("MatMul", ["state", "sel_Tc"], "Tc_"); N("Sub", ["Tc_", "Tcool"], "Tc_m"); N("Mul", ["d", "Tc_m"], "loss")
    N("Mul", ["r", "q"], "rq"); N("Sub", ["rq", "loss"], "dTc")
    N("Concat", ["dTe", "dTc"], "deriv", axis=1)
    N("Sub", ["loga_f", "logk"], "logR"); N("Exp", ["logR"], "Rstar")

    g = helper.make_graph(
        nodes, "ohp_twin",
        [helper.make_tensor_value_info("state", TensorProto.FLOAT, [1, 3]),
         helper.make_tensor_value_info("fluid", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("deriv", TensorProto.FLOAT, [1, 2]),
         helper.make_tensor_value_info("Rstar", TensorProto.FLOAT, [1, 1])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 9
    onnx.checker.check_model(m); onnx.save(m, path)
    return path


class ODETwin:
    def __init__(self, path):
        self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def f(self, Te, Tc, Tv, fl):
        oh = np.zeros((1, 3), np.float32); oh[0, FLUIDS.index(fl)] = 1
        dv, R = self.s.run(None, {"state": np.array([[Te, Tc, Tv]], np.float32), "fluid": oh})
        return dv[0, 0], dv[0, 1], R[0, 0]

    def replay(self, fl, Tv_stream, y0, h=1.0):
        """Tv_stream: vessel temperature sampled every h seconds. Returns Te, Tc, R* arrays (same length)."""
        n = len(Tv_stream); Te = np.zeros(n); Tc = np.zeros(n); Rs = np.zeros(n)
        y = np.array(y0, float); Te[0], Tc[0] = y; Rs[0] = self.f(y[0], y[1], Tv_stream[0], fl)[2]
        for i in range(n - 1):
            v0, v1 = Tv_stream[i], Tv_stream[i + 1]; vm = 0.5 * (v0 + v1)
            F = lambda yy, v: np.array(self.f(yy[0], yy[1], v, fl)[:2], float)
            k1 = F(y, v0); k2 = F(y + 0.5 * h * k1, vm); k3 = F(y + 0.5 * h * k2, vm); k4 = F(y + h * k3, v1)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Te[i + 1], Tc[i + 1] = y; Rs[i + 1] = self.f(y[0], y[1], v1, fl)[2]
        return Te, Tc, Rs


SCENARIOS = [(amp, tau) for amp in (30.0, 50.0, 70.0) for tau in (150.0, 300.0, 600.0)]   # (final rise [K], time constant [s])


def scenario_Tv(amp, tau, T_start=20.0, dur=1800, h=1.0):
    t = np.arange(0, dur + 1e-9, h)
    return t, T_start + amp * (1.0 - np.exp(-t / tau))


def sweep(twin, fluids=FLUIDS, Tc0=18.0):
    """Returns dict[(fluid, scenario_idx)] -> dict(peakTe, extraction, Rstar_mean)."""
    out = {}
    for si, (amp, tau) in enumerate(SCENARIOS):
        t, Tv = scenario_Tv(amp, tau)
        for fl in fluids:
            Te, Tc, Rs = twin.replay(fl, Tv, [Tv[0], Tc0])
            out[(fl, si)] = dict(peakTe=float(Te.max()), finalTe=float(Te[-1]), gap=float(np.mean(Tv[200:] - Te[200:])),
                                 Rstar=float(np.mean(Rs[200:])), Tc_rise=float(Tc[-1] - Tc[0]))
    return out
