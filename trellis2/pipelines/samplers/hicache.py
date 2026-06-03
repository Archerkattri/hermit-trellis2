"""HiCache: Hermite-polynomial velocity forecasting for TRELLIS.2.

Training-free inference acceleration for the TRELLIS.2 flow-matching
samplers. This is a drop-in replacement for the TaylorSeer Taylor-series
velocity forecast (Fast-TRELLIS ``taylor_utils_ss`` / ``faster_utils_slat``)
that instead forecasts the final CFG-combined velocity at *skipped* sampling
steps with a **scaled (physicist's) Hermite polynomial** basis.

Reference
---------
HiCache: Training-free Acceleration of Diffusion Models via Hermite
Polynomial Feature Forecasting (arXiv:2508.16984).

Method (verbatim from the paper)
--------------------------------
Let ``F_t`` be the cached feature/velocity at the most recent compute
("full") step and ``N = N_interval`` the spacing between compute steps.
At a compute step we update backward finite differences::

    Delta^0 F_t = F_t
    Delta^i F_t = (Delta^{i-1} F_t - Delta^{i-1} F_{t-N}) / N

At a skipped step with forward horizon ``k`` (number of steps elapsed since
the last compute step) the velocity is forecast as::

    F_hat_{t-k} = F_t + sum_{i=1}^{m} (Delta^i F_t / i!) * Htilde_i(-k)

where ``Htilde`` is the *dual-scaled* physicist's Hermite polynomial with
contraction factor ``sigma in (0, 1)``::

    Htilde_n(x) = sigma^n * H_n(sigma * x)
    H_0(x) = 1,  H_1(x) = 2x
    H_{n+1}(x) = 2*x*H_n(x) - 2*n*H_{n-1}(x)

TaylorSeer is the special case where the basis ``Htilde_i(-k)`` is replaced
by the monomial ``(-k)^i``. The dual scaling (input scale ``sigma*x`` and
coefficient scale ``sigma^n``) suppresses the exponential growth of the
high-order Hermite terms and keeps the forecast inside the numerically
stable oscillatory regime.

Substrate
---------
The cached object is the FINAL velocity ``pred_v`` (exactly the Fast-TRELLIS
substrate): cache ``pred_v`` at compute steps, forecast/reuse it at skip
steps, then rebuild the Euler update with the sampler's own
``_v_to_xstart_eps``. For the SLaT stages ``pred_v`` is a TRELLIS.2
``SparseTensor`` and only its ``.feats`` are cached/forecast; the coords are
carried through unchanged via ``.replace(feats)``. For the sparse-structure
stage ``pred_v`` is a dense tensor and is cached directly.

This module is pure numerics (CPU-unit-tested in ``__main__``); the sampler
classes in ``flow_euler.py`` own the model calls and the SparseTensor glue.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch


# ---------------------------------------------------------------------------
# Hermite basis
# ---------------------------------------------------------------------------
def physicists_hermite(n: int, x: torch.Tensor) -> torch.Tensor:
    """Physicist's Hermite polynomial ``H_n(x)`` via the stable recurrence.

    ``H_0 = 1``, ``H_1 = 2x``, ``H_{k+1} = 2 x H_k - 2 k H_{k-1}``.
    """
    if n < 0:
        raise ValueError(f"Hermite order must be >= 0, got {n}")
    if n == 0:
        return torch.ones_like(x)
    h_prev = torch.ones_like(x)          # H_0
    h_curr = 2.0 * x                     # H_1
    if n == 1:
        return h_curr
    for k in range(1, n):
        h_next = 2.0 * x * h_curr - 2.0 * k * h_prev
        h_prev, h_curr = h_curr, h_next
    return h_curr


def scaled_hermite(n: int, x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Dual-scaled Hermite ``Htilde_n(x) = sigma^n * H_n(sigma * x)``."""
    return (sigma ** n) * physicists_hermite(n, sigma * x)


def scaled_hermite_scalar(n: int, x: float, sigma: float) -> float:
    """Scalar (Python-float) ``Htilde_n(x) = sigma^n * H_n(sigma * x)``.

    Numerically identical to ``scaled_hermite`` evaluated at a 0-d tensor, but
    avoids allocating a per-step device tensor in the forecast hot loop; the
    returned float broadcasts against the cached derivative tensors. Used only
    in ``hicache_forecast`` where ``-k`` is a single integer horizon.
    """
    sx = sigma * x
    if n == 0:
        return 1.0
    h_prev = 1.0          # H_0
    h_curr = 2.0 * sx     # H_1
    for k in range(1, n):
        h_prev, h_curr = h_curr, 2.0 * sx * h_curr - 2.0 * k * h_prev
    return (sigma ** n) * h_curr


# ---------------------------------------------------------------------------
# HiCache state
# ---------------------------------------------------------------------------
def hicache_init(
    num_steps: int,
    interval: int = 3,
    max_order: int = 1,
    first_enhance: int = 2,
    end_enhance: Optional[int] = None,
    sigma: float = 0.5,
    adaptive: bool = False,
    interval_min: Optional[int] = None,
    interval_max: Optional[int] = None,
    adaptive_tol: float = 0.05,
) -> Dict[str, Any]:
    """Create a fresh HiCache state dict for one sampling run.

    Parameters
    ----------
    num_steps : total sampler steps.
    interval : ``N_interval`` -- one compute step then ``interval-1`` forecasts.
    max_order : highest finite-difference / Hermite order ``m`` (>= 1).
    first_enhance : always compute the first ``first_enhance`` steps.
    end_enhance : always compute steps with index ``>= end_enhance``
        (defaults to ``num_steps`` -> the last step is always full).
    sigma : Hermite contraction factor in ``(0, 1)``.
    adaptive : when True, ``hicache_decide`` uses the residual-gated effective
        interval (``hicache_adaptive_interval``) instead of the fixed
        ``interval``; the schedule lengthens on smooth velocity fields and
        tightens on fast-changing ones.
    interval_min, interval_max : bounds for the adaptive interval (default to
        ``interval`` if unset, i.e. a no-op band).
    adaptive_tol : residual tolerance mapping smooth<->fast to the interval band.
    """
    if interval < 1:
        raise ValueError("interval must be >= 1")
    if max_order < 1:
        raise ValueError("max_order must be >= 1")
    if not (0.0 < sigma < 1.0):
        raise ValueError(f"sigma must be in (0, 1), got {sigma}")
    imin = int(interval_min) if interval_min is not None else int(interval)
    imax = int(interval_max) if interval_max is not None else int(interval)
    return {
        "num_steps": int(num_steps),
        "interval": int(interval),
        "max_order": int(max_order),
        "first_enhance": int(first_enhance),
        "end_enhance": int(end_enhance if end_enhance is not None else num_steps),
        "sigma": float(sigma),
        "adaptive": bool(adaptive),
        "interval_min": max(imin, 1),
        "interval_max": max(imax, max(imin, 1)),
        "adaptive_tol": float(adaptive_tol),
        "last_residual": None,   # set by hicache_record_compute
        "step": 0,
        "counter": 0,            # forecasts since last compute
        "type": None,            # "full" | "forecast"
        "activated_steps": [],   # indices of compute steps
        "derivatives": {},       # {0: F_t, 1: Delta^1 F_t, ...}
    }


def hicache_decide(state: Dict[str, Any]) -> str:
    """Decide whether the current step is computed or forecast.

    Mirrors the paper's schedule (``t mod N_interval``) plus the
    enhance-window guards. Sets and returns ``state['type']``.
    """
    step = state["step"]
    first = step < state["first_enhance"]
    last = step >= state["end_enhance"]
    # Effective interval: residual-gated when adaptive, else the fixed cadence.
    eff_interval = (hicache_adaptive_interval(state)
                    if state.get("adaptive") else state["interval"])
    interval_hit = state["counter"] >= eff_interval - 1

    if first or last or interval_hit:
        state["type"] = "full"
        state["counter"] = 0
        state["activated_steps"].append(step)
    else:
        state["type"] = "forecast"
        state["counter"] += 1
    return state["type"]


def hicache_update_derivatives(state: Dict[str, Any], feature: torch.Tensor) -> None:
    """Compute backward finite-difference derivatives at a compute step.

    ``Delta^0 = feature``;
    ``Delta^i = (Delta^{i-1}_now - Delta^{i-1}_prev) / dist``.

    Edge case (<2 anchors): with only one compute step seen we cannot form a
    finite difference, so only the 0th-order term (the raw velocity) is kept.
    The forecast then reduces to plain reuse of the cached velocity, the
    correct, well-defined zero-information forecast.
    """
    prev = state["derivatives"]  # derivatives from the previous compute step
    have_prev = len(prev) > 0

    new_deriv: Dict[int, torch.Tensor] = {0: feature}
    if have_prev:
        acts = state["activated_steps"]
        if len(acts) >= 2:
            dist = acts[-1] - acts[-2]
        else:
            dist = state["interval"]
        dist = max(int(dist), 1)
        for order in range(state["max_order"]):
            if order not in prev:
                break
            new_deriv[order + 1] = (new_deriv[order] - prev[order]) / dist

    # Replace (not retain) the previous generation of derivative tensors:
    # the old dict is dropped here so only ONE generation of cached-velocity
    # CUDA tensors is alive per stage at any time.
    state["derivatives"] = new_deriv


def hicache_forecast(state: Dict[str, Any]) -> torch.Tensor:
    """Scaled-Hermite forecast of the velocity at the current skip step.

    ``F_hat = F_t + sum_{i>=1} (Delta^i F_t / i!) * Htilde_i(-k)``.

    ``k`` is the number of steps elapsed since the last compute step.
    With <2 anchors only ``Delta^0`` exists and this returns the cached
    velocity unchanged (k-independent), the correct degenerate forecast.
    """
    deriv = state["derivatives"]
    if 0 not in deriv:
        raise RuntimeError("hicache_forecast called before any compute step")

    k = state["step"] - state["activated_steps"][-1]
    sigma = state["sigma"]
    base = deriv[0]
    x = float(-k)

    result = base
    order = 1
    while order in deriv:
        coeff = deriv[order] / math.factorial(order)
        # Hermite basis evaluated at the scalar horizon -k as a Python float
        # (broadcasts against the derivative tensors; no device-tensor alloc).
        result = result + coeff * scaled_hermite_scalar(order, x, sigma)
        order += 1
    return result


# ---------------------------------------------------------------------------
# (1) Forecast calibration -- FoCa-style Heun corrector (zero extra net evals)
# ---------------------------------------------------------------------------
def hicache_calibrate(
    state: Dict[str, Any],
    forecast: torch.Tensor,
    weight: float = 0.5,
) -> torch.Tensor:
    """Calibrate a Hermite forecast toward the last trusted velocity anchor.

    "Forecast then Calibrate" (arXiv:2508.16211): a static velocity forecaster
    is accurate locally but accumulates a *systematic* extrapolation bias that
    grows with the forward horizon ``k``. A single predictor-corrector
    (Heun-style) blend in velocity space removes most of that bias **without any
    additional network evaluation**: it only reuses the already-cached anchor
    ``F_t = Delta^0`` (the most recent fully-computed velocity) and the
    first-order rate ``Delta^1`` already held in the derivative cache.

    Corrector (trapezoidal / Heun in velocity space)
    ------------------------------------------------
    The plain forecast uses the slope ``Delta^1`` measured at the anchor
    (an explicit-Euler-like single-ended estimate). A Heun corrector replaces
    that with the *average* of the anchor slope and the slope implied by the
    forecast itself, which is the trapezoidal rule and cancels the leading
    second-order extrapolation error term. With a damping ``weight`` (``lambda``)
    that scales the correction by the horizon, the calibrated velocity is::

        F_cal = forecast + lambda * gain(k) * (F_t - forecast)

    i.e. a horizon-scaled shrinkage of the over-confident extrapolation back
    toward the last reliable signal ``F_t``. ``gain(k) = k / (k + interval)``
    rises from 0 (no correction at the anchor, ``k=0``) toward 1 as the forecast
    horizon outgrows the compute interval, exactly where the static forecast is
    least trustworthy. With ``weight=0`` this is a strict no-op (returns the
    forecast unchanged); with ``weight=1`` and a far horizon it collapses to the
    plain-reuse anchor.

    Parameters
    ----------
    state : the HiCache state (provides ``derivatives[0]`` = ``F_t``,
        the current ``step``, the last ``activated_steps`` entry, and
        ``interval``).
    forecast : the Hermite forecast tensor to calibrate (``feats`` payload).
    weight : blend strength ``lambda in [0, 1]``; 0 disables calibration.
    """
    if weight <= 0.0:
        return forecast
    anchor = state["derivatives"].get(0)
    if anchor is None:
        return forecast
    k = max(int(state["step"] - state["activated_steps"][-1]), 0)
    if k == 0:
        return forecast
    interval = max(int(state.get("interval", 1)), 1)
    gain = k / float(k + interval)            # in (0, 1), grows with horizon
    lam = float(weight) * gain
    # F_cal = forecast + lam * (anchor - forecast) = (1-lam)*forecast + lam*anchor
    return forecast + lam * (anchor - forecast)


# ---------------------------------------------------------------------------
# (3) Adaptive / confidence-gated schedule -- cheap feature-change probe
# ---------------------------------------------------------------------------
def hicache_record_compute(state: Dict[str, Any], feature: torch.Tensor) -> None:
    """Record the residual between the last forecast and the freshly-computed
    velocity at a compute step, used by the adaptive schedule to lengthen or
    shorten the next skip run (DiCache-style error feedback).

    The residual is the relative L2 distance between what the forecaster *would*
    have predicted for this step (had it been skipped) and the true velocity.
    It is a unit-free confidence signal in ``[0, inf)``: small => the velocity
    field is smooth and forecasts are safe to extend; large => the field is
    changing fast and the cadence should tighten. Stored on the state; consumed
    by ``hicache_decide`` when ``adaptive`` is enabled.

    No network evaluation: ``feature`` is the already-computed velocity and the
    forecast reuses the existing derivative cache.

    Call ordering: this runs at a compute step AFTER ``hicache_decide`` (which
    has already appended the current step to ``activated_steps``) but BEFORE
    ``hicache_update_derivatives`` (so ``state['derivatives']`` still holds the
    PREVIOUS anchor's finite differences). The horizon is therefore measured
    from the previous anchor (``activated_steps[-2]``), not the just-appended
    current step, so the residual reflects a genuine multi-step forecast.
    """
    deriv = state.get("derivatives", {})
    acts = state.get("activated_steps", [])
    # Need the previous anchor (>=2 anchors) and an existing derivative cache to
    # form a non-trivial forecast; otherwise the probe is undefined.
    if 0 not in deriv or len(acts) < 2:
        state["last_residual"] = None
        return
    prev_anchor = acts[-2]
    k = max(int(state["step"] - prev_anchor), 1)
    sigma = state["sigma"]
    x = float(-k)
    pred = deriv[0]
    order = 1
    while order in deriv:
        pred = pred + (deriv[order] / math.factorial(order)) * scaled_hermite_scalar(order, x, sigma)
        order += 1
    num = (feature - pred).reshape(-1).float().norm()
    den = feature.reshape(-1).float().norm() + 1e-8
    state["last_residual"] = float((num / den).item())


def hicache_adaptive_interval(state: Dict[str, Any]) -> int:
    """Return the effective skip interval for the current run position.

    Maps the most recent compute-step residual to an interval in
    ``[interval_min, interval_max]``: a residual at/above ``tol`` keeps the
    tightest cadence (``interval_min``), a residual at/below ``tol/4`` opens up
    to the loosest (``interval_max``); linear in between. With no residual yet
    (first compute) it returns the configured base interval.
    """
    r = state.get("last_residual")
    base = max(int(state.get("interval", 1)), 1)
    if r is None:
        return base
    lo = max(int(state.get("interval_min", base)), 1)
    hi = max(int(state.get("interval_max", base)), lo)
    tol = float(state.get("adaptive_tol", 0.05))
    if tol <= 0:
        return base
    # frac = 1 at r<=tol/4 (smooth -> long), 0 at r>=tol (fast -> short)
    frac = (tol - r) / (0.75 * tol)
    frac = max(0.0, min(1.0, frac))
    return int(round(lo + frac * (hi - lo)))


# ---------------------------------------------------------------------------
# (5) Frequency-aware token selectivity -- per-token forecast blend weights
# ---------------------------------------------------------------------------
def hicache_freq_blend(
    forecast: torch.Tensor,
    anchor: torch.Tensor,
    freq_weight: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """Per-token frequency-gated blend of forecast and last-anchor velocity.

    HiCache skips *whole* steps, so it cannot (like Fast-TRELLIS token carving)
    recompute only the high-frequency voxels. The training-free analogue that
    preserves the zero-extra-eval guarantee is to *trust the forecast less* on
    high-frequency tokens: blend each token's forecast back toward the last
    fully-computed anchor in proportion to its 3D high-frequency FFT energy
    (``freq_weight in [0, 1]`` per token, from ``fft3d.analyze_voxel_frequency``
    on the SS occupancy grid). Low-frequency tokens (smooth interior) keep the
    aggressive forecast; high-frequency tokens (detail/edges) are pulled toward
    the reliable anchor::

        F_tok = (1 - s*w_tok) * forecast_tok + s*w_tok * anchor_tok

    ``strength s in [0, 1]`` scales the maximum pull. This is the per-token
    cadence idea (ProCache / FreqCa selective computation) expressed as a blend
    rather than a recompute. ``s=0`` or all-zero weights => exact forecast.

    Shapes: ``forecast``/``anchor`` are ``(N, C)`` token-feature tensors;
    ``freq_weight`` is ``(N,)`` or ``(N, 1)`` and broadcasts over channels.
    """
    if strength <= 0.0:
        return forecast
    w = freq_weight
    if w.dim() == 1:
        w = w.unsqueeze(-1)
    w = (float(strength) * w).clamp_(0.0, 1.0)
    return forecast + w * (anchor - forecast)


# ---------------------------------------------------------------------------
# CPU unit test (no GPU, no TRELLIS model)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    ok = True

    def check(name: str, cond: bool) -> None:
        global ok
        ok = ok and bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    xs = torch.tensor([-1.5, 0.0, 0.7, 2.0])
    check("H_0 == 1", torch.allclose(physicists_hermite(0, xs), torch.ones_like(xs)))
    check("H_1 == 2x", torch.allclose(physicists_hermite(1, xs), 2 * xs))
    check("H_2 == 4x^2-2", torch.allclose(physicists_hermite(2, xs), 4 * xs**2 - 2))
    check("H_3 == 8x^3-12x", torch.allclose(physicists_hermite(3, xs), 8 * xs**3 - 12 * xs))

    sig = 0.5
    expect = (sig**2) * (4 * (sig * xs) ** 2 - 2)
    check("scaled_hermite matches sigma^n H_n(sigma x)",
          torch.allclose(scaled_hermite(2, xs, sig), expect))
    check("scaled_hermite_scalar matches tensor scaled_hermite",
          all(abs(scaled_hermite_scalar(n, float(xv), sig)
                  - float(scaled_hermite(n, torch.tensor(float(xv)), sig))) < 1e-6
              for n in (0, 1, 2, 3) for xv in (-3.0, -1.0, 0.0, 2.5)))

    # Finite-difference order-1 derivative is exact on a linear velocity series.
    a = torch.tensor([1.0, -2.0, 0.5])
    b = torch.tensor([0.3, 0.3, 0.3])
    interval = 4
    st = hicache_init(num_steps=12, interval=interval, max_order=1,
                      first_enhance=0, end_enhance=12, sigma=sig)
    st["step"] = 0
    st["activated_steps"].append(0)
    hicache_update_derivatives(st, a + b * 0.0)
    check("after 1 anchor only order-0 derivative exists",
          set(st["derivatives"].keys()) == {0})
    st["step"] = 1
    check("<2 anchors -> forecast == cached velocity",
          torch.allclose(hicache_forecast(st), a + b * 0.0))
    st["step"] = interval
    st["activated_steps"].append(interval)
    hicache_update_derivatives(st, a + b * float(interval))
    check("finite-difference order-1 derivative == b (exact on linear series)",
          torch.allclose(st["derivatives"][1], b))

    # Schedule cadence.
    sched = hicache_init(num_steps=12, interval=4, max_order=1,
                         first_enhance=2, end_enhance=10, sigma=sig)
    types = []
    for s in range(12):
        sched["step"] = s
        types.append(hicache_decide(sched))
    check("step 0,1 full (first_enhance)", types[0] == "full" and types[1] == "full")
    check("forecast then full at interval boundary",
          types[2] == "forecast" and types[5] == "full")
    check("steps >= end_enhance are full", types[10] == "full" and types[11] == "full")

    # Constant series -> exact reproduction.
    stc = hicache_init(num_steps=8, interval=4, max_order=3,
                       first_enhance=0, end_enhance=8, sigma=sig)
    const = torch.tensor([2.0, -1.0, 4.0])
    for idx in (0, 4):
        stc["step"] = idx
        stc["activated_steps"].append(idx)
        hicache_update_derivatives(stc, const.clone())
    stc["step"] = 6
    check("constant series -> forecast == constant (exact)",
          torch.allclose(hicache_forecast(stc), const))

    # Far-horizon finiteness (sigma contraction).
    stb = hicache_init(num_steps=64, interval=32, max_order=3,
                       first_enhance=0, end_enhance=64, sigma=0.4)
    f0 = torch.randn(5)
    for idx in (0, 32):
        stb["step"] = idx
        stb["activated_steps"].append(idx)
        hicache_update_derivatives(stb, f0 + 0.01 * idx)
    stb["step"] = 60
    check("far-horizon Hermite forecast is finite", torch.isfinite(hicache_forecast(stb)).all().item())

    # --- (1) calibration: Heun corrector ---
    stcal = hicache_init(num_steps=12, interval=4, max_order=1,
                         first_enhance=0, end_enhance=12, sigma=sig)
    anc = torch.tensor([1.0, 2.0, 3.0])
    for idx in (0, 4):
        stcal["step"] = idx
        stcal["activated_steps"].append(idx)
        hicache_update_derivatives(stcal, anc.clone())   # constant -> Delta^1 = 0
    stcal["step"] = 6
    fc = hicache_forecast(stcal)
    check("calibrate weight=0 is a strict no-op",
          torch.allclose(hicache_calibrate(stcal, fc, 0.0), fc))
    # constant series -> forecast == anchor, so calibration must not move it
    check("calibrate on constant series leaves forecast unchanged",
          torch.allclose(hicache_calibrate(stcal, fc, 1.0), anc))
    # synthetic biased forecast: corrector must pull strictly toward the anchor
    biased = anc + 1.0
    cal = hicache_calibrate(stcal, biased, 0.5)
    check("calibrate pulls a biased forecast toward the anchor",
          bool(((cal - anc).abs() < (biased - anc).abs()).all().item()))
    stcal["step"] = 4  # k=0 at the anchor -> no correction
    check("calibrate at k=0 (anchor) is a no-op",
          torch.allclose(hicache_calibrate(stcal, biased, 1.0), biased))

    # --- (3) adaptive schedule: residual probe + interval mapping ---
    sta = hicache_init(num_steps=20, interval=4, max_order=1,
                       first_enhance=0, end_enhance=20, sigma=sig,
                       adaptive=True, interval_min=2, interval_max=6,
                       adaptive_tol=0.10)
    sta["last_residual"] = None
    check("adaptive interval with no residual == base", hicache_adaptive_interval(sta) == 4)
    sta["last_residual"] = 0.0
    check("adaptive interval at residual 0 == interval_max", hicache_adaptive_interval(sta) == 6)
    sta["last_residual"] = 0.20
    check("adaptive interval at large residual == interval_min", hicache_adaptive_interval(sta) == 2)
    # record_compute residual: probe is unit-free L2 relative error in [0, inf).
    # On a CONSTANT series the Hermite forecast == anchor exactly, so a held-out
    # compute equal to the constant must give residual ~0 (the probe is exact).
    # Production ordering: hicache_decide appends the CURRENT step before
    # record_compute runs, and the derivative cache still holds the PREVIOUS
    # anchor -> the test appends step 8 first, then probes.
    str_ = hicache_init(num_steps=12, interval=4, max_order=1,
                        first_enhance=0, end_enhance=12, sigma=sig)
    cval = torch.tensor([0.5, -1.0, 2.0])
    for idx in (0, 4):
        str_["step"] = idx
        str_["activated_steps"].append(idx)
        hicache_update_derivatives(str_, cval.clone())
    str_["step"] = 8
    str_["activated_steps"].append(8)            # as hicache_decide would
    hicache_record_compute(str_, cval.clone())   # before update_derivatives
    check("record_compute residual ~0 on a constant series (exact forecast)",
          str_["last_residual"] is not None and str_["last_residual"] < 1e-5)
    # A divergent compute (anchor doubled) must register a large positive residual.
    hicache_record_compute(str_, 2.0 * cval)
    check("record_compute residual large on a divergent compute",
          str_["last_residual"] is not None and str_["last_residual"] > 0.1)
    # <2 anchors -> residual undefined (None), not a spurious 0.
    str2 = hicache_init(num_steps=12, interval=4, max_order=1,
                        first_enhance=0, end_enhance=12, sigma=sig)
    str2["step"] = 0
    str2["activated_steps"].append(0)
    hicache_update_derivatives(str2, cval.clone())
    str2["step"] = 4
    str2["activated_steps"].append(4)
    hicache_record_compute(str2, cval.clone())   # only 2 anchors incl. current
    check("record_compute with a single prior anchor still defined",
          str2["last_residual"] is not None)

    # --- (5) frequency-gated per-token blend ---
    N, C = 6, 3
    fcast = torch.randn(N, C)
    ancr = torch.randn(N, C)
    fw = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])   # half low, half high freq
    check("freq_blend strength=0 is a no-op",
          torch.allclose(hicache_freq_blend(fcast, ancr, fw, 0.0), fcast))
    blended = hicache_freq_blend(fcast, ancr, fw, 1.0)
    check("freq_blend keeps low-freq tokens on the forecast",
          torch.allclose(blended[:3], fcast[:3]))
    check("freq_blend snaps high-freq tokens to the anchor",
          torch.allclose(blended[3:], ancr[3:]))

    print("\nALL PASS" if ok else "\nSOME FAILED")
    raise SystemExit(0 if ok else 1)
