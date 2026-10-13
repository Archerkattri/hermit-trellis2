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

    F_hat_{t+k} = F_t + sum_{i=1}^{m} (Delta^i F_t / i!) * Htilde_i(k)

where ``Htilde`` is the *dual-scaled* physicist's Hermite polynomial with
contraction factor ``sigma in (0, 1)``::

    Htilde_n(x) = sigma^n * H_n(sigma * x)
    H_0(x) = 1,  H_1(x) = 2x
    H_{n+1}(x) = 2*x*H_n(x) - 2*n*H_{n-1}(x)

TaylorSeer is the special case where the basis ``Htilde_i(k)`` is replaced
by the monomial ``k^i``. The dual scaling (input scale ``sigma*x`` and
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


# Public compatibility surface for TRELLIS.2 integrations.  This checkout
# ships the Hermite (polynomial) basis only; DMD lives in the sibling
# hermit-trellis2-plus-plus repo.  The single-backend switch keeps lifecycle,
# reset, and telemetry semantics identical across the two checkouts.
HICACHE_BACKENDS = ("hermite",)


def normalize_backend(backend: str) -> str:
    """Return the supported forecast backend or raise a useful error."""
    value = "hermite" if backend is None else str(backend).lower().strip()
    if value not in HICACHE_BACKENDS:
        choices = ", ".join(repr(item) for item in HICACHE_BACKENDS)
        raise ValueError(f"HiCache backend must be one of {choices}, got {backend!r}")
    return value


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
    in ``hicache_forecast`` where ``k`` is a single integer horizon.
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
    backend: str = "hermite",
    stage: str = "unknown",
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
    backend : forecast basis; this checkout ships ``"hermite"`` only.
    stage : telemetry label for the model stage owning this cache.
    """
    if interval < 1:
        raise ValueError("interval must be >= 1")
    if max_order < 1:
        raise ValueError("max_order must be >= 1")
    if not (0.0 < sigma < 1.0):
        raise ValueError(f"sigma must be in (0, 1), got {sigma}")
    backend = normalize_backend(backend)
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
        "backend": backend,       # "hermite" (polynomial HiCache)
        "stage": str(stage),      # telemetry only; the carved SLaT path is separate
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

    ``F_hat = F_t + sum_{i>=1} (Delta^i F_t / i!) * Htilde_i(k)``.

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
    x = float(k)

    result = base
    order = 1
    while order in deriv:
        coeff = deriv[order] / math.factorial(order)
        # Hermite basis evaluated at the scalar horizon -k as a Python float
        # (broadcasts against the derivative tensors; no device-tensor alloc).
        result = result + coeff * scaled_hermite_scalar(order, x, sigma)
        order += 1
    return result


def hicache_forecast_state(state: Dict[str, Any]) -> torch.Tensor:
    """Forecast from the backend selected in ``state``.

    This is the compatibility switch used by the sampler.  Keeping dispatch
    here makes Hermite share lifecycle, reset, and telemetry semantics with
    the sibling DMD checkout.
    """
    normalize_backend(state.get("backend", "hermite"))
    return hicache_forecast(state)


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
    x = float(k)
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
