"""Adaptive-CFG: training-free CFG-pass skipping for TRELLIS.2.

Implements *Adaptive Guidance* (Castillo et al., arXiv:2312.12487) on top of
the TRELLIS.2 ``FlowEulerGuidanceIntervalSampler`` CFG path, with the same
caching substrate as Fast-TRELLIS: the **final velocity** is the cached /
forecast quantity.

The paper, exactly
------------------
Classifier-free guidance combines two model evaluations per step::

    v_cfg = guidance_strength * v_cond + (1 - guidance_strength) * v_uncond

(TRELLIS.2's convention; equivalent to ``(1+w) v_cond - w v_uncond`` with
``w = guidance_strength - 1``). The unconditional pass doubles the cost.

Adaptive Guidance observes that the conditional and unconditional predictions
become increasingly *aligned* as sampling proceeds, measured by the cosine
similarity (the paper's gamma_t)::

    gamma_t = <v_cond, v_uncond> / (||v_cond|| * ||v_uncond||)

with ``lim_{t->0} gamma_t = 1``. Once ``gamma_t >= gamma_bar`` the
unconditional pass carries no new directional information, so AG drops it.

Caching substrate (Fast-TRELLIS-compatible)
-------------------------------------------
Plain AG discards guidance entirely on skip steps (``v_cfg -> v_cond``). We
keep a faithful reconstruction: the CFG *guidance term*

    g_t = v_cfg - v_cond = (guidance_strength - 1) * (v_cond - v_uncond)

is a smooth function of the diffusion step. At a *compute* step we evaluate
both passes and cache ``g`` (anchored at the step index). At a *skip* step we
do only the conditional pass and **forecast** ``g_t`` from the cached anchors
(Newton divided-difference extrapolation, exact for polynomial guidance),
then return ``v_cond + g_forecast``. With the 0th-order forecast this reduces
to plain AG.

This module is pure numerics; the sampler in ``flow_euler.py`` owns the model
calls and the SparseTensor glue.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# tensor / SparseTensor helpers                                               #
# --------------------------------------------------------------------------- #
def is_sparse(x: Any) -> bool:
    return hasattr(x, "feats") and hasattr(x, "replace")


def payload(x: Any) -> torch.Tensor:
    return x.feats if is_sparse(x) else x


def with_payload(template: Any, feats: torch.Tensor) -> Any:
    """Rebuild an object of the same kind as ``template`` carrying ``feats``."""
    if is_sparse(template):
        return template.replace(feats)
    return feats


def cosine_sim(a: Any, b: Any, eps: float = 1e-12) -> float:
    """gamma_t: cosine similarity of two (possibly sparse) predictions."""
    fa = payload(a).reshape(-1).float()
    fb = payload(b).reshape(-1).float()
    num = torch.dot(fa, fb)
    den = fa.norm() * fb.norm() + eps
    return float((num / den).item())


# --------------------------------------------------------------------------- #
# guidance forecast                                                           #
# --------------------------------------------------------------------------- #
def forecast_guidance(
    anchors: List[Tuple[int, torch.Tensor]],
    step: int,
    max_order: int = 1,
) -> torch.Tensor:
    """Forecast the guidance term ``g`` at integer ``step`` from cached anchors.

    Newton divided-difference extrapolator through the most recent
    ``max_order + 1`` anchors -- the unique polynomial of degree
    ``<= max_order`` interpolating those anchors, hence exact whenever the
    true guidance series is polynomial of that degree.

    Edge cases (explicit, no silent fallback):
      * 0 anchors -> ValueError (caller must guarantee >= 1).
      * 1 anchor  -> 0th-order hold (plain-AG behaviour).
      * >= 2 anchors with max_order >= 1 -> divided-difference extrapolation.
    """
    if len(anchors) == 0:
        raise ValueError("forecast_guidance requires at least one anchor")

    if len(anchors) == 1 or max_order < 1:
        return anchors[-1][1].clone()

    used = anchors[-(max_order + 1):]
    xs = [float(s) for s, _ in used]
    ys = [g.clone() for _, g in used]
    n = len(used)

    coeffs = [ys[0]]
    col = ys
    for k in range(1, n):
        new_col = []
        for i in range(n - k):
            denom = xs[i + k] - xs[i]
            new_col.append((col[i + 1] - col[i]) / denom)
        col = new_col
        coeffs.append(col[0])

    x = float(step)
    result = coeffs[-1].clone()
    for k in range(n - 2, -1, -1):
        result = result * (x - xs[k]) + coeffs[k]
    return result


def forecast_guidance_hermite(
    anchors: List[Tuple[int, torch.Tensor]],
    step: int,
    max_order: int = 1,
    sigma: float = 0.5,
    calibrate_weight: float = 0.0,
) -> torch.Tensor:
    """Scaled-Hermite forecast of the guidance term ``g`` (the (6) upgrade).

    Plain ``forecast_guidance`` uses an *undamped* Newton divided-difference
    polynomial, which can overshoot when the anchors are far apart. This variant
    reuses HiCache's dual-scaled physicist's-Hermite basis on the backward
    finite differences of ``g`` (evenly spaced anchors assumed, spacing taken
    from the two most recent anchor step indices), which contracts the
    high-order terms and stays in the stable oscillatory regime. With
    ``calibrate_weight > 0`` it then applies the same FoCa-style shrinkage toward
    the most recent anchor used by the velocity path.

    Falls back to the plain Newton extrapolator for <2 anchors / order<1, so the
    degenerate cases match ``forecast_guidance`` exactly.
    """
    if len(anchors) == 0:
        raise ValueError("forecast_guidance_hermite requires at least one anchor")
    if len(anchors) == 1 or max_order < 1:
        return anchors[-1][1].clone()

    try:                       # package context (production)
        from .hicache import scaled_hermite_scalar
    except ImportError:        # standalone CPU unit-test context
        from hicache import scaled_hermite_scalar

    used = anchors[-(max_order + 1):]
    steps_ = [int(s) for s, _ in used]
    gs = [g for _, g in used]
    last_step = steps_[-1]
    dist = max(int(steps_[-1] - steps_[-2]), 1)

    # Backward finite differences of g at the most recent anchor (Delta^i).
    # Delta^0 = g_last ; Delta^i = (Delta^{i-1}[j] - Delta^{i-1}[j-1]) / dist.
    diffs = [gs[-1].clone()]
    col = [g.clone() for g in gs]
    order = 1
    while len(col) >= 2 and order <= max_order:
        col = [(col[j] - col[j - 1]) / dist for j in range(1, len(col))]
        diffs.append(col[-1].clone())
        order += 1

    k = step - last_step          # forward horizon from the last anchor
    x = float(k)
    result = diffs[0]
    for i in range(1, len(diffs)):
        result = result + (diffs[i] / math.factorial(i)) * scaled_hermite_scalar(i, x, sigma)

    if calibrate_weight > 0.0 and k > 0:
        gain = k / float(k + dist)
        lam = float(calibrate_weight) * gain
        result = result + lam * (diffs[0] - result)   # shrink toward last anchor
    return result


# --------------------------------------------------------------------------- #
# state + decision                                                            #
# --------------------------------------------------------------------------- #
def adaptive_cfg_init(
    num_steps: int,
    gamma_bar: float = 0.94,
    warmup: int = 2,
    max_order: int = 1,
    reuse_guidance: bool = True,
    hermite: bool = False,
    hermite_sigma: float = 0.5,
    hermite_calibrate_weight: float = 0.0,
    gamma_bar_schedule: bool = False,
) -> Dict[str, Any]:
    """Create per-``sample()`` state for one diffusion trajectory.

    ``hermite`` (the (6) upgrade): forecast the guidance term with the
    dual-scaled Hermite extrapolator (optionally FoCa-calibrated) instead of the
    undamped Newton polynomial.
    ``gamma_bar_schedule``: relax the alignment threshold linearly across the
    trajectory (stricter early, looser late, mirroring gamma_t -> 1 as t -> 0),
    so more uncond passes are skipped in the back half where they matter least.
    """
    if not (0.0 <= gamma_bar <= 1.0):
        raise ValueError(f"gamma_bar must be in [0,1], got {gamma_bar}")
    return {
        "num_steps": int(num_steps),
        "gamma_bar": float(gamma_bar),
        "warmup": int(warmup),
        "max_order": int(max_order),
        "reuse_guidance": bool(reuse_guidance),
        "hermite": bool(hermite),
        "hermite_sigma": float(hermite_sigma),
        "hermite_calibrate_weight": float(hermite_calibrate_weight),
        "gamma_bar_schedule": bool(gamma_bar_schedule),
        "step": 0,
        "anchors": [],          # list[(step, g)] cached guidance terms
        "last_gamma": None,
        "n_full": 0,
        "n_skip": 0,
    }


def adaptive_cfg_decide(state: Dict[str, Any], gamma: Optional[float]) -> bool:
    """Return True if the current CFG step must run the full (uncond) pass.

    A skip requires ALL of: past warmup, at least one cached anchor, not the
    final step, and the last measured cosine similarity above threshold.
    """
    step = state["step"]
    if step < state["warmup"]:
        return True
    if step >= state["num_steps"] - 1:          # always anchor the final step
        return True
    if len(state["anchors"]) == 0:
        return True
    if gamma is None:
        return True
    thresh = state["gamma_bar"]
    if state.get("gamma_bar_schedule"):
        # Relax the threshold across the run: stricter early (keep more uncond
        # passes while structure forms), looser late. Linear from gamma_bar at
        # the start to a slightly lower floor by the end.
        frac = step / max(state["num_steps"] - 1, 1)
        thresh = state["gamma_bar"] - 0.06 * frac
    return gamma < thresh


# --------------------------------------------------------------------------- #
