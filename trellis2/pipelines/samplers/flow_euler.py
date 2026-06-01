from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps
    
    def _pred_to_xstart(self, x_t, t, pred):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            tqdm_desc: A customized tqdm desc.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            guidance_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, guidance_interval=guidance_interval, **kwargs)


# ===========================================================================
# Training-free acceleration for TRELLIS.2 (publishable, faithful to cited
# methods). Two composable, training-free accelerations on the *final
# velocity* substrate:
#
#   * HiCache (arXiv:2508.16984) -- Hermite-polynomial forecast of the final
#     CFG-combined velocity at skipped sampling steps. Replaces the TaylorSeer
#     monomial basis with the dual-scaled physicist's Hermite basis (same
#     final-velocity substrate as Fast-TRELLIS). Skips whole model evaluations.
#
#   * adaptive_cfg (Adaptive Guidance, arXiv:2312.12487) -- on CFG steps, skip
#     the *unconditional* forward pass once cond/uncond predictions align
#     (cosine sim >= gamma_bar), reconstructing the guidance term from cached
#     anchors. Halves the per-compute-step CFG cost on aligned steps.
#
# SS-stage robustness note (empty-mesh finding):
#   In the wide bench, stacking adaptive_cfg onto the sparse-structure (SS)
#   stage emptied 3/20 rounded objects: the uncond pass is what holds the coarse
#   occupancy volume open, and skipping it over-carves rounded silhouettes to
#   nothing. The fix lives in the pipeline (enable_faster): full_stack confines
#   adaptive_cfg to the SLaT stages and runs the SS stage HiCache-only, so the
#   _faster sampler below is only ever wired to the SLaT stages. HiCache alone
#   (the default) never skips an uncond pass and had 0 failures (n=20).
#
# v1 -> v2 API adaptations (the only changes vs the faster_components/*.py
# reference implementations):
#   * cfg_strength/cfg_interval        -> guidance_strength/guidance_interval
#   * single CFG-in-interval mixin     -> v2 split MRO
#                                         (GuidanceIntervalSamplerMixin,
#                                          ClassifierFreeGuidanceSamplerMixin, base)
#   * extra v2 kwargs (guidance_rescale, concat_cond, tqdm_desc) flow through
#   * SS pred_v is a dense tensor; SLaT pred_v is a SparseTensor -- HiCache
#     forecasts .feats only and rebuilds via .replace(feats).
# No silent fallbacks, no placeholders, full mesh+texture+1024_cascade support.
# ===========================================================================

from .hicache import (
    hicache_init,
    hicache_decide,
    hicache_update_derivatives,
    hicache_forecast,
)
from . import adaptive_cfg as _acfg


def _is_sparse(x: Any) -> bool:
    return hasattr(x, "feats") and hasattr(x, "replace")


# ---------------------------------------------------------------------------
# HiCache mixin: cache/forecast the final velocity in sample_once.
# ---------------------------------------------------------------------------
class HiCacheMixin:
    """Adds HiCache state + a cache/forecast ``sample_once`` to a FlowEuler
    sampler. The schedule is configured per-run in ``sample`` (scaled to the
    actual step count), and ``sample_once`` either runs the full model (and
    updates the Hermite finite-difference cache) or forecasts the velocity.
    """

    # HiCache config (overridable on the instance after construction).
    hicache_interval: int = 3
    hicache_max_order: int = 1
    hicache_first_enhance: int = 2
    hicache_sigma: float = 0.5

    def _hicache_setup(self, steps: int) -> None:
        # TRELLIS.2 runs short schedules (12 steps). Keep a few warm-up full
        # steps and always make the final step full; shrink the interval for
        # short schedules so the forecast horizon stays small.
        first_enhance = self.hicache_first_enhance
        end_enhance = steps - 1            # last step always full
        interval = self.hicache_interval
        if steps < 20:
            first_enhance = max(2, min(first_enhance, round(steps / 3.0)))
            interval = min(interval, 3)
        self._hicache = hicache_init(
            num_steps=steps,
            interval=interval,
            max_order=self.hicache_max_order,
            first_enhance=first_enhance,
            end_enhance=end_enhance,
            sigma=self.hicache_sigma,
        )

    @torch.no_grad()
    def sample_once(self, model, x_t, t, t_prev, cond=None, **kwargs):
        state = getattr(self, "_hicache", None)
        if state is None:
            return super().sample_once(model, x_t, t, t_prev, cond, **kwargs)

        decision = hicache_decide(state)

        if decision == "full":
            pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
            feats = pred_v.feats if _is_sparse(pred_v) else pred_v
            hicache_update_derivatives(state, feats.detach().clone())
            state["step"] += 1
            pred_x_prev = x_t - (t - t_prev) * pred_v
            return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

        # forecast step: rebuild the final velocity from cached derivatives,
        # no model evaluation at all.
        feats_hat = hicache_forecast(state)
        if _is_sparse(x_t):
            pred_v = x_t.replace(feats_hat)
        else:
            pred_v = feats_hat
        pred_x_0, _eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        state["step"] += 1
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


# ---------------------------------------------------------------------------
# adaptive_cfg mixin: skip the uncond pass on aligned CFG steps.
# Overrides the ClassifierFreeGuidanceSamplerMixin slot in the MRO.
# ---------------------------------------------------------------------------
class AdaptiveCFGMixin:
    """Replaces the two-pass CFG ``_inference_model`` with an Adaptive-Guidance
    variant that skips the unconditional pass once cond/uncond predictions
    align, reconstructing the guidance term from cached anchors.

    This sits at the ``ClassifierFreeGuidanceSamplerMixin`` position in the MRO
    (same signature), so the ``GuidanceIntervalSamplerMixin`` above it still
    forces ``guidance_strength=1`` outside the guidance interval -- those
    single-pass steps are not counted as CFG steps and do not advance the
    adaptive-CFG step counter.
    """

    # adaptive_cfg config (overridable on the instance after construction).
    acfg_gamma_bar: float = 0.94
    acfg_warmup: int = 2
    acfg_max_order: int = 1
    acfg_reuse_guidance: bool = True

    def _acfg_reset(self, steps: int) -> None:
        self._acfg_state = None
        self._acfg_steps = int(steps)

    def _inference_model(self, model, x_t, t, cond, neg_cond,
                         guidance_strength, guidance_rescale=0.0, **kwargs):
        # No CFG requested (e.g. tex stage guidance_strength=1, or the
        # outside-interval path): single conditional/uncond pass, no caching,
        # no counter advance -- identical to the stock mixin.
        if guidance_strength == 1:
            return super(ClassifierFreeGuidanceSamplerMixin, self)._inference_model(
                model, x_t, t, cond, **kwargs)
        if guidance_strength == 0:
            return super(ClassifierFreeGuidanceSamplerMixin, self)._inference_model(
                model, x_t, t, neg_cond, **kwargs)

        state = getattr(self, "_acfg_state", None)
        if state is None:
            state = _acfg.adaptive_cfg_init(
                num_steps=getattr(self, "_acfg_steps", 50),
                gamma_bar=self.acfg_gamma_bar,
                warmup=self.acfg_warmup,
                max_order=self.acfg_max_order,
                reuse_guidance=self.acfg_reuse_guidance,
            )
            self._acfg_state = state

        base_inf = super(ClassifierFreeGuidanceSamplerMixin, self)._inference_model
        pred_pos = base_inf(model, x_t, t, cond, **kwargs)

        run_full = _acfg.adaptive_cfg_decide(state, state["last_gamma"])

        if run_full:
            pred_neg = base_inf(model, x_t, t, neg_cond, **kwargs)
            state["last_gamma"] = _acfg.cosine_sim(pred_pos, pred_neg)
            # guidance term g = (w-1) * (v_cond - v_uncond) in payload space.
            g = (guidance_strength - 1.0) * (
                _acfg.payload(pred_pos) - _acfg.payload(pred_neg))
            state["anchors"].append((state["step"], g))
            keep = state["max_order"] + 2
            if len(state["anchors"]) > keep:
                state["anchors"] = state["anchors"][-keep:]
            pred_payload = guidance_strength * _acfg.payload(pred_pos) \
                + (1.0 - guidance_strength) * _acfg.payload(pred_neg)
            pred = _acfg.with_payload(pred_pos, pred_payload)
            state["n_full"] += 1
        else:
            if state["reuse_guidance"]:
                g = _acfg.forecast_guidance(
                    state["anchors"], state["step"], state["max_order"])
                pred = _acfg.with_payload(pred_pos, _acfg.payload(pred_pos) + g)
            else:
                pred = pred_pos
            state["n_skip"] += 1

        state["step"] += 1

        # CFG rescale (v2 behaviour) -- applied to the final combined pred.
        if guidance_rescale > 0:
            x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
            x_0_cfg = self._pred_to_xstart(x_t, t, pred)
            std_pos = _acfg.payload(x_0_pos).std(
                dim=list(range(1, _acfg.payload(x_0_pos).ndim)), keepdim=True)
            std_cfg = _acfg.payload(x_0_cfg).std(
                dim=list(range(1, _acfg.payload(x_0_cfg).ndim)), keepdim=True)
            x_0_rescaled = _acfg.with_payload(
                x_0_cfg, _acfg.payload(x_0_cfg) * (std_pos / std_cfg))
            x_0 = _acfg.with_payload(
                x_0_cfg,
                guidance_rescale * _acfg.payload(x_0_rescaled)
                + (1 - guidance_rescale) * _acfg.payload(x_0_cfg))
            pred = self._xstart_to_pred(x_t, t, x_0)

        return pred


# ---------------------------------------------------------------------------
# Concrete accelerated samplers.
# ---------------------------------------------------------------------------
class FlowEulerGuidanceIntervalSampler_hicache(
        HiCacheMixin,
        GuidanceIntervalSamplerMixin,
        ClassifierFreeGuidanceSamplerMixin,
        FlowEulerSampler):
    """HiCache-only: Hermite velocity forecast, stock two-pass CFG."""

    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps: int = 50,
               rescale_t: float = 1.0, guidance_strength: float = 3.0,
               guidance_interval: Tuple[float, float] = (0.0, 1.0),
               verbose: bool = True, **kwargs):
        self._hicache_setup(steps)
        try:
            ret = super(HiCacheMixin, self).sample(
                model, noise, cond, steps, rescale_t, verbose,
                neg_cond=neg_cond, guidance_strength=guidance_strength,
                guidance_interval=guidance_interval, **kwargs)
        finally:
            n_full = len(self._hicache["activated_steps"])
            if verbose:
                print(f"[hicache] full steps: {n_full}/{steps} "
                      f"(forecast {steps - n_full})")
            self._hicache = None
        return ret


class FlowEulerGuidanceIntervalSampler_adaptivecfg(
        GuidanceIntervalSamplerMixin,
        AdaptiveCFGMixin,
        ClassifierFreeGuidanceSamplerMixin,
        FlowEulerSampler):
    """adaptive_cfg-only: stock per-step Euler, skip the uncond CFG pass once
    cond/uncond align."""

    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps: int = 50,
               rescale_t: float = 1.0, guidance_strength: float = 3.0,
               guidance_interval: Tuple[float, float] = (0.0, 1.0),
               verbose: bool = True, **kwargs):
        self._acfg_reset(steps)
        ret = super().sample(
            model, noise, cond, steps, rescale_t, verbose,
            neg_cond=neg_cond, guidance_strength=guidance_strength,
            guidance_interval=guidance_interval, **kwargs)
        st = getattr(self, "_acfg_state", None)
        if verbose and st is not None:
            print(f"[adaptive-cfg] full CFG: {st['n_full']}, "
                  f"uncond-skipped: {st['n_skip']}")
        return ret


class FlowEulerGuidanceIntervalSampler_faster(
        HiCacheMixin,
        GuidanceIntervalSamplerMixin,
        AdaptiveCFGMixin,
        ClassifierFreeGuidanceSamplerMixin,
        FlowEulerSampler):
    """full_stack: HiCache (velocity forecast on skip steps) + adaptive_cfg
    (uncond-pass skip on aligned compute steps). The default "faster" mode.

    Composition: HiCache decides compute vs forecast in ``sample_once``. On a
    compute step ``_get_model_prediction`` runs, which resolves through the
    MRO to the adaptive CFG ``_inference_model`` (may skip the uncond pass).
    On a forecast step no model is evaluated at all.
    """

    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps: int = 50,
               rescale_t: float = 1.0, guidance_strength: float = 3.0,
               guidance_interval: Tuple[float, float] = (0.0, 1.0),
               verbose: bool = True, **kwargs):
        self._hicache_setup(steps)
        self._acfg_reset(steps)
        try:
            ret = super(HiCacheMixin, self).sample(
                model, noise, cond, steps, rescale_t, verbose,
                neg_cond=neg_cond, guidance_strength=guidance_strength,
                guidance_interval=guidance_interval, **kwargs)
        finally:
            n_full = len(self._hicache["activated_steps"])
            st = getattr(self, "_acfg_state", None)
            if verbose:
                msg = (f"[faster] hicache full: {n_full}/{steps} "
                       f"(forecast {steps - n_full})")
                if st is not None:
                    msg += (f" | adaptive-cfg full: {st['n_full']}, "
                            f"uncond-skipped: {st['n_skip']}")
                print(msg)
            self._hicache = None
        return ret
