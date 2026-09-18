from typing import *
import copy
import json
import time
import uuid
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
# Training-free acceleration for TRELLIS.2. Two composable accelerations that
# act on the *final velocity* substrate:
#
#   * HiCache (arXiv:2508.16984) -- Hermite-polynomial forecast of the final
#     CFG-combined velocity at skipped sampling steps, using a dual-scaled
#     physicist's Hermite basis. Skips whole model evaluations.
#
#   * adaptive_cfg (Adaptive Guidance, arXiv:2312.12487) -- on CFG steps, skip
#     the *unconditional* forward pass once cond/uncond predictions align
#     (cosine sim >= gamma_bar), reconstructing the guidance term from cached
#     anchors. Halves the per-compute-step CFG cost on aligned steps.
#
# Sparse-structure vs SLaT stages:
#   adaptive_cfg is applied only to the SLaT stages; the sparse-structure (SS)
#   stage keeps the unconditional pass, which shapes the coarse occupancy
#   volume. The _faster sampler below is wired to the SLaT stages by the
#   pipeline (enable_faster), while the SS stage runs the HiCache-only sampler.
#   The default HiCache mode never skips an unconditional pass on any stage.
#
# Stage tensor types:
#   The SS stage pred_v is a dense tensor; the SLaT stages pred_v is a
#   SparseTensor, so HiCache forecasts .feats only and rebuilds the tensor via
#   .replace(feats). Full mesh + texture + 1024_cascade support throughout.
# ===========================================================================

from .hicache import (
    hicache_init,
    hicache_decide,
    hicache_update_derivatives,
    hicache_forecast,
    hicache_calibrate,
    hicache_record_compute,
    hicache_freq_blend,
    hicache_forecast_state,
)
from . import adaptive_cfg as _acfg
# Single sparse-tensor predicate, shared with AdaptiveCFGMixin (no duplicate).
from .adaptive_cfg import is_sparse as _is_sparse
from hicache_pp.budget import CacheBudget, CacheBudgetRuntime, RunIdentity, stable_digest


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
    # forecast basis: "hermite" only in this checkout (DMD lives in the
    # sibling hermit-trellis2-plus-plus repo).
    hicache_backend: str = "hermite"
    hicache_stage: str = "unknown"
    # Deployment envelope.  ``None`` keeps the historical schedule while the
    # runtime still emits an identity-bound manifest with the default envelope.
    hicache_budget = None
    hicache_max_horizon: Optional[int] = None
    hicache_max_memory_mb: Optional[float] = None
    hicache_audit_budget: int = 0

    # --- improvement toggles (all default-OFF so shipped behaviour is unchanged) ---
    # (2) higher-order forecast: hicache_max_order in {1,2,3}; when
    # hicache_order_sigma_retune is on, sigma is auto-lowered for order>=2 so the
    # higher Hermite terms stay in the stable contracted regime.
    hicache_order_sigma_retune: bool = False
    # (1) FoCa-style Heun calibration of the forecast.
    hicache_calibrate: bool = False
    hicache_calibrate_weight: float = 0.5
    # (3) confidence-gated adaptive skip schedule.
    hicache_adaptive_schedule: bool = False
    hicache_interval_min: int = 2
    hicache_interval_max: int = 5
    hicache_adaptive_tol: float = 0.05
    # (5) frequency-aware per-token forecast selectivity.
    hicache_freq_carve: bool = False
    hicache_freq_strength: float = 0.8
    # per-token high-frequency weight (N,) injected from the SS stage; None=off.
    _hicache_freq_weight = None

    def configure_hicache_budget(self, budget=None, *, max_horizon: Optional[int] = None,
                                 max_memory_mb: Optional[float] = None,
                                 audit_budget: int = 0):
        """Set the explicit quality/latency/memory envelope for future runs."""
        self.hicache_budget = budget
        self.hicache_max_horizon = max_horizon
        self.hicache_max_memory_mb = max_memory_mb
        self.hicache_audit_budget = audit_budget
        return self

    def set_freq_weight(self, freq_weight) -> None:
        """Inject the SS-stage per-token high-frequency weight (N,) in [0,1].

        Mirrors fast-trellis2's ``set_coords_scores``: the pipeline computes a
        3D-FFT high-frequency score per occupied voxel and hands it to the SLaT
        sampler, where ``hicache_freq_carve`` uses it to blend high-frequency
        tokens' forecasts back toward the last computed anchor. Auto-disabled
        (shape mismatch / None) on the cascade-upsample and texture stages where
        the token coordinates are re-derived -- exactly as the original did.
        """
        self._hicache_freq_weight = freq_weight

    def _release_accel_state(self) -> None:
        """Drop the HiCache derivative cache (CUDA tensors) and free memory.

        Cooperative: chains to any sibling acceleration mixin's
        ``_release_accel_state`` (e.g. AdaptiveCFGMixin in the full_stack
        sampler) before emptying the CUDA cache once. The state is rebuilt fresh
        by ``_hicache_setup`` at the start of every run, so this does not affect
        the output.
        """
        runtime = getattr(self, "_budget_runtime", None)
        if runtime is not None:
            self._last_hicache_manifest = runtime.manifest.as_dict()
        self._budget_runtime = None
        self._hicache = None
        self._hicache_freq_weight = None
        parent = getattr(super(), "_release_accel_state", None)
        if callable(parent):
            parent()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _hicache_setup(self, steps: int) -> None:
        # TRELLIS.2 runs short schedules (12 steps). Keep a few warm-up full
        # steps and always make the final step full; shrink the interval for
        # short schedules so the forecast horizon stays small.
        first_enhance = self.hicache_first_enhance
        end_enhance = steps - 1            # last step always full
        interval = self.hicache_interval
        imin = self.hicache_interval_min
        imax = self.hicache_interval_max
        if steps < 20:
            first_enhance = max(2, min(first_enhance, round(steps / 3.0)))
            # The <20-step interval clamp is the conservative default; when the
            # calibrate/adaptive toggles are on we trust longer forecast runs,
            # so the clamp only applies to the plain (uncalibrated, fixed) path.
            relaxed = self.hicache_calibrate or self.hicache_adaptive_schedule
            if not relaxed:
                interval = min(interval, 3)
            imax = min(imax, max(interval, 3) + 2)
            imin = max(min(imin, interval), 1)
        # (2) order<->sigma interaction: raising the forecast order pairs with a
        # lower contraction sigma so the high-order Hermite terms stay damped
        # (each extra order multiplies sigma by 0.7: order2 -> 0.35*base,
        # order3 -> ~0.245*base), floored at 0.15.
        sigma = self.hicache_sigma
        if self.hicache_order_sigma_retune and self.hicache_max_order >= 2:
            sigma = sigma * (0.7 ** (self.hicache_max_order - 1))
            sigma = max(sigma, 0.15)
        budget = self.hicache_budget
        if budget is None:
            budget = CacheBudget(
                backend=self.hicache_backend,
                allowed_stages=(self.hicache_stage,),
                max_horizon=max(1, int(interval) - 1)
                if self.hicache_max_horizon is None else self.hicache_max_horizon,
                quality_preset="adapter-default",
                max_memory_mb=self.hicache_max_memory_mb,
                audit_budget=self.hicache_audit_budget,
                fallback="full",
            )
        elif not isinstance(budget, CacheBudget):
            budget = CacheBudget.from_mapping(budget)
        self._hicache_budget = budget
        self._budget_runtime = None
        self._last_hicache_manifest = None
        self._hicache = hicache_init(
            num_steps=steps,
            interval=interval,
            max_order=self.hicache_max_order,
            first_enhance=first_enhance,
            end_enhance=end_enhance,
            sigma=sigma,
            adaptive=self.hicache_adaptive_schedule,
            interval_min=imin,
            interval_max=imax,
            adaptive_tol=self.hicache_adaptive_tol,
            backend=self.hicache_backend,
            stage=self.hicache_stage,
        )

    def _ensure_budget_runtime(self, model, x_t, cond) -> CacheBudgetRuntime:
        runtime = getattr(self, "_budget_runtime", None)
        if runtime is not None:
            return runtime
        state = self._hicache
        state.setdefault("run_id", uuid.uuid4().hex)
        model_id = f"{type(model).__module__}.{type(model).__qualname__}"
        schedule = {
            "num_steps": int(state["num_steps"]),
            "interval": int(state["interval"]),
            "first_enhance": int(state["first_enhance"]),
            "end_enhance": int(state["end_enhance"]),
            "backend": str(state["backend"]),
            "stage": str(self.hicache_stage),
        }
        condition_shape = tuple(getattr(cond, "shape", ()))
        token_shape = tuple(getattr(getattr(x_t, "feats", x_t), "shape", ()))
        identity = RunIdentity(
            model_id=model_id,
            run_id=state["run_id"],
            schedule_digest=stable_digest(schedule),
            cfg_branch="cfg-combined",
            conditioning_id=stable_digest({"condition_shape": condition_shape}),
            stage=str(self.hicache_stage),
            token_layout_digest=stable_digest({"token_shape": token_shape}),
            dtype=str(getattr(x_t, "dtype", "")),
            device=str(getattr(x_t, "device", "")),
        )
        self._budget_runtime = CacheBudgetRuntime(
            self._hicache_budget,
            identity,
            model_digest=stable_digest({"model_id": model_id}),
            config_digest=stable_digest({"budget": self._hicache_budget.as_dict(), "schedule": schedule}),
            input_digest=stable_digest({"token_shape": token_shape, "condition_shape": condition_shape}),
        )
        return self._budget_runtime

    @staticmethod
    def _budget_memory_mb(value) -> Optional[float]:
        value = getattr(value, "feats", value)
        try:
            return float(value.numel() * value.element_size()) / (1024.0 * 1024.0)
        except (AttributeError, TypeError, ValueError):
            return None

    def hicache_status(self) -> dict:
        """Return the selected backend/stage and current cache counters.

        The status is deliberately descriptive only: it never infers quality
        or speed from configuration.  ``backend`` is the actual state selected
        by the compatibility switch after setup, while ``stage`` identifies
        the model stage that owns this cache.
        """
        state = getattr(self, "_hicache", None)
        if state is None:
            return {
                "enabled": False,
                "backend": "none",
                "stage": str(getattr(self, "hicache_stage", "unknown")),
                "full_steps": 0,
                "forecast_steps": 0,
            }
        full_steps = len(state.get("activated_steps", ()))
        return {
            "enabled": True,
            "backend": state.get("backend", self.hicache_backend),
            "stage": state.get("stage", self.hicache_stage),
            "full_steps": full_steps,
            "forecast_steps": max(int(state.get("step", 0)) - full_steps, 0),
        }

    @torch.no_grad()
    def sample_once(self, model, x_t, t, t_prev, cond=None, **kwargs):
        state = getattr(self, "_hicache", None)
        if state is None:
            return super().sample_once(model, x_t, t, t_prev, cond, **kwargs)

        scheduled = hicache_decide(state)
        runtime = self._ensure_budget_runtime(model, x_t, cond)
        forecast_requested = scheduled == "forecast"
        budget_decision = runtime.decide(
            str(self.hicache_stage),
            horizon=max(1, int(state.get("counter", 0))) if forecast_requested else 0,
            method=runtime.budget.backend,
            supported=True,
            memory_mb=self._budget_memory_mb(x_t),
            force_full=(not forecast_requested or runtime.budget.backend == "full"),
            audit=forecast_requested and runtime.audits_remaining > 0,
            controller_selected=True,
        )
        serve_forecast = forecast_requested and budget_decision.mode == "forecast"
        if forecast_requested and not serve_forecast:
            # hicache_decide has already advanced its skip counter.  A guard or
            # budgeted audit becomes a fresh anchor at this diffusion index.
            state["type"] = "full"
            state["counter"] = 0
            if not state["activated_steps"] or state["activated_steps"][-1] != state["step"]:
                state["activated_steps"].append(state["step"])

        work_started = time.perf_counter()

        if not serve_forecast:
            # eps branch is unused on this path; take x_0 and v only.
            pred_x_0, _, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
            feats = pred_v.feats if _is_sparse(pred_v) else pred_v
            feats = feats.detach().clone()
            # (3) adaptive schedule: probe the forecast error at this compute
            # step BEFORE overwriting the derivative cache, so the next skip run
            # is gated on how well the forecaster is currently tracking. Reuses
            # the existing cache only -- no network evaluation.
            if state.get("adaptive"):
                hicache_record_compute(state, feats)
            hicache_update_derivatives(state, feats)
            state["step"] += 1
            runtime.record_measurement(
                str(self.hicache_stage), "full",
                wall_time_ms=(time.perf_counter() - work_started) * 1000.0,
            )
            pred_x_prev = x_t - (t - t_prev) * pred_v
            return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

        # forecast step: rebuild the final velocity from cached derivatives,
        # no model evaluation at all. Only x_0 is needed downstream, so compute
        # it directly (avoids recomputing the discarded eps branch on the
        # SparseTensor each forecast step). Numerically identical to
        # _v_to_xstart_eps(...)[0].
        feats_hat = hicache_forecast_state(state)
        # (1) FoCa-style Heun calibration: shrink the over-confident forecast
        # toward the last trusted anchor (reuses cached derivatives; zero evals).
        if self.hicache_calibrate:
            feats_hat = hicache_calibrate(state, feats_hat, self.hicache_calibrate_weight)
        # (5) frequency-aware selectivity: pull high-frequency tokens' forecasts
        # back toward the last computed anchor. Auto-disabled when no SS-stage
        # weight is present or its length does not match the current tokens (the
        # cascade-upsample / texture stages, where coords are re-derived).
        fw = self._hicache_freq_weight
        if self.hicache_freq_carve and fw is not None:
            anchor = state["derivatives"].get(0)
            if (anchor is not None and feats_hat.dim() == 2
                    and fw.shape[0] == feats_hat.shape[0]):
                fw = fw.to(feats_hat.device, feats_hat.dtype)
                feats_hat = hicache_freq_blend(
                    feats_hat, anchor, fw, self.hicache_freq_strength)
        if _is_sparse(x_t):
            pred_v = x_t.replace(feats_hat)
        else:
            pred_v = feats_hat
        pred_x_0 = self._pred_to_xstart(x_t, t, pred_v)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        state["step"] += 1
        runtime.record_measurement(
            str(self.hicache_stage), state.get("backend", "hermite"),
            wall_time_ms=(time.perf_counter() - work_started) * 1000.0,
        )
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    def get_hicache_manifest(self):
        """Return the latest portable budget/identity manifest."""
        manifest = getattr(self, "_last_hicache_manifest", None)
        if manifest is None:
            runtime = getattr(self, "_budget_runtime", None)
            manifest = runtime.manifest.as_dict() if runtime is not None else None
        return copy.deepcopy(manifest) if manifest is not None else None

    def save_hicache_manifest(self, destination):
        """Write the latest manifest without tensors, prompts or local paths."""
        manifest = self.get_hicache_manifest()
        if manifest is None:
            raise RuntimeError("no completed HiCache run is available")
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        return destination


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
    # (6) adaptive-CFG upgrade toggles (default-OFF).
    acfg_hermite: bool = False                 # Hermite (+calibrate) guidance forecast
    acfg_hermite_sigma: float = 0.5
    acfg_hermite_calibrate_weight: float = 0.0
    acfg_gamma_bar_schedule: bool = False      # schedule-adaptive gamma_bar

    def _acfg_reset(self, steps: int) -> None:
        self._acfg_state = None
        self._acfg_steps = int(steps)

    def _release_accel_state(self) -> None:
        """Drop the adaptive_cfg guidance-anchor state (CUDA tensors) and free
        memory. Cooperative: chains to any sibling acceleration mixin's
        ``_release_accel_state`` before emptying the CUDA cache once. The state
        is rebuilt fresh by ``_acfg_reset`` each run, so the output is unchanged.
        """
        self._acfg_state = None
        parent = getattr(super(), "_release_accel_state", None)
        if callable(parent):
            parent()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _inference_model(self, model, x_t, t, cond, neg_cond,
                         guidance_strength, guidance_rescale=0.0, **kwargs):
        # No CFG requested (e.g. tex stage guidance_strength=1, or the
        # outside-interval path): single conditional/unconditional pass, no
        # caching and no counter advance, matching the standard CFG mixin.
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
                hermite=self.acfg_hermite,
                hermite_sigma=self.acfg_hermite_sigma,
                hermite_calibrate_weight=self.acfg_hermite_calibrate_weight,
                gamma_bar_schedule=self.acfg_gamma_bar_schedule,
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
                if state.get("hermite"):
                    g = _acfg.forecast_guidance_hermite(
                        state["anchors"], state["step"], state["max_order"],
                        sigma=state["hermite_sigma"],
                        calibrate_weight=state["hermite_calibrate_weight"])
                else:
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
            # Release the cached-derivative CUDA tensors held by the HiCache
            # state at the end of every run. The state is rebuilt fresh on the
            # next run, so the output is unchanged.
            self._release_accel_state()
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
        try:
            ret = super().sample(
                model, noise, cond, steps, rescale_t, verbose,
                neg_cond=neg_cond, guidance_strength=guidance_strength,
                guidance_interval=guidance_interval, **kwargs)
            st = getattr(self, "_acfg_state", None)
            if verbose and st is not None:
                print(f"[adaptive-cfg] full CFG: {st['n_full']}, "
                      f"uncond-skipped: {st['n_skip']}")
        finally:
            # Release the guidance-anchor CUDA tensors held in _acfg_state at the
            # end of every run. The anchors are rebuilt on the next run's reset.
            self._release_accel_state()
        return ret


class FlowEulerGuidanceIntervalSampler_faster(
        HiCacheMixin,
        GuidanceIntervalSamplerMixin,
        AdaptiveCFGMixin,
        ClassifierFreeGuidanceSamplerMixin,
        FlowEulerSampler):
    """full_stack: HiCache (velocity forecast on skip steps) + adaptive_cfg
    (unconditional-pass skip on aligned compute steps).

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
            # Release both the HiCache derivative cache and the adaptive_cfg
            # guidance anchors (both hold CUDA tensors) at the end of the run.
            self._release_accel_state()
        return ret
