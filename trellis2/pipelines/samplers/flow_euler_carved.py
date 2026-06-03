"""Carved SLaT sampler (ported verbatim from fast-trellis2).

Easy delta-cache + learned-k temporal skip + spatial token carving. Used as
the SLaT sampler in the "carved" hybrid mode so the SLaT stage matches
fast-trellis2 cost, while the SS stage keeps HiCache. Renamed _faster->_carved
to avoid colliding with the HiCache full_stack sampler in flow_euler.py."""
from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .flow_euler import FlowEulerSampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
from ...modules import sparse as sp

# Faster sampler (SLaT): easy delta-cache + learned-k skip + token carving.
from token_slat.token_leader import TokenLeader
from token_slat.token_argparser import parse_token_args
from token_slat.selection import AdvancedStabilityTracker
from faster_utils_slat import faster_cal_type, faster_init


class FlowEulerSampler_carved(FlowEulerSampler):
    def __init__(
        self,
        sigma_min: float,
    ):
        super().__init__(sigma_min)

        self.LEADER = TokenLeader()
        self.stability_tracker = AdvancedStabilityTracker()
        self.args = parse_token_args()
        self.coords_scores = None
        self.coords_raw = None
        self.cache_dic = None
        self.current = None

        # Scalars used to (re)seed the per-run delta-cache state in sample();
        # the cache itself is built per-run via faster_init(steps). Env-overridable
        # so carving aggressiveness is tunable without code edits. The carved
        # default is MORE aggressive than fast-trellis2 (carve 25% vs 10%): the
        # hybrid has quality headroom (cheaper HiCache SS) to spend on a faster SLaT.
        import os
        self.thresh = float(os.environ.get("GF_CARVE_THRESH", 5.0))
        self.ret_steps = int(os.environ.get("GF_CARVE_RET_STEPS", 2))
        self.carving_ratio = float(os.environ.get("GF_CARVE_RATIO", 0.25))

    # Inject the SS-stage spatial scores used for token selection.
    def set_coords_scores(self, coords_scores):
        self.coords_scores = coords_scores

    def _init_token_state(self, x_t_shape, device, args, model):
        self.LEADER.set_parameters(args)
        if hasattr(model, 'dtype'):
            self.model_dtype = model.dtype
        elif hasattr(model, 'parameters'):
            try:
                self.model_dtype = next(model.parameters()).dtype
            except StopIteration:
                pass

        self.stability_tracker.reset(device=device, num_tokens=x_t_shape[0], latent_channels=x_t_shape[1])
        self.stability_tracker.coords_scores = self.coords_scores

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
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        self.cache_dic, self.current = faster_init(steps)
        self.cache_dic['thresh'] = self.thresh
        self.cache_dic['first_enhance'] = self.ret_steps

        # Token carving requires per-token spatial scores from the SS stage and
        # cannot run when the model concatenates a full-resolution conditioning
        # tensor (texture stage: concat_cond), since the carved coordinates
        # would not match. In those cases the easy delta-cache still applies and
        # only carving is disabled.
        has_concat_cond = kwargs.get('concat_cond', None) is not None
        token_available = (
            self.coords_scores is not None
            and self.coords_scores.shape[0] == sample.feats.shape[0]
            and not has_concat_cond
        )
        if not token_available:
            if self.coords_scores is not None and not has_concat_cond:
                print(f"[faster-slat] coords_scores shape {tuple(self.coords_scores.shape)} != tokens "
                      f"{sample.feats.shape[0]}; token carving disabled for this stage.")
            self.current['use_token'] = False

        N, C = sample.feats.shape
        self.args.effective_steps = steps
        self._init_token_state((N, C), sample.device, self.args, model)

        self.LEADER.total_tokens = N

        self.coords_raw = sample.coords

        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            cache = self.cache_dic['cache']
            self.current['is_token_active'] = False
            current_step = self.LEADER.current_step

            if self.current['use_token'] and cache['prev_v'] is not None and current_step >= self.LEADER.full_sampling_steps:
                self.current['num_to_skip'] = int(self.carving_ratio * N)
                if self.current['num_to_skip'] > 0 and self.current['num_to_skip'] < N:
                    self.current['is_token_active'] = True

            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

        print("[faster-slat] Activated steps:   ", self.current['activated_steps'])
        ret.samples = sample
        # Release the SLaT delta-cache feature tensors, the carved-coordinates
        # reference, and the stability-tracker buffers held on the sampler
        # instance so the pipeline's torch.cuda.empty_cache() can reclaim them
        # between stages and runs.
        self.free_cache()
        return ret

    def free_cache(self):
        """Drop CUDA tensors retained by the SLaT delta-cache after a run."""
        if self.cache_dic is not None:
            cache = self.cache_dic.get('cache', None)
            if cache is not None:
                for key in ('prev_x', 'prev_prev_x', 'prev_v', 'easy', 'feature'):
                    cache[key] = None
        self.cache_dic = None
        self.current = None
        self.coords_raw = None
        self.stability_tracker.free()

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
        should_calc = faster_cal_type(self.cache_dic, self.current, x_t.feats)

        if should_calc:
            if self.current['is_token_active'] and self.current['use_token']:
                coords_scores = self.stability_tracker.coords_scores
                self.current['cached_indices'], self.current['fast_update_indices'] = \
                    self.stability_tracker.update_and_select_combined(
                        self.cache_dic['cache']['prev_v'], self.current['num_to_skip'],
                        t=0, coords_scores=coords_scores, spatial_weight=0.3,
                    )

                x_input_feats = x_t.feats[self.current['fast_update_indices'], :]
                x_input_coords = x_t.coords[self.current['fast_update_indices'], :]
                x_input = sp.SparseTensor(feats=x_input_feats, coords=x_input_coords)

                pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_input, t, cond, **kwargs)
                velocity_feats = pred_v.feats
            else:
                pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
                velocity_feats = pred_v.feats

            # Restore the carved tokens from cache to recover the full token set.
            if self.current['is_token_active'] and self.current['use_token']:
                final_v_tokens = self.cache_dic['cache']['prev_v'].clone()
                final_v_tokens[self.current['fast_update_indices'], :] = velocity_feats.to(final_v_tokens.dtype)
                velocity_feats = final_v_tokens

            prev_x = self.cache_dic['cache']['prev_x']
            prev_prev_x = self.cache_dic['cache']['prev_prev_x']
            prev_v = self.cache_dic['cache']['prev_v']
            k = self.cache_dic['cache']['k']

            if prev_x is not None and prev_prev_x is not None:
                output_change = (velocity_feats - prev_v).abs().mean()
                prev_input_change = (prev_x - prev_prev_x).abs().mean() + 1e-8
                current_k = output_change / prev_input_change
                if k is None:
                    self.cache_dic['cache']['k'] = current_k
                else:
                    self.cache_dic['cache']['k'] = 0.7 * k + 0.3 * current_k

            if prev_x is not None:
                self.cache_dic['cache']['prev_prev_x'] = prev_x
            self.cache_dic['cache']['prev_x'] = x_t.feats.detach().clone()
            self.cache_dic['cache']['prev_v'] = velocity_feats.detach().clone()
            self.cache_dic['cache']['easy'] = velocity_feats - x_t.feats
        else:
            # Reuse the cached velocity delta (easy cache).
            velocity_feats = x_t.feats + self.cache_dic['cache']['easy']
            self.cache_dic['cache']['prev_x'] = x_t.feats.detach().clone()
            self.cache_dic['cache']['prev_v'] = velocity_feats.detach().clone()

        velocity = sp.SparseTensor(
            feats=velocity_feats,
            coords=self.coords_raw,
        )
        pred_v = velocity
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        pred_x_prev = x_t - (t - t_prev) * pred_v

        output = edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

        self.current['step'] += 1
        self.LEADER.increase_step()
        return output


class FlowEulerGuidanceIntervalSampler_carved(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler_carved):
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
        return super().sample(
            model, noise, cond, steps, rescale_t, verbose,
            neg_cond=neg_cond, guidance_strength=guidance_strength,
            guidance_interval=guidance_interval, **kwargs
        )
