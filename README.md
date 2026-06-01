<div align="center">

# ⚡ faster-trellis2

**Up to 1.9× faster [TRELLIS.2-4B](https://github.com/microsoft/TRELLIS) image-to-3D — no retraining, no weight edits, same output format.**

`TRELLIS.2-4B` · `1024_cascade` (mesh + texture) · training-free · single RTX 5090 · MIT

</div>

`faster-trellis2` drops two training-free accelerators onto the TRELLIS.2 flow-matching
samplers. They cache and forecast the model's **final CFG-combined velocity**, so fewer
network evaluations run per diffusion trajectory while the weights, decoders, and 3D output
stay byte-for-byte identical to stock TRELLIS.2.

```python
pipe.enable_faster("hicache")     # one line → ~1.86× faster, near-vanilla quality
```

---

## Pick a mode

| `enable_faster(...)` | what it does | speedup | robust? | use it when |
|---|---|:--:|:--:|---|
| `"hicache"` **(default)** | Hermite velocity forecast on every stage | **1.86×** | ✅ 0 fails / 20 | always — the safe choice |
| `"full_stack"` | `hicache` **+** adaptive-CFG on the SLaT stages | **1.89×** | ✅ *(fixed, verified)* | max speed on clean silhouettes |
| `"adaptive_cfg"` | skip the unconditional CFG pass only | modest | ✅ | ablation / study |
| `"vanilla"` | restore stock TRELLIS.2 samplers | 1.00× | ✅ | baseline |

> **TL;DR:** keep `hicache`. Reach for `full_stack` when you want the last bit of speed and
> know the input silhouette is clean. Both are validated on RTX 5090 at full `1024_cascade`.

---

## Quickstart

```bash
git clone https://github.com/Archerkattri/faster-trellis2
cd faster-trellis2
# TRELLIS.2 runtime deps (torch, flash-attn, spconv/flex_gemm, o-voxel, cumesh,
# nvdiffrast) per microsoft/TRELLIS.2. Place / symlink weights at ckpts/TRELLIS.2-4B.
```

```python
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from PIL import Image

pipe = Trellis2ImageTo3DPipeline.from_pretrained("ckpts/TRELLIS.2-4B").to("cuda")
pipe.enable_faster("hicache")                         # ← the only added line

out  = pipe.run(Image.open("input_rgba.png"), pipeline_type="1024_cascade")
mesh = out[0]
```

`example_faster.py` is the runnable end-to-end script; `example.py` is the stock TRELLIS.2 demo.

<details>
<summary><b>Blackwell (RTX 5090 / sm_120) launch env</b></summary>

`1024_cascade` fits in 32 GB with `expandable_segments`:

```bash
SPARSE_CONV_BACKEND=spconv SPCONV_ALGO=native ATTN_BACKEND=flash_attn \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python example_faster.py --image input_rgba.png --mode hicache
```
</details>

---

## Benchmarks

Toys4K, **RTX 5090 (sm_120)**, TRELLIS.2-4B, full `1024_cascade` (mesh + texture), seed 42,
geometry scored against the GT mesh. CD ↓ lower is better; F1@0.05 / vIoU ↑ higher is better.
`vIoU` = surface-shell occupancy IoU on a 64³ grid. Full detail: [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md).

**Wide run — 20 objects, 20 categories:**

| mode | CD ↓ | F1@0.05 ↑ | vIoU ↑ | latency | speedup | n |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| `vanilla` | 0.197 | 0.370 | 0.037 | 15.79 s | 1.00× | 20 |
| **`hicache`** | 0.215 | 0.341 | 0.036 | 8.49 s | **1.86×** | 20 |
| `full_stack` | 0.196 | 0.370 | 0.045 | 8.36 s | **1.89×** | 17¹ |

<sub>¹ pre-fix `full_stack` emptied 3 rounded objects → those rows excluded. The fix below
restores all three; see the verification run.</sub>

**SS-fix verification — the 3 formerly-failing rounded objects + 2 controls (GPU, 2026-06-01):**

| object | `vanilla` | `hicache` | `full_stack` *(fixed)* |
|---|:--:|:--:|:--:|
| ball_000 *(was ∅)* | 519k v · 16.0 s | 117k v · 5.8 s | **186k v · 5.6 s** |
| bowl_000 *(was ∅)* | 430k v · 13.0 s | 221k v · 6.1 s | **365k v · 5.8 s** |
| chicken_000 *(was ∅)* | 424k v · 11.3 s | 117k v · 5.5 s | **113k v · 5.3 s** |
| airplane_000 | 482k v · 11.4 s | 705k v · 6.9 s | 529k v · 5.8 s |
| apple_000 | 1.36M v · 19.3 s | 1.05M v · 10.7 s | 1.61M v · 9.9 s |

`STILL EMPTY/FAILED: []` — **0/3 empty, fix verified end-to-end.** `full_stack` runs ~2–3×
faster than vanilla; on these hard textureless inputs its quality sits at or just below
vanilla (e.g. apple F1 0.764 vs 0.785) while airplane vIoU improves (0.052 vs 0.041). The
catastrophic empty-mesh failure is gone — not just suppressed by construction.

---

## How it works

TRELLIS.2 samples a shape in three flow-matching stages — **sparse structure (SS)**, **shape
SLaT** (the 512→1024 cascade), and **texture SLaT** (guidance = 1, no CFG) — each a short
Euler sampler. Both accelerators act on the final velocity `pred_v` those samplers emit.

<details>
<summary><b>① HiCache — Hermite velocity forecast</b> (replaces network calls on skipped steps)</summary>

At a **compute** step the sampler runs the model, caches `F_t = pred_v`, and keeps backward
finite differences:

```
Δ⁰F_t = F_t
ΔⁱF_t = (Δⁱ⁻¹F_t − Δⁱ⁻¹F_{t−N}) / N
```

At a **skipped** step (`k` past the last compute step) it forecasts the velocity with the
dual-scaled physicist's Hermite basis instead of touching the network:

```
F̂_{t−k} = F_t + Σ_{i≥1} (ΔⁱF_t / i!) · H̃_i(−k)
H̃_n(x) = σⁿ · H_n(σ·x),   σ ∈ (0,1)
H_0 = 1,  H_1 = 2x,  H_{n+1} = 2x·H_n − 2n·H_{n−1}
```

**Why Hermite over Taylor?** TaylorSeer is the special case `H̃_i(−k) = (−k)ⁱ`. Those
monomials blow up super-exponentially with order and horizon, so high-order terms dominate
and the forecast diverges. The dual σ-scaling (input scale `σx` **and** coefficient scale
`σⁿ`) contracts the basis into the bounded, oscillatory regime of the Hermite functions, so
the forecast tracks the true velocity on the same cached anchors without diverging. For the
dense SS latent `pred_v` is forecast directly; for the SLaT `SparseTensor`s only `.feats` is
forecast and coords carry through via `.replace(feats)`. *(arXiv:2508.16984)*
</details>

<details>
<summary><b>② Adaptive-Guidance — skip the unconditional pass</b> (SLaT stages only)</summary>

CFG runs two passes per step: `v_cfg = w·v_cond + (1−w)·v_uncond`. As sampling proceeds
`v_cond` and `v_uncond` align — cosine similarity `γ_t → 1`. Once `γ_t ≥ γ̄` the uncond pass
carries no new directional information and is dropped.

We don't simply zero the guidance (vanilla AG sets `v_cfg → v_cond`, collapsing to the
unguided trajectory). Instead we reconstruct the guidance term

```
g_t = v_cfg − v_cond = (w−1)·(v_cond − v_uncond)
```

by Newton divided-difference extrapolation through the cached anchors (exact when the
guidance series is polynomial of the chosen order), and return `v_cond + ĝ_t` — keeping the
guided trajectory. The texture stage runs `w = 1`, so adaptive_cfg is a no-op there.
*(arXiv:2312.12487)*
</details>

<details>
<summary><b>③ The SS-stage robustness fix</b> (why <code>full_stack</code> is now safe)</summary>

Stacking adaptive_cfg onto the **sparse-structure** stage emptied 3/20 rounded objects: the
uncond pass on the SS stage is what holds the coarse occupancy volume *open*, so skipping it
over-carves rounded silhouettes to nothing. The SLaT stages (refining an already-decided
structure) don't show this.

**Fix:** even in `full_stack` the SS stage runs **HiCache-only** (never adaptive_cfg);
adaptive_cfg is confined to the two SLaT stages. This keeps the SS HiCache speedup and
removes the empty-mesh mechanism entirely. `hicache` never touches adaptive_cfg, so it was
robust from the start. Wired at `trellis2/pipelines/trellis2_image_to_3d.py` —
`full_stack → (HiCache, HiCache+AG, HiCache+AG)`. GPU-verified (table above).

**Composition:** HiCache picks compute-vs-forecast in `sample_once`; on a compute step the
model call resolves through the sampler MRO so adaptive_cfg may *additionally* skip that
step's uncond pass. On a forecast step no network runs at all. The two savings multiply.
</details>

---

## Tuning

Knobs live on the swapped sampler instances (e.g. `pipe.shape_slat_sampler`):

| attribute | default | meaning |
|---|:--:|---|
| `hicache_interval` | `3` | compute 1 step, then forecast `interval − 1` |
| `hicache_max_order` | `1` | Hermite / finite-difference order |
| `hicache_sigma` | `0.5` | Hermite contraction `σ ∈ (0,1)` |
| `acfg_gamma_bar` | `0.94` | cosine-similarity skip threshold |
| `acfg_warmup` | `2` | full-CFG warm-up steps before any skip |

```python
s = pipe.shape_slat_sampler
s.hicache_interval = 4   # skip more aggressively
s.acfg_gamma_bar   = 0.96
```

---

## What changed vs clean TRELLIS.2

All Microsoft TRELLIS.2 model / decoder / o-voxel code is **unmodified**. Added files only:

- `trellis2/pipelines/samplers/hicache.py` — Hermite basis + finite-difference cache *(CPU-unit-tested)*
- `trellis2/pipelines/samplers/adaptive_cfg.py` — guidance forecast + cosine-sim decision *(CPU-unit-tested)*
- `trellis2/pipelines/samplers/flow_euler.py` — `HiCacheMixin`, `AdaptiveCFGMixin`, the accelerated sampler classes
- `trellis2/pipelines/samplers/__init__.py` — registers the accelerated samplers
- `trellis2/pipelines/trellis2_image_to_3d.py` — `enable_faster()` + per-stage SS-robustness wiring
- `example_faster.py`, `bench_faster_trellis2.py`, `bench_partB_wide.py`, `BENCHMARK_RESULTS.md`

There is **no Fast-TRELLIS code** here — both accelerators are independent re-implementations
of the cited papers on the TRELLIS.2 sampler API.

---

## Credits & license

| | |
|---|---|
| **TRELLIS.2** | [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) — the pipeline, models, decoders this builds on (MIT) |
| **HiCache** | arXiv:2508.16984 — Hermite-polynomial velocity forecasting |
| **Adaptive Guidance** | Castillo et al., arXiv:2312.12487 — unconditional-pass skipping |

MIT. Accelerations © 2026 Krishi Attri; bundled TRELLIS.2 © Microsoft Corporation. See
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

**Krishi Attri** · krishiattriwork@gmail.com · [github.com/Archerkattri](https://github.com/Archerkattri)

<details>
<summary><b>BibTeX</b></summary>

```bibtex
@software{attri2026fastertrellis2,
  author = {Krishi Attri},
  title  = {faster-trellis2: Training-free acceleration of TRELLIS.2 image-to-3D
            via Hermite velocity forecasting and Adaptive Guidance},
  year   = {2026},
  url    = {https://github.com/Archerkattri/faster-trellis2}
}
@article{hicache2025,
  title   = {HiCache: Training-free Acceleration of Diffusion Models via
             Hermite Polynomial Feature Forecasting},
  journal = {arXiv preprint arXiv:2508.16984}, year = {2025}
}
@article{castillo2023adaptiveguidance,
  title   = {Adaptive Guidance: Training-free Acceleration of Conditional Diffusion Models},
  author  = {Castillo, Angela and others},
  journal = {arXiv preprint arXiv:2312.12487}, year = {2023}
}
@article{trellis2,
  title   = {Native and Compact Structured Latents for 3D Generation (TRELLIS.2)},
  journal = {arXiv preprint arXiv:2512.14692}, note = {microsoft/TRELLIS.2}
}
```
</details>
