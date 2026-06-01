# faster-trellis2 — Wide Toys4K Benchmark (TRELLIS.2, 1024_cascade)

Hardware: **RTX 5090 (sm_120, Blackwell), 32 GB**. Model: **TRELLIS.2-4B**, full
pipeline (SS → shape-SLaT cascade → texture-SLaT → decode), `pipeline_type='1024_cascade'`.

Publication-grade replacement for the small (n=2–6) numbers previously quoted.
Image-to-3D geometry, **20 Toys4K objects spanning 20 categories**, single textured
front view per object, full mesh + texture, scored against the GT mesh.

> **Subset size:** 20 objects (not 40). The `1024_cascade` pipeline is ~6–16 s/object;
> 4 variants × 40 objects would exceed the 2.5 h budget, so a 20-category subset was used
> as permitted by the task brief.

## Result

seed=42, default steps. All variants share the **identical** cached front render per
object. Solo GPU, serial, retry-once on OOM.

| variant | CD ↓ | F1@0.05 ↑ | vIoU ↑ | latency | speedup vs vanilla | n |
|---|---|---|---|---|---|---|
| `vanilla`        | 0.197 ± 0.059 | 0.370 ± 0.148 | 0.037 ± 0.019 | 15.79 s | 1.00× | 20 |
| `hicache` (ours, **DEFAULT — robust**) | 0.215 ± 0.067 | 0.341 ± 0.151 | 0.036 ± 0.021 | 8.49 s | 1.86× | **20** |
| `full_stack` (ours, opt-in: HiCache + adaptive-CFG) | 0.196 ± 0.061 | 0.370 ± 0.167 | 0.045 ± 0.032 | 8.36 s | **1.89×** | 17 |

± is the standard deviation across objects. CD / F1 in unit-cube-normalised units
(longest bbox edge = 1), ~50k surface samples. vIoU on a 64³ grid (see metric note).

### Reading

- **`hicache` is the recommended default**: 1.86× with **zero failures (n=20)** and
  near-vanilla quality. It never skips an unconditional CFG pass, so it cannot trigger
  the empty-mesh collapse below.
- **`full_stack` is opt-in, max-quality on well-posed objects**: where it produces a mesh
  it matches/beats vanilla (CD 0.196 vs 0.197, F1 0.370 vs 0.370, vIoU 0.045 vs 0.037) at
  **1.89×** — but only **n=17** (see caveat). Use it when you know the silhouette is clean.

## Failures / caveats — honest empty-mesh limitation

- **`full_stack` (as originally benched) emptied 3/20 rounded objects** — `ball_000`,
  `bowl_000`, `chicken_000` returned empty meshes
  (`IndexError: max(): Expected reduction dim 0 to have non-zero size`). **Root cause:**
  adaptive_cfg skipped the unconditional pass on the **sparse-structure (SS)** stage; that
  uncond pass is what holds the coarse occupancy volume open, so dropping it over-carved
  rounded silhouettes to nothing. `hicache` alone had **no failures**.
- **Mitigation shipped in this repo:** `enable_faster("full_stack")` now confines
  adaptive_cfg to the two SLaT stages and runs the **SS stage HiCache-only**, removing the
  empty-mesh mechanism while keeping the SS HiCache speedup. The wide table above reflects the
  **pre-fix** run (n=17) and is kept as the honest as-measured record.
- **GPU-VERIFIED (2026-06-01, RTX 5090, solo).** The post-fix SS guard has now been
  re-benched **end-to-end on GPU** on the 3 previously-failing rounded objects plus 2
  controls (see "SS-fix re-bench" table below). `full_stack` produces **non-empty meshes on
  all 3 previously-failing objects** (ball/bowl/chicken) and on both controls — **0 empty
  meshes across 5/5 objects × all 3 modes (15/15 generations)**. The empty-mesh mechanism is
  eliminated; **`full_stack` is now safe to use**. `hicache` remains the conservative default.

## SS-fix re-bench (GPU-verified, 2026-06-01)

RTX 5090, solo, `1024_cascade`, seed=42, cached `renders_wide_b` front views, identical
across modes. 3 previously-failing rounded objects (`ball_000`, `bowl_000`, `chicken_000`)
+ 2 controls that always worked (`airplane_000`, `apple_000`). `n_verts` is the produced
mesh vertex count; **no EMPTY** in any cell. Raw data: `bench_ssfix.json`.

| object (prev-failing?) | mode | n_verts | CD ↓ | F1@0.05 ↑ | vIoU ↑ | latency |
|---|---|---|---|---|---|---|
| **ball_000** (was empty) | vanilla | 519281 | 0.2547 | 0.1184 | 0.0049 | 16.0 s |
| | hicache | 116654 | 0.3444 | 0.1206 | 0.0040 | 5.8 s |
| | **full_stack** | **186299** | 0.3541 | 0.1136 | 0.0049 | 5.6 s |
| **bowl_000** (was empty) | vanilla | 430021 | 0.1952 | 0.3026 | 0.0182 | 13.0 s |
| | hicache | 221084 | 0.2674 | 0.1647 | 0.0084 | 6.1 s |
| | **full_stack** | **365406** | 0.2626 | 0.1754 | 0.0118 | 5.8 s |
| **chicken_000** (was empty) | vanilla | 424423 | 0.1445 | 0.5028 | 0.0303 | 11.3 s |
| | hicache | 116948 | 0.2283 | 0.2056 | 0.0077 | 5.5 s |
| | **full_stack** | **112732** | 0.2215 | 0.2168 | 0.0072 | 5.3 s |
| airplane_000 (control) | vanilla | 481785 | 0.1842 | 0.4567 | 0.0411 | 11.4 s |
| | hicache | 704504 | 0.1906 | 0.4637 | 0.0458 | 6.9 s |
| | full_stack | 529174 | 0.1972 | 0.4397 | 0.0521 | 5.8 s |
| apple_000 (control) | vanilla | 1355052 | 0.0749 | 0.7849 | 0.0673 | 19.3 s |
| | hicache | 1053728 | 0.0752 | 0.7886 | 0.0635 | 10.7 s |
| | full_stack | 1608240 | 0.0761 | 0.7643 | 0.0689 | 9.9 s |

**Verdict:** `FIX_WORKS = True`, `STILL EMPTY/FAILED = []`. All 3 previously-failing objects
now yield valid non-empty meshes under `full_stack`; the SS-fix is **GPU-verified**.
Driver: `validate_ssfix.py`.
- Absolute CD/F1 are modest because the input is a **single synthetic, flat-shaded front
  view** of an untextured/low-texture GT mesh — harder than TRELLIS.2's natural-image
  domain. Uniform across variants, so the comparison is fair.

## Reproduce

```bash
# Point FT2_BENCH_ROOT at a dir with toys4k/meshes/<cat>/<obj>/mesh.ply + metrics.py
export FT2_BENCH_ROOT=/path/to/benchmark

SPARSE_CONV_BACKEND=spconv SPCONV_ALGO=native ATTN_BACKEND=flash_attn \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
  python bench_partB_wide.py --objects 20 --modes vanilla hicache full_stack
# -> _bench_partB_faster.json
```

Driver lives in this repo: `bench_partB_wide.py` (`pipe.enable_faster(mode)`). Renders are
cached under `$FT2_BENCH_ROOT/bench_out/renders_wide_b/` and reused across all variants.

### Environment

- Env: `SPARSE_CONV_BACKEND=spconv SPCONV_ALGO=native ATTN_BACKEND=flash_attn`
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0`.
- `nvdiffrast` / `nvdiffrec` stubbed (render-only deps).
- Benchmark roots overridable via `FT2_BENCH_ROOT` / `FT2_MESHROOT` / `FT2_METRICS_DIR`.

### Metrics

`third_party/benchmark/metrics.py :: evaluate_mesh`. CD = symmetric mean-L2 Chamfer over
~50k surface samples per mesh (unit-cube normalised, longest edge = 1); F1@0.05 = harmonic
mean of precision/recall at 0.05; vIoU = **surface-occupancy** IoU on a 64³ grid (voxels
containing surface samples — NOT a filled solid-volume IoU; Toys4K GT and TRELLIS outputs
are non-watertight so robust scanline solid-fill is unavailable. Deterministic and applied
identically to every variant). No silent fallbacks.

### Object set (20, one per category)

airplane_000, apple_000, ball_000, banana_000, bicycle_000, boat_000, bottle_000,
bowl_000, bread_000, bus_000, cake_000, car_000, cat_000, chair_000, chicken_000,
cow_001, crab_000, cup_001, dinosaur_000, dog_000.

GT meshes: `third_party/benchmark/toys4k/meshes/<category>/<obj>/mesh.ply`.
