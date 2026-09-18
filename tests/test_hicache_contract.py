"""CPU contract tests for the TRELLIS.2 acceleration seam.

These tests deliberately load the sampler modules without the optional GPU
runtime dependencies.  They exercise cache lifecycle and backend selection,
not model quality or end-to-end inference.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from hicache_pp import CacheBudget


ROOT = Path(__file__).resolve().parents[1]


class _EasyDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _load_sampler_modules():
    """Load the pure sampler modules while bypassing trellis2's heavy package init."""
    names = {
        "trellis2": ROOT / "trellis2",
        "trellis2.pipelines": ROOT / "trellis2" / "pipelines",
        "trellis2.pipelines.samplers": ROOT / "trellis2" / "pipelines" / "samplers",
    }
    for name, path in names.items():
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules.setdefault(name, pkg)
    easy = types.ModuleType("easydict")
    easy.EasyDict = _EasyDict
    sys.modules.setdefault("easydict", easy)

    def load(name):
        path = ROOT.joinpath(*name.split("."))
        path = path.with_suffix(".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    hicache = load("trellis2.pipelines.samplers.hicache")
    load("trellis2.pipelines.samplers.base")
    load("trellis2.pipelines.samplers.classifier_free_guidance_mixin")
    load("trellis2.pipelines.samplers.guidance_interval_mixin")
    load("trellis2.pipelines.samplers.adaptive_cfg")
    flow = load("trellis2.pipelines.samplers.flow_euler")
    return hicache, flow


def test_backend_dispatch_and_validation():
    hicache, _ = _load_sampler_modules()
    assert hicache.normalize_backend(None) == "hermite"
    assert hicache.normalize_backend(" Hermite ") == "hermite"
    for bad in ("dmd", "carved"):
        try:
            hicache.normalize_backend(bad)
        except ValueError as exc:
            assert "hermite" in str(exc)
        else:
            raise AssertionError(f"unsupported backend was accepted: {bad}")

    state = hicache.hicache_init(8, interval=2, first_enhance=0,
                                 end_enhance=8, backend="hermite", stage="sparse_structure")
    assert state["backend"] == "hermite"
    assert state["stage"] == "sparse_structure"


def test_hicache_lifecycle_reset_and_actual_status():
    _, flow = _load_sampler_modules()

    class Harness(flow.HiCacheMixin, flow.FlowEulerSampler):
        hicache_interval = 2
        hicache_first_enhance = 0
        hicache_stage = "sparse_structure"

    sampler = Harness(sigma_min=0.001)
    sampler._hicache_setup(8)
    first_state = sampler._hicache
    assert sampler.hicache_status()["backend"] == "hermite"
    assert sampler.hicache_status()["stage"] == "sparse_structure"

    calls = {"n": 0}

    def model(x, t, cond, **kwargs):
        calls["n"] += 1
        return x + 0.1

    x = torch.zeros(2, 3)
    for step in range(4):
        sampler.sample_once(model, x, 1.0 - step * 0.1, 0.9 - step * 0.1,
                            cond=None)
    assert calls["n"] < 4, "a forecast step must avoid a model call"
    assert sampler.hicache_status()["forecast_steps"] > 0

    sampler._release_accel_state()
    assert sampler.hicache_status()["enabled"] is False
    sampler._hicache_setup(8)
    assert sampler._hicache is not first_state
    assert sampler.hicache_status()["full_steps"] == 0


def test_cfg_path_preserves_guidance_identity_on_full_step():
    _, flow = _load_sampler_modules()
    sampler = flow.FlowEulerGuidanceIntervalSampler_hicache(sigma_min=0.001)
    sampler._hicache_setup(4)
    calls = []

    def model(x, t, cond, **kwargs):
        calls.append(cond)
        return x + cond

    x = torch.zeros(1, 2)
    sampler.sample_once(model, x, 1.0, 0.75, cond=torch.ones(1, 2),
                        neg_cond=torch.zeros(1, 2), guidance_strength=3.0,
                        guidance_interval=(0.0, 1.0))
    # CFG convention in this checkout is w*positive+(1-w)*negative.
    assert len(calls) == 2
    assert torch.allclose(sampler._hicache["derivatives"][0], 3 * torch.ones(1, 2))


def test_budget_manifest_records_guarded_fallback_and_is_portable(tmp_path):
    _, flow = _load_sampler_modules()

    class Harness(flow.HiCacheMixin, flow.FlowEulerSampler):
        hicache_interval = 3
        hicache_first_enhance = 0
        hicache_stage = "sparse_structure"

    sampler = Harness(sigma_min=0.001)
    sampler.configure_hicache_budget(CacheBudget(
        backend="hermite",
        allowed_stages=("sparse_structure",),
        max_horizon=1,
        quality_preset="test",
        fallback="full",
    ))
    sampler._hicache_setup(8)

    def model(x, t, cond, **kwargs):
        return x + 0.1

    x = torch.zeros(2, 3)
    for step in range(4):
        sampler.sample_once(model, x, 1.0 - step * 0.1, 0.9 - step * 0.1)

    manifest = sampler.get_hicache_manifest()
    assert manifest["schema"] == "hicache-pp.run-manifest.v1"
    assert manifest["identity"]["stage"] == "sparse_structure"
    assert manifest["counts"]["fallback"] >= 1
    assert len(manifest["measurements"]) == 4
    output = tmp_path / "manifest.json"
    sampler.save_hicache_manifest(output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == manifest
    assert all("tensor" not in key.lower() for key in json.dumps(saved).split('"'))


def test_pipeline_preset_reports_hermite_and_keeps_carved_slat_separate():
    """The user-facing preset selects Hermite only for SS and reports all stages."""
    # Importing the pipeline normally requires the full TRELLIS.2 runtime.  A
    # small module shim is enough to exercise its configuration contract.
    import importlib.util

    class Sampler:
        def __init__(self, sigma_min=0.001):
            self.sigma_min = sigma_min

    class HiCache(Sampler):
        hicache_backend = "hermite"
        hicache_stage = "sparse_structure"

    class Carved(Sampler):
        carving_ratio = 0.25

    sampler_mod = types.ModuleType("trellis2.pipelines.samplers")
    sampler_mod.Sampler = Sampler
    sampler_mod.FlowEulerGuidanceIntervalSampler = Sampler
    sampler_mod.FlowEulerGuidanceIntervalSampler_hicache = HiCache
    sampler_mod.FlowEulerGuidanceIntervalSampler_carved = Carved
    sys.modules["trellis2.pipelines.samplers"] = sampler_mod
    freq_mod = types.ModuleType("trellis2.pipelines.samplers.hicache_freq")
    freq_mod.ss_high_freq_weight = lambda *args, **kwargs: None
    sys.modules["trellis2.pipelines.samplers.hicache_freq"] = freq_mod
    sys.modules["trellis2.pipelines.rembg"] = types.ModuleType("trellis2.pipelines.rembg")
    pil_mod = types.ModuleType("PIL")
    pil_mod.Image = types.SimpleNamespace(Image=object)
    sys.modules["PIL"] = pil_mod

    base_mod = types.ModuleType("trellis2.pipelines.base")
    class Pipeline:
        pass
    base_mod.Pipeline = Pipeline
    sys.modules["trellis2.pipelines.base"] = base_mod
    modules_pkg = types.ModuleType("trellis2.modules")
    modules_pkg.__path__ = []
    sys.modules["trellis2.modules"] = modules_pkg
    sparse_mod = types.ModuleType("trellis2.modules.sparse")
    sparse_mod.SparseTensor = object
    sys.modules["trellis2.modules.sparse"] = sparse_mod
    sys.modules["trellis2.modules.image_feature_extractor"] = types.ModuleType(
        "trellis2.modules.image_feature_extractor")
    rep_mod = types.ModuleType("trellis2.representations")
    rep_mod.Mesh = object
    rep_mod.MeshWithVoxel = object
    sys.modules["trellis2.representations"] = rep_mod

    name = "trellis2.pipelines.trellis2_image_to_3d_contract"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "trellis2" / "pipelines" / "trellis2_image_to_3d.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    pipe = module.Trellis2ImageTo3DPipeline.__new__(
        module.Trellis2ImageTo3DPipeline)
    pipe.sparse_structure_sampler = Sampler()
    pipe.shape_slat_sampler = Sampler()
    pipe.tex_slat_sampler = Sampler()
    try:
        pipe.enable_faster("dmd")
    except ValueError:
        pass
    else:
        raise AssertionError("dmd mode was accepted by the Hermite-only checkout")
    pipe.enable_faster("hermite")
    status = pipe.acceleration_status()
    assert status["backend"] == "hermite"
    assert status["stages"]["sparse_structure"] == "hicache:hermite"
    assert status["stages"]["shape_slat"] == "carved_slat"
    assert status["stages"]["texture_slat"] == "carved_slat"
    assert pipe.shape_slat_sampler.carving_ratio == 0.1
    pipe.enable_faster("base")
    assert pipe.acceleration_status()["enabled"] is False


if __name__ == "__main__":
    import tempfile

    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            # pytest supplies tmp_path; direct runs get a TemporaryDirectory.
            wants_tmp = "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]
            if wants_tmp:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
            print(f"[PASS] {name}")
