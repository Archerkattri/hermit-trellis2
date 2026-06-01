#!/usr/bin/env python3
"""SOLO GPU validation of the SS empty-mesh fix in faster-trellis2.

Targets exactly 5 objects: 3 previously-failing rounded (ball/bowl/chicken_000)
+ 2 controls (airplane/apple_000). Modes: vanilla, hicache, full_stack.
Reuses cached renders in benchmark/bench_out/renders_wide_b/. 1024_cascade.
Records n_verts (CONFIRM full_stack non-empty), CD/F1@0.05/vIoU, latency.
"""
import os, sys, time, json, types, gc, traceback
import numpy as np

for _m in ("nvdiffrast", "nvdiffrast.torch", "nvdiffrec"):
    sys.modules.setdefault(_m, types.ModuleType(_m))
sys.modules["nvdiffrast"].torch = sys.modules["nvdiffrast.torch"]
sys.modules["nvdiffrast.torch"].RasterizeCudaContext = lambda *a, **k: None
sys.modules["nvdiffrast.torch"].RasterizeGLContext = lambda *a, **k: None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH_ROOT = "/home/krishi/workspace/gaussianfeels/third_party/benchmark"
MESHROOT = os.path.join(BENCH_ROOT, "toys4k", "meshes")
RENDERDIR = os.path.join(BENCH_ROOT, "bench_out", "renders_wide_b")
sys.path.insert(0, BENCH_ROOT)

import torch, trimesh
from PIL import Image
from metrics import evaluate_mesh
DEV = torch.device("cuda:0")

# (category, object) -> previously failing flag
TARGETS = [
    ("ball", "ball_000", True),
    ("bowl", "bowl_000", True),
    ("chicken", "chicken_000", True),
    ("airplane", "airplane_000", False),
    ("apple", "apple_000", False),
]
CKPT = os.path.join(HERE, "ckpts/TRELLIS.2-4B")
PTYPE = "1024_cascade"
SEED = 42
OUT = os.path.join(HERE, "_bench_ssfix.json")
MODES = ["vanilla", "hicache", "full_stack"]


def main():
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    print("Loading TRELLIS.2-4B ...", flush=True)
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(CKPT); pipe.to("cuda")

    renders = []
    for cat, obj, failflag in TARGETS:
        png = os.path.join(RENDERDIR, f"{obj}.png")
        gt = os.path.join(MESHROOT, cat, obj, "mesh.ply")
        assert os.path.exists(png), f"missing render {png}"
        assert os.path.exists(gt), f"missing GT mesh {gt}"
        m = trimesh.load(gt, process=False)
        renders.append((obj, png, np.asarray(m.vertices, np.float32),
                        np.asarray(m.faces, np.int64), failflag))
    print(f"RENDERED={len(renders)}", flush=True)

    def run_one(image):
        for attempt in (1, 2):
            try:
                torch.cuda.synchronize(); t0 = time.time()
                out = pipe.run(image, seed=SEED, preprocess_image=False, pipeline_type=PTYPE)
                torch.cuda.synchronize(); return out[0], time.time() - t0
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                if attempt == 2: raise
                print("  OOM, retry after 15s ...", flush=True); time.sleep(15)

    results = {}
    for mode in MODES:
        print(f"\n===== MODE {mode} =====", flush=True)
        pipe.enable_faster(mode)
        rows = []
        for obj, png, gv, gf, failflag in renders:
            try:
                image = Image.open(png).convert("RGBA"); image = pipe.preprocess_image(image)
                mesh, lat = run_one(image)
                pv = mesh.vertices.detach().cpu().numpy(); pf = mesh.faces.detach().cpu().numpy()
                nverts = int(len(pv))
                if nverts == 0:
                    rows.append({"obj": obj, "prev_failing": failflag, "n_verts": 0,
                                 "CD": float('nan'), "F1@0.05": float('nan'),
                                 "vIoU": float('nan'), "latency_s": round(lat, 2),
                                 "EMPTY": True})
                    print(f"  {obj}: *** EMPTY MESH (0 verts) *** lat={lat:.1f}s", flush=True)
                else:
                    res = evaluate_mesh(pv, pf, gv, gf)
                    rows.append({"obj": obj, "prev_failing": failflag, "n_verts": nverts,
                                 **res, "latency_s": round(lat, 2), "EMPTY": False})
                    print(f"  {obj}: nverts={nverts} CD={res['CD']:.4f} "
                          f"F1={res['F1@0.05']:.4f} vIoU={res['vIoU']:.4f} lat={lat:.1f}s", flush=True)
                del mesh, pv, pf; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                rows.append({"obj": obj, "prev_failing": failflag, "n_verts": -1,
                             "CD": float('nan'), "F1@0.05": float('nan'), "vIoU": float('nan'),
                             "latency_s": float('nan'), "error": repr(e)})
                print(f"  {obj}: ERROR {repr(e)}", flush=True); traceback.print_exc()
                gc.collect(); torch.cuda.empty_cache()
        results[mode] = rows
        json.dump({"repo": "faster-trellis2", "pipeline_type": PTYPE,
                   "targets": [t[1] for t in TARGETS], "results": results},
                  open(OUT, "w"), indent=2)

    print("\n==== VERDICT (full_stack on previously-failing objects) ====", flush=True)
    fs = results["full_stack"]
    empties = [r["obj"] for r in fs if r.get("prev_failing") and r.get("n_verts", 0) <= 0]
    nonempty = [(r["obj"], r["n_verts"]) for r in fs if r.get("prev_failing") and r.get("n_verts", 0) > 0]
    print("non-empty (was failing):", nonempty, flush=True)
    print("STILL EMPTY/FAILED:", empties, flush=True)
    print("FIX_WORKS =", len(empties) == 0, flush=True)
    print("DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
