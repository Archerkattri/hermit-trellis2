#!/usr/bin/env python3
"""Benchmark faster-trellis2 (HiCache + adaptive_cfg) vs vanilla TRELLIS.2 on Toys4K.

Loads TRELLIS.2-4B ONCE, renders each Toys4K input view with pytorch3d, then runs
image->3D in several acceleration modes (vanilla / hicache / adaptive_cfg / full_stack),
scoring geometry against the GT mesh with the shared Toys4K metrics (CD, F1@0.05, vIoU).

Designed to be GPU-frugal and OOM-tolerant: serial, retries once on OOM.

Run (from the faster-trellis2 dir, TRELLIS.2 venv + Blackwell env):
    SPARSE_CONV_BACKEND=spconv SPCONV_ALGO=native ATTN_BACKEND=flash_attn \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LD_LIBRARY_PATH=/tmp/ssl11:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 \
    .../python bench_faster_trellis2.py --pipeline-type 1024_cascade --objects 2 \
        --modes vanilla full_stack
"""
import argparse, os, sys, time, json, types, gc, traceback

# stub render-only deps that we don't use (decode_latent imports them lazily)
for _m in ("nvdiffrast", "nvdiffrast.torch", "nvdiffrec"):
    sys.modules.setdefault(_m, types.ModuleType(_m))
sys.modules["nvdiffrast"].torch = sys.modules["nvdiffrast.torch"]
sys.modules["nvdiffrast.torch"].RasterizeCudaContext = lambda *a, **k: None
sys.modules["nvdiffrast.torch"].RasterizeGLContext = lambda *a, **k: None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Benchmark dataset roots (toys4k GT meshes + metrics helpers). Override via env.
BENCH_ROOT = os.environ.get("FT2_BENCH_ROOT", os.path.join(HERE, "benchmark"))
MESHROOT = os.environ.get("FT2_MESHROOT", os.path.join(BENCH_ROOT, "toys4k", "meshes"))
METRICS_DIR = os.environ.get("FT2_METRICS_DIR", BENCH_ROOT)
sys.path.insert(0, METRICS_DIR)

import numpy as np
import torch
import trimesh
from PIL import Image
from metrics import evaluate_mesh

DEV = torch.device("cuda:0")


# --------------------------------------------------------------- input rendering
def render_view(ply, out_png, img_size=512):
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer,
        SoftPhongShader, PointLights, TexturesVertex, look_at_view_transform,
    )
    m = trimesh.load(ply, process=False)
    v = np.asarray(m.vertices, np.float32)
    f = np.asarray(m.faces, np.int64)
    c = 0.5 * (v.min(0) + v.max(0)); v = v - c
    s = float(np.linalg.norm(v, axis=1).max()); v = v / (s + 1e-9)
    col = None
    try:
        vc = m.visual.to_color().vertex_colors
        if vc is not None and len(vc) == len(v):
            col = np.asarray(vc, np.float32)[:, :3] / 255.0
    except Exception:
        col = None
    if col is None:
        col = np.full((len(v), 3), 0.7, np.float32)
    verts = torch.tensor(v, device=DEV); faces = torch.tensor(f, device=DEV)
    tex = TexturesVertex(verts_features=torch.tensor(col, device=DEV)[None])
    mesh = Meshes(verts=[verts], faces=[faces], textures=tex)
    R, T = look_at_view_transform(dist=2.2, elev=20.0, azim=45.0, device=DEV)
    cam = FoVPerspectiveCameras(device=DEV, R=R, T=T, fov=45.0)
    raster = RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1)
    lights = PointLights(device=DEV, location=[[2.0, 2.0, 2.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cam, raster_settings=raster),
        shader=SoftPhongShader(device=DEV, cameras=cam, lights=lights),
    )
    img = renderer(mesh)[0, ..., :3].clamp(0, 1).cpu().numpy()
    frag = MeshRasterizer(cameras=cam, raster_settings=raster)(mesh)
    mask = (frag.pix_to_face[0, ..., 0] >= 0).cpu().numpy()
    # RGBA with alpha = object mask (preprocess_image uses the alpha directly)
    rgba = np.zeros((*img.shape[:2], 4), np.float32)
    rgba[..., :3] = img
    rgba[..., 3] = mask.astype(np.float32)
    Image.fromarray((rgba * 255).astype(np.uint8)).save(out_png)
    del renderer, mesh, verts, faces, tex
    torch.cuda.empty_cache()
    return out_png


# --------------------------------------------------------------- object selection
CANDIDATES = [
    ("airplane", "airplane_000"), ("apple", "apple_000"), ("chair", "chair_000"),
    ("bottle", "bottle_000"), ("banana", "banana_000"), ("car", "car_000"),
    ("dog", "dog_000"), ("guitar", "guitar_001"),
]


def pick_objects(n):
    out = []
    for cat, obj in CANDIDATES:
        for cand in (os.path.join(MESHROOT, cat, obj, "mesh.ply"),
                     os.path.join(MESHROOT, cat, obj, "mesh.obj")):
            if os.path.exists(cand):
                out.append((cat, obj, cand)); break
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpts/TRELLIS.2-4B"))
    ap.add_argument("--pipeline-type", default="1024_cascade",
                    choices=["512", "1024", "1024_cascade", "1536_cascade"])
    ap.add_argument("--objects", type=int, default=2)
    ap.add_argument("--modes", nargs="+",
                    default=["vanilla", "hicache", "adaptive_cfg", "full_stack"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "_bench_faster.json"))
    args = ap.parse_args()

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    objs = pick_objects(args.objects)
    print("OBJECTS:", [o[1] for o in objs])
    rdir = os.path.join(HERE, "_bench_renders"); os.makedirs(rdir, exist_ok=True)

    print("Loading TRELLIS.2-4B ...")
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(args.ckpt)
    pipe.to("cuda")

    # pre-render input views + cache GT mesh arrays
    renders = []
    for cat, obj, ply in objs:
        png = os.path.join(rdir, f"{obj}.png")
        render_view(ply, png)
        m = trimesh.load(ply, process=False)
        renders.append((obj, png, np.asarray(m.vertices, np.float32),
                        np.asarray(m.faces, np.int64)))
        print("rendered", obj)

    def run_one(image):
        for attempt in (1, 2):
            try:
                torch.cuda.synchronize(); t0 = time.time()
                out = pipe.run(image, seed=args.seed, preprocess_image=False,
                               pipeline_type=args.pipeline_type)
                torch.cuda.synchronize(); lat = time.time() - t0
                return out[0], lat
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                if attempt == 2:
                    raise
                print("  OOM, retry after 10s ..."); time.sleep(10)

    results = {}
    for mode in args.modes:
        print(f"\n===== MODE {mode} =====")
        pipe.enable_faster(mode)
        per_obj = []
        for obj, png, gt_v, gt_f in renders:
            try:
                image = Image.open(png).convert("RGBA")
                image = pipe.preprocess_image(image)
                mesh, lat = run_one(image)
                pv = mesh.vertices.detach().cpu().numpy()
                pf = mesh.faces.detach().cpu().numpy()
                res = evaluate_mesh(pv, pf, gt_v, gt_f)
                per_obj.append({"obj": obj, **res, "latency_s": round(lat, 2),
                                "n_verts": int(len(pv))})
                print(f"  {obj}: CD={res['CD']:.5f} F1={res['F1@0.05']:.4f} "
                      f"vIoU={res['vIoU']:.4f} lat={lat:.1f}s")
                del mesh, pv, pf
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                per_obj.append({"obj": obj, "error": repr(e)})
                print(f"  {obj}: ERROR {repr(e)}"); traceback.print_exc()
                gc.collect(); torch.cuda.empty_cache()
        results[mode] = per_obj
        with open(args.out, "w") as fp:
            json.dump({"pipeline_type": args.pipeline_type,
                       "objects": [o[1] for o in objs], "results": results}, fp, indent=2)

    # summary table
    print("\n================ SUMMARY ================")
    print(f"{'mode':<14}{'CD':>9}{'F1@0.05':>10}{'vIoU':>8}{'lat_s':>9}")
    base_lat = None
    for mode in args.modes:
        valid = [r for r in results[mode] if "error" not in r]
        if not valid:
            print(f"{mode:<14}{'all-failed':>9}"); continue
        cd = np.mean([r["CD"] for r in valid])
        f1 = np.mean([r["F1@0.05"] for r in valid])
        viou = np.mean([r["vIoU"] for r in valid])
        lat = np.mean([r["latency_s"] for r in valid])
        if mode == "vanilla" or base_lat is None:
            base_lat = lat
        sp = base_lat / lat if lat > 0 else float("nan")
        print(f"{mode:<14}{cd:>9.5f}{f1:>10.4f}{viou:>8.4f}{lat:>9.1f}  ({sp:.2f}x)")
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
