#!/usr/bin/env python3
"""PART B wide Toys4K benchmark for faster-trellis2 (OUR repo).

Variants via pipe.enable_faster(mode): vanilla / hicache / full_stack.
1024_cascade full mesh+texture. Serial, solo GPU, OOM-tolerant (retry once).
Renders cached under bench_out/renders_wide_b/ and reused across variants.
Writes per-object rows + mean/std summary to _bench_partB_faster.json.
"""
import argparse, os, sys, time, json, types, gc, traceback
import numpy as np

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
RENDERDIR = os.environ.get("FT2_RENDERDIR", os.path.join(BENCH_ROOT, "bench_out", "renders_wide_b"))
sys.path.insert(0, METRICS_DIR)
os.makedirs(RENDERDIR, exist_ok=True)

import torch, trimesh
from PIL import Image
from metrics import evaluate_mesh
DEV = torch.device("cuda:0")

CATS = ["airplane","apple","ball","banana","bicycle","boat","bottle","bowl","bread","bus",
        "cake","car","cat","chair","chicken","cow","crab","cup","dinosaur","dog",
        "dolphin","dragon","elephant","fish","fox","frog","giraffe","guitar","hamburger","hammer",
        "hat","helicopter","horse","knife","laptop","lion","monkey","mug","penguin","robot"]

def render_view(ply, out_png, img_size=512):
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (FoVPerspectiveCameras, RasterizationSettings, MeshRenderer,
        MeshRasterizer, SoftPhongShader, PointLights, TexturesVertex, look_at_view_transform)
    m = trimesh.load(ply, process=False)
    v = np.asarray(m.vertices, np.float32); f = np.asarray(m.faces, np.int64)
    c = 0.5*(v.min(0)+v.max(0)); v = v-c; s = float(np.linalg.norm(v,axis=1).max()); v = v/(s+1e-9)
    col=None
    try:
        vc=m.visual.to_color().vertex_colors
        if vc is not None and len(vc)==len(v): col=np.asarray(vc,np.float32)[:,:3]/255.0
    except Exception: col=None
    if col is None: col=np.full((len(v),3),0.7,np.float32)
    verts=torch.tensor(v,device=DEV); faces=torch.tensor(f,device=DEV)
    tex=TexturesVertex(verts_features=torch.tensor(col,device=DEV)[None])
    mesh=Meshes(verts=[verts],faces=[faces],textures=tex)
    R,T=look_at_view_transform(dist=2.2,elev=20.0,azim=45.0,device=DEV)
    cam=FoVPerspectiveCameras(device=DEV,R=R,T=T,fov=45.0)
    raster=RasterizationSettings(image_size=img_size,blur_radius=0.0,faces_per_pixel=1)
    lights=PointLights(device=DEV,location=[[2.0,2.0,2.0]])
    renderer=MeshRenderer(rasterizer=MeshRasterizer(cameras=cam,raster_settings=raster),
                          shader=SoftPhongShader(device=DEV,cameras=cam,lights=lights))
    img=renderer(mesh)[0,...,:3].clamp(0,1).cpu().numpy()
    frag=MeshRasterizer(cameras=cam,raster_settings=raster)(mesh)
    mask=(frag.pix_to_face[0,...,0]>=0).cpu().numpy()
    rgba=np.zeros((*img.shape[:2],4),np.float32); rgba[...,:3]=img; rgba[...,3]=mask.astype(np.float32)
    Image.fromarray((rgba*255).astype(np.uint8)).save(out_png)
    del renderer,mesh,verts,faces,tex; torch.cuda.empty_cache(); return out_png

def pick_objects(n):
    out=[]
    for cat in CATS:
        cd=os.path.join(MESHROOT,cat)
        if not os.path.isdir(cd): continue
        for obj in sorted(os.listdir(cd)):
            p=os.path.join(cd,obj,"mesh.ply")
            if os.path.exists(p): out.append((cat,obj,p)); break
        if len(out)>=n: break
    return out

def stats(xs):
    xs=[x for x in xs if x==x]
    if not xs: return (float('nan'),float('nan'))
    return (float(np.mean(xs)), float(np.std(xs)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE,"ckpts/TRELLIS.2-4B"))
    ap.add_argument("--pipeline-type", default="1024_cascade")
    ap.add_argument("--objects", type=int, default=20)
    ap.add_argument("--modes", nargs="+", default=["vanilla","hicache","full_stack"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE,"_bench_partB_faster.json"))
    args=ap.parse_args()

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    objs=pick_objects(args.objects)
    print(f"N_OBJECTS={len(objs)}",flush=True)
    print("OBJECTS:",[o[1] for o in objs],flush=True)

    print("Loading TRELLIS.2-4B ...",flush=True)
    pipe=Trellis2ImageTo3DPipeline.from_pretrained(args.ckpt); pipe.to("cuda")

    renders=[]
    for cat,obj,ply in objs:
        png=os.path.join(RENDERDIR,f"{obj}.png")
        if not os.path.exists(png):
            try: render_view(ply,png)
            except Exception as e: print(f"RENDER FAIL {obj}: {e!r}",flush=True); continue
        m=trimesh.load(ply,process=False)
        renders.append((obj,png,np.asarray(m.vertices,np.float32),np.asarray(m.faces,np.int64)))
    print(f"RENDERED={len(renders)}",flush=True)

    def run_one(image):
        for attempt in (1,2):
            try:
                torch.cuda.synchronize(); t0=time.time()
                out=pipe.run(image,seed=args.seed,preprocess_image=False,pipeline_type=args.pipeline_type)
                torch.cuda.synchronize(); return out[0], time.time()-t0
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                if attempt==2: raise
                print("  OOM, retry after 15s ...",flush=True); time.sleep(15)

    results={}
    for mode in args.modes:
        print(f"\n===== MODE {mode} =====",flush=True)
        pipe.enable_faster(mode)
        rows=[]
        for obj,png,gv,gf in renders:
            try:
                image=Image.open(png).convert("RGBA"); image=pipe.preprocess_image(image)
                mesh,lat=run_one(image)
                pv=mesh.vertices.detach().cpu().numpy(); pf=mesh.faces.detach().cpu().numpy()
                res=evaluate_mesh(pv,pf,gv,gf)
                rows.append({"obj":obj,**res,"latency_s":round(lat,2),"n_verts":int(len(pv))})
                print(f"  {obj}: CD={res['CD']:.4f} F1={res['F1@0.05']:.4f} vIoU={res['vIoU']:.4f} lat={lat:.1f}s",flush=True)
                del mesh,pv,pf; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                rows.append({"obj":obj,"CD":float('nan'),"F1@0.05":float('nan'),"vIoU":float('nan'),
                             "latency_s":float('nan'),"error":repr(e)})
                print(f"  {obj}: ERROR {repr(e)}",flush=True); traceback.print_exc()
                gc.collect(); torch.cuda.empty_cache()
        results[mode]=rows
        json.dump({"repo":"faster-trellis2","pipeline_type":args.pipeline_type,
                   "objects":[o[1] for o in objs],"results":results},open(args.out,"w"),indent=2)

    print("\n==== SUMMARY (mean +/- std) ====",flush=True)
    for mode in args.modes:
        r=results[mode]
        cd=stats([x.get('CD') for x in r]); f1=stats([x.get('F1@0.05') for x in r])
        vi=stats([x.get('vIoU') for x in r]); lat=stats([x.get('latency_s') for x in r])
        n=len([x for x in r if x.get('CD')==x.get('CD')])
        print(f"{mode:12s} n={n} CD={cd[0]:.4f}+/-{cd[1]:.4f} F1={f1[0]:.4f}+/-{f1[1]:.4f} "
              f"vIoU={vi[0]:.4f}+/-{vi[1]:.4f} lat={lat[0]:.1f}s",flush=True)
    print("DONE ->",args.out,flush=True)

if __name__=="__main__":
    main()
