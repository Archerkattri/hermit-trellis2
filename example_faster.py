#!/usr/bin/env python3
"""faster-trellis2: TRELLIS.2-4B + training-free acceleration (HiCache + adaptive_cfg).

Minimal image->3D example showing how to enable the acceleration. The pipeline is
the full, unpruned TRELLIS.2 (mesh + texture + 1024_cascade); only the per-step
diffusion compute is reduced.

Acceleration is one call::

    pipeline.enable_faster("hicache")      # DEFAULT: Hermite velocity forecast,
                                           #   robust (n=20, 0 failures), 1.86x
    pipeline.enable_faster("full_stack")   # opt-in: + adaptive_cfg on SLaT stages,
                                           #   1.89x but see empty-mesh caveat
    pipeline.enable_faster("adaptive_cfg") # skip the uncond CFG pass only
    pipeline.enable_faster("vanilla")      # restore stock samplers

Blackwell (RTX 5090 / sm_120) environment, run as::

    SPARSE_CONV_BACKEND=spconv SPCONV_ALGO=native ATTN_BACKEND=flash_attn \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LD_LIBRARY_PATH=/tmp/ssl11:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 \
    <trellis2-venv>/python example_faster.py --image <rgba.png>
"""
import argparse, os, sys, types


def _stub_render_deps():
    # decode_latent lazily imports these render-only deps; stub if absent.
    for m in ("nvdiffrast", "nvdiffrast.torch", "nvdiffrec"):
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.modules["nvdiffrast"].torch = sys.modules["nvdiffrast.torch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/TRELLIS.2-4B")
    ap.add_argument("--image", required=True, help="RGBA PNG (alpha = object mask)")
    ap.add_argument("--mode", default="hicache",
                    choices=["hicache", "full_stack", "adaptive_cfg", "vanilla"])
    ap.add_argument("--pipeline-type", default="1024_cascade",
                    choices=["512", "1024", "1024_cascade", "1536_cascade"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="faster_trellis2_out.glb")
    args = ap.parse_args()

    _stub_render_deps()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import time
    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(args.ckpt)
    pipe.to("cuda")

    # Enable training-free acceleration.
    pipe.enable_faster(args.mode)

    image = Image.open(args.image).convert("RGBA")

    torch.cuda.synchronize(); t0 = time.time()
    out = pipe.run(image, seed=args.seed, preprocess_image=True,
                   pipeline_type=args.pipeline_type)
    torch.cuda.synchronize()
    print(f"[{args.mode}] image->3D in {time.time() - t0:.1f}s "
          f"({args.pipeline_type})")

    mesh = out[0]
    print(f"mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    # Export if trimesh / glb writer is available.
    try:
        import trimesh
        v = mesh.vertices.detach().cpu().numpy()
        f = mesh.faces.detach().cpu().numpy()
        trimesh.Trimesh(vertices=v, faces=f).export(args.out)
        print(f"saved {args.out}")
    except Exception as e:
        print(f"(skipped export: {e})")


if __name__ == "__main__":
    main()
