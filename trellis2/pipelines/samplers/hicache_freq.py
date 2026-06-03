"""3D high-frequency voxel scoring for HiCache frequency-aware token carving.

Self-contained port of the Fast-TRELLIS ``fft.fft3d`` high-frequency-energy
scorer (the prototype the task asks to resurrect), with the Plotly
visualisation stripped. Given the post-maxpool sparse-structure occupancy grid
and the SAME ``argwhere``-ordered ``coords`` the pipeline uses for the SLaT
tokens, it returns a per-token high-frequency weight in ``[0, 1]`` aligned 1:1
with ``coords`` rows.

Method
------
The occupancy grid is transformed with a 3D FFT; a spherical low-frequency mask
(radius ``filter_radius`` about the spectrum centre) is zeroed; the inverse
transform gives a per-voxel high-frequency intensity which is min-max
normalised to ``[0, 1]``. High-scoring voxels carry spatial detail/edges; low
scorers are smooth interior. HiCache uses this to forecast low-frequency tokens
aggressively and pull high-frequency tokens back toward the last computed anchor
(``hicache_freq_blend``).

Pure numerics (numpy + torch); no rendering, no filesystem side effects.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def ss_high_freq_weight(
    occupancy: torch.Tensor,
    coords: torch.Tensor,
    filter_radius: int = 8,
) -> Optional[torch.Tensor]:
    """Per-token high-frequency weight aligned 1:1 with ``coords``.

    Parameters
    ----------
    occupancy : (1, 1, G, G, G) or (1, G, G, G) bool/float occupancy grid (the
        post-maxpool sparse-structure decode the pipeline thresholds).
    coords : (N, 4) int tensor ``[batch, z, y, x]`` exactly as produced by
        ``torch.argwhere(decoded)[:, [0, 2, 3, 4]]`` -- the SLaT token coords.
    filter_radius : low-frequency cutoff radius (voxels) in the centred spectrum.

    Returns
    -------
    (N,) float32 weight in ``[0, 1]`` on ``coords.device``, or ``None`` if the
    grid is empty or the token count cannot be matched (caller then disables
    carving for that stage).
    """
    occ = occupancy
    while occ.dim() > 3:
        occ = occ[0]
    if occ.dim() != 3:
        return None
    grid = occ.detach().float().cpu().numpy()
    G = grid.shape[0]
    if grid.shape != (G, G, G) or grid.sum() <= 0:
        return None

    c = coords.detach().cpu().numpy()
    if c.ndim != 2 or c.shape[0] == 0:
        return None
    zs = c[:, 1].astype(int)
    ys = c[:, 2].astype(int)
    xs = c[:, 3].astype(int)
    valid = (zs >= 0) & (zs < G) & (ys >= 0) & (ys < G) & (xs >= 0) & (xs < G)
    if not valid.all():
        # coords reference cells outside the scored grid -> cannot align safely.
        return None

    f = np.fft.fftn(grid)
    fshift = np.fft.fftshift(f)

    cz = cy = cx = G // 2
    z_idx, y_idx, x_idx = np.ogrid[:G, :G, :G]
    dist_sq = (z_idx - cz) ** 2 + (y_idx - cy) ** 2 + (x_idx - cx) ** 2
    freq_mask = np.ones((G, G, G), dtype=np.float32)
    freq_mask[dist_sq < filter_radius ** 2] = 0.0      # drop the low-freq core

    fshift_filtered = fshift * freq_mask
    img_back = np.fft.ifftn(np.fft.ifftshift(fshift_filtered))
    intensity = np.abs(img_back)
    tok = intensity[zs, ys, xs].astype(np.float32)

    rng = tok.max() - tok.min()
    if rng > 1e-12:
        tok = (tok - tok.min()) / rng
    else:
        tok = np.zeros_like(tok)
    return torch.from_numpy(tok).to(coords.device)


if __name__ == "__main__":
    # CPU smoke test: a grid with a sharp high-frequency speckle vs a smooth
    # blob -> speckle voxels must score higher.
    G = 16
    grid = torch.zeros(1, 1, G, G, G)
    # smooth low-freq blob in one octant
    grid[0, 0, 2:6, 2:6, 2:6] = 1.0
    # high-freq checkerboard speckle in another octant
    for z in range(9, 15):
        for y in range(9, 15):
            for x in range(9, 15):
                if (z + y + x) % 2 == 0:
                    grid[0, 0, z, y, x] = 1.0
    occ = grid > 0.5
    coords = torch.argwhere(occ)[:, [0, 2, 3, 4]].int()
    w = ss_high_freq_weight(occ.float(), coords, filter_radius=4)
    assert w is not None and w.shape[0] == coords.shape[0], "shape mismatch"
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0, "out of [0,1]"
    # mean speckle-octant weight should exceed mean blob-octant weight
    z = coords[:, 1]
    blob = w[z < 8].mean().item()
    speckle = w[z >= 8].mean().item()
    print(f"[freq] blob={blob:.3f} speckle={speckle:.3f}")
    assert speckle > blob, "high-freq speckle should outscore smooth blob"
    print("ALL PASS")
