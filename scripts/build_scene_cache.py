"""Precompute per-condition scene geometry so the objective runs without raw point clouds.

For every (material, file_index) the Scene needs only three quantities derived from the
raw assets: the fitted material-surface polynomial ``para`` (from the point cloud) and the
measured start/end bucket-tip y coordinates ``tsy``/``tey`` (from the excavation log).
This script evaluates them once with the original platform code and stores them in
``data/scene_cache.npz``; ``pgcd_paper.scene`` then reads the cache when the raw
point clouds are not available.

Requires the full platform (``PGCD_PLATFORM_ROOT``) with OriginalData present.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgcd_paper import config as C  # noqa: E402
from pgcd_paper import scene as SC  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-idx", type=int, default=100, help="file indices 1..n per material")
    ap.add_argument("--output", default=str(ROOT / "data" / "scene_cache.npz"))
    args = ap.parse_args()

    G = SC._models()[0]
    kne = G.Kinematic()
    import PhysicsModel.PointCloudFit as pcf
    print(f"[scene-cache] platform={C.ROOT}\n[scene-cache] PointCloudFit={pcf.__file__}", flush=True)

    keys, paras, tsys, teys = [], [], [], []
    t0 = time.time()
    for mat, cfg in C.MATERIALS.items():
        for fi in range(1, args.n_idx + 1):
            try:
                para, tsy, tey = SC.raw_geometry(G, kne, mat, fi, offset=0)
            except Exception as e:  # missing / unreadable files are skipped, as in cond_data
                print(f"   skip {mat}#{fi}: {type(e).__name__}", flush=True)
                continue
            keys.append(f"{mat}#{fi}")
            paras.append(np.asarray(para, float).reshape(-1))
            tsys.append(float(tsy))
            teys.append(float(tey))
        print(f"[scene-cache] {mat} done; {len(keys)} scenes, {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, keys=np.array(keys), para=np.stack(paras), tsy=np.array(tsys), tey=np.array(teys))
    print(f"[scene-cache] wrote {len(keys)} scenes to {args.output}")


if __name__ == "__main__":
    main()
