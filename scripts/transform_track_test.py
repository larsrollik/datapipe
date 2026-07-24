#!/usr/bin/env python
"""TEST: transform brainglobe-segmentation sample-space tracks -> atlas space.

Reads ``<reg>/segmentation/sample_space/tracks/*.points`` (pandas-HDF of
downsampled-space points), transforms them to atlas space via
``brainglobe_utils.brainreg.transform`` (the forward transform brainmapper uses for
cells), and writes ``<reg>/segmentation/atlas_space/tracks/*.points`` in the SAME
format brainglobe-segmentation writes — so opening the sample in atlas space
auto-loads the transformed track. Also prints the atlas region each point lands in,
as an immediate correctness check.

Run with the brainglobe env python:
  ~/.venvs/brainglobe/bin/python transform_track_test.py <registration_dir> [atlas]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_utils.brainreg.transform import (
    transform_points_from_downsampled_to_atlas_space,
)


def main() -> int:
    reg = Path(sys.argv[1])
    atlas_name = sys.argv[2] if len(sys.argv) > 2 else "allen_mouse_10um"

    sample_tracks = reg / "segmentation" / "sample_space" / "tracks"
    atlas_tracks = reg / "segmentation" / "atlas_space" / "tracks"
    points_files = sorted(sample_tracks.glob("*.points"))
    if not points_files:
        sys.exit(f"no .points in {sample_tracks}")

    deform = [reg / f"deformation_field_{i}.tiff" for i in range(3)]
    for d in deform:
        if not d.exists():
            sys.exit(f"missing deformation field: {d}")

    atlas = BrainGlobeAtlas(atlas_name)
    ann = atlas.annotation
    atlas_tracks.mkdir(parents=True, exist_ok=True)

    for pf in points_files:
        pts = pd.read_hdf(pf, key="df").to_numpy(dtype=float)
        print(f"\n== {pf.name}: {len(pts)} points (downsampled space) ==")

        atlas_pts, oob = transform_points_from_downsampled_to_atlas_space(
            pts, atlas, deform, warn_out_of_bounds=True
        )
        atlas_pts = np.asarray(atlas_pts, dtype=float)

        # write in the exact format brainglobe-segmentation uses (so reload loads it)
        out = atlas_tracks / pf.name
        pd.DataFrame(atlas_pts).to_hdf(out, key="df", mode="w")
        print(f"   -> wrote {out}  ({len(atlas_pts)} pts, {len(oob)} out-of-bounds)")

        for i, p in enumerate(atlas_pts):
            z, y, x = (int(round(v)) for v in p)
            if 0 <= z < ann.shape[0] and 0 <= y < ann.shape[1] and 0 <= x < ann.shape[2]:
                sid = int(ann[z, y, x])
                reg_name = (
                    atlas.structures[sid]["acronym"]
                    if sid in atlas.structures else f"id{sid}?"
                )
            else:
                reg_name = "OUT-OF-BOUNDS"
            print(f"     pt{i:>2}: atlas(z,y,x)=({z},{y},{x}) -> {reg_name}")

    print(
        "\nDone. Open this sample in ATLAS space "
        "(datapipe segment <...> --atlas-space) — the transformed track loads "
        "from segmentation/atlas_space/tracks/."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
