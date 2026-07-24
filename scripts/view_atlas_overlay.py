#!/usr/bin/env python
"""Open a registration in napari to compare backends (brainreg vs elastix).

sample space (default): brain (``downsampled.tiff``) + each backend's atlas warped
  into sample space as Labels (``registered_atlas.tiff`` / ``elastix/registered_atlas_elastix.tiff``).

atlas space (--space atlas [--atlas NAME]): the reference atlas (template image +
  annotation Labels) + each backend's brain warped into atlas space
  (``downsampled_standard.tiff`` / ``elastix/downsampled_standard_elastix.tiff``),
  so you see which backend's warp better matches the atlas boundaries.

Run with a napari env python:
  ~/.venvs/brainglobe/bin/python view_atlas_overlay.py <registration_dir> [--space atlas|sample] [--atlas NAME]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import napari
import tifffile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("registration_dir", type=Path)
    ap.add_argument("--space", choices=["sample", "atlas"], default="sample")
    ap.add_argument("--atlas", default="allen_mouse_10um")
    args = ap.parse_args()
    reg = args.registration_dir
    viewer = napari.Viewer()

    if args.space == "atlas":
        from brainglobe_atlasapi import BrainGlobeAtlas

        atlas = BrainGlobeAtlas(args.atlas)
        viewer.add_image(atlas.reference, name="atlas_template", colormap="gray")
        viewer.add_labels(atlas.annotation.astype("uint32"), name="atlas_regions", opacity=0.3)
        layers = [("brainreg", reg / "downsampled_standard.tiff", "green")]
        cmaps = ["magenta", "cyan", "yellow", "red", "bop orange"]
        variants = sorted((reg / "elastix").glob("downsampled_standard_elastix*.tiff"))
        for i, path in enumerate(variants):
            layers.append((path.stem.replace("downsampled_standard_", ""), path, cmaps[i % len(cmaps)]))
        for name, path, cmap in layers:
            if path.exists():
                viewer.add_image(tifffile.imread(path), name=name, colormap=cmap, blending="additive")
    else:
        viewer.add_image(tifffile.imread(reg / "downsampled.tiff"), name="brain", colormap="gray")
        for name, path in [
            ("brainreg", reg / "registered_atlas.tiff"),
            ("elastix", reg / "elastix" / "registered_atlas_elastix.tiff"),
        ]:
            if path.exists():
                viewer.add_labels(tifffile.imread(path).astype("uint32"), name=name, opacity=0.4)

    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
