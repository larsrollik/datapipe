#!/usr/bin/env python
"""Headless elastix registration test (brainglobe-registration core, no napari).

Registers a sample's atlas-oriented downsampled brain (brainreg's ``downsampled.tiff``
— reoriented + scaled to the atlas grid, pre-warp, so it's a fair/neutral input)
against the BrainGlobe atlas via itk-elastix, with a **tuned bspline** (finer control
grid + more resolutions/iterations than the stock profile).

Two output spaces (``--space``), each directly comparable to a brainreg output on
the same grid:
  atlas  (default) : sample warped INTO atlas space -> ``elastix/downsampled_standard_elastix.tiff``
                     (compare with brainreg's ``downsampled_standard.tiff`` against the reference atlas)
  sample           : atlas annotation warped INTO sample space -> ``elastix/registered_atlas_elastix.tiff``
                     (compare with brainreg's ``registered_atlas.tiff`` on ``downsampled.tiff``)

Run with the brainglobe-registration env python (no CLI in that package):
  ~/.venvs/bg-registration/bin/python elastix_register_test.py <registration_dir> \
      [--space atlas|sample] [--atlas allen_mouse_10um] [--profile brainglobe_registration]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import brainglobe_registration
import itk
import numpy as np
import tifffile
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_registration.elastix.register import (
    run_registration,
    transform_annotation_image,
    transform_image,
)
from brainglobe_registration.utils.file import open_parameter_file

# --- bspline tuning presets (override the stock brainglobe_registration profile) ---
# Stock: NumberOfResolutions 3, FinalGridSpacingInVoxels 50 (=500um @10um),
# 500 iterations, 10000 samples, single MI metric (NO regularization).
#   none : stock profile unchanged (control).
#   fine : finer 25-vox grid + more pyramids/iterations, NO penalty -> OVER-FITS
#          (deformation folds; atlas-space NCC collapsed to ~0). Kept to show the trap.
#   reg  : the SAME finer grid but with a TransformBendingEnergyPenalty added as a
#          second metric (MultiMetric) -> regularizes the warp so it stays smooth.
_FINE = {
    "NumberOfResolutions": ["4"],
    "FinalGridSpacingInVoxels": ["25.000000", "25.000000", "25.000000"],  # 250um @10um
    "MaximumNumberOfIterations": ["1000"],
    "NumberOfSpatialSamples": ["20000"],
}
TUNES = {
    "none": {},
    "fine": _FINE,
    "reg": {
        **_FINE,
        "Registration": ["MultiMetricMultiResolutionRegistration"],
        "Metric": ["AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty"],
        "Metric0Weight": ["1.0"],
        "Metric1Weight": ["10.0"],  # bending-energy weight (smoothness regulariser)
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("registration_dir", type=Path)
    ap.add_argument("--space", choices=["atlas", "sample"], default="atlas")
    ap.add_argument("--tune", choices=list(TUNES), default="fine")
    ap.add_argument("--out-suffix", default=None, help="output filename suffix (default: --tune value)")
    ap.add_argument("--atlas", default="allen_mouse_10um")
    ap.add_argument("--profile", default="brainglobe_registration")
    args = ap.parse_args()
    suffix = args.out_suffix or args.tune

    reg = args.registration_dir
    moving_path = reg / "downsampled.tiff"
    if not moving_path.exists():
        ap.error(f"missing {moving_path} — run brainreg first (atlas-oriented brain)")

    atlas = BrainGlobeAtlas(args.atlas)
    ref = np.asarray(atlas.reference)
    sample = tifffile.imread(moving_path)
    print(f"atlas ref {ref.shape} | sample {sample.shape} | space={args.space} | profile={args.profile}")

    param_dir = Path(brainglobe_registration.__file__).parent / "parameters" / args.profile
    affine = open_parameter_file(param_dir / "affine.txt")
    bspline = open_parameter_file(param_dir / "bspline.txt")
    tune = TUNES[args.tune]
    bspline.update(tune)
    print(f"tune={args.tune}: {', '.join(f'{k}={v}' for k, v in tune.items()) or 'stock'}")
    parameter_lists = [("affine", affine), ("bspline", bspline)]

    out = reg / "elastix"
    out.mkdir(parents=True, exist_ok=True)

    # run_registration wires ElastixRegistrationMethod.New(moving_image, atlas_image)
    # = New(fixed, moving). So fixed = the moving_image arg, moving = the atlas_image arg.
    t0 = time.time()
    if args.space == "atlas":
        # fixed = atlas ref, moving = sample -> transform resamples sample onto atlas grid.
        transform_parameters = run_registration(
            atlas_image=sample, moving_image=ref,
            parameter_lists=parameter_lists, output_directory=None, filter_images=True,
        )
        print(f"elastix registration: {time.time() - t0:.0f}s")
        warped = np.asarray(transform_image(sample, transform_parameters))
        dst = out / f"downsampled_standard_elastix_{suffix}.tiff"
        tifffile.imwrite(dst, warped)
        print(f"wrote {dst}  shape={warped.shape}  (sample in ATLAS space)")
        print(
            f"COMPARE in atlas space: reference atlas annotation + brainreg's "
            f"{reg / 'downsampled_standard.tiff'} vs {dst}"
        )
    else:  # sample space
        # fixed = sample, moving = atlas ref -> annotation warped onto sample grid.
        transform_parameters = run_registration(
            atlas_image=ref, moving_image=sample,
            parameter_lists=parameter_lists, output_directory=None, filter_images=True,
        )
        print(f"elastix registration: {time.time() - t0:.0f}s")
        ann = np.asarray(atlas.annotation).astype(np.uint32)
        warped = np.asarray(transform_annotation_image(ann, transform_parameters))
        dst = out / f"registered_atlas_elastix_{suffix}.tiff"
        tifffile.imwrite(dst, warped.astype(np.uint32))
        print(f"wrote {dst}  shape={warped.shape}  (atlas in SAMPLE space)")
        print(
            f"COMPARE in sample space: overlay on {reg / 'downsampled.tiff'} vs "
            f"brainreg's {reg / 'registered_atlas.tiff'}"
        )

    # persist the transform (one file per map; single-string overload works)
    for i in range(transform_parameters.GetNumberOfParameterMaps()):
        itk.ParameterObject.WriteParameterFile(
            transform_parameters.GetParameterMap(i),
            str(out / f"TransformParameters.{args.space}.{suffix}.{i}.txt"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
