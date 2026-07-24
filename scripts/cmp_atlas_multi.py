#!/usr/bin/env python
"""Compare every atlas-space warped brain in a registration against the atlas template.

Measures NCC + mutual information (brain mask) of brainreg's downsampled_standard and
each elastix variant (downsampled_standard_elastix*.tiff) vs the reference template.
Higher = better aligned.

  ~/.venvs/bg-registration/bin/python cmp_atlas_multi.py <registration_dir> [atlas]
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
from brainglobe_atlasapi import BrainGlobeAtlas

reg = Path(sys.argv[1])
atlas = BrainGlobeAtlas(sys.argv[2] if len(sys.argv) > 2 else "allen_mouse_10um")
ref = np.asarray(atlas.reference)
mask = np.asarray(atlas.annotation) > 0
nmask = int(mask.sum())


def ncc(a):
    a = a[mask].astype(np.float64)
    r = ref[mask].astype(np.float64)
    a -= a.mean()
    r -= r.mean()
    return float((a * r).sum() / np.sqrt((a * a).sum() * (r * r).sum()))


def mi(a, bins=64):
    h, _, _ = np.histogram2d(a[mask].astype(np.float32), ref[mask].astype(np.float32), bins=bins)
    p = h / h.sum()
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (pa[:, None] * pb[None, :])[nz])).sum())


cands = [("brainreg", reg / "downsampled_standard.tiff")]
cands += sorted(("elastix:" + p.stem.split("elastix_")[-1], p)
                for p in (reg / "elastix").glob("downsampled_standard_elastix_*.tiff"))
old = reg / "elastix" / "downsampled_standard_elastix.tiff"
if old.exists():
    cands.append(("elastix:fine(old)", old))

print(f"mask vox {nmask:,} | vs reference template (higher = better aligned)")
print(f"{'variant':<22}{'NCC':>10}{'MI':>10}")
for name, path in cands:
    if not path.exists():
        continue
    img = tifffile.imread(path)
    print(f"{name:<22}{ncc(img):>10.4f}{mi(img):>10.4f}")
