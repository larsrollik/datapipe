"""Serial-section (brainsaw/StitchIt) dataset model + loader.

This is the **single home** for serial-section layout/recipe knowledge — locate
the standard dirs (``stitchedImages_*``, ``downsampled_stacks/``), parse
``recipe_*.yml`` (sample ID, acquisition datetime, voxel sizes, channels), and
report completeness. Other tools (datapipe stages, and eventually datapack /
a downstream pipeline) should import this rather than re-deriving the layout.

A StitchIt sample (``<sample>/``) typically contains::

    recipe_<id>_<scandt>.yml      FINISHED
    stitchedImages_100/<ch>/      full-res stitched sections (per channel: 1,2,3,4)
    downsampled_stacks/<NNN>_micron/ds_<...>_<res>_<res>_ch0<N>_chan_<N>_<color>.tif

Voxel sizes come from the recipe's top-level ``VoxelSize: {X, Y, Z}`` (µm).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_RECIPE_GLOBS = ("recipe*.yml", "recipe*.yaml")
_STITCHED_GLOB = "stitchedImages_*"


class SerialSectionError(Exception):
    """Raised when a directory is not a parseable serial-section sample."""


@dataclass(frozen=True)
class SerialSectionSample:
    """One brainsaw/StitchIt whole-brain scan."""

    path: Path
    sample_id: str | None
    acq_datetime: str | None
    voxel_um: tuple[float, float, float] | None  # (z, y, x) µm
    recipe_path: Path | None
    finished: bool
    stitched_dir: Path | None  # e.g. stitchedImages_100
    channels: dict[str, Path]  # "1".. -> stitchedImages_100/<n>
    downsampled_res: dict[str, Path]  # "025_micron" -> dir

    def stitched_channel(self, channel: str | int) -> Path:
        """Full-res stitched channel dir, e.g. channel 3 -> stitchedImages_100/3."""
        key = str(channel)
        if key not in self.channels:
            raise SerialSectionError(
                f"{self.path.name}: no stitched channel {key!r} "
                f"(have {sorted(self.channels)})"
            )
        return self.channels[key]

    def downsampled_stack(self, res_micron: int, channel: str | int) -> Path:
        """The ds_*_ch0<channel>*.tif stack at the given resolution (µm)."""
        res_dir = self.downsampled_res.get(f"{res_micron:03d}_micron")
        if res_dir is None:
            raise SerialSectionError(
                f"{self.path.name}: no {res_micron:03d}_micron downsampled stacks "
                f"(have {sorted(self.downsampled_res)})"
            )
        matches = sorted(res_dir.glob(f"ds_*_ch{int(channel):02d}_*.tif"))
        if not matches:
            raise SerialSectionError(
                f"{self.path.name}: no ds stack for channel {channel} in {res_dir}"
            )
        return matches[0]


def _find_recipe(path: Path) -> Path | None:
    cands: list[Path] = []
    for g in _RECIPE_GLOBS:
        cands += sorted(path.glob(g))
    return cands[0] if cands else None


def _parse_voxel(recipe: Any) -> tuple[float, float, float] | None:
    if not isinstance(recipe, dict):
        return None
    vs = recipe.get("VoxelSize")
    if isinstance(vs, dict) and all(k in vs for k in ("X", "Y", "Z")):
        return (float(vs["Z"]), float(vs["Y"]), float(vs["X"]))
    # fallback: ScannerSettings (XY pixel size + plane spacing)
    ss = recipe.get("ScannerSettings")
    if isinstance(ss, dict):
        xy = ss.get("micronsPerPixel_cols")
        z = ss.get("micronsBetweenOpticalPlanes")
        if xy is not None and z is not None:
            return (float(z), float(xy), float(xy))
    return None


def _channels(stitched_dir: Path) -> dict[str, Path]:
    return {
        p.name: p
        for p in sorted(stitched_dir.iterdir())
        if p.is_dir() and p.name.isdigit()
    }


def is_sample(path: Path) -> bool:
    """True if ``path`` looks like a serial-section sample (recipe + stitched dir)."""
    path = Path(path)
    return (
        path.is_dir()
        and _find_recipe(path) is not None
        and any(path.glob(_STITCHED_GLOB))
    )


def load_sample(path: Path | str) -> SerialSectionSample:
    """Load one serial-section sample directory."""
    path = Path(path)
    recipe_path = _find_recipe(path)
    sample_id = acq = None
    voxel = None
    if recipe_path is not None:
        try:
            recipe = yaml.safe_load(recipe_path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise SerialSectionError(f"{path.name}: recipe unparseable: {exc}") from exc
        if isinstance(recipe, dict):
            sample = recipe.get("sample", {})
            acqd = recipe.get("Acquisition", {})
            sample_id = sample.get("ID") if isinstance(sample, dict) else None
            acq = acqd.get("acqStartTime") if isinstance(acqd, dict) else None
            voxel = _parse_voxel(recipe)

    stitched_dirs = sorted(path.glob(_STITCHED_GLOB))
    stitched_dir = stitched_dirs[0] if stitched_dirs else None
    channels = _channels(stitched_dir) if stitched_dir else {}

    ds_root = path / "downsampled_stacks"
    downsampled_res = (
        {p.name: p for p in sorted(ds_root.iterdir()) if p.is_dir()}
        if ds_root.is_dir()
        else {}
    )

    return SerialSectionSample(
        path=path,
        sample_id=sample_id,
        acq_datetime=acq,
        voxel_um=voxel,
        recipe_path=recipe_path,
        finished=(path / "FINISHED").exists(),
        stitched_dir=stitched_dir,
        channels=channels,
        downsampled_res=downsampled_res,
    )


def find_samples(root: Path | str) -> list[SerialSectionSample]:
    """Find serial-section samples directly under ``root`` (or ``root`` itself)."""
    root = Path(root)
    if is_sample(root):
        return [load_sample(root)]
    return [load_sample(p) for p in sorted(root.iterdir()) if is_sample(p)]
