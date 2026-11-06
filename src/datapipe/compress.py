"""Compress stage — delegate to the datapack CLI (no datapack import).

datapack owns all compression; this stage just invokes its dataset-aware drivers
per sample (consistent with how datapipe shells out to brainreg/napari from their
own envs, keeping datapipe's env lean). For one sample it compresses:

- the **raw acquisition** tree (``stitchedImages_*``/``downsampled_stacks`` + metadata)
  via ``datapack serial-section`` — in-place lossless recompress by default
  (``--archive-raw`` to bundle the raw dirs into zips instead);
- the **analysis** tree (``registration/`` + ``manual_segmentation/``), if present,
  via ``datapack histology tree`` — in-place lossless recompress, browsable, with
  ``.npy`` tracks / JSON copied verbatim.

Both write to ``<dest>/<sample>/…`` so the compressed raw + derived land in one tree
(ready for the finalize stage). The source is only read.

Resolve the datapack CLI via ``--datapack`` / ``$DATAPACK`` / PATH / the datapack venv.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

#: Last resort after an explicit path, $DATAPACK and PATH have all come up empty. A bare name,
#: so the failure reads as "datapack not found" rather than naming one machine's virtualenv.
DEFAULT_DATAPACK = "datapack"


class CompressError(RuntimeError):
    """Raised when datapack can't be resolved or a compress sub-step fails."""


def resolve_datapack(explicit: str | None = None) -> str:
    """Find the datapack CLI (arg > $DATAPACK > PATH)."""
    cand = explicit or os.environ.get("DATAPACK") or shutil.which("datapack") or DEFAULT_DATAPACK
    if Path(cand).exists():
        return str(cand)
    found = shutil.which(cand)
    if found:
        return found
    raise CompressError(
        f"datapack CLI not found ({cand!r}) — install datapack, set $DATAPACK, or "
        "pass --datapack"
    )


def compress_commands(
    datapack: str,
    sample: Path | str,
    dest: Path | str,
    *,
    analysis_root: Path | str | None = None,
    archive_raw: bool = False,
    codec: str = "deflate",
) -> list[tuple[str, list[str]]]:
    """Build the datapack command(s) for one sample. Returns (label, argv) pairs."""
    sample = Path(sample)
    dst_sample = Path(dest) / sample.name
    cmds: list[tuple[str, list[str]]] = []

    raw = [datapack, "serial-section", str(sample), str(dst_sample), "--codec", codec]
    if archive_raw:
        raw.append("--archive-raw")
    cmds.append(("raw", raw))

    if analysis_root is not None:
        ana = Path(analysis_root) / sample.name
        if ana.is_dir():
            cmds.append(
                ("analysis",
                 [datapack, "histology", "tree", str(ana), str(dst_sample), "--codec", codec])
            )
    return cmds
