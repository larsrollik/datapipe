"""Per-sample manual-segmentation launcher (napari + brainglobe-segmentation).

Segmentation is inherently manual (a napari plugin — a human draws regions/tracks),
so this stage doesn't automate it: it opens napari with the brainglobe-segmentation
widget **preloaded** on a sample's brainreg output, so you start drawing immediately.
Output saves to ``manual_segmentation/`` next to the registration. It's the manual
gate: GUI, workstation, not parallelizable.

napari + brainglobe-segmentation live in a separate env (the uv ``~/.venvs/brainglobe``,
no conda); resolve its python via ``--napari-python`` / ``$DATAPIPE_NAPARI_PYTHON`` /
the default. This module shells out to that python — it is not a dependency of datapipe.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from datapipe.register import OUTPUT_DIRNAME  # "registration" — keep dir name in sync

DEFAULT_NAPARI_PYTHON = "~/.venvs/brainglobe/bin/python"

# Runs in the brainglobe env. Mirrors what the widget's "load brainreg directory"
# file-dialog does (get_brainreg_directory), minus the dialog — preloads directly.
_LAUNCH_SCRIPT = """\
import sys
from pathlib import Path
import napari
from brainglobe_segmentation.segment import SegmentationWidget

reg_dir = Path(sys.argv[1])
atlas_space = sys.argv[2] == "atlas"

viewer = napari.Viewer(title="datapipe segment - " + reg_dir.parent.name)
widget = SegmentationWidget(viewer)
viewer.window.add_dock_widget(widget, name="brainglobe-segmentation", area="right")

widget.directory = reg_dir
widget.atlas_space = atlas_space
widget.plugin = (
    "brainglobe-napari-io.brainreg_read_dir_atlas_space"
    if atlas_space
    else "brainglobe-napari-io.brainreg_read_dir"
)
widget.load_brainreg_directory()

napari.run()
"""


class SegmentError(RuntimeError):
    """Raised when the sample isn't registered or napari can't be resolved."""


def resolve_napari_python(explicit: str | None = None) -> str:
    """Find the python that has napari + brainglobe-segmentation."""
    cand = explicit or os.environ.get("DATAPIPE_NAPARI_PYTHON") or DEFAULT_NAPARI_PYTHON
    path = Path(cand).expanduser()
    if not path.exists():
        raise SegmentError(
            f"napari python not found at {path} — install napari + "
            "brainglobe-segmentation in a uv env and pass --napari-python / set "
            "$DATAPIPE_NAPARI_PYTHON"
        )
    return str(path)


def registration_dir(sample: Path | str, analysis_root: Path | str | None = None) -> Path:
    """Where this sample's brainreg output lives (mirrors register's layout)."""
    sample = Path(sample)
    base = Path(analysis_root) / sample.name if analysis_root else sample
    return base / OUTPUT_DIRNAME


def resolve_registration(
    sample: Path | str, analysis_root: Path | str | None = None
) -> Path:
    """Resolve the brainreg output dir, accepting either form:

    - a path that is *already* a brainreg output (contains ``brainreg.json``) — used as-is;
    - a sample dir + ``analysis_root`` (root) → ``<analysis_root>/<sample>/registration``;
    - a sample dir with in-place output → ``<sample>/registration``.
    """
    path = Path(sample)
    if (path / "brainreg.json").exists():
        return path
    return registration_dir(path, analysis_root)


def build_cmd(napari_python: str, reg_dir: Path, *, atlas_space: bool) -> list[str]:
    return [
        napari_python, "-c", _LAUNCH_SCRIPT,
        str(reg_dir), "atlas" if atlas_space else "sample",
    ]


def launch_segment(
    sample: Path | str,
    *,
    analysis_root: Path | str | None = None,
    atlas_space: bool = True,
    napari_python: str | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Open napari + brainglobe-segmentation preloaded on the sample's registration.

    Loads in **atlas space by default** — tracing happens in atlas space so tracks
    are born in the reference frame (no sample→atlas transform needed). Pass
    ``atlas_space=False`` only for legacy sample-space work.
    """
    reg = resolve_registration(sample, analysis_root)
    if not (reg / "brainreg.json").exists():
        raise SegmentError(
            f"{reg} is not a brainreg output (no brainreg.json). Pass either the "
            "registration dir directly, or a sample dir with --analysis-root <ROOT> "
            "(the analysis ROOT, not the per-sample registration path) — or register first."
        )
    cmd = build_cmd(resolve_napari_python(napari_python), reg, atlas_space=atlas_space)
    if not dry_run:
        subprocess.run(cmd)  # blocks until napari is closed (the manual gate)
    return cmd
