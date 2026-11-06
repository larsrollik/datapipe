"""brainreg registration runner for serial-section samples.

Wraps ``brainreg`` with the pre-processing it needs: locate the sample, pull the
registration channel + voxel sizes (from the recipe), and run brainreg with the
lab defaults (atlas ``allen_mouse_10um``, orientation ``psr``), writing the output
**next to the stitched dirs** (``<sample>/registration``). The source is only read.

``brainreg`` is resolved from ``--brainreg`` / ``$BRAINREG`` / PATH and is *not* a
Python dependency — run this in whatever env has brainreg installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from datapipe.models.serialsection import SerialSectionSample

DEFAULT_ATLAS = "allen_mouse_10um"
# psr confirmed correct for the lab's brainsaw scans (scan P->A, dorsal up; the
# mediolateral axis is right-origin). psl mirrors the hemispheres — see datapipe docs.
DEFAULT_ORIENTATION = "psr"
DEFAULT_CHANNEL = "3"  # autofluorescence / registration channel (green)
OUTPUT_DIRNAME = "registration"

#: what brainreg leaves behind when it finishes. A crashed run leaves the directory, its log and a
#: niftyreg/ subdirectory, which a presence test reads as done.
REGISTRATION_OUTPUTS = (
    "brainreg.json",
    "registered_atlas.tiff",
    "registered_hemispheres.tiff",
    "boundaries.tiff",
    "volumes.csv",
)


def output_state(output_dir: Path | str) -> tuple[str, list[str]]:
    """('absent' | 'partial' | 'complete', missing_files) for a registration output directory.

    The distinction is the whole point. Skipping a COMPLETE output is idempotence, and re-running a
    cohort should not fail on the samples already done. Skipping a PARTIAL one hides a registration
    that never happened: t013 crashed in the atlas load, left brainreg.json, a log and niftyreg/,
    and the re-run then skipped it and exited 0, so the batch system recorded a success for a brain
    with no registered atlas.
    """
    d = Path(output_dir)
    if not d.is_dir():
        return "absent", list(REGISTRATION_OUTPUTS)
    found = {f.name for f in d.rglob("*") if f.is_file()}
    missing = [f for f in REGISTRATION_OUTPUTS if f not in found]
    if not found:
        return "absent", missing
    return ("complete", []) if not missing else ("partial", missing)


class RegistrationError(RuntimeError):
    """Raised when a sample can't be registered or brainreg is unavailable."""


def resolve_brainreg(explicit: str | None = None) -> str:
    """Find the brainreg executable (arg > $BRAINREG > PATH); fail loud if absent."""
    exe = explicit or os.environ.get("BRAINREG") or shutil.which("brainreg")
    if not exe:
        raise RegistrationError(
            "brainreg not found — install it, set $BRAINREG, or pass --brainreg "
            "(run this in the conda/venv env that has brainreg)"
        )
    return exe


def _fmt(v: float) -> str:
    return f"{v:g}"


@dataclass(frozen=True)
class RegistrationPlan:
    """A resolved brainreg invocation for one sample (build the cmd, then run)."""

    sample_id: str | None
    input_path: Path
    output_dir: Path
    voxel_zyx: tuple[float, float, float]
    atlas: str
    orientation: str
    additional_paths: tuple[Path, ...] = ()  # -a: extra channels, same space as main
    extra_args: tuple[
        str, ...
    ] = ()  # forwarded verbatim to brainreg (e.g. --n-free-cpus)

    def cmd(self, brainreg: str) -> list[str]:
        z, y, x = self.voxel_zyx
        cmd = [
            brainreg,
            str(self.input_path),
            str(self.output_dir),
            "-v",
            _fmt(z),
            _fmt(y),
            _fmt(x),
        ]
        if self.additional_paths:
            # -a is variadic; bound it by the following --orientation flag.
            cmd += ["-a", *[str(p) for p in self.additional_paths]]
        cmd += [
            "--orientation",
            self.orientation,
            "--atlas",
            self.atlas,
            *self.extra_args,
        ]
        return cmd


def plan_registration(
    sample: SerialSectionSample,
    *,
    channel: str = DEFAULT_CHANNEL,
    atlas: str = DEFAULT_ATLAS,
    orientation: str = DEFAULT_ORIENTATION,
    source: str = "full",  # "full" (stitched + recipe voxel) | "downsampled"
    res_micron: int = 25,
    analysis_root: Path | str | None = None,
    output_dir: Path | str | None = None,
    extra_args: tuple[str, ...] = (),
    additional_channels: tuple[str, ...] = (),
) -> RegistrationPlan:
    """Resolve input path + voxel sizes for one sample's brainreg run.

    Output goes to ``<analysis_root>/<sample_dirname>/registration`` when
    ``analysis_root`` is given — mirroring the raw sample structure so a read-only
    raw tree and the writable analysis tree overlay cleanly — otherwise in-place at
    ``<sample>/registration``.

    ``output_dir`` names the directory outright, for a tree whose layout is not
    ``<root>/<sample>/registration``: the consuming data tree files a registration as its
    own acquisition, ``derivative/<subject>/<session>__histology/<session>__brainreg``, and
    composing that from a root is not possible. It takes precedence over ``analysis_root``.
    """
    if source == "full":
        input_path = sample.stitched_channel(channel)
        if sample.voxel_um is None:
            raise RegistrationError(
                f"{sample.path.name}: no voxel size in recipe (cannot register full-res)"
            )
        voxel = sample.voxel_um
    elif source == "downsampled":
        input_path = sample.downsampled_stack(res_micron, channel)
        voxel = (float(res_micron), float(res_micron), float(res_micron))
    else:
        raise RegistrationError(f"unknown source {source!r} (use full|downsampled)")

    # additional channels: same source/space as the main channel, so the same
    # registration transform applies and they auto-load alongside it.
    additional_paths: list[Path] = []
    for ac in additional_channels:
        if source == "full":
            additional_paths.append(sample.stitched_channel(ac))
        else:
            additional_paths.append(sample.downsampled_stack(res_micron, ac))

    if output_dir is not None:
        out = Path(output_dir)
    elif analysis_root is not None:
        out = Path(analysis_root) / sample.path.name / OUTPUT_DIRNAME
    else:
        out = sample.path / OUTPUT_DIRNAME

    return RegistrationPlan(
        sample_id=sample.sample_id,
        input_path=input_path,
        output_dir=out,
        voxel_zyx=voxel,
        atlas=atlas,
        orientation=orientation,
        additional_paths=tuple(additional_paths),
        extra_args=tuple(extra_args),
    )


@dataclass(frozen=True)
class RegistrationResult:
    plan: RegistrationPlan
    status: str  # "planned" | "ok" | "failed" | "skipped_exists"
    returncode: int | None = None


def run_registration(
    plan: RegistrationPlan,
    *,
    brainreg: str,
    log_path: Path | str | None = None,
) -> int:
    """Run brainreg for a resolved plan; return its exit code.

    If ``log_path`` is given, brainreg's stdout+stderr go there (useful when running
    samples in parallel so their output doesn't interleave on the console).
    """
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = plan.cmd(brainreg)
    if log_path is not None:
        with Path(log_path).open("w") as log:
            return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd).returncode


def register_sample(
    sample: SerialSectionSample,
    *,
    brainreg: str | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    log_path: Path | str | None = None,
    **plan_kw: object,
) -> RegistrationResult:
    """Per-sample wrapper: derive everything from the recipe, then run brainreg.

    Resolves the brainreg invocation for one sample (input channel + voxel sizes
    from the recipe, lab-default atlas/orientation, output beside the stitched
    dirs). Skips if ``registration/`` already exists; dry-run plans only.
    """
    plan = plan_registration(sample, **plan_kw)  # type: ignore[arg-type]

    if plan.output_dir.exists() and any(plan.output_dir.iterdir()) and not overwrite:
        return RegistrationResult(plan, "skipped_exists")
    if dry_run:
        return RegistrationResult(plan, "planned")

    rc = run_registration(plan, brainreg=resolve_brainreg(brainreg), log_path=log_path)
    return RegistrationResult(plan, "ok" if rc == 0 else "failed", rc)
