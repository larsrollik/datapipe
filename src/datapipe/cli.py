"""datapipe CLI. First stage: ``datapipe register`` — one sample per invocation.

Per-sample by design: no top-level crawler. Each call registers exactly one
serial_section sample, so chaining/parallelism is done by the caller (a shell
loop, GNU parallel, or a SLURM array — one task per sample).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from datapipe.compress import CompressError, compress_commands, resolve_datapack

from datapipe.models.serialsection import (
    SerialSectionError,
    is_sample,
    load_sample,
)
from datapipe.place import (
    PlacementError,
    append_step,
    datapack_command,
    placement_record,
    plan_placement,
    write_manifests,
)
from datapipe.register import (
    output_state,
    DEFAULT_ATLAS,
    DEFAULT_CHANNEL,
    DEFAULT_ORIENTATION,
    RegistrationError,
    plan_registration,
    resolve_brainreg,
    run_registration,
)
from datapipe.segment import SegmentError, launch_segment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datapipe", description="Dataset-aware processing & moving pipeline."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser(
        "register",
        help="brainreg-register ONE serial_section sample (output beside stitched dirs)",
    )
    r.add_argument("sample", type=Path, help="a single serial_section sample directory")
    r.add_argument("--channel", default=DEFAULT_CHANNEL, help="autofluo/registration channel")
    r.add_argument("--atlas", default=DEFAULT_ATLAS)
    r.add_argument("--orientation", default=DEFAULT_ORIENTATION)
    r.add_argument(
        "--source", choices=["full", "downsampled"], default="full",
        help="full = stitched channel + recipe voxel; downsampled = ds stack at --res",
    )
    r.add_argument("--res", type=int, default=25, help="µm, when --source downsampled")
    r.add_argument(
        "--main-only", action="store_true",
        help="register ONLY the main channel (default: all stitched channels, so every "
        "channel transforms into atlas space and auto-loads together)",
    )
    r.add_argument(
        "--additional", nargs="*", metavar="CH", default=None,
        help="specific extra channel numbers to register (-a); default = all other channels",
    )
    r.add_argument(
        "--analysis-root", type=Path,
        help="write derived output to <analysis_root>/<sample>/registration (mirrors the "
        "raw tree so RO-raw + RW-analysis overlay cleanly); default = in-place beside the data",
    )
    r.add_argument(
        "--output", type=Path,
        help="write the registration to THIS directory, for a tree that is not "
             "<root>/<sample>/registration (e.g. the namespace's "
             "derivative/<subject>/<session>__histology/<session>__brainreg). "
             "Takes precedence over --analysis-root.",
    )
    r.add_argument("--brainreg", help="path to brainreg (or $BRAINREG / PATH)")
    r.add_argument(
        "--brainreg-extra", default="",
        help="extra args forwarded verbatim to brainreg, as a quoted string "
        "(e.g. --brainreg-extra '--n-free-cpus 8' to cap RAM/CPU)",
    )
    r.add_argument("--overwrite", action="store_true", help="re-run even if registration/ exists")
    r.add_argument("--dry-run", action="store_true", help="print the brainreg command only")
    r.set_defaults(func=_cmd_register)

    g = sub.add_parser(
        "segment",
        help="open napari + brainglobe-segmentation preloaded on ONE sample's registration",
    )
    g.add_argument(
        "sample", type=Path,
        help="a registration dir (contains brainreg.json) OR a sample dir (+ --analysis-root)",
    )
    g.add_argument(
        "--analysis-root", type=Path,
        help="analysis ROOT (not the per-sample path); resolves "
        "<analysis_root>/<sample>/registration. Ignored if the positional is already a "
        "registration dir. Default: in-place <sample>/registration",
    )
    g.add_argument(
        "--sample-space", action="store_true",
        help="segment in sample space (default: atlas space — trace in the reference "
        "frame so tracks need no sample→atlas transform)",
    )
    g.add_argument(
        "--napari-python",
        help="python with napari + brainglobe-segmentation (or $DATAPIPE_NAPARI_PYTHON)",
    )
    g.add_argument("--dry-run", action="store_true", help="print the launch command only")
    g.set_defaults(func=_cmd_segment)

    c = sub.add_parser(
        "compress",
        help="compress ONE sample's data via datapack (raw + analysis trees) -> dest/<sample>",
    )
    c.add_argument("sample", type=Path, help="a serial_section sample directory (raw)")
    c.add_argument("dest", type=Path, help="destination root (output -> dest/<sample>)")
    c.add_argument(
        "--analysis-root", type=Path,
        help="also compress <analysis_root>/<sample> (registration/segmentation)",
    )
    c.add_argument(
        "--archive-raw", action="store_true",
        help="bundle raw stitched/downsampled dirs to zips (default: in-place recompress)",
    )
    c.add_argument("--codec", choices=["deflate", "zstd"], default="deflate")
    c.add_argument("--datapack", help="path to the datapack CLI (or $DATAPACK / PATH)")
    c.add_argument("--dry-run", action="store_true", help="print the datapack commands only")
    c.set_defaults(func=_cmd_compress)

    pl = sub.add_parser(
        "place",
        help="file ONE serial_section sample into a data tree, refusing on identity mismatch",
    )
    pl.add_argument("sample", type=Path, help="a serial_section sample directory")
    pl.add_argument("subject", help="subject to file it under; checked against the scan's own record")
    pl.add_argument("data_root", type=Path, help="data tree root (<root>/<subject>/... is created)")
    pl.add_argument(
        "--roster",
        type=Path,
        help="LEGACY only: subject map relating a pre-migration rig name to a namespace id. "
             "Current scans name their own subject and need none.",
    )
    pl.add_argument("--codec", choices=["deflate", "zstd"], default="deflate")
    pl.add_argument("--datapack", help="path to the datapack CLI (or $DATAPACK / PATH)")
    pl.add_argument(
        "--dry-run", action="store_true",
        help="check identity and print the destination and command; write nothing",
    )
    pl.set_defaults(func=_cmd_place)
    return parser


def _cmd_place(args: argparse.Namespace) -> int:
    """Check, then file. The check is not a separate step that can be skipped."""
    try:
        plan = plan_placement(args.sample, args.subject, args.data_root, roster=args.roster)
    except (PlacementError, SerialSectionError) as exc:
        print(f"datapipe place: {exc}", file=sys.stderr)
        return 2

    print(f"  scan identity : {sorted(plan.identity.names)}")
    print(f"  datetime      : {plan.datetime_str}  (from {plan.datetime_source})")
    print(f"  destination   : {plan.acq_dir}")

    try:
        datapack = resolve_datapack(args.datapack)
    except CompressError as exc:
        print(f"datapipe place: {exc}", file=sys.stderr)
        return 2
    cmd = datapack_command(plan, datapack, codec=args.codec)

    if args.dry_run:
        print(f"  would run     : {shlex.join(cmd)}")
        return 0

    write_manifests(plan)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"datapipe place: datapack exited {rc}; manifests written, data not packed",
              file=sys.stderr)
        return rc
    append_step(plan.acq_dir, placement_record(plan))
    print(f"  placed        : {plan.acq_dir}")
    return 0


def _cmd_compress(args: argparse.Namespace) -> int:
    datapack = "datapack" if args.dry_run else resolve_datapack(args.datapack)
    cmds = compress_commands(
        datapack, args.sample, args.dest,
        analysis_root=args.analysis_root, archive_raw=args.archive_raw, codec=args.codec,
    )
    for label, cmd in cmds:
        print(f"[{label}] {shlex.join(cmd)}")
    if args.dry_run:
        return 0
    for label, cmd in cmds:
        if subprocess.run(cmd).returncode != 0:
            print(f"FAIL: {label} compress failed", file=sys.stderr)
            return 1
    print(f"ok: {args.sample.name} -> {args.dest / args.sample.name}")
    return 0


def _cmd_segment(args: argparse.Namespace) -> int:
    cmd = launch_segment(
        args.sample,
        analysis_root=args.analysis_root,
        atlas_space=not args.sample_space,
        napari_python=args.napari_python,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(shlex.join(cmd[:2]) + " <launch-script> " + shlex.join(cmd[3:]))
    return 0


def _cmd_register(args: argparse.Namespace) -> int:
    if not is_sample(args.sample):
        print(
            f"error: {args.sample} is not a serial_section sample "
            "(need recipe_*.yml + stitchedImages_*/)",
            file=sys.stderr,
        )
        return 2

    sample = load_sample(args.sample)
    if args.additional is not None:
        additional = tuple(args.additional)
    elif args.main_only:
        additional = ()
    else:  # default: register every other stitched channel too
        additional = tuple(c for c in sorted(sample.channels) if c != args.channel)
    try:
        plan = plan_registration(
            sample,
            channel=args.channel,
            atlas=args.atlas,
            orientation=args.orientation,
            source=args.source,
            res_micron=args.res,
            analysis_root=args.analysis_root,
            output_dir=args.output,
            extra_args=tuple(shlex.split(args.brainreg_extra)),
            additional_channels=additional,
        )
    except (RegistrationError, SerialSectionError) as exc:
        print(f"error: {sample.path.name}: {exc}", file=sys.stderr)
        return 1

    brainreg = "brainreg" if args.dry_run else resolve_brainreg(args.brainreg)
    # The point of the wrapper: show the exact args handed to brainreg.
    print(shlex.join(plan.cmd(brainreg)))
    if args.dry_run:
        return 0

    if plan.output_dir.exists() and any(plan.output_dir.iterdir()) and not args.overwrite:
        state, missing = output_state(plan.output_dir)
        if state == "complete":
            # idempotent: re-running a cohort must not fail on the samples already registered
            print(f"skip: {plan.output_dir} is already a complete registration")
            return 0
        print(
            f"REFUSE: {plan.output_dir} holds a PARTIAL registration, missing "
            f"{', '.join(missing)}. A skip here would report a registration that never ran as a "
            f"success. Clear the directory and re-run, or pass --overwrite.",
            file=sys.stderr,
        )
        return 3

    rc = run_registration(plan, brainreg=brainreg)
    if rc != 0:
        print(f"FAIL: {sample.path.name}: brainreg exited {rc}", file=sys.stderr)
        return 1
    print(f"ok: {sample.path.name} -> {plan.output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (RegistrationError, SerialSectionError, SegmentError, CompressError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
