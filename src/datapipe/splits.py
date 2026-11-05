"""Find acquisitions that are one recording the software broke in two, and tell them from stubs.

When the recorder crashes mid-session it is restarted, and what lands on disk is two acquisition
directories with two timestamps. Nothing in the tree says they belong together, so each is filed,
ingested and sorted as an independent session -- one electrode penetration becomes two unrelated
datasets with two unit lists.

The trap is that a short gap between two acquisitions is NOT evidence of a crash. Deliberate
protocol blocks sit minutes apart too. The discriminator is the gap between the FIRST acquisition
*ending* and the second *starting*, which needs each one's duration, not just its start time.

And once durations are in hand a second distinction appears, which matters more than the first: most
close pairs are a recording that died seconds-to-minutes in and was restarted. Those are **stubs** --
there is nothing in them to join, and appending one to a real session would corrupt the session. A
genuine split has substance on both sides. Measured on one corpus of 146 acquisitions: 3 genuine
splits against 44 stubs. A detector that reported only "close pairs" would have been wrong about 44
of 47 of them, so both classes are always reported and neither is acted on automatically.

Durations come from a stream's ``sample_numbers.npy`` via ``numpy.load(mmap_mode="r")`` -- the public
API, read for its shape alone, so nothing is loaded. Do not compute it from the file size and an
assumed header length: the header is not a fixed 128 bytes, and reaching for a private reader to
avoid that is how a sibling tool broke silently under a numpy upgrade. Two acquisitions in the
corpus hold a 65,656-byte sidecar containing a zero-length array; a size-based count calls that
8,191 samples.

WHICH stream you measure changes the stub boundary and not the splits. An acquisition has several
streams and they need not agree -- two here recorded on the auxiliary input while a probe stream is
empty. Measured on the same 146 acquisitions: the probe streams alone give 3 splits and 44 stubs,
the longest stream of any kind gives the same 3 splits and 45. So pass ``stream_hint`` for the
stream you actually intend to join (for ephys, the probe); the splits are robust to the choice, one
borderline stub is not.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

#: `<subject>__<YYYYmmdd_HHMMSS>__<kind>`, the acquisition naming this walks.
ACQ_NAME = re.compile(r"^(?P<subject>.+?)__(?P<stamp>\d{8}_\d{6})__(?P<kind>.+)$")

#: A second acquisition starting within this of the first ending is a candidate continuation.
#: Negative allows for a start stamp a little before the measured end, which rounding can produce.
GAP_MIN_S, GAP_MAX_S = -60.0, 300.0

#: Below this, a side has no usable recording in it and the pair is a stub, not a split.
MIN_HALF_S = 900.0


class SplitsError(RuntimeError):
    """Raised when the tree cannot be read as acquisitions at all."""


@dataclass(frozen=True)
class Acquisition:
    path: Path
    subject: str
    started: _dt.datetime
    duration_s: float
    kind: str

    @property
    def ended(self) -> _dt.datetime:
        return self.started + _dt.timedelta(seconds=self.duration_s)


@dataclass(frozen=True)
class Pair:
    first: Acquisition
    second: Acquisition
    gap_s: float
    verdict: str  # "split" | "stub"
    reason: str


@dataclass
class Report:
    root: Path
    scanned: int = 0
    unparsed: list[str] = field(default_factory=list)
    unreadable: list[str] = field(
        default_factory=list
    )  # no sidecar could be read: duration UNKNOWN
    empty: list[str] = field(
        default_factory=list
    )  # sidecar read, zero samples: recorded NOTHING
    splits: list[Pair] = field(default_factory=list)
    stubs: list[Pair] = field(default_factory=list)


def stream_samples(acq: Path, stream_hint: str = "") -> tuple[int, bool]:
    """``(largest sample count, any sidecar was readable)``.

    The two halves of the answer are different facts and must not be collapsed. A sidecar that
    cannot be read leaves the duration UNKNOWN; a sidecar that reads as an empty array says the
    acquisition recorded NOTHING, which is a real outcome worth reporting and worth keeping in the
    sequence -- an empty acquisition sitting between two others is itself a false start, and
    dropping it would silently re-pair its neighbours.

    Two acquisitions in the corpus this was written against are exactly this: 65,656-byte sidecars
    holding a zero-length array. Deriving the count from the file size instead reports 8,191
    samples for them, which is how a size-based duration invents a quarter-second of data that was
    never recorded.
    """
    import numpy as np

    best = 0
    readable = False
    for sidecar in acq.rglob("sample_numbers.npy"):
        if stream_hint and stream_hint not in str(sidecar):
            continue
        try:
            arr = np.load(sidecar, mmap_mode="r")
        except Exception:  # noqa: S112, BLE001 - not swallowed: an acquisition whose sidecars all
            # fail to read ends with 0 samples, and the caller lists it under `undated` rather than
            # treating it as zero-length. Logging here would duplicate that report.
            continue
        readable = True
        best = max(best, int(arr.shape[0]) if arr.shape else 0)
    return best, readable


def read_acquisitions(
    root: Path, suffix: str, fs: float, stream_hint: str = ""
) -> tuple[list[Acquisition], Report]:
    """Every `*__<suffix>` directory under ``root``, with its start time and measured duration."""
    root = Path(root)
    if not root.is_dir():
        raise SplitsError(f"not a directory: {root}")
    report = Report(root=root)
    out: list[Acquisition] = []
    for acq in sorted(root.glob(f"*/*/*__{suffix}")):
        if not acq.is_dir():
            continue
        report.scanned += 1
        m = ACQ_NAME.match(acq.name)
        if not m:
            report.unparsed.append(acq.name)
            continue
        n, readable = stream_samples(acq, stream_hint)
        if not readable:
            report.unreadable.append(acq.name)
            continue
        if n == 0:
            # Recorded nothing. Kept in the sequence at duration 0 -- see stream_samples.
            report.empty.append(acq.name)
        out.append(
            Acquisition(
                path=acq,
                subject=m["subject"],
                # Naive on purpose (ruff DTZ007). The stamp carries no zone, and these times are
                # used ONLY to test adjacency between two acquisitions written by the same machine
                # minutes apart -- a difference, where a missing zone cancels. No recorded data is
                # ever aligned to a wall clock here; that is what the sync signals are for.
                started=_dt.datetime.strptime(m["stamp"], "%Y%m%d_%H%M%S"),  # noqa: DTZ007
                duration_s=n / fs,
                kind=m["kind"],
            )
        )
    return out, report


def classify(
    acqs: list[Acquisition], min_half_s: float = MIN_HALF_S
) -> tuple[list[Pair], list[Pair]]:
    """Consecutive same-subject acquisitions whose gap says "continuation", split from stub."""
    by_subject: dict[str, list[Acquisition]] = defaultdict(list)
    for a in acqs:
        by_subject[a.subject].append(a)

    splits: list[Pair] = []
    stubs: list[Pair] = []
    for items in by_subject.values():
        items.sort(key=lambda a: a.started)
        for first, second in pairwise(items):
            gap = (second.started - first.ended).total_seconds()
            if not GAP_MIN_S <= gap <= GAP_MAX_S:
                continue
            shorter = min(first.duration_s, second.duration_s)
            if shorter > min_half_s:
                splits.append(
                    Pair(
                        first,
                        second,
                        gap,
                        "split",
                        f"both halves over {min_half_s / 60:.0f} min; restarted {gap:.0f} s after the first ended",
                    )
                )
            else:
                stubs.append(
                    Pair(
                        first,
                        second,
                        gap,
                        "stub",
                        f"shorter side is {shorter / 60:.1f} min; a false start, not a half to join",
                    )
                )
    splits.sort(key=lambda p: (p.first.subject, p.first.started))
    stubs.sort(key=lambda p: (p.first.subject, p.first.started))
    return splits, stubs


def find_splits(
    root: Path,
    suffix: str = "pxi",
    fs: float = 30000.0,
    stream_hint: str = "",
    min_half_s: float = MIN_HALF_S,
) -> Report:
    acqs, report = read_acquisitions(root, suffix, fs, stream_hint)
    report.splits, report.stubs = classify(acqs, min_half_s)
    return report


def render(report: Report) -> str:
    lines = [f"{report.scanned} acquisitions scanned under {report.root}"]
    if report.unparsed:
        lines.append(
            f"  {len(report.unparsed)} name(s) did not parse: {report.unparsed[:3]}"
        )
    if report.unreadable:
        lines.append(
            f"  {len(report.unreadable)} with no readable sample count (duration UNKNOWN): "
            f"{report.unreadable[:3]}"
        )
    if report.empty:
        lines.append(
            f"  {len(report.empty)} recorded NOTHING (zero samples, kept in the sequence): "
            f"{report.empty[:3]}"
        )
    lines.append("")
    lines.append(
        f"GENUINE SPLITS ({len(report.splits)}) -- one recording in two parts, safe to join"
        " once drift is checked:"
    )
    for p in report.splits:
        lines.append(f"  {p.first.subject}  gap {p.gap_s:6.0f} s")
        lines.append(f"      A {p.first.duration_s / 60:7.1f} min  {p.first.path.name}")
        lines.append(
            f"      B {p.second.duration_s / 60:7.1f} min  {p.second.path.name}"
        )
    lines.append("")
    lines.append(
        f"STUBS ({len(report.stubs)}) -- false starts; EXCLUDE, never append to a session:"
    )
    for p in report.stubs:
        lines.append(
            f"  {p.first.subject}  A {p.first.duration_s / 60:6.1f} min + "
            f"B {p.second.duration_s / 60:6.1f} min  gap {p.gap_s:5.0f} s"
        )
    lines.append("")
    lines.append(
        "Nothing was changed. A split still needs its two halves checked for probe movement"
        " before they are sorted as one recording."
    )
    return "\n".join(lines)


def as_json(report: Report) -> str:
    def pair(p: Pair) -> dict:
        return {
            "subject": p.first.subject,
            "verdict": p.verdict,
            "reason": p.reason,
            "gap_s": round(p.gap_s, 1),
            "first": {
                "name": p.first.path.name,
                "path": str(p.first.path),
                "duration_s": round(p.first.duration_s, 1),
            },
            "second": {
                "name": p.second.path.name,
                "path": str(p.second.path),
                "duration_s": round(p.second.duration_s, 1),
            },
        }

    return json.dumps(
        {
            "root": str(report.root),
            "scanned": report.scanned,
            "unparsed": report.unparsed,
            "unreadable": report.unreadable,
            "empty": report.empty,
            "splits": [pair(p) for p in report.splits],
            "stubs": [pair(p) for p in report.stubs],
        },
        indent=2,
    )
