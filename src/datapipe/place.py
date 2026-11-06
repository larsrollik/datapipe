"""File a serial-section scan into a data tree, checking that it is the animal it is claimed to be.

Placement and identity checking are one operation here, deliberately. A scan arrives as a directory
and a subject to file it under, and the subject is a *claim* -- someone's recollection of which brain
went on the microtome. The scan itself carries the answer: the recipe records the sample ID, and the
acquisition log records the directory the microscope wrote every section into. Comparing the two
costs nothing and is the only thing standing between a mislabelled scan and a subject that quietly
acquires someone else's anatomy.

That is not hypothetical. One scan in this dataset was filed under EJT48 while its recipe and
acquisition log both say EJT38; it was found long afterwards, by which time the staging area that
would have explained it was gone. The brain it was attributed to now has no histology at all. A
check that ran at placement would have caught it while somebody still remembered the session.

So this refuses on disagreement rather than warning. A warning is a check that can be ignored, and
the moment to resolve an identity question is before the data is filed, not after something has been
built on top of it.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from datapipe.models.serialsection import SerialSectionSample, load_sample

# BakingTray writes every section into the RUN directory: "... in directory F:\EJT38\...". One run
# can hold several brains, and then that name carries each of their ids: "HL_t013_t017_FPOF10".
_ACQLOG_SAMPLE = re.compile(r"in directory\s+[A-Za-z]:\\([^\\]+)\\", re.IGNORECASE)
_ACQ_TIME = "%Y/%m/%d %H:%M:%S"


class PlacementError(Exception):
    """Raised when a scan cannot be filed as asked. Never downgraded to a warning."""


def _id_tokens(name: str) -> set[str]:
    """The identifier-shaped tokens of a name: those carrying a digit, lowercased.

    "t013" and "GPe160" say which animal this is; "HL", "sample" and "GPe" do not, so they cannot
    reconcile two names that disagree.
    """
    return {
        part.lower()
        for part in re.split(r"[^A-Za-z0-9]+", name or "")
        if part and any(ch.isdigit() for ch in part)
    }


@dataclass
class Identity:
    """What the scan says about itself, and where each claim came from."""

    recipe_id: str | None = None
    acqlog_ids: set[str] = field(default_factory=set)
    acq_start: dt.datetime | None = None

    @property
    def names(self) -> set[str]:
        out = set(self.acqlog_ids)
        if self.recipe_id:
            out.add(self.recipe_id)
        return out

    @property
    def unreconciled(self) -> set[str]:
        """Log names that cannot be reconciled with the recipe's sample ID.

        The log names the acquisition RUN, not the sample, and a run can hold several brains: a
        two-brain run writes "HL_t013_t017_FPOF10" into every section's path and is split per
        animal afterwards. A run whose name carries the recipe's own identifier corroborates it
        rather than contradicting it, so comparing the two strings for equality reports eleven of
        the fifteen GPe scans as self-contradictory when none of them is.

        Reconciliation is on identifier-shaped tokens only, the ones carrying a digit, so
        "sample_161" is reconciled with "HL_GPe_161_202" through 161 while the words they share,
        "sample", "HL" and "GPe", count for nothing. That makes this corroboration between two
        records of the same run; binding the run to an animal is the subject check below.
        """
        if not self.recipe_id:
            # Nothing to reconcile against: two different run names are then a real ambiguity.
            return set() if len(self.acqlog_ids) <= 1 else set(self.acqlog_ids)
        wanted = _id_tokens(self.recipe_id)
        return {
            name
            for name in self.acqlog_ids
            if name != self.recipe_id and not (wanted & _id_tokens(name))
        }

    @property
    def internally_consistent(self) -> bool:
        """Do the recipe and the log agree? They are written by different parts of the rig."""
        return not self.unreconciled


def read_identity(sample: SerialSectionSample) -> Identity:
    """Read the scan's own account of which animal it is, from two independent places."""
    ident = Identity(recipe_id=sample.sample_id)

    if sample.acq_datetime:
        try:
            ident.acq_start = dt.datetime.strptime(str(sample.acq_datetime), _ACQ_TIME)
        except ValueError:
            pass

    for log in sorted(sample.path.glob("acqLog*.txt")):
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        ident.acqlog_ids.update(_ACQLOG_SAMPLE.findall(text))
        if ident.acq_start is None:
            first = text.splitlines()[0] if text else ""
            try:
                ident.acq_start = dt.datetime.strptime(first.split(" -- ")[0].strip(), _ACQ_TIME)
            except (ValueError, IndexError):
                pass
    return ident


def subject_aliases(subject: str, roster: Path | str | None) -> set[str]:
    """Every name this subject is known by.

    Normally just the subject itself: identifiers do not change any more, so a scan acquired under a
    subject should name that subject, and anything else is a real disagreement.

    A roster is the LEGACY path only. Scans from before the identifier migration carry the name the
    rig used rather than the namespace id, and for those the two can only be related through the
    mapping. Passing one for current data would weaken the check for no benefit -- it can only ever
    widen what counts as a match.
    """
    names = {subject}
    if roster is None:
        return names
    data = yaml.safe_load(Path(roster).read_text()) or {}
    # ``lgcy_id`` is the roster's own field name for a subject's earlier identifier, kept as-is
    # because the rosters on disk spell it that way; renaming it here would stop reading them.
    for entry in data.get("subjects", []) or []:
        if entry.get("lgcy_id") != subject and entry.get("name") != subject:
            continue
        for key in ("lgcy_id", "name"):
            if entry.get(key):
                names.add(str(entry[key]))
        names.update(str(s) for s in entry.get("src_names", []) or [])
    return names


def check_identity(ident: Identity, subject: str, roster: Path | str | None) -> list[str]:
    """Return the reasons this scan may not be filed under `subject`. Empty means it may."""
    problems: list[str] = []

    if not ident.names:
        problems.append(
            "the scan states no sample identity: no recipe sample ID and no acquisition log naming "
            "a sample directory. There is nothing to check the subject against."
        )
        return problems

    if not ident.internally_consistent:
        problems.append(
            f"the scan disagrees with itself: the recipe names {ident.recipe_id!r} and the "
            f"acquisition log names {sorted(ident.unreconciled)}, sharing no identifier. The two "
            f"are written by different parts of the rig, so neither can be trusted without the "
            f"acquisition record."
        )

    aliases = subject_aliases(subject, roster)
    if ident.names & aliases:
        return problems

    if roster is None:
        problems.append(
            f"scan says {sorted(ident.names)}, filing was requested under {subject!r}. Identifiers "
            f"do not change, so these are different animals. If this is a scan from before the "
            f"identifier migration, pass the roster that maps its rig name to a namespace id."
        )
    else:
        problems.append(
            f"scan says {sorted(ident.names)}, but {subject!r} is known as {sorted(aliases)}. "
            f"These are different animals unless the roster is wrong."
        )
    return problems


# recipe_<sample>_<YYMMDD>_<HHMMSS>.yml -- the rig stamps the recipe file when it writes it
_RECIPE_STAMP = re.compile(r"recipe_.*?_(\d{6})_(\d{6})\.ya?ml$", re.IGNORECASE)


def namespace_datetime(ident: Identity, sample: SerialSectionSample) -> tuple[str, str]:
    """The datetime for the namespace, and where it came from.

    Taken from the scan rather than its directory: a directory can be renamed, copied or staged
    under another name, whereas the recipe travels with the recording.

    The recipe FILENAME stamp is preferred over the recorded acqStartTime, even though the latter
    sounds more authoritative. Every serial-section session already in this dataset is named from the
    filename stamp, and acqStartTime runs one to two seconds earlier -- the rig writes the recipe,
    then begins acquiring. Preferring acqStartTime would be defensible in isolation and would rename
    every existing scan by a second or two, which is a worse outcome than matching the convention
    that is already there.
    """
    if sample.recipe_path is not None:
        m = _RECIPE_STAMP.search(sample.recipe_path.name)
        if m:
            return f"20{m.group(1)}_{m.group(2)}", "recipe filename"
    if ident.acq_start:
        return ident.acq_start.strftime("%Y%m%d_%H%M%S"), "recipe acqStartTime"
    m = re.search(r"(\d{8}_\d{6})", sample.path.name)
    if m:
        return m.group(1), "directory name"
    raise PlacementError(
        f"{sample.path.name}: no timestamp in the recipe filename, the recorded start time, or the "
        f"directory name. The scan cannot be given a namespace position."
    )


@dataclass
class Placement:
    subject: str
    session_dir: Path
    acq_dir: Path
    datetime_str: str
    datetime_source: str
    identity: Identity
    source: Path


def plan_placement(
    source: Path | str,
    subject: str,
    data_root: Path | str,
    *,
    roster: Path | str | None = None,
) -> Placement:
    """Work out where a scan belongs, refusing if it is not the animal it is claimed to be.

    `roster` is for scans predating the identifier migration, whose rig name differs from the
    namespace id. Current data needs none: the scan names the subject it was acquired under.
    """
    sample = load_sample(source)
    ident = read_identity(sample)

    problems = check_identity(ident, subject, roster)
    if problems:
        raise PlacementError(
            f"refusing to file {Path(source).name} under {subject!r}:\n  "
            + "\n  ".join(problems)
            + "\n  Resolve the identity against the acquisition record before filing."
        )

    stamp, stamp_src = namespace_datetime(ident, sample)
    session = Path(data_root) / subject / f"{subject}__{stamp}__histology"
    return Placement(
        subject=subject,
        session_dir=session,
        acq_dir=session / f"{subject}__{stamp}__serial_section",
        datetime_str=stamp,
        datetime_source=stamp_src,
        identity=ident,
        source=Path(source),
    )


def write_manifests(p: Placement) -> None:
    """Session and acquisition manifests, recording what the scan says about itself.

    The identity is written down even though it matched: the next person to wonder whether this scan
    belongs here should not have to re-read the acquisition log to find out.
    """
    ident_record = {
        "recipe_sample_id": p.identity.recipe_id,
        "acqlog_sample_dirs": sorted(p.identity.acqlog_ids),
        "acq_start": p.identity.acq_start.isoformat() if p.identity.acq_start else None,
        "checked_against_subject": p.subject,
    }
    session = {
        "msw_manifest_version": 1,
        "msw_namespace_version": "4.3",
        "type": "session",
        "session_name": p.session_dir.name,
        "acquisitions": [{"basename": p.acq_dir.name, "status": "complete"}],
        "metadata": {
            "subject": p.subject,
            "datetime": p.datetime_str,
            "session_type": "histology",
            "datetime_source": p.datetime_source,
            "scan_identity": ident_record,
            "placed_from": str(p.source),
        },
    }
    p.session_dir.mkdir(parents=True, exist_ok=True)
    (p.session_dir / "session_manifest.yaml").write_text(yaml.safe_dump(session, sort_keys=False))

    acq = {
        "msw_manifest_version": 1,
        "msw_namespace_version": "4.3",
        "type": "acquisition",
        "acquisition_name": p.acq_dir.name,
        "metadata": {"acq_type": "serial_section", "scan_identity": ident_record},
    }
    p.acq_dir.mkdir(parents=True, exist_ok=True)
    (p.acq_dir / "acquisition_manifest.yaml").write_text(yaml.safe_dump(acq, sort_keys=False))


def datapack_command(p: Placement, datapack: str, *, codec: str = "deflate") -> list[str]:
    """The command that moves the bytes. Placement decides where; datapack does the work."""
    return [datapack, "serial-section", str(p.source), str(p.acq_dir), "--codec", codec]


def placement_record(p: Placement) -> dict:
    return {
        "step": "placed",
        "at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tool": "datapipe.place",
        "source": str(p.source),
        "subject": p.subject,
        "session": p.session_dir.name,
        "acquisition": p.acq_dir.name,
        "identity_checked": sorted(p.identity.names),
    }


def append_step(acq_dir: Path, record: dict) -> None:
    """Append one line to the acquisition's step log. Append-only, never rewritten."""
    with (acq_dir / "_datapipe.steps.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\n")
