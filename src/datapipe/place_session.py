"""File an acquisition SESSION (ephys / video / behaviour) into a data tree, identity-checked.

The sibling of `place.py`, which does this for serial-section scans, and it exists for the same
reason: the subject a session is filed under is a **claim**, and the session carries its own answer
in places the claim cannot reach. Filing without comparing them is how a subject quietly acquires
another animal's recordings -- it happened once already with a histology scan (EJT48 / EJT38), and
nothing about that failure was specific to histology.

An msw session states its subject in four independent places:

1. the session directory name              ``<subject>__<datetime>__ephys``
2. ``session_manifest.yaml``               ``session_name``, written by the acquisition software
3. every acquisition basename in it        ``<subject>__<ts>__msw__<task>`` and friends
4. ``*.settings.ephys.json``               a ``subject`` field, written by the Open Ephys controller

(1) travels with the directory and so is worth the least: renaming or copying a directory changes it
and nothing else. (2)-(4) are written at acquisition time and do not move, which is what makes the
comparison meaningful -- a session copied into the wrong subject's folder still says what it is.

The convention is that everything before the first ``__`` is the subject, in the directory name and
in every acquisition basename alike.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from datapipe.place import PlacementError, subject_aliases


def _subject_of(basename: str) -> str | None:
    """The subject token of an msw name: everything before the first ``__``."""
    head = basename.split("__", 1)[0].strip()
    return head or None


@dataclass
class SessionIdentity:
    """What a session says about itself, and which record each claim came from."""

    dir_subject: str | None = None
    manifest_subject: str | None = None
    acquisition_subjects: set[str] = field(default_factory=set)
    settings_subjects: set[str] = field(default_factory=set)
    session_name: str | None = None
    n_acquisitions: int = 0

    @property
    def names(self) -> set[str]:
        out = set(self.acquisition_subjects) | set(self.settings_subjects)
        for one in (self.dir_subject, self.manifest_subject):
            if one:
                out.add(one)
        return out

    @property
    def recorded_names(self) -> set[str]:
        """The claims that do NOT come from the directory name.

        A directory name is the one thing a mistaken copy changes, so when it is the only evidence
        there is nothing to check against and the session has to be refused rather than trusted.
        """
        out = set(self.acquisition_subjects) | set(self.settings_subjects)
        if self.manifest_subject:
            out.add(self.manifest_subject)
        return out

    @property
    def internally_consistent(self) -> bool:
        return len(self.names) <= 1


def read_session_identity(session_dir: Path | str) -> SessionIdentity:
    """Read a session's own account of which animal it is, from every record that carries one."""
    session_dir = Path(session_dir)
    ident = SessionIdentity(dir_subject=_subject_of(session_dir.name))

    manifest = session_dir / "session_manifest.yaml"
    if manifest.is_file():
        try:
            data = yaml.safe_load(manifest.read_text()) or {}
        except (OSError, yaml.YAMLError):
            data = {}
        name = data.get("session_name")
        if isinstance(name, str):
            ident.session_name = name
            ident.manifest_subject = _subject_of(name)
        acquisitions = data.get("acquisitions") or []
        ident.n_acquisitions = len(acquisitions)
        for entry in acquisitions:
            basename = (
                (entry or {}).get("basename") if isinstance(entry, dict) else None
            )
            if isinstance(basename, str):
                subject = _subject_of(basename)
                if subject:
                    ident.acquisition_subjects.add(subject)

    # The acquisition software's own record, independent of every name on disk.
    for settings in sorted(session_dir.glob("*.settings.*.json")):
        try:
            data = json.loads(settings.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        subject = data.get("subject")
        if isinstance(subject, str) and subject.strip():
            ident.settings_subjects.add(subject.strip())

    # Fall back to the directories on disk when no manifest listed them.
    if not ident.acquisition_subjects:
        for child in sorted(session_dir.iterdir()):
            if child.is_dir() and "__" in child.name:
                subject = _subject_of(child.name)
                if subject:
                    ident.acquisition_subjects.add(subject)
    return ident


def check_session_identity(
    ident: SessionIdentity, subject: str, roster: Path | str | None = None
) -> list[str]:
    """Reasons this session may not be filed under `subject`. Empty means it may."""
    problems: list[str] = []

    if not ident.recorded_names:
        problems.append(
            "the session states no subject of its own: no session_manifest.yaml, no acquisition "
            "basenames and no settings JSON carrying one. Only the directory name says what this "
            "is, and a directory name is exactly what a misfiled copy changes."
        )
        return problems

    if not ident.internally_consistent:
        problems.append(
            f"the session disagrees with itself: {sorted(ident.names)} appear across its directory "
            f"name, manifest, acquisitions and settings. These records are written by different "
            f"parts of the rig, so none can be trusted without the acquisition record."
        )

    aliases = subject_aliases(subject, roster)
    if ident.recorded_names & aliases:
        return problems

    problems.append(
        f"the session records {sorted(ident.recorded_names)}, filing was requested under "
        f"{subject!r}. Identifiers do not change, so these are different animals."
    )
    return problems


@dataclass
class SessionPlacement:
    subject: str
    session_dir: Path
    datetime_str: str
    identity: SessionIdentity
    source: Path


def plan_session_placement(
    source: Path | str,
    subject: str,
    data_root: Path | str,
    *,
    roster: Path | str | None = None,
) -> SessionPlacement:
    """Where this session belongs, refusing if it is not the animal it is claimed to be.

    The destination keeps the session's own directory name. Unlike a serial-section scan, an msw
    session is already named to the namespace convention by the acquisition software, so renaming it
    here would invent a second naming authority and the manifests inside would no longer match the
    directory around them.
    """
    source = Path(source)
    if not source.is_dir():
        raise PlacementError(f"{source} is not a directory")

    ident = read_session_identity(source)
    problems = check_session_identity(ident, subject, roster)
    if problems:
        raise PlacementError(
            f"refusing to file {source.name} under {subject!r}:\n  "
            + "\n  ".join(problems)
            + "\n  Resolve the identity against the acquisition record before filing."
        )

    parts = source.name.split("__")
    stamp = parts[1] if len(parts) > 1 else ""
    return SessionPlacement(
        subject=subject,
        session_dir=Path(data_root) / subject / source.name,
        datetime_str=stamp,
        identity=ident,
        source=source,
    )


def looks_like_session(path: Path | str) -> bool:
    """Whether this directory is an acquisition session rather than a serial-section scan.

    Used to route `datapipe place` without a mode flag: a session carries a session manifest, or
    acquisition subdirectories named to the msw convention.
    """
    path = Path(path)
    if (path / "session_manifest.yaml").is_file():
        return True
    if any(path.glob("*.settings.*.json")):
        return True
    return (
        any(child.is_dir() and "__msw__" in child.name for child in path.iterdir())
        if path.is_dir()
        else False
    )
