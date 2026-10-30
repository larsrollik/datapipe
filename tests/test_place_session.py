"""Session placement identity checks, on sessions built to be right and to be wrong."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from datapipe.place import PlacementError
from datapipe.place_session import (
    check_session_identity,
    looks_like_session,
    plan_session_placement,
    read_session_identity,
)

SUBJECT = "seq216_m1103158_308"
SESSION = f"{SUBJECT}__20261028_100628__ephys"


def _session(
    tmp_path: Path,
    dir_name: str = SESSION,
    manifest_subject: str | None = SUBJECT,
    settings_subject: str | None = SUBJECT,
    acquisition_subject: str | None = SUBJECT,
) -> Path:
    """A session directory, with each identity record independently settable."""
    root = tmp_path / dir_name
    root.mkdir(parents=True)
    if manifest_subject is not None:
        acquisitions = []
        if acquisition_subject is not None:
            acquisitions = [
                {"basename": f"{acquisition_subject}__20261028_101111__msw__sequence"},
                {"basename": f"{acquisition_subject}__20261028_101111__video_flir__v1"},
            ]
        (root / "session_manifest.yaml").write_text(
            yaml.safe_dump(
                {
                    "type": "session",
                    "session_name": f"{manifest_subject}__20261028_100628__ephys",
                    "acquisitions": acquisitions,
                }
            )
        )
    if settings_subject is not None:
        (root / f"{dir_name}.settings.ephys.json").write_text(
            json.dumps({"subject": settings_subject, "acquisition_extension": "ephys"})
        )
    return root


def test_a_consistent_session_is_placed_keeping_its_own_name(tmp_path):
    placement = plan_session_placement(_session(tmp_path), SUBJECT, tmp_path / "data")
    assert placement.session_dir == tmp_path / "data" / SUBJECT / SESSION
    assert placement.datetime_str == "20261028_100628"
    assert placement.identity.n_acquisitions == 2


def test_a_session_copied_into_the_wrong_subject_folder_is_refused(tmp_path):
    """The failure this check exists for: the directory was renamed, the records were not."""
    misfiled = _session(tmp_path, dir_name="seq999_m0000000_000__20261028_100628__ephys")
    with pytest.raises(PlacementError) as exc:
        plan_session_placement(misfiled, "seq999_m0000000_000", tmp_path / "data")
    assert "disagrees with itself" in str(exc.value)
    assert SUBJECT in str(exc.value)


def test_filing_under_an_unrelated_subject_is_refused(tmp_path):
    with pytest.raises(PlacementError) as exc:
        plan_session_placement(_session(tmp_path), "seq999_m0000000_000", tmp_path / "data")
    assert "different animals" in str(exc.value)


def test_a_session_with_only_a_directory_name_is_refused(tmp_path):
    """A name is not evidence: it is the one thing a mistaken copy changes."""
    bare = _session(tmp_path, manifest_subject=None, settings_subject=None)
    problems = check_session_identity(read_session_identity(bare), SUBJECT)
    assert problems and "states no subject of its own" in problems[0]


def test_the_settings_json_alone_is_enough_to_check_against(tmp_path):
    """A session whose manifest never arrived can still be verified by the controller's record."""
    partial = _session(tmp_path, manifest_subject=None)
    assert read_session_identity(partial).settings_subjects == {SUBJECT}
    assert check_session_identity(read_session_identity(partial), SUBJECT) == []


def test_one_disagreeing_acquisition_is_caught(tmp_path):
    """A session assembled from two animals' acquisitions is not a session."""
    mixed = _session(tmp_path, acquisition_subject="seq777_m1111111_111")
    problems = check_session_identity(read_session_identity(mixed), SUBJECT)
    assert problems and "disagrees with itself" in problems[0]


def test_looks_like_session_distinguishes_a_scan_from_a_session(tmp_path):
    assert looks_like_session(_session(tmp_path))
    scan = tmp_path / "sample_202"
    (scan / "stitchedImages_100").mkdir(parents=True)
    (scan / "recipe_sample_202_260514_125011.yml").write_text("sample: {ID: sample_202}\n")
    assert not looks_like_session(scan)


def test_a_missing_directory_is_refused_rather_than_guessed(tmp_path):
    with pytest.raises(PlacementError):
        plan_session_placement(tmp_path / "nope", SUBJECT, tmp_path / "data")
