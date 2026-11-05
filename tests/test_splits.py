"""The split/stub distinction, which is the whole point of the detector.

A detector that only reported "two acquisitions close together" would have been wrong about 44 of
47 pairs in the corpus this was written for: nearly all of them are a recording that died seconds
in and was restarted. Appending one of those to a real session would corrupt the session, so the
two classes are asserted separately here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datapipe.splits import SplitsError, find_splits, read_acquisitions

FS = 30000.0


def _acq(
    root: Path, subject: str, stamp: str, minutes: float, kind: str = "pxi"
) -> Path:
    """An acquisition with a probe stream whose sample sidecar encodes the duration."""
    acq = (
        root / subject / f"{subject}__{stamp}__session" / f"{subject}__{stamp}__{kind}"
    )
    stream = (
        acq
        / "Record Node 101"
        / "experiment1"
        / "recording1"
        / "continuous"
        / "Neuropix-PXI-100.ProbeA"
    )
    stream.mkdir(parents=True)
    np.save(
        stream / "sample_numbers.npy", np.arange(int(minutes * 60 * FS), dtype="int64")
    )
    return acq


def test_a_real_split_is_found(tmp_path):
    """Second starts ~2 min after the first ENDS, and both sides have substance."""
    _acq(tmp_path, "sub001", "20260405_120000", 30.0)  # ends 12:30:00
    _acq(tmp_path, "sub001", "20260405_123200", 100.0)  # starts 12:32:00 -> gap 120 s
    rep = find_splits(tmp_path, fs=FS)
    assert rep.scanned == 2
    assert len(rep.splits) == 1 and not rep.stubs
    p = rep.splits[0]
    assert 110 < p.gap_s < 130
    assert p.verdict == "split"


def test_a_false_start_is_a_stub_not_a_split(tmp_path):
    """One minute then a restart: nothing to join. This is the common case, by a wide margin."""
    _acq(tmp_path, "sub001", "20260405_120000", 1.0)  # ends 12:01:00
    _acq(tmp_path, "sub001", "20260405_120200", 100.0)  # gap 60 s
    rep = find_splits(tmp_path, fs=FS)
    assert not rep.splits
    assert len(rep.stubs) == 1
    assert "false start" in rep.stubs[0].reason


def test_a_long_gap_is_not_a_continuation(tmp_path):
    """Deliberate separate blocks hours apart must not be paired."""
    _acq(tmp_path, "sub001", "20260405_120000", 30.0)
    _acq(tmp_path, "sub001", "20260405_180000", 30.0)
    rep = find_splits(tmp_path, fs=FS)
    assert not rep.splits and not rep.stubs


def test_gap_is_measured_from_the_END_not_the_start(tmp_path):
    """Two starts 35 min apart are NOT close; the first runs for 30 of those minutes.

    Pairing on start times alone would call this a continuation at a 35 min "gap", and would equally
    miss a true continuation whose starts are hours apart because the first half ran for hours.
    """
    _acq(tmp_path, "sub001", "20260405_120000", 30.0)
    _acq(
        tmp_path, "sub001", "20260405_123500", 60.0
    )  # gap from end = 300 s, just inside
    rep = find_splits(tmp_path, fs=FS)
    assert len(rep.splits) == 1
    assert 290 < rep.splits[0].gap_s < 310


def test_different_subjects_are_never_paired(tmp_path):
    _acq(tmp_path, "sub001", "20260405_120000", 30.0)
    _acq(tmp_path, "sub002", "20260405_123200", 100.0)
    rep = find_splits(tmp_path, fs=FS)
    assert not rep.splits and not rep.stubs


def test_unreadable_duration_is_reported_not_guessed(tmp_path):
    """An acquisition with no sample sidecar is listed, never treated as zero-length."""
    acq = (
        tmp_path
        / "sub001"
        / "sub001__20260405_120000__session"
        / "sub001__20260405_120000__pxi"
    )
    acq.mkdir(parents=True)
    _acq(tmp_path, "sub001", "20260405_123200", 100.0)
    rep = find_splits(tmp_path, fs=FS)
    assert rep.unreadable == ["sub001__20260405_120000__pxi"]
    assert not rep.splits and not rep.stubs


def test_min_half_threshold_is_the_split_stub_boundary(tmp_path):
    _acq(tmp_path, "sub001", "20260405_120000", 10.0)
    _acq(tmp_path, "sub001", "20260405_121100", 100.0)
    assert (
        len(find_splits(tmp_path, fs=FS, min_half_s=900).stubs) == 1
    )  # 15 min -> stub
    assert (
        len(find_splits(tmp_path, fs=FS, min_half_s=300).splits) == 1
    )  # 5 min -> split


def test_missing_root_refuses(tmp_path):
    with pytest.raises(SplitsError, match="not a directory"):
        find_splits(tmp_path / "nope")


def test_samples_come_from_the_npy_shape_not_the_file_size(tmp_path):
    """Public API, read for shape only. A size-minus-128 guess breaks when the header is not 128 B."""
    acq = _acq(tmp_path, "sub001", "20260405_120000", 2.0)
    acqs, _rep = read_acquisitions(tmp_path, "pxi", FS)
    assert len(acqs) == 1
    assert abs(acqs[0].duration_s - 120.0) < 1e-6
    # The duration above is exact, and it is NOT what a size-minus-128 guess would give unless the
    # header happens to be 128 bytes. The shape is the source of truth.
    sidecar = next(acq.rglob("sample_numbers.npy"))
    guessed = (sidecar.stat().st_size - 128) // 8
    assert guessed <= int(2.0 * 60 * FS)
