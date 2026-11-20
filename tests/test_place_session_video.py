"""Placing a packed session whose video is a raw-H.264 remux next to a HEVC re-encode.

After packing, a session holds two kinds of ``.mp4``: camera ensemble ``.h264`` streams
remuxed without re-encoding (with a datapack manifest record carrying the frame rate,
the coded-slice hashes and the verdict), and FLIR ``.avi`` video re-encoded to HEVC.
Placement files the session as a whole; both kinds, the record and the frame time
files must arrive unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from datapipe.cli import main
from datapipe.compress import CompressError, resolve_datapack

SUBJECT = "subj042_m1000001_001"
STAMP = "20261028_100628"
SESSION = f"{SUBJECT}__{STAMP}__ephys"
RCE = f"{SUBJECT}__20261028_101111__video_rce__v1"
FLIR = f"{SUBJECT}__20261028_101111__video_flir__v1"
RCE_BASE = f"{RCE}.rce.pi3.20261028101112"
FLIR_BASE = f"{FLIR}.flir.cam0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _session_shell(root: Path) -> Path:
    """A session directory with its identity records and two video acquisitions."""
    session = root / SESSION
    (session / RCE).mkdir(parents=True)
    (session / FLIR).mkdir(parents=True)
    (session / "session_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "type": "session",
                "session_name": SESSION,
                "acquisitions": [{"basename": RCE}, {"basename": FLIR}],
            }
        )
    )
    (session / f"{SESSION}.settings.ephys.json").write_text(
        json.dumps({"subject": SUBJECT, "acquisition_extension": "ephys"})
    )
    # frame times stay next to the video; neither the remux nor placement touches them
    (session / RCE / f"{RCE_BASE}.ttl_out.npz").write_bytes(b"ttl-out")
    (session / FLIR / f"{FLIR_BASE}.timestamps.csv").write_text("0,0.0\n1,0.033\n")
    return session


def _remux_record(mp4: Path, src_name: str) -> dict:
    """A datapack manifest record of a verified remux, as the remux mode writes it."""
    slice_hash = hashlib.sha256(b"coded slices").hexdigest()
    details = {
        "in_frames": 48,
        "out_frames": 48,
        "size_ratio": 1.0004,
        "fps": 100.0,
        "fps_tag": 100.0,
        "slice_sha256_h264": slice_hash,
        "slice_sha256_mp4": slice_hash,
        "slice_hash_match": True,
        "failed_checks": [],
        "verdict": "match",
        "source_deletable": True,
    }
    return {
        "source_path": str(mp4.with_name(src_name)),
        "source_size": mp4.stat().st_size,
        "modality": "video",
        "engine": "mp4box-remux",
        "params": {"mode": "remux", "remuxer": "mp4box", "fps": 100.0},
        "output_path": str(mp4),
        "output_size": mp4.stat().st_size,
        "lossless": True,
        "verification": {
            "ok": True,
            "method": "framecount+size+fps+slice_sha256",
            "details": details,
        },
        "ready_to_offload": True,
        "extra": details,
    }


def _place(session: Path, data_root: Path) -> Path:
    assert main(["place", str(session), SUBJECT, str(data_root)]) == 0
    placed = data_root / SUBJECT / SESSION
    assert placed.is_dir()
    assert not session.exists()  # a session is moved, not copied
    return placed


def _assert_placed_unchanged(before: dict[str, str], placed: Path) -> None:
    after = {
        str(p.relative_to(placed)): _sha256(p)
        for p in sorted(placed.rglob("*"))
        if p.is_file()
    }
    assert after == before


def _snapshot(session: Path) -> dict[str, str]:
    return {
        str(p.relative_to(session)): _sha256(p)
        for p in sorted(session.rglob("*"))
        if p.is_file()
    }


def test_session_with_remuxed_and_hevc_video_is_placed(tmp_path: Path) -> None:
    session = _session_shell(tmp_path / "incoming")
    remuxed = session / RCE / f"{RCE_BASE}.video.h264.mp4"
    remuxed.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"h264-remux" * 64)
    hevc = session / FLIR / f"{FLIR_BASE}.mp4"
    hevc.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"hevc-encode" * 64)
    (session / RCE / f"{RCE_BASE}.debug_acq_meta.yaml").write_text(
        "configs:\n  settings:\n    framerate: 100\n"
    )
    record = _remux_record(remuxed, f"{RCE_BASE}.video.h264")
    (session / RCE / "datapack_manifest.jsonl").write_text(json.dumps(record) + "\n")
    before = _snapshot(session)

    placed = _place(session, tmp_path / "data")

    _assert_placed_unchanged(before, placed)
    [line] = (placed / RCE / "datapack_manifest.jsonl").read_text().splitlines()
    kept = json.loads(line)
    assert kept["extra"]["verdict"] == "match"
    assert kept["extra"]["slice_sha256_h264"] == kept["extra"]["slice_sha256_mp4"]
    assert kept["params"]["fps"] == 100.0


# --- the same with real video packed by the datapack CLI -----------------------


def _datapack_with_remux() -> str | None:
    try:
        datapack = resolve_datapack()
    except CompressError:
        return None
    try:
        help_text = subprocess.run(
            [datapack, "video", "--help"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return datapack if "--mode" in help_text else None


def _encoders(ffmpeg: str) -> str:
    return subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout


def _tools() -> tuple[str, str, str] | None:
    datapack = _datapack_with_remux()
    ffmpeg = os.environ.get("DATAPACK_FFMPEG") or shutil.which("ffmpeg")
    if not (datapack and ffmpeg and shutil.which("MP4Box")):
        return None
    enc = _encoders(ffmpeg)
    h264 = next((e for e in ("libx264", "libopenh264") if e in enc), None)
    if h264 is None or not ("libx265" in enc or "hevc_nvenc" in enc):
        return None
    return datapack, ffmpeg, h264


def test_session_packed_by_datapack_is_placed(tmp_path: Path) -> None:
    tools = _tools()
    if tools is None:
        pytest.skip(
            "needs the datapack CLI with the remux mode, ffmpeg with H.264 and HEVC "
            "encoders, and MP4Box"
        )
    datapack, ffmpeg, h264_encoder = tools
    session = _session_shell(tmp_path / "incoming")

    h264 = session / RCE / f"{RCE_BASE}.video.h264"
    rate = ["-qp", "8"] if h264_encoder == "libx264" else ["-b:v", "20M"]
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-f", "lavfi"]
        + ["-i", "testsrc=size=320x240:rate=25,noise=alls=40:allf=t"]
        + ["-frames:v", "48", "-c:v", h264_encoder, *rate, "-bf", "0"]
        + ["-pix_fmt", "yuv420p", "-f", "h264", str(h264)],
        check=True,
    )
    (session / RCE / f"{RCE_BASE}.debug_acq_meta.yaml").write_text(
        "configs:\n  camera_config:\n    framerate: null\n  settings:\n"
        "    framerate: 100\n"
    )
    avi = session / FLIR / f"{FLIR_BASE}.avi"
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-f", "lavfi"]
        + ["-i", "testsrc2=size=320x240:rate=30:duration=1"]
        + ["-c:v", "mjpeg", "-q:v", "3", str(avi)],
        check=True,
    )

    report = session / RCE / "datapack_manifest.jsonl"
    remuxed = h264.with_name(h264.name + ".mp4")
    hevc = avi.with_suffix(".mp4")
    subprocess.run(
        [datapack, "video", str(h264), str(remuxed), "--report", str(report)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [datapack, "video", str(avi), str(hevc), "--allow-cpu"]
        + ["--report", str(session / FLIR / "datapack_manifest.jsonl")],
        check=True,
        capture_output=True,
    )
    # the verified remux makes the raw stream deletable; the caller deletes it
    [rec] = [json.loads(x) for x in report.read_text().splitlines()]
    assert rec["extra"]["verdict"] == "match" and rec["ready_to_offload"]
    h264.unlink()
    avi.unlink()
    before = _snapshot(session)

    placed = _place(session, tmp_path / "data")

    _assert_placed_unchanged(before, placed)
    [flir_rec] = [
        json.loads(x)
        for x in (placed / FLIR / "datapack_manifest.jsonl").read_text().splitlines()
    ]
    assert rec["params"]["mode"] == "remux" and rec["lossless"] is True
    assert flir_rec["params"]["mode"] == "encode" and flir_rec["lossless"] is False
