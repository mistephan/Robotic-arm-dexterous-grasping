"""Verify wrc_demo pins ultralytics text-encoder lookup to a stable absolute
path so demo launch never triggers a CWD-relative GitHub download.

Background
----------
YOLOE's text encoder (``mobileclip2_b.ts`` for yoloe-26l, ``mobileclip_blt.ts``
for older yoloe variants) is resolved by ultralytics as a CWD-relative
path. If the CWD doesn't already contain the file, ultralytics falls
through to a GitHub download that stalls the watcher on a slow link.

The YOLO WEIGHTS are already pinned via ``${repo}/models/...`` (absolute).
This test pins the text encoder the same way, via the monkey-patch in
``wrc_demo/__init__.py``.

Run::

    python -m pytest \\
        tests/test_mobileclip_pin.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def test_mobileclip_resolves_to_absolute_path(tmp_path, monkeypatch):
    """From any CWD, attempt_download_asset('mobileclip2_b.ts') must return
    an absolute path to a file that already exists locally — no download."""
    repo_root = Path(__file__).resolve().parents[1]
    expected = repo_root / "models" / "mobileclip2_b.ts"
    if not expected.exists():
        expected = repo_root / "mobileclip2_b.ts"
    if not expected.exists():
        pytest.skip(f"mobileclip2_b.ts not present at {expected}")

    # Run from a tmp CWD that does NOT contain mobileclip2_b.ts
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "mobileclip2_b.ts").exists()

    # Force fresh import of wrc_demo with this CWD
    sys.path.insert(0, str(repo_root / "src"))
    import wrc_demo  # noqa: F401 — trigger the patch

    from ultralytics.utils import downloads

    result = downloads.attempt_download_asset("mobileclip2_b.ts")
    assert os.path.isabs(result), f"resolved to non-absolute path: {result}"
    assert Path(result).exists(), f"resolved path doesn't exist: {result}"
    # Must NOT be a tmp-path file (would mean CWD-relative path was used)
    assert not str(result).startswith(str(tmp_path))


def test_mobileclip_pin_does_not_break_other_downloads(monkeypatch):
    """The patch only intercepts mobileclip*.ts; other downloads pass through."""
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    import wrc_demo  # noqa: F401

    from ultralytics.utils import downloads

    # Non-encoder file: must go through to the original implementation
    wrapped = downloads.attempt_download_asset
    assert wrapped.__name__ == "_attempt_download_asset_pinned", (
        "patch not installed"
    )
    # A random filename that doesn't exist anywhere should return the
    # original CWD-relative path (it'll then trigger a download via the
    # original function — that's expected behavior we're not changing).
    sentinel = "definitely_not_a_real_file_xyz123.pt"
    result = wrapped(sentinel)
    assert Path(result).name == sentinel