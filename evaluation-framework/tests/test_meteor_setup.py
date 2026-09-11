"""
Tests for meteor_setup.py's decision logic, using mocks so these run
deterministically in CI regardless of whether the test runner actually
has network access to nltk's data server (it usually won't).
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

from metrics.common import meteor_setup


def test_main_returns_zero_when_already_available():
    with patch.object(meteor_setup, "_check_available", return_value=True):
        with patch("sys.argv", ["aixpert-setup-meteor"]):
            assert meteor_setup.main() == 0


def test_main_detects_permission_warning_and_fails_with_guidance(capsys):
    with patch.object(meteor_setup, "_check_available", return_value=False):
        with patch.object(
            meteor_setup, "_attempt_download",
            return_value=(True, ["will not authorize the non-private download directory '/home/x'"]),
        ):
            with patch("sys.argv", ["aixpert-setup-meteor"]):
                exit_code = meteor_setup.main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--nltk-data-dir" in captured.err
    assert "NLTK_DATA" in captured.err


def test_main_succeeds_on_clean_download(capsys):
    with patch.object(meteor_setup, "_check_available", side_effect=[False, True]):
        with patch.object(meteor_setup, "_attempt_download", return_value=(True, [])):
            with patch("sys.argv", ["aixpert-setup-meteor"]):
                exit_code = meteor_setup.main()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "installed and verified" in captured.out


def test_main_handles_download_exception_with_clear_message(capsys):
    with patch.object(meteor_setup, "_check_available", return_value=False):
        with patch.object(meteor_setup, "_attempt_download", side_effect=ConnectionError("no route to host")):
            with patch("sys.argv", ["aixpert-setup-meteor"]):
                exit_code = meteor_setup.main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "network restriction" in captured.err
    assert "administrator" in captured.err


def test_main_handles_silent_failure_after_reported_success(capsys):
    """The real failure mode observed on a network-restricted sandbox:
    nltk.download() reports success (True) but the corpus is still not
    actually present afterward -- no exception raised, so this needs its
    own branch rather than relying on exception handling alone."""
    with patch.object(meteor_setup, "_check_available", side_effect=[False, False]):
        with patch.object(meteor_setup, "_attempt_download", return_value=(True, [])):
            with patch("sys.argv", ["aixpert-setup-meteor"]):
                exit_code = meteor_setup.main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "network restriction" in captured.err
