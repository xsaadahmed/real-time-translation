"""Tests for the supported-Python gate."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rtt.python_compat import REQUIRED_MINOR, supported_python


def test_supported_python_accepts_3_12():
    info = type(sys.version_info)(3, 12, 0, "final", 0)
    assert supported_python(info=info) is True


def test_supported_python_rejects_3_14():
    info = type(sys.version_info)(3, 14, 0, "final", 0)
    assert supported_python(info=info) is False


def test_supported_python_rejects_3_11():
    info = type(sys.version_info)(3, 11, 9, "final", 0)
    assert supported_python(info=info) is False


def test_required_minor_is_twelve():
    assert REQUIRED_MINOR == 12
