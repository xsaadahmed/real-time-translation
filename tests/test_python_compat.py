"""Tests for the supported-Python gate."""

import pathlib
import sys
from collections import namedtuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rtt.python_compat import REQUIRED_MINOR, supported_python

# sys.version_info's type is a structseq that CPython does not allow
# constructing directly (attempting to raises "cannot create
# 'sys.version_info' instances"). supported_python() only reads .major/.minor
# off whatever it's given, so a plain duck-typed namedtuple stands in fine.
_VersionInfo = namedtuple("_VersionInfo", "major minor micro releaselevel serial")


def test_supported_python_accepts_3_12():
    info = _VersionInfo(3, 12, 0, "final", 0)
    assert supported_python(info=info) is True


def test_supported_python_rejects_3_14():
    info = _VersionInfo(3, 14, 0, "final", 0)
    assert supported_python(info=info) is False


def test_supported_python_rejects_3_11():
    info = _VersionInfo(3, 11, 9, "final", 0)
    assert supported_python(info=info) is False


def test_required_minor_is_twelve():
    assert REQUIRED_MINOR == 12
