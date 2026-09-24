"""Unit tests for utils.agnostic_path: joining path components into a forward-slash-normalized
``pathlib.Path``, used throughout series_selection.py (e.g. for input/output_dirpath) so path
comparisons and string handling downstream don't have to care whether a component arrived with
Windows backslashes or POSIX forward slashes.
"""

import pathlib as plb

from musiq.utils import agnostic_path


def test_joins_multiple_components():
    result = agnostic_path("a", "b", "c")

    assert isinstance(result, plb.Path)
    assert result.as_posix() == "a/b/c"


def test_normalizes_windows_backslashes_to_forward_slashes():
    result = agnostic_path("a\\b", "c")

    assert result.as_posix() == "a/b/c"
    assert result.parts == ("a", "b", "c")


def test_stringifies_non_string_components():
    result = agnostic_path("a", 5, "b")

    assert result.as_posix() == "a/5/b"


def test_single_component_round_trips():
    assert agnostic_path("just_one").as_posix() == "just_one"
