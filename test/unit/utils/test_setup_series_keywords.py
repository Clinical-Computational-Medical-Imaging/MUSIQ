"""Unit tests for utils.setup_series_keywords: resolving the per-modality PRIMARY/SECONDARY/
EXCLUSION keyword lists series_selection.py uses to auto-pick series, backfilling from
config.yaml wherever a modality's keywords weren't passed explicitly.

test_entrypoint.py already exercises this indirectly (via series_selection_entrypoint's CLI
wiring), for the "no keywords -> config default" and "explicit empty list -> disable filtering"
cases. These add a direct unit test of the function itself, plus the partial-specification
backfill and missing-config-file branches entrypoint-level tests can't reach.
"""

import pytest

from musiq.utils import setup_series_keywords

# Mirrors src/musiq/config.yaml's SERIES_KEYWORDS defaults.
_DEFAULT_CT_PRIMARY = ["knochen", "i30f"]
_DEFAULT_CT_SECONDARY = ["weichteil", "i70f"]


def test_no_keywords_provided_uses_config_yaml_defaults(caplog):
    keywords = setup_series_keywords()

    assert keywords["CT"]["PRIMARY"] == _DEFAULT_CT_PRIMARY
    assert keywords["CT"]["SECONDARY"] == _DEFAULT_CT_SECONDARY
    assert "No series keywords provided" in caplog.text


def test_explicit_keywords_override_the_config_default():
    keywords = setup_series_keywords(ct_primary_keywords=["custom1", "custom2"])

    assert keywords["CT"]["PRIMARY"] == ["custom1", "custom2"]


def test_unspecified_lists_are_backfilled_from_config_defaults():
    """Only CT PRIMARY is given explicitly; CT SECONDARY (and every other modality/field) must
    still be backfilled from config.yaml, not left empty."""
    keywords = setup_series_keywords(ct_primary_keywords=["custom1"])

    assert keywords["CT"]["PRIMARY"] == ["custom1"]
    assert keywords["CT"]["SECONDARY"] == _DEFAULT_CT_SECONDARY


def test_any_explicit_empty_list_disables_filtering_for_every_modality(caplog):
    """An explicitly empty keyword list (argparse nargs="*" with no values) means "select every
    series" -- and, per the function's own warning, this switches ALL modalities to
    all-empty/unfiltered, not just the one whose flag was passed empty."""
    keywords = setup_series_keywords(pt_primary_keywords=[])

    assert keywords == {
        "CT": {"PRIMARY": [], "SECONDARY": [], "EXCLUSION": []},
        "PT": {"PRIMARY": [], "SECONDARY": [], "EXCLUSION": []},
        "MR": {"PRIMARY": [], "SECONDARY": [], "EXCLUSION": []},
    }
    assert "disabling keyword filtering" in caplog.text


def test_missing_config_file_raises_file_not_found(mocker):
    mocker.patch("musiq.utils.plb.Path.exists", return_value=False)

    with pytest.raises(FileNotFoundError):
        setup_series_keywords()
