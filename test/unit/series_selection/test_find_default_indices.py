import pytest


def _series(modality, desc, **extra):
    return {"Modality": modality, "SeriesDescription": desc, **extra}


@pytest.fixture()
def empty_keywords():
    return {
        "CT": {"PRIMARY": [], "SECONDARY": [], "EXCLUSION": []},
        "PT": {"PRIMARY": [], "SECONDARY": [], "EXCLUSION": []},
        "MR": {"PRIMARY": [], "SECONDARY": [], "EXCLUSION": []},
    }


@pytest.fixture()
def none_keywords():
    return {
        "CT": {"PRIMARY": None, "SECONDARY": None, "EXCLUSION": None},
        "PT": {"PRIMARY": [], "SECONDARY": [], "EXCLUSION": []},
        "MR": {"PRIMARY": [], "SECONDARY": [], "EXCLUSION": []},
    }


def test_no_keywords_selects_all_series(make_collector, empty_keywords):
    collector = make_collector(series_keywords=empty_keywords)
    series_list = [_series("CT", "series a"), _series("CT", "series b"), _series("PT", "series c")]

    indices, should_flag = collector.find_default_indices(series_list)

    assert indices == [0, 1, 2]
    assert should_flag is False


def test_none_valued_keywords_selects_all_series(make_collector, none_keywords):
    collector = make_collector(series_keywords=none_keywords)
    series_list = [_series("CT", "series a"), _series("PT", "series b")]

    indices, should_flag = collector.find_default_indices(series_list)

    assert indices == [0, 1]
    assert should_flag is False


def test_primary_keyword_match_is_not_flagged(collector):
    series_list = [_series("CT", "ct knochen thin slice"), _series("CT", "unrelated series")]

    indices, should_flag = collector.find_default_indices(series_list)

    assert indices == [0]
    assert should_flag is False


def test_unique_secondary_keyword_match_is_not_flagged(collector):
    """secondary_used is only set when a tie among duplicate descriptions is broken; a single,
    unambiguous secondary match does not flag the study by itself."""
    series_list = [_series("CT", "ct weichteil"), _series("CT", "unrelated series")]

    indices, should_flag = collector.find_default_indices(series_list)

    assert indices == [0]
    assert should_flag is True


def test_duplicate_secondary_matches_are_flagged(mocker, collector):
    """Two series sharing a description that only matches SECONDARY keywords: the best-of-duplicates
    tie-break sets secondary_used, which flags the study."""
    series_list = [
        _series("CT", "ct weichteil", SeriesPath="path0"),
        _series("CT", "ct weichteil", SeriesPath="path1"),
    ]
    mocker.patch.object(collector, "get_number_of_slices", side_effect=lambda p: {"path0": 10, "path1": 20}[p])

    indices, should_flag = collector.find_default_indices(series_list)

    assert indices == [1]
    assert should_flag is True


def test_no_match_returns_empty_indices_and_is_flagged(collector):
    series_list = [_series("CT", "unrelated series")]

    indices, should_flag = collector.find_default_indices(series_list)

    assert indices == []
    assert should_flag is True


def test_exclusion_keyword_skips_series_entirely(collector):
    # "nac motion free" is both a PT exclusion keyword; ensure it never lands in preselected indices.
    # The PT branch always ranks candidates by slice count (even a single match), so NumSlices
    # must be present to avoid a real get_number_of_slices() filesystem lookup.
    series_list = [_series("PT", "nac motion free"), _series("PT", "wb ctac", NumSlices=1)]

    indices, should_flag = collector.find_default_indices(series_list)

    assert indices == [1]
    assert should_flag is False


def test_empty_series_description_is_skipped(collector):
    series_list = [_series("CT", ""), _series("CT", "knochen")]

    indices, should_flag = collector.find_default_indices(series_list)

    assert indices == [1]
    assert should_flag is False


def test_modality_without_configured_keywords_is_skipped(collector):
    series_list = [_series("SC", "knochen"), _series("CT", "knochen")]

    indices, should_flag = collector.find_default_indices(series_list)

    assert indices == [1]
    assert should_flag is False


def test_duplicate_description_picks_series_with_most_slices(mocker, collector):
    series_list = [
        _series("CT", "knochen", SeriesPath="path0"),
        _series("CT", "knochen", SeriesPath="path1"),
    ]

    def fake_slices(series_path):
        return {"path0": 50, "path1": 200}[series_path]

    mocker.patch.object(collector, "get_number_of_slices", side_effect=fake_slices)

    indices, should_flag = collector.find_default_indices(series_list)

    assert indices == [1]
    assert should_flag is False
    assert series_list[0]["NumSlices"] == 50
    assert series_list[1]["NumSlices"] == 200


def test_duplicate_description_reuses_precomputed_num_slices(mocker, collector):
    """If NumSlices was already computed for a series, get_number_of_slices must not be called again for it."""
    series_list = [
        _series("CT", "knochen", SeriesPath="path0", NumSlices=5),
        _series("CT", "knochen", SeriesPath="path1"),
    ]

    count_calls = mocker.patch.object(collector, "get_number_of_slices", return_value=99)

    indices, _ = collector.find_default_indices(series_list)

    count_calls.assert_called_once_with("path1")
    assert indices == [1]


def test_pt_single_series_per_study_ranks_by_keyword_priority_then_slices(mocker, collector):
    """Exactly one PT series is ever selected per study: among several PRIMARY matches, the
    earliest keyword in PRIMARY order wins over a later one regardless of slice count, and ties
    on keyword rank are broken by the most slices."""
    series_list = [
        _series("PT", "qc fx", SeriesPath="path0"),  # matches PRIMARY[1] -> lower priority
        _series("PT", "pet gk ctac", SeriesPath="path1"),  # matches PRIMARY[0], fewer slices
        _series("PT", "pet gk ctac", SeriesPath="path2"),  # matches PRIMARY[0], more slices -> wins
    ]
    mocker.patch.object(
        collector, "get_number_of_slices", side_effect=lambda p: {"path0": 1, "path1": 5, "path2": 50}[p]
    )

    indices, should_flag = collector.find_default_indices(series_list)

    assert indices == [2]
    assert should_flag is False


@pytest.fixture()
def pt_primary_and_secondary_keywords():
    return {"PT": {"PRIMARY": ["wb ctac"], "SECONDARY": ["delayed pet"], "EXCLUSION": []}}


def test_pt_second_match_after_primary_resolved_is_skipped(mocker, make_collector, pt_primary_and_secondary_keywords):
    """Once a study's single PT slot is filled from a PRIMARY match, a second PT series matching
    only a SECONDARY keyword must not be appended as a further selection."""
    collector = make_collector(series_keywords=pt_primary_and_secondary_keywords)
    series_list = [
        _series("PT", "wb ctac", NumSlices=1),
        _series("PT", "delayed pet", NumSlices=1),
    ]

    indices, should_flag = collector.find_default_indices(series_list)

    assert indices == [0]
    assert should_flag is False


def test_pt_secondary_only_match_flags_study(mocker, make_collector, pt_primary_and_secondary_keywords):
    """A study whose only PT series matches a SECONDARY (not PRIMARY) keyword is still selected,
    but marks secondary_used so the study gets flagged for review."""
    collector = make_collector(series_keywords=pt_primary_and_secondary_keywords)
    series_list = [_series("PT", "delayed pet", NumSlices=1)]

    indices, should_flag = collector.find_default_indices(series_list)

    assert indices == [0]
    assert should_flag is True
