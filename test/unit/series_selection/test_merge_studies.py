def test_merge_adds_new_study_date(collector):
    existing = {"20240101": {"Modalities": {"CT": [{"series_a": {}}]}}}
    new = {"20240202": {"Modalities": {"PT": [{"series_b": {}}]}}}

    merged = collector._merge_studies(existing, new)

    assert set(merged.keys()) == {"20240101", "20240202"}
    assert merged["20240202"] == new["20240202"]
    assert merged["20240101"] == existing["20240101"]


def test_merge_adds_new_modality_to_existing_study(collector):
    existing = {"20240101": {"Modalities": {"CT": [{"series_a": {}}]}}}
    new = {"20240101": {"Modalities": {"PT": [{"series_b": {}}]}}}

    merged = collector._merge_studies(existing, new)

    assert merged["20240101"]["Modalities"]["CT"] == [{"series_a": {}}]
    assert merged["20240101"]["Modalities"]["PT"] == [{"series_b": {}}]


def test_merge_appends_new_series_for_existing_modality(collector):
    existing = {"20240101": {"Modalities": {"CT": [{"series_a": {"CTPath": "a.nii.gz"}}]}}}
    new = {"20240101": {"Modalities": {"CT": [{"series_b": {"CTPath": "b.nii.gz"}}]}}}

    merged = collector._merge_studies(existing, new)

    assert merged["20240101"]["Modalities"]["CT"] == [
        {"series_a": {"CTPath": "a.nii.gz"}},
        {"series_b": {"CTPath": "b.nii.gz"}},
    ]


def test_merge_does_not_duplicate_existing_series(collector):
    existing = {"20240101": {"Modalities": {"CT": [{"series_a": {"CTPath": "a.nii.gz"}}]}}}
    new = {"20240101": {"Modalities": {"CT": [{"series_a": {"CTPath": "a.nii.gz"}}]}}}

    merged = collector._merge_studies(existing, new)

    assert merged["20240101"]["Modalities"]["CT"] == [{"series_a": {"CTPath": "a.nii.gz"}}]


def test_merge_leaves_existing_untouched_when_new_is_empty(collector):
    existing = {"20240101": {"Modalities": {"CT": [{"series_a": {}}]}}}

    merged = collector._merge_studies(existing, {})

    assert merged == existing
