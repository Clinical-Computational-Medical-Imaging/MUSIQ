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


def test_series_identity_uses_input_dirpath_basename(collector):
    series_dict = {"knochen ct": {"InputDirPath": "/raw/patient1/study1/1.2.3.4/"}}

    assert collector._series_identity(series_dict) == "1.2.3.4"


def test_series_identity_falls_back_to_key_without_input_dirpath(collector):
    series_dict = {"Missing_SeriesDesc_ABCDE": {"CTPath": "/out/CT.nii.gz"}}

    assert collector._series_identity(series_dict) == "Missing_SeriesDesc_ABCDE"


def test_series_identity_of_empty_dict_is_empty_string(collector):
    assert collector._series_identity({}) == ""


def test_merge_deduplicates_series_recorded_under_a_different_random_key_via_input_dirpath(collector):
    """A description-less series gets a fresh random `Missing_SeriesDesc_*` key each run, but its
    InputDirPath (the series' raw directory) is stable — so a re-run must still recognize it as
    the same series instead of appending a duplicate."""
    existing = {
        "20240101": {
            "Modalities": {
                "CT": [{"Missing_SeriesDesc_AAAAA": {"InputDirPath": "/raw/p1/s1/1.2.3.4", "CTPath": "/out/CT.nii.gz"}}]
            }
        }
    }
    new = {
        "20240101": {
            "Modalities": {
                "CT": [{"Missing_SeriesDesc_ZZZZZ": {"InputDirPath": "/raw/p1/s1/1.2.3.4", "CTPath": "/out/CT.nii.gz"}}]
            }
        }
    }

    merged = collector._merge_studies(existing, new)

    assert len(merged["20240101"]["Modalities"]["CT"]) == 1


def test_refresh_conversion_fields_returns_early_when_existing_inner_is_not_a_dict(collector):
    existing_series = {"pet ac": "not-a-dict"}
    new_series = {"pet ac": {"DICOM": {"SUVFactor": 0.5}}}

    collector._refresh_conversion_fields(existing_series, new_series)

    assert existing_series == {"pet ac": "not-a-dict"}


def test_refresh_conversion_fields_returns_early_when_new_dicom_is_not_a_dict(collector):
    existing_series = {"pet ac": {"DICOM": {"SUVFactor": 0.1}}}
    new_series = {"pet ac": {"DICOM": "not-a-dict"}}

    collector._refresh_conversion_fields(existing_series, new_series)

    assert existing_series["pet ac"]["DICOM"]["SUVFactor"] == 0.1


def test_refresh_conversion_fields_updates_only_allow_listed_pet_fields(collector):
    existing_series = {
        "pet ac": {
            "SULPath": "/out/SUL.nii.gz",
            "DICOM": {"SUVFactor": 0.1, "PatientWeight": 80.0},
        }
    }
    new_series = {
        "pet ac": {
            "DICOM": {
                "SUVFactor": 0.2,
                "AcquisitionTime": "121000",
                "DecayCorrectionReference": "121000",
                "RadiopharmaceuticalStartTime": "113000",
                "InjectedRadioactivity": 3e8,
                "RadionuclideHalfLife": 6588.0,
                "PatientWeight": 999.0,  # not in the allow-list: must be left untouched
            }
        }
    }

    collector._refresh_conversion_fields(existing_series, new_series)

    dicom = existing_series["pet ac"]["DICOM"]
    assert dicom["SUVFactor"] == 0.2
    assert dicom["AcquisitionTime"] == "121000"
    assert dicom["DecayCorrectionReference"] == "121000"
    assert dicom["RadiopharmaceuticalStartTime"] == "113000"
    assert dicom["InjectedRadioactivity"] == 3e8
    assert dicom["RadionuclideHalfLife"] == 6588.0
    # Untouched: not in _PET_CONVERSION_DICOM_FIELDS and a later-stage key.
    assert dicom["PatientWeight"] == 80.0
    assert existing_series["pet ac"]["SULPath"] == "/out/SUL.nii.gz"


def test_merge_refreshes_conversion_fields_of_an_already_recorded_series(collector):
    """A re-run's corrected SUVFactor/decay reference for an already-recorded series must land in
    the merged result, without dropping later-stage keys like SULPath."""
    existing = {
        "20240101": {
            "Modalities": {
                "PT": [
                    {
                        "wb ctac": {
                            "InputDirPath": "/raw/p1/s1/1.2.3.4",
                            "SULPath": "/out/SUL.nii.gz",
                            "DICOM": {"SUVFactor": 0.1},
                        }
                    }
                ]
            }
        }
    }
    new = {
        "20240101": {
            "Modalities": {"PT": [{"wb ctac": {"InputDirPath": "/raw/p1/s1/1.2.3.4", "DICOM": {"SUVFactor": 0.2}}}]}
        }
    }

    merged = collector._merge_studies(existing, new)

    series = merged["20240101"]["Modalities"]["PT"][0]["wb ctac"]
    assert series["DICOM"]["SUVFactor"] == 0.2
    assert series["SULPath"] == "/out/SUL.nii.gz"
