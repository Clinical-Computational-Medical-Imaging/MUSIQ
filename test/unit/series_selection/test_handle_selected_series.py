import json


def _seed_patient_results(collector, patient_id="P999"):
    collector.patient_results = {patient_id: {"Studies": {}}}
    return collector


def _series_entry(**overrides):
    entry = {
        "PatientID": "P999",
        "StudyDate": "20240101",
        "Modality": "CT",
        "SeriesDescription": "knochen ct",
        "StudyDescription": "study desc",
        "SeriesPath": "/dicom/series1",
        "StudyPath": "/dicom/study1",
        "PatientPath": "/dicom",
        "Manufacturer": "manufacturer",
    }
    entry.update(overrides)
    return entry


def test_single_series_populates_study_and_modality(mocker, collector):
    patient_id = "P999"
    _seed_patient_results(collector, patient_id)
    mocker.patch("musiq.series_selection.extract_dicom_data", return_value={"PatientAge": "050Y"})
    mocker.patch.object(
        collector, "start_dcm2nii", return_value=(False, {"knochen ct": {"CTPath": "/out/CT.nii.gz"}})
    )

    selected_series = {patient_id: [_series_entry()]}
    flags = collector.handle_selected_series(selected_series)

    study = collector.patient_results[patient_id]["Studies"]["20240101"]
    assert study["InputDirPath"] == "/dicom/study1"
    assert study["StudyDescription"] == "study desc"
    assert study["PatientAge"] == "050Y"
    assert study["Modalities"]["CT"] == [{"knochen ct": {"CTPath": "/out/CT.nii.gz"}}]
    assert flags == []


def test_multiple_modalities_in_same_study(mocker, collector):
    patient_id = "P999"
    _seed_patient_results(collector, patient_id)
    mocker.patch("musiq.series_selection.extract_dicom_data", return_value={})
    mocker.patch.object(
        collector,
        "start_dcm2nii",
        side_effect=[
            (False, {"knochen ct": {"CTPath": "/out/CT.nii.gz"}}),
            (False, {"wb ctac": {"PTPath": "/out/PET.nii.gz"}}),
        ],
    )

    selected_series = {
        patient_id: [
            _series_entry(Modality="CT", SeriesDescription="knochen ct"),
            _series_entry(Modality="PT", SeriesDescription="wb ctac"),
        ]
    }
    collector.handle_selected_series(selected_series)

    modalities = collector.patient_results[patient_id]["Studies"]["20240101"]["Modalities"]
    assert set(modalities.keys()) == {"CT", "PT"}


def test_two_series_same_modality_are_both_appended(mocker, collector):
    patient_id = "P999"
    _seed_patient_results(collector, patient_id)
    mocker.patch("musiq.series_selection.extract_dicom_data", return_value={})
    mocker.patch.object(
        collector,
        "start_dcm2nii",
        side_effect=[
            (False, {"series_a": {"CTPath": "/out/a.nii.gz"}}),
            (False, {"series_b": {"CTPath": "/out/b.nii.gz"}}),
        ],
    )

    selected_series = {
        patient_id: [
            _series_entry(SeriesDescription="series_a"),
            _series_entry(SeriesDescription="series_b"),
        ]
    }
    collector.handle_selected_series(selected_series)

    ct_series = collector.patient_results[patient_id]["Studies"]["20240101"]["Modalities"]["CT"]
    assert ct_series == [{"series_a": {"CTPath": "/out/a.nii.gz"}}, {"series_b": {"CTPath": "/out/b.nii.gz"}}]


def test_conversion_failure_is_collected_as_flag(mocker, collector):
    patient_id = "P999"
    _seed_patient_results(collector, patient_id)
    mocker.patch("musiq.series_selection.extract_dicom_data", return_value={})
    mocker.patch.object(collector, "start_dcm2nii", return_value=(True, {}))

    selected_series = {patient_id: [_series_entry(SeriesPath="/dicom/broken")]}
    flags = collector.handle_selected_series(selected_series)

    assert flags == [[patient_id, "20240101", "knochen ct", "/dicom/broken"]]
    # A failed conversion contributes nothing to the modality's series list.
    assert collector.patient_results[patient_id]["Studies"]["20240101"]["Modalities"]["CT"] == []


def test_dynamic_sibling_paths_are_forwarded_to_start_dcm2nii(mocker, collector):
    patient_id = "P999"
    _seed_patient_results(collector, patient_id)
    mocker.patch("musiq.series_selection.extract_dicom_data", return_value={})
    start_dcm2nii = mocker.patch.object(collector, "start_dcm2nii", return_value=(False, {}))

    selected_series = {patient_id: [_series_entry(DynamicSiblingPaths=["/a", "/b", "/c"])]}
    collector.handle_selected_series(selected_series)

    _, kwargs = start_dcm2nii.call_args
    assert kwargs["dynamic_sibling_dirs"] == ["/a", "/b", "/c"]


def test_missing_series_weight_falls_back_to_recorded_patient_info(mocker, tmp_path, make_collector):
    """No selected series carries a positive PatientWeight (e.g. an anonymized PET tag is 0/missing);
    the study-level weight already recorded in patient_info.json from an earlier run is reused instead
    of leaving fallback_weight unset."""
    patient_id = "P999"
    collector = make_collector(output_dirpath=tmp_path)
    _seed_patient_results(collector, patient_id)
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    with open(patient_dir / "patient_info.json", "w") as f:
        json.dump({"Studies": {"20240101": {"PatientWeight": 72.5}}}, f)

    mocker.patch("musiq.series_selection.extract_dicom_data", return_value={})
    start_dcm2nii = mocker.patch.object(collector, "start_dcm2nii", return_value=(False, {}))

    selected_series = {patient_id: [_series_entry()]}
    collector.handle_selected_series(selected_series)

    _, kwargs = start_dcm2nii.call_args
    assert kwargs["fallback_weight"] == 72.5


def test_series_weight_used_directly_without_fallback_lookup(mocker, tmp_path, make_collector):
    """A selected series with its own valid, positive PatientWeight is used directly and takes
    priority over any weight already recorded in patient_info.json."""
    patient_id = "P999"
    collector = make_collector(output_dirpath=tmp_path)
    _seed_patient_results(collector, patient_id)
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    with open(patient_dir / "patient_info.json", "w") as f:
        json.dump({"Studies": {"20240101": {"PatientWeight": 40.0}}}, f)

    mocker.patch("musiq.series_selection.extract_dicom_data", return_value={"PatientWeight": "80"})
    start_dcm2nii = mocker.patch.object(collector, "start_dcm2nii", return_value=(False, {}))

    selected_series = {patient_id: [_series_entry()]}
    collector.handle_selected_series(selected_series)

    _, kwargs = start_dcm2nii.call_args
    assert kwargs["fallback_weight"] == 80.0


def test_corrupt_recorded_patient_info_is_ignored_during_weight_fallback(mocker, tmp_path, make_collector):
    """A corrupt patient_info.json encountered while looking up the fallback weight must not crash
    the run; the lookup is simply skipped and no fallback weight is used."""
    patient_id = "P999"
    collector = make_collector(output_dirpath=tmp_path)
    _seed_patient_results(collector, patient_id)
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    (patient_dir / "patient_info.json").write_text("{not valid json")

    mocker.patch("musiq.series_selection.extract_dicom_data", return_value={})
    start_dcm2nii = mocker.patch.object(collector, "start_dcm2nii", return_value=(False, {}))

    selected_series = {patient_id: [_series_entry()]}
    collector.handle_selected_series(selected_series)

    _, kwargs = start_dcm2nii.call_args
    assert kwargs["fallback_weight"] is None
