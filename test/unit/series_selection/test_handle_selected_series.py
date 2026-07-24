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
