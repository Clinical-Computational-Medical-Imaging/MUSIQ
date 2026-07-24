import json

def test_unknown_patient_is_a_no_op(mocker, tmp_path, make_collector):
    collector = make_collector(output_dirpath=tmp_path)
    validate_output = mocker.patch.object(collector, "validate_output")

    collector._finalize_patient("unknown", user_flags={}, patient_conversion_flags={})

    validate_output.assert_not_called()
    assert not (tmp_path / "unknown").exists()


def test_writes_patient_info_json(mocker, tmp_path, make_collector):
    patient_id = "P999"
    collector = make_collector(output_dirpath=tmp_path)
    collector.patient_results = {patient_id: {"PatientID": patient_id, "Studies": {"20240101": {"Modalities": {}}}}}
    mocker.patch.object(collector, "validate_output")

    collector._finalize_patient(patient_id, user_flags={patient_id: [False]}, patient_conversion_flags={})

    json_path = tmp_path / patient_id / "patient_info.json"
    assert json_path.is_file()
    with open(json_path) as f:
        written = json.load(f)
    assert written["PatientID"] == patient_id
    assert written["Studies"] == {"20240101": {"Modalities": {}}}


def test_merges_with_existing_patient_info(mocker, tmp_path, make_collector):
    patient_id = "P999"
    collector = make_collector(output_dirpath=tmp_path)
    existing_dir = tmp_path / patient_id
    existing_dir.mkdir()
    existing_info = {
        "PatientID": patient_id,
        "Studies": {"20240101": {"Modalities": {"CT": [{"series_a": {"CTPath": "a.nii.gz"}}]}}},
    }
    with open(existing_dir / "patient_info.json", "w") as f:
        json.dump(existing_info, f)

    collector.patient_results = {
        patient_id: {
            "PatientID": patient_id,
            "Studies": {"20240101": {"Modalities": {"PT": [{"series_b": {"PTPath": "b.nii.gz"}}]}}},
        }
    }
    mocker.patch.object(collector, "validate_output")

    collector._finalize_patient(patient_id, user_flags={}, patient_conversion_flags={})

    with open(existing_dir / "patient_info.json") as f:
        written = json.load(f)
    modalities = written["Studies"]["20240101"]["Modalities"]
    assert modalities["CT"] == [{"series_a": {"CTPath": "a.nii.gz"}}]
    assert modalities["PT"] == [{"series_b": {"PTPath": "b.nii.gz"}}]


def test_validate_output_called_with_aggregated_user_flag_and_conversion_flags(mocker, tmp_path, make_collector):
    patient_id = "P999"
    collector = make_collector(output_dirpath=tmp_path)
    collector.patient_results = {patient_id: {"PatientID": patient_id, "Studies": {}}}
    validate_output = mocker.patch.object(collector, "validate_output")

    collector._finalize_patient(
        patient_id,
        user_flags={patient_id: [False, True, False]},
        patient_conversion_flags={patient_id: [["P999", "20240101", "desc", "path"]]},
    )

    _, kwargs = validate_output.call_args
    assert kwargs["user_flag"] is True
    assert kwargs["conversion_flags"] == [["P999", "20240101", "desc", "path"]]
