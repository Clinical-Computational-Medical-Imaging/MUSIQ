import logging
import pathlib as plb
from types import SimpleNamespace


def _series_entry(patient_id="P1", study_date="20240101", **overrides):
    entry = {
        "PatientID": patient_id,
        "StudyDate": study_date,
        "Modality": "CT",
        "SeriesDescription": "knochen ct",
        "StudyDescription": "study desc",
        "SeriesPath": "/dicom/s1",
        "StudyPath": "/dicom",
        "PatientPath": "/dicom",
        "Manufacturer": "manufacturer",
    }
    entry.update(overrides)
    return entry


def _prepared(collector, mocker, find_default_return=([0], False), handle_return=None):
    """Wire up a collector fixture with the collaborators interactive_selection() orchestrates,
    so tests can assert on the orchestration itself without exercising the real conversion path."""
    mocker.patch("musiq.series_selection.extract_dicom_data", return_value={})
    mocker.patch.object(collector, "find_default_indices", return_value=find_default_return)
    mocker.patch.object(collector, "handle_selected_series", return_value=handle_return or [])
    mocker.patch.object(collector, "_finalize_patient")
    return collector


def test_eof_on_input_falls_back_to_non_interactive_preselected_indices(mocker, caplog, collector):
    collector = _prepared(collector, mocker)
    entry = _series_entry()
    collector.grouped_series = {("P1", "20240101"): [entry]}
    mocker.patch("builtins.input", side_effect=EOFError)

    with caplog.at_level(logging.WARNING):
        collector.interactive_selection()

    assert "No interactive terminal detected" in caplog.text
    collector.handle_selected_series.assert_called_once_with({"P1": [entry]})
    collector._finalize_patient.assert_called_once_with("P1", {"P1": [False]}, {"P1": []})


def test_invalid_selection_answer_defaults_to_preselected_indices(mocker, caplog, collector):
    collector = _prepared(collector, mocker)
    entry = _series_entry()
    collector.grouped_series = {("P1", "20240101"): [entry]}
    mocker.patch("builtins.input", return_value="maybe")

    with caplog.at_level(logging.WARNING):
        collector.interactive_selection()

    assert "Starting without interactive" in caplog.text
    collector.handle_selected_series.assert_called_once_with({"P1": [entry]})


def test_flag_x_uses_preselected_indices_and_sets_user_flag(mocker, collector):
    collector = _prepared(collector, mocker, find_default_return=([0], False))
    entry_a = _series_entry(SeriesDescription="a")
    entry_b = _series_entry(SeriesDescription="b")
    collector.grouped_series = {("P1", "20240101"): [entry_a, entry_b]}
    mocker.patch("builtins.input", side_effect=["y", "1,x"])

    collector.interactive_selection()

    # 'x' forces the preselected indices (index 0) even though the user typed "1".
    collector.handle_selected_series.assert_called_once_with({"P1": [entry_a]})
    collector._finalize_patient.assert_called_once_with("P1", {"P1": [True]}, {"P1": []})


def test_manual_selection_parses_comma_separated_indices(mocker, collector):
    collector = _prepared(collector, mocker)
    entry_a = _series_entry(SeriesDescription="a")
    entry_b = _series_entry(SeriesDescription="b")
    collector.grouped_series = {("P1", "20240101"): [entry_a, entry_b]}
    mocker.patch("builtins.input", side_effect=["y", "1"])

    collector.interactive_selection()

    collector.handle_selected_series.assert_called_once_with({"P1": [entry_b]})


def test_empty_study_info_is_skipped_without_conversion(mocker, caplog, collector):
    collector = _prepared(collector, mocker)
    collector.grouped_series = {("P1", "20240101"): []}
    mocker.patch("builtins.input", side_effect=EOFError)

    with caplog.at_level(logging.WARNING):
        collector.interactive_selection()

    assert "Skipping empty study" in caplog.text
    collector.handle_selected_series.assert_not_called()
    # The final flush still fires for the last-seen patient; the real (unmocked) _finalize_patient
    # would no-op internally since P1 was never added to patient_results.
    collector._finalize_patient.assert_called_once_with("P1", {}, {})


def test_no_selected_indices_skips_conversion_for_that_study(mocker, caplog, collector):
    collector = _prepared(collector, mocker, find_default_return=([], True))
    entry = _series_entry()
    collector.grouped_series = {("P1", "20240101"): [entry]}
    mocker.patch("builtins.input", side_effect=EOFError)

    with caplog.at_level(logging.WARNING):
        collector.interactive_selection()

    assert "No series selected" in caplog.text
    collector.handle_selected_series.assert_not_called()


def test_finalize_called_per_patient_boundary_and_once_at_the_end(mocker, collector):
    collector = _prepared(collector, mocker)
    entry_p1 = _series_entry(patient_id="P1")
    entry_p2 = _series_entry(patient_id="P2")
    collector.grouped_series = {
        ("P1", "20240101"): [entry_p1],
        ("P2", "20240101"): [entry_p2],
    }
    mocker.patch("builtins.input", side_effect=EOFError)

    collector.interactive_selection()

    called_patient_ids = [call.args[0] for call in collector._finalize_patient.call_args_list]
    assert called_patient_ids == ["P1", "P2"]


def test_multiple_studies_for_same_patient_finalize_only_once(mocker, collector):
    collector = _prepared(collector, mocker)
    entry_1 = _series_entry(study_date="20240101")
    entry_2 = _series_entry(study_date="20240202")
    collector.grouped_series = {
        ("P1", "20240101"): [entry_1],
        ("P1", "20240202"): [entry_2],
    }
    mocker.patch("builtins.input", side_effect=EOFError)

    collector.interactive_selection()

    collector._finalize_patient.assert_called_once_with("P1", {"P1": [False, False]}, {"P1": []})


def test_run_calls_collect_series_then_interactive_selection(mocker, collector):
    grouped = {("P1", "20240101"): []}
    mocker.patch.object(collector, "collect_series", return_value=grouped)
    interactive_selection = mocker.patch.object(collector, "interactive_selection")

    collector.run()

    assert collector.grouped_series == grouped
    interactive_selection.assert_called_once()


def test_series_with_precomputed_num_slices_logs_slice_count(mocker, caplog, collector):
    collector = _prepared(collector, mocker)
    entry = _series_entry(NumSlices=42)
    collector.grouped_series = {("P1", "20240101"): [entry]}
    mocker.patch("builtins.input", side_effect=EOFError)

    with caplog.at_level(logging.INFO):
        collector.interactive_selection()

    assert "slices: 42" in caplog.text


def test_dynamic_mr_siblings_are_recorded_on_selected_series(mocker, collector, tmp_path):
    collector = _prepared(collector, mocker, find_default_return=([0], False))
    entries = []
    for i in range(3):
        series_dir = tmp_path / f"series{i}"
        series_dir.mkdir()
        (series_dir / "file1.dcm").touch()
        entries.append(_series_entry(Modality="MR", SeriesDescription="dyn scan", SeriesPath=series_dir))
    collector.grouped_series = {("P1", "20240101"): entries}
    mocker.patch("builtins.input", side_effect=EOFError)
    mocker.patch(
        "musiq.series_selection.pydicom.dcmread",
        side_effect=lambda path, *a, **k: SimpleNamespace(AcquisitionTime=plb.Path(path).parent.name, ImageType=[]),
    )

    collector.interactive_selection()

    assert entries[0]["DynamicSiblingPaths"] == [e["SeriesPath"] for e in entries]


def test_conversion_flags_from_handle_selected_series_are_accumulated(mocker, collector):
    collector = _prepared(collector, mocker, handle_return=[["P1", "20240101", "desc", "/path"]])
    entry = _series_entry()
    collector.grouped_series = {("P1", "20240101"): [entry]}
    mocker.patch("builtins.input", side_effect=EOFError)

    collector.interactive_selection()

    collector._finalize_patient.assert_called_once_with(
        "P1", {"P1": [False]}, {"P1": [["P1", "20240101", "desc", "/path"]]}
    )


def test_answer_y_prompts_again_with_preselected_default_shown(mocker, collector):
    collector = _prepared(collector, mocker, find_default_return=([0], False))
    entry_a = _series_entry(SeriesDescription="a")
    entry_b = _series_entry(SeriesDescription="b")
    collector.grouped_series = {("P1", "20240101"): [entry_a, entry_b]}
    mock_input = mocker.patch("builtins.input", side_effect=["y", "0"])

    collector.interactive_selection()

    assert mock_input.call_count == 2
    per_study_prompt = mock_input.call_args_list[1].args[0]
    assert "default: 0" in per_study_prompt
    collector.handle_selected_series.assert_called_once_with({"P1": [entry_a]})


def test_answer_n_skips_per_study_prompt_and_uses_preselected(mocker, caplog, collector):
    collector = _prepared(collector, mocker, find_default_return=([0], False))
    entry_a = _series_entry(SeriesDescription="a")
    entry_b = _series_entry(SeriesDescription="b")
    collector.grouped_series = {("P1", "20240101"): [entry_a, entry_b]}
    mock_input = mocker.patch("builtins.input", return_value="n")

    with caplog.at_level(logging.INFO):
        collector.interactive_selection()

    # Only the initial y/n prompt is asked; the per-study selection prompt must not fire.
    assert mock_input.call_count == 1
    assert "Using preselected indices: 0" in caplog.text
    collector.handle_selected_series.assert_called_once_with({"P1": [entry_a]})


def test_uppercase_n_answer_is_normalized_and_treated_as_no(mocker, collector):
    collector = _prepared(collector, mocker, find_default_return=([0], False))
    entry = _series_entry()
    collector.grouped_series = {("P1", "20240101"): [entry]}
    mock_input = mocker.patch("builtins.input", return_value="N")

    collector.interactive_selection()

    assert mock_input.call_count == 1
    collector.handle_selected_series.assert_called_once_with({"P1": [entry]})


def test_uppercase_y_answer_is_normalized_and_triggers_second_prompt(mocker, collector):
    collector = _prepared(collector, mocker, find_default_return=([0], False))
    entry = _series_entry()
    collector.grouped_series = {("P1", "20240101"): [entry]}
    mock_input = mocker.patch("builtins.input", side_effect=["Y", "0"])

    collector.interactive_selection()

    assert mock_input.call_count == 2
    collector.handle_selected_series.assert_called_once_with({"P1": [entry]})


def test_series_listing_marks_preselected_index_and_shows_modality_description(mocker, caplog, collector):
    collector = _prepared(collector, mocker, find_default_return=([1], False))
    entry_a = _series_entry(Modality="CT", SeriesDescription="a")
    entry_b = _series_entry(Modality="MR", SeriesDescription="b")
    collector.grouped_series = {("P1", "20240101"): [entry_a, entry_b]}
    mocker.patch("builtins.input", side_effect=EOFError)

    with caplog.at_level(logging.INFO):
        collector.interactive_selection()

    assert "[ ] [ 0]  CT | a" in caplog.text
    assert "[*] [ 1]  MR | b" in caplog.text


def test_fallback_flag_logs_warning_with_primary_and_secondary_keywords(mocker, caplog, collector):
    collector = _prepared(collector, mocker, find_default_return=([0], True))
    entry = _series_entry(Modality="CT")
    collector.grouped_series = {("P1", "20240101"): [entry]}
    mocker.patch("builtins.input", side_effect=EOFError)

    with caplog.at_level(logging.INFO):
        collector.interactive_selection()

    primary = collector.series_keywords["CT"]["PRIMARY"]
    secondary = collector.series_keywords["CT"]["SECONDARY"]
    assert f"No {primary} found" in caplog.text
    assert f"defaulted to {secondary}" in caplog.text
    assert "and flagged study." in caplog.text


def test_study_header_and_manufacturer_are_logged(mocker, caplog, collector):
    collector = _prepared(collector, mocker)
    patient_id, study_date = "P1", "20240101"
    study_description, manufacturer = "my study", "ACME"
    entry = _series_entry(
        patient_id=patient_id, study_date=study_date, StudyDescription=study_description, Manufacturer=manufacturer
    )
    collector.grouped_series = {(patient_id, study_date): [entry]}
    mocker.patch("builtins.input", side_effect=EOFError)

    with caplog.at_level(logging.INFO):
        collector.interactive_selection()

    study_count = len(collector.grouped_series)
    assert (
        f"Study 1 of {study_count} — Patient ID: {patient_id} — "
        f"Study Date: {study_date} - Study Desc: {study_description}"
    ) in caplog.text
    assert f"Manufacturer: {manufacturer}" in caplog.text
    assert "Available Series:" in caplog.text
