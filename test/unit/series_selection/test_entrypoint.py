from musiq.series_selection import series_selection_entrypoint


def _patch_common(mocker):
    mocker.patch("musiq.utils.create_logger", return_value=mocker.MagicMock())
    # series_selection_entrypoint() does `global logger; logger = create_logger(...)`, permanently
    # rebinding the module-level `logger` for the rest of the test session (not undone by mocking
    # create_logger alone, since that only affects the assignment's source, not the module
    # attribute it writes to). Patching the attribute itself ensures it's restored on teardown, so
    # later tests' caplog-based assertions against the real logger keep working.
    mocker.patch("musiq.series_selection.logger")
    return mocker.patch("musiq.series_selection.SeriesSelection")


def test_entrypoint_wires_cli_args_into_series_selection(mocker):
    series_selection_cls = _patch_common(mocker)
    mocker.patch(
        "sys.argv",
        [
            "musiq_series_selection",
            "--input-dir",
            "/data/raw",
            "--output-dir",
            "/data/processed",
            # Deliberately NOT config.yaml's real CT PRIMARY keywords ("knochen"/"i30f") — using
            # those would make this test pass even if the CLI value were silently dropped and
            # setup_series_keywords fell back to the config default instead, since both would
            # coincidentally match.
            "--ct-primary-keywords",
            "testkw1",
            "testkw2",
        ],
    )

    series_selection_entrypoint()

    series_selection_cls.assert_called_once()
    _, kwargs = series_selection_cls.call_args
    assert kwargs["input_dirpath"] == "/data/raw"
    assert kwargs["output_dirpath"] == "/data/processed"
    assert kwargs["series_keywords"]["CT"]["PRIMARY"] == ["testkw1", "testkw2"]
    series_selection_cls.return_value.run.assert_called_once()


def test_entrypoint_defaults_keywords_to_config_yaml_when_not_passed(mocker):
    series_selection_cls = _patch_common(mocker)
    mocker.patch(
        "sys.argv",
        ["musiq_series_selection", "--input-dir", "/data/raw", "--output-dir", "/data/processed"],
    )

    series_selection_entrypoint()

    _, kwargs = series_selection_cls.call_args
    # No keyword flags passed at all -> setup_series_keywords backfills from config.yaml.
    assert kwargs["series_keywords"]["CT"]["PRIMARY"] == ["knochen", "i30f"]


def test_entrypoint_empty_keyword_flag_disables_filtering_for_that_list(mocker):
    series_selection_cls = _patch_common(mocker)
    mocker.patch(
        "sys.argv",
        [
            "musiq_series_selection",
            "--input-dir",
            "/data/raw",
            "--output-dir",
            "/data/processed",
            "--pt-primary-keywords",
        ],
    )

    series_selection_entrypoint()

    _, kwargs = series_selection_cls.call_args
    # A keyword flag passed with no values -> argparse yields [] (distinct from absent=None).
    assert kwargs["series_keywords"]["PT"]["PRIMARY"] == []
