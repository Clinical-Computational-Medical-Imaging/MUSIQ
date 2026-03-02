import logging
from pprint import pprint
import pytest
import sys
from unittest.mock import MagicMock, Mock

sys.modules["SimpleITK"] = MagicMock()
sys.modules["gdcm"] = MagicMock()

from musiq.series_selection import SeriesSelection

def test_collect_series_dir_path(monkeypatch, dummy_dicom_dataset, tmp_input_output, get_series_keywords):
    input_dir, output_dir = tmp_input_output
    series_keywords = get_series_keywords
    exclution = [".zip", ".inf", ".jar", ".icns", ".info", ".exe", ".pdf",
                 ".txt", ".ini", ".xml", ".bmp", ".sh"]

    def dummy_dcmread(file_path, *args, **kwargs):
        if "P001" in str(file_path):
            return dummy_dicom_dataset["P001"] 
        elif "P002" in str(file_path):
            return dummy_dicom_dataset["P002"]  
        else:
            raise FileNotFoundError("Unexpectet file")


    monkeypatch.setattr("pydicom.dcmread", dummy_dcmread)

    collector = SeriesSelection(input_dirpath = input_dir, output_dirpath = output_dir, series_keywords=series_keywords)

    collector.logger = Mock()

    grouped = collector.collect_series()

    assert grouped != {}
    for pat, ds in dummy_dicom_dataset.items():
        assert ("P001", "20240101") in grouped
        assert grouped[(pat, ds.StudyDate)][0]["Manufacturer"] == dummy_dicom_dataset[pat].Manufacturer
        assert grouped[(pat, ds.StudyDate)][0]["Modality"] == dummy_dicom_dataset[pat].Modality
        assert grouped[(pat, ds.StudyDate)][0]["PatientID"] == dummy_dicom_dataset[pat].PatientID
        assert grouped[(pat, ds.StudyDate)][0]["SeriesDescription"] == dummy_dicom_dataset[pat].SeriesDescription
        assert grouped[(pat, ds.StudyDate)][0]["StudyDate"] == dummy_dicom_dataset[pat].StudyDate
        assert grouped[(pat, ds.StudyDate)][0]["StudyDescription"] == dummy_dicom_dataset[pat].StudyDescription

        actual_series_path = grouped[(pat, ds.StudyDate)][0]["SeriesPath"] 
        expect_path = tmp_input_output[0] / pat / "10000000" / "10000001" / "100000E0"

        actual_study_path = grouped[(pat, ds.StudyDate)][0]["StudyPath"] 
        assert actual_series_path.parent == actual_study_path

        actual_patient_path = grouped[(pat, ds.StudyDate)][0]["PatientPath"] 
        expect_patient_path = tmp_input_output[0] / pat
        assert actual_patient_path == expect_patient_path

    for _, series_list in grouped.items():
        for series in series_list:
            path = (series["SeriesPath"])
            assert path.suffix.lower() not in exclution and path.suffix != ".DS_Store" and path.name != "DeepUnity Media Viewer Mac"


def test_file_not_found_error(get_series_keywords): 
    """ 
    Tests for wrong path and checks if the FileNotFoundError is thrown. 
    """ 
    collector = SeriesSelection( input_dirpath = "invalid/test/input/path", 
                                output_dirpath = "invalid/test/output/path", 
                                series_keywords = get_series_keywords ) 
    
    with pytest.raises(FileNotFoundError): 
        collector.collect_series() 

def test_log_error_dcmread_fail(caplog, mocker, tmp_input_output, get_series_keywords): 
    input_dir, output_dir = tmp_input_output 

    mocker.patch("musiq.series_selection.pydicom.dcmread", 
                side_effect=Exception("Thrown exception")) 
    
    collector = SeriesSelection(input_dirpath=input_dir, 
                                output_dirpath=output_dir, 
                                series_keywords=get_series_keywords) 
    
    with caplog.at_level(logging.ERROR): 
        grouped = collector.collect_series() 

    assert grouped == {} 
    assert "Failed to read DICOM" in caplog.text 
    assert "Thrown exception" in caplog.text
