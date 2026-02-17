from pprint import pprint
import pydicom
import sys
from unittest.mock import MagicMock, Mock

sys.modules["SimpleITK"] = MagicMock()
sys.modules["gdcm"] = MagicMock()

from musiq.series_selection import SeriesSelection

def test_collect_series_dir_path(monkeypatch, dummy_dicom_dataset, tmp_input_output, get_series_keywords):
    input_dir, output_dir = tmp_input_output
    series_keywords = get_series_keywords

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

    pprint(grouped)



    assert ("P001", "20240101") in grouped
    assert grouped[("P001", "20240101")][0]["Manufacturer"] == "manufacturer"
    assert grouped[("P001", "20240101")][0]["Modality"] == "CT"
    assert grouped[("P001", "20240101")][0]["PatientID"] == "P001"
    assert grouped[("P001", "20240101")][0]["SeriesDescription"] == "series_desc"
    assert grouped[("P001", "20240101")][0]["StudyDate"] == "20240101"
    assert grouped[("P001", "20240101")][0]["StudyDescription"] == "study_desc"

    actual_series_path = grouped[("P001", "20240101")][0]["SeriesPath"] 
    expect_path = tmp_input_output[0] / "P001" / "10000000" / "10000001" / "100000E0"

    actual_study_path = grouped[("P001", "20240101")][0]["StudyPath"] 
    assert actual_series_path.parent == actual_study_path

    actual_patient_path = grouped[("P001", "20240101")][0]["PatientPath"] 
    expect_patient_path = tmp_input_output[0] / "P001"
    assert actual_patient_path == expect_patient_path

    assert ("P002", "20240102") in grouped
    assert grouped[("P002", "20240102")][0]["Manufacturer"] == "manufacturer2"
    assert grouped[("P002", "20240102")][0]["Modality"] == "MR"
    assert grouped[("P002", "20240102")][0]["PatientID"] == "P002"
    assert grouped[("P002", "20240102")][0]["SeriesDescription"] == "series_desc2"
    assert grouped[("P002", "20240102")][0]["StudyDate"] == "20240102"
    assert grouped[("P002", "20240102")][0]["StudyDescription"] == "study_desc2"
    
    actual_series_path = grouped[("P002", "20240102")][0]["SeriesPath"] 
    expect_series_path = tmp_input_output[0] / "P002"  / "1000031B" / "1000031C" / "100004A9"

    actual_study_path = grouped[("P002", "20240102")][0]["StudyPath"] 
    expect_study_path = tmp_input_output[0] / "P002" / "1000031B" / "1000031C"
    assert actual_series_path.parent == actual_study_path

    actual_patient_path = grouped[("P002", "20240102")][0]["PatientPath"] 
    expect_patient_path = tmp_input_output[0] / "P002"
    assert actual_patient_path == expect_patient_path