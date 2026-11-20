"""Unit tests for utils.make_json_safe: converting pydicom/NumPy value types into plain,
json.dump-able Python types before series_selection.py writes ``patient_results`` to
patient_info.json. Every DICOM tag pulled via ``extract_dicom_data`` can carry one of these
non-JSON-safe types (PersonName, UID, DS/IS numeric strings, multi-valued lists, NumPy scalars
from segmentation/metrics code), so each branch is exercised directly here rather than only
indirectly through whichever tags happen to appear in a real series.
"""

import numpy as np
from pydicom.multival import MultiValue
from pydicom.uid import UID
from pydicom.valuerep import IS, DSfloat, PersonName

from musiq.utils import make_json_safe


def test_person_name_becomes_plain_string():
    result = make_json_safe(PersonName("Doe^Jane"))

    assert result == "Doe^Jane"
    assert isinstance(result, str)


def test_uid_becomes_plain_string():
    result = make_json_safe(UID("1.2.840.10008.5.1.4.1.1.2"))

    assert result == "1.2.840.10008.5.1.4.1.1.2"
    assert isinstance(result, str)


def test_integer_string_stays_numeric_not_stringified():
    """IS (Integer String) values like PatientWeight must stay numeric downstream (e.g. LBM
    arithmetic in totalsegmentator_muscle_fat.py), not get stringified."""
    result = make_json_safe(IS("80"))

    assert result == 80
    assert isinstance(result, int)


def test_decimal_string_stays_numeric_not_stringified():
    result = make_json_safe(DSfloat("72.5"))

    assert result == 72.5
    assert isinstance(result, float)


def test_multi_value_recurses_into_each_element():
    mv = MultiValue(DSfloat, ["1.0", "2.5", "3.0"])

    result = make_json_safe(mv)

    assert result == [1.0, 2.5, 3.0]
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


def test_numpy_integer_becomes_plain_int():
    result = make_json_safe(np.int64(7))

    assert result == 7
    assert isinstance(result, int)


def test_numpy_floating_becomes_plain_float():
    result = make_json_safe(np.float32(1.5))

    assert result == 1.5
    assert isinstance(result, float)


def test_numpy_ndarray_becomes_plain_list():
    result = make_json_safe(np.array([1, 2, 3]))

    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_dict_recurses_into_values():
    result = make_json_safe({"weight": IS("80"), "name": PersonName("Doe^Jane")})

    assert result == {"weight": 80, "name": "Doe^Jane"}


def test_nested_list_of_dicts_recurses_fully():
    result = make_json_safe([{"w": IS("80")}, {"w": IS("65")}])

    assert result == [{"w": 80}, {"w": 65}]


def test_basic_types_pass_through_unchanged():
    assert make_json_safe("plain string") == "plain string"
    assert make_json_safe(42) == 42
    assert make_json_safe(None) is None
    assert make_json_safe(True) is True
