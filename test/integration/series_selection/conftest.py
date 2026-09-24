import pytest

from musiq.series_selection import SeriesSelection


@pytest.fixture()
def collector():
    return SeriesSelection(input_dirpath=".", output_dirpath=".", series_keywords={})
