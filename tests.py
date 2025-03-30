# tests.py

import polars as pl
import pytest
from utils import filter_chinese_license_plates

TEST_DATA = [
    # Valid Standard Plates (7 characters)
    ("京A12345", True),
    ("粤B12345", True),
    ("沪D1234F", True),  # F is allowed in standard plates
    ("京BU0330", True),
    ("京A1234E", True),  # E is allowed in standard plates
    # Valid New Energy Plates (8 characters)
    ("津C5678D", True),
    ("粤BD1234F", True),
    # Invalid Plates
    ("京A1234d", False),  # lowercase d in standard plate
    ("京a12345", False),  # lowercase province code
    ("京I12345", False),  # contains I
    ("京O12345", False),  # contains O
    ("京A1234", False),  # too short
    ("京A123456", False),  # too long
    ("A12345", False),  # missing Chinese character
    ("京BU0330??2019-11-25 23:56:43", False),  # extra characters
    ("", False),  # empty string
]


@pytest.fixture
def test_df():
    """Create a test DataFrame with license plate column"""
    data = {"license_plate": [plate for plate, _ in TEST_DATA]}
    return pl.LazyFrame(data, schema={"license_plate": pl.String})


def test_filter_chinese_license_plates(test_df):
    """Test the license plate filtering function"""
    filtered = filter_chinese_license_plates(test_df).collect()

    expected_valid = [plate for plate, is_valid in TEST_DATA if is_valid]
    result_plates = filtered["license_plate"].to_list()

    # Check all expected plates are present
    assert sorted(result_plates) == sorted(expected_valid)


def test_filter_with_different_column_name():
    """Test function works with custom column name"""
    data = {"custom_col": ["京A12345", "invalid", "粤B54321"]}
    ldf = pl.LazyFrame(data, schema={"custom_col": pl.String})
    filtered = filter_chinese_license_plates(ldf, col="custom_col").collect()
    assert filtered.shape[0] == 2
    assert set(filtered["custom_col"].to_list()) == {"京A12345", "粤B54321"}


def test_empty_dataframe():
    """Test function handles empty DataFrame correctly"""
    empty_df = pl.LazyFrame({"license_plate": []}, schema={"license_plate": pl.String})
    filtered = filter_chinese_license_plates(empty_df).collect()
    assert filtered.shape[0] == 0


def test_non_string_columns():
    """Test function handles string conversion"""
    data = {"license_plate": ["1234567", "京A12345"]}
    ldf = pl.LazyFrame(data, schema={"license_plate": pl.String})
    filtered = filter_chinese_license_plates(ldf).collect()
    assert filtered.shape[0] == 1
    assert filtered["license_plate"].to_list() == ["京A12345"]
