# tests.py

import polars as pl
import pytest
from utils import filter_chinese_license_plates

TEST_DATA = [
    ("京A12345", True),  # Standard valid plate (7 chars)
    ("粤B12345", True),  # Standard valid plate (7 chars)
    ("沪D1234F", False),  # New energy must end with D/d
    ("津C5678D", True),  # New energy valid plate (8 chars)
    ("京BU0330", True),  # Valid standard plate (7 chars)
    ("京BU0330??2019-11-25 23:56:43", False),  # Invalid - extra chars
    ("京I12345", False),  # Invalid - contains I
    ("京O12345", False),  # Invalid - contains O
    ("京A1234", False),  # Invalid - too short (6 chars)
    ("京A123456", False),  # Invalid - too long (9 chars)
    ("A12345", False),  # Invalid - missing Chinese char
    ("京a12345", False),  # Invalid - lowercase letter
    ("京A1234E", False),  # Invalid new energy - ends with E
    ("京A1234d", True),  # Valid new energy - lowercase d
    (None, False),  # Invalid - null value
    ("", False),  # Invalid - empty string
]


@pytest.fixture
def test_df():
    """Create a test DataFrame with license plate column"""
    data = {"license_plate": [plate for plate, _ in TEST_DATA]}
    return pl.LazyFrame(data, schema={"license_plate": pl.String})


def test_filter_chinese_license_plates(test_df):
    """Test the license plate filtering function"""
    filtered = filter_chinese_license_plates(test_df).collect()

    # Get the expected valid plates
    expected_valid = [plate for plate, is_valid in TEST_DATA if is_valid]

    # Verify all valid plates are included
    result_plates = filtered["license_plate"].to_list()
    assert sorted(result_plates) == sorted(expected_valid)
    assert len(result_plates) == len(expected_valid)


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


def test_case_sensitivity():
    """Test that new energy plates ending with D/d are accepted"""
    data = {"license_plate": ["京A1234D", "京B5678d"]}
    ldf = pl.LazyFrame(data, schema={"license_plate": pl.String})
    filtered = filter_chinese_license_plates(ldf).collect()
    assert filtered.shape[0] == 2
    assert set(filtered["license_plate"].to_list()) == {"京A1234D", "京B5678d"}
