import polars as pl
import pytest
from utils import filter_chinese_license_plates

# Test data with valid and invalid license plates
TEST_DATA = [
    ("京A12345", True),  # Standard valid plate
    ("粤B12345", True),  # Standard valid plate
    ("沪D1234F", True),  # New energy valid plate (F)
    ("津C5678D", True),  # New energy valid plate (D)
    ("京BU0330", True),  # Valid plate with letters
    ("京BU0330??2019-11-25 23:56:43", False),  # Invalid - extra characters
    ("京I12345", False),  # Invalid - contains I
    ("京O12345", False),  # Invalid - contains O
    ("京A1234", False),  # Invalid - too short
    ("京A123456", False),  # Invalid - too long
    ("A12345", False),  # Invalid - missing Chinese character
    ("京a12345", False),  # Invalid - lowercase letter
    ("京A1234E", False),  # Invalid new energy - ends with E
    (
        "京A1234d",
        False,
    ),  # Invalid new energy - lowercase d (but should pass if case-insensitive)
    (None, False),  # Invalid - null value
    ("", False),  # Invalid - empty string
]


@pytest.fixture
def test_df():
    """Create a test DataFrame with license plate column"""
    data = {"license_plate": [plate for plate, _ in TEST_DATA]}
    return pl.LazyFrame(data)


def test_filter_chinese_license_plates(test_df):
    """Test the license plate filtering function"""
    filtered = filter_chinese_license_plates(test_df).collect()

    # Get the expected valid plates
    expected_valid = [plate for plate, is_valid in TEST_DATA if is_valid]

    # Verify all valid plates are included
    assert sorted(filtered["license_plate"].to_list()) == sorted(expected_valid)

    # Verify no invalid plates slipped through
    assert filtered.shape[0] == len(expected_valid)


def test_filter_with_different_column_name():
    """Test function works with custom column name"""
    data = {"custom_col": ["京A12345", "invalid", "粤B54321"]}
    ldf = pl.LazyFrame(data)
    filtered = filter_chinese_license_plates(ldf, col="custom_col").collect()

    assert filtered.shape[0] == 2
    assert set(filtered["custom_col"].to_list()) == {"京A12345", "粤B54321"}


def test_empty_dataframe():
    """Test function handles empty DataFrame correctly"""
    empty_df = pl.LazyFrame({"license_plate": []})
    filtered = filter_chinese_license_plates(empty_df).collect()
    assert filtered.shape[0] == 0


def test_non_string_columns():
    """Test function handles non-string columns (should cast to string)"""
    data = {"license_plate": [1234567, "京A12345"]}  # Mixed types
    ldf = pl.LazyFrame(data)
    filtered = filter_chinese_license_plates(ldf).collect()
    assert filtered.shape[0] == 1
    assert filtered["license_plate"].to_list() == ["京A12345"]
