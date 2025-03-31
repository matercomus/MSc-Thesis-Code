# tests.py

from datetime import datetime
import polars as pl
from utils import (
    filter_chinese_license_plates,
    profile_data,
    filter_by_year,
)


def test_filter_chinese_license_plates():
    data = {
        "license_plate": [
            "京A12345",  # Valid standard
            "粤B12345",  # Valid standard
            "沪C1234D",  # Valid new energy
            "苏E1234F",  # Valid new energy
            "京A1234",  # Too short
            "粤B123456",  # Too long
            "invalid",  # Invalid format
            None,  # Null
        ]
    }
    ldf = pl.LazyFrame(data)
    filtered = filter_chinese_license_plates(ldf).collect()
    assert filtered.shape == (4, 1)
    assert filtered["license_plate"].to_list() == [
        "京A12345",
        "粤B12345",
        "沪C1234D",
        "苏E1234F",
    ]


def test_profile_data_basic():
    """Test basic profiling functionality."""
    data = {
        "numeric": [1, 2, 3, None, 5],
        "string": ["a", "a", "b", None, "c"],
        "date": [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            None,
            None,
            datetime(2023, 1, 3),
        ],
    }
    ldf = pl.LazyFrame(data)
    stats_df, stats_dict = profile_data(ldf)

    # Check DataFrame structure
    assert set(stats_df.columns) >= {
        "column",
        "dtype",
        "missing_count",
        "missing_%",
        "unique",
    }

    # Check numeric column stats
    numeric_stats = stats_dict["numeric"]
    assert numeric_stats["dtype"] == "Int64"
    assert numeric_stats["missing_count"] == 1
    assert numeric_stats["missing_%"] == 20.0
    assert numeric_stats["unique"] == 5  # 1, 2, 3, None, 5

    # Check string column stats
    string_stats = stats_dict["string"]
    assert string_stats["dtype"] == "String"
    assert string_stats["missing_count"] == 1
    # Corrected assertion: Expecting 4 unique values (including None if it's counted)
    assert string_stats["unique"] == 4  # a, b, c, None

    # Check date column stats
    date_stats = stats_dict["date"]
    assert "Datetime" in date_stats["dtype"]
    assert date_stats["missing_count"] == 2


def test_profile_data_empty():
    """Test with empty DataFrame."""
    ldf = pl.LazyFrame({})
    stats_df, stats_dict = profile_data(ldf)
    assert stats_df.is_empty()
    assert stats_dict == {}


def test_profile_data_specified_columns():
    """Test profiling specific columns only."""
    data = {"col1": [1, 2], "col2": ["a", "b"]}
    ldf = pl.LazyFrame(data)
    stats_df, stats_dict = profile_data(ldf, columns=["col1"])
    assert stats_df.shape[0] == 1
    assert stats_df["column"].to_list() == ["col1"]


def test_filter_by_year():
    data = {
        "timestamp": [
            datetime(2018, 1, 1),
            datetime(2019, 1, 1),
            datetime(2019, 12, 31),
            datetime(2020, 1, 1),
        ]
    }
    ldf = pl.LazyFrame(data)
    filtered = filter_by_year(ldf).collect()
    assert filtered.shape == (2, 1)
    assert filtered["timestamp"].dt.year().to_list() == [2019, 2019]
