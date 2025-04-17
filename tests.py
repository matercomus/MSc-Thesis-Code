# tests.py

from datetime import datetime
import polars as pl
import pytest
from utils import (
    filter_chinese_license_plates,
    profile_data,
    filter_by_date,
    add_time_distance_calcs,
    add_abnormality_flags,
    add_implied_speed,
    select_final_columns,
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


def test_filter_by_date():
    data = {
        "timestamp": [
            datetime(2018, 1, 1),
            datetime(2019, 1, 1),
            datetime(2019, 12, 31),
            datetime(2020, 1, 1),
        ]
    }
    ldf = pl.LazyFrame(data)
    filtered = filter_by_date(ldf).collect()
    assert filtered.shape == (2, 1)
    assert filtered["timestamp"].dt.year().to_list() == [2019, 2019]


def test_filter_by_date_range():
    data = {
        "timestamp": [
            datetime(2019, 1, 1),
            datetime(2019, 6, 15),
            datetime(2019, 12, 31),
            datetime(2020, 1, 1),
        ]
    }
    ldf = pl.LazyFrame(data)
    filtered = filter_by_date(
        ldf,
        start_date=datetime(2019, 6, 1),
        end_date=datetime(2019, 6, 30),
    ).collect()
    assert filtered.shape == (1, 1)
    assert filtered["timestamp"].to_list() == [datetime(2019, 6, 15)]


def test_filter_by_single_day():
    data = {
        "timestamp": [
            datetime(2019, 1, 1),
            datetime(2019, 1, 2),
            datetime(2019, 1, 3),
        ]
    }
    ldf = pl.LazyFrame(data)
    filtered = filter_by_date(
        ldf,
        start_date=datetime(2019, 1, 2),
        end_date=datetime(2019, 1, 2),
    ).collect()
    assert filtered.shape == (1, 1)
    assert filtered["timestamp"].to_list() == [datetime(2019, 1, 2)]


def test_filter_by_month():
    data = {
        "timestamp": [
            datetime(2019, 1, 1),
            datetime(2019, 2, 1),
            datetime(2019, 2, 15),
            datetime(2019, 3, 1),
        ]
    }
    ldf = pl.LazyFrame(data)
    filtered = filter_by_date(
        ldf,
        start_date=datetime(2019, 2, 1),
        end_date=datetime(2019, 2, 28),
    ).collect()
    assert filtered.shape == (2, 1)
    assert filtered["timestamp"].to_list() == [
        datetime(2019, 2, 1),
        datetime(2019, 2, 15),
    ]


def test_filter_by_date_with_date_range():
    data = {
        "timestamp": [
            datetime(2018, 12, 31),
            datetime(2019, 1, 1),
            datetime(2019, 12, 31),
            datetime(2020, 1, 1),
        ]
    }
    ldf = pl.LazyFrame(data)
    filtered = filter_by_date(
        ldf,
        correct_year=2019,
        start_date=datetime(2019, 1, 1),
        end_date=datetime(2019, 12, 31),
    ).collect()
    assert filtered.shape == (2, 1)
    assert filtered["timestamp"].to_list() == [
        datetime(2019, 1, 1),
        datetime(2019, 12, 31),
    ]


def test_filter_by_date_empty_dataframe():
    ldf = pl.LazyFrame({"timestamp": []}, schema={"timestamp": pl.Datetime})
    filtered = filter_by_date(ldf).collect()
    assert filtered.is_empty()


def test_filter_by_date_invalid_column():
    ldf = pl.LazyFrame({"invalid_column": [1, 2, 3]})
    with pytest.raises(ValueError, match="Column 'timestamp' not found in DataFrame"):
        filter_by_date(ldf).collect()


def test_filter_by_date_invalid_date_type():
    ldf = pl.LazyFrame({"timestamp": ["not a date", "also not a date"]})
    with pytest.raises(
        ValueError, match="Column 'timestamp' must be a datetime or date type"
    ):
        filter_by_date(ldf).collect()


@pytest.fixture
def sample_data():
    return pl.LazyFrame(
        {
            "license_plate": ["A", "A", "B", "B"],
            "timestamp": [
                "2023-01-01T00:00:00",
                "2023-01-01T00:00:05",
                "2023-01-01T00:00:00",
                "2023-01-01T00:00:01",
            ],
            "latitude": [0.0, 1.0, 45.0, 45.001],
            "longitude": [0.0, 0.0, 0.0, 0.0],
        }
    ).with_columns(pl.col("timestamp").str.strptime(pl.Datetime))


def test_time_distance_calcs(sample_data):
    df = add_time_distance_calcs(
        sample_data.sort(["license_plate", "timestamp"])
    ).collect()
    assert df["time_diff_seconds"].to_list() == [None, 5.0, None, 1.0]
    assert df["distance_km"][1] == pytest.approx(111.195, rel=1e-3)
    assert df["distance_km"][3] == pytest.approx(0.11119, rel=1e-2)


@pytest.mark.parametrize(
    "time_diff, distance, expected_speed",
    [
        (5.0, 1.0, 720.0),  # 1km in 5s = 720 km/h
        (0.0, 1.0, 0.0),  # Zero time case
        (3600.0, 100.0, 100.0),  # 100km in 1h = 100 km/h
    ],
)
def test_implied_speed(time_diff, distance, expected_speed):
    df = pl.LazyFrame({"time_diff_seconds": [time_diff], "distance_km": [distance]})
    result = add_implied_speed(df).collect()
    assert result["implied_speed_kph"][0] == pytest.approx(expected_speed)


def test_abnormality_flags(sample_data):
    df = (
        sample_data.pipe(add_time_distance_calcs)
        .pipe(add_implied_speed)
        .pipe(add_abnormality_flags, 4.0, 100.0)
        .collect()
    )
    assert df["is_temporal_gap"].to_list() == [None, True, None, False]
    assert df["is_position_jump"].to_list() == [False, True, False, True]

def test_final_columns(sample_data):
    df = (
        sample_data.pipe(add_time_distance_calcs)
        .pipe(add_implied_speed)
        .pipe(add_abnormality_flags, 60.0, 100.0)
        .pipe(select_final_columns)
        .collect()
    )
    assert set(df.columns) == {
        "license_plate",
        "timestamp",
        "longitude",
        "latitude",
        "time_diff_seconds",
        "distance_km",
        "implied_speed_kph",
        "is_temporal_gap",
        "is_position_jump",
    }
