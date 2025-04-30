# tests.py

from datetime import datetime, timedelta
import polars as pl
import pytest
from utils import (
    filter_chinese_license_plates,
    profile_data,
    filter_by_date,
    add_time_distance_calcs,
    add_abnormality_flags,
    add_implied_speed,
    add_period_id,
    summarize_periods,
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
    stats_df, _ = profile_data(ldf, columns=["col1"])
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


@pytest.fixture
def sample_data_periods():
    base_time = datetime(2023, 1, 1, 0, 0, 0)

    return pl.DataFrame(
        {
            "license_plate": ["A", "A", "A", "A", "B", "B"],
            "timestamp": [
                base_time,
                base_time + timedelta(microseconds=1),
                base_time + timedelta(microseconds=2),
                base_time + timedelta(microseconds=3),
                base_time,
                base_time + timedelta(microseconds=1),
            ],
            "latitude": [39.9, 39.91, 39.92, 39.93, 40.0, 40.01],
            "longitude": [116.4, 116.41, 116.42, 116.43, 116.5, 116.51],
            "occupancy_status": ["1", "1", "0", "0", "0", "1"],
            "implied_speed_kph": [50.0, 45.0, 0.0, 0.0, 30.0, 120.0],
            "time_diff_seconds": [60, 60, 60, 60, 60, 60],
            "distance_km": [0.5, 0.45, 0.0, 0.0, 0.3, 1.2],
        }
    )


@pytest.fixture
def sample_data_multiple_periods():
    """More complex sample with explicitly included required columns."""
    base_time = datetime(2023, 1, 1, 0, 0, 0)

    return pl.DataFrame(
        {
            "license_plate": ["A"] * 5 + ["B"] * 5,
            "timestamp": [
                base_time,
                base_time + timedelta(seconds=1),
                base_time + timedelta(seconds=2),
                base_time + timedelta(seconds=3),
                base_time + timedelta(seconds=4),
                base_time,
                base_time + timedelta(seconds=1),
                base_time + timedelta(seconds=2),
                base_time + timedelta(seconds=3),
                base_time + timedelta(seconds=4),
            ],
            "occupancy_status": ["1", "1", "0", "1", "0", "0", "0", "1", "0", "0"],
            "implied_speed_kph": [40, 42, 0, 44, 46, 30, 35, 32, 33, 34],
            "time_diff_seconds": [60] * 10,
            "distance_km": [0.6, 0.7, 0, 0.8, 0.9, 0.2, 0.3, 0.25, 0.3, 0.35],
        }
    )


def test_add_period_id_assigns_correctly(sample_data_periods):
    df = sample_data_periods.sort(["license_plate", "timestamp"])
    df_with_period = add_period_id(df)

    periods_a = (
        df_with_period.filter(pl.col("license_plate") == "A")
        .select("period_id")
        .unique()
        .to_series()
        .to_list()
    )
    periods_b = (
        df_with_period.filter(pl.col("license_plate") == "B")
        .select("period_id")
        .unique()
        .to_series()
        .to_list()
    )

    assert sorted(periods_a) == [1, 2]
    assert sorted(periods_b) == [3, 4]
    assert df_with_period["period_id"][0] == 1


def test_summarize_periods_basic(sample_data_periods):
    df = sample_data_periods.sort(["license_plate", "timestamp"])
    df_with_period = add_period_id(df)

    summary = summarize_periods(df_with_period)

    # First period of A
    a_period_1 = summary.filter(
        (pl.col("license_plate") == "A") & (pl.col("period_id") == 1)
    ).to_dicts()[0]

    assert a_period_1["duration"] == timedelta(microseconds=1)
    assert a_period_1["count_rows"] == 2
    assert abs(a_period_1["avg_implied_speed"] - 47.5) < 1e-6

    # Second period of A
    a_period_2 = summary.filter(
        (pl.col("license_plate") == "A") & (pl.col("period_id") == 2)
    ).to_dicts()[0]

    assert a_period_2["duration"] == timedelta(microseconds=1)
    assert a_period_2["count_rows"] == 2
    assert abs(a_period_2["avg_implied_speed"] - 0.0) < 1e-6


def test_summarize_periods_multiple_periods(sample_data_multiple_periods):
    df = sample_data_multiple_periods.sort(["license_plate", "timestamp"])
    df_with_period = add_period_id(df)

    summary = summarize_periods(df_with_period)

    periods_a = (
        summary.filter(pl.col("license_plate") == "A")
        .select("period_id")
        .unique()
        .to_series()
        .to_list()
    )
    periods_b = (
        summary.filter(pl.col("license_plate") == "B")
        .select("period_id")
        .unique()
        .to_series()
        .to_list()
    )

    # Plate A has 4 periods due to occupancy changes at rows 2 & 3
    assert sorted(periods_a) == [1, 2, 3, 4]
    # Plate B has 3 periods as expected
    assert sorted(periods_b) == [5, 6, 7]

    for period_row in summary.to_dicts():
        assert period_row["count_rows"] >= 1
        assert isinstance(period_row["duration"], timedelta)
        assert period_row["avg_implied_speed"] >= 0


def test_summarize_periods_empty():
    empty_df = pl.DataFrame(
        {
            "license_plate": pl.Series([], dtype=pl.Utf8),
            "timestamp": pl.Series([], dtype=pl.Datetime("us")),
            "occupancy_status": pl.Series([], dtype=pl.Utf8),
            "implied_speed_kph": pl.Series([], dtype=pl.Float64),
            "time_diff_seconds": pl.Series([], dtype=pl.Float64),
            "distance_km": pl.Series([], dtype=pl.Float64),
            "period_id": pl.Series([], dtype=pl.Int32),
        }
    )
    summary = summarize_periods(empty_df)
    assert summary.is_empty()
    
def test_detect_outliers_pd_basic():
    import pytest
    pytest.importorskip("sklearn", reason="scikit-learn is required for outlier detection tests")
    import pandas as pd
    from utils import detect_outliers_pd
    # Simple linear trajectory for a single vehicle
    df = pd.DataFrame({
        'license_plate': ['A', 'A', 'A'],
        'timestamp': ['2023-01-01T00:00:00'] * 3,
        'latitude': [0.0, 1.0, 2.0],
        'longitude': [0.0, 1.0, 2.0],
    })
    # Perform outlier detection
    out = detect_outliers_pd(df, contamination=0.1, random_state=0)
    # Check output type and values
    assert isinstance(out, pd.Series)
    assert len(out) == 3
    assert set(out.unique()).issubset({-1, 1})


def test_period_id_increments_on_license_plate_change():
    df = pl.DataFrame(
        {
            "license_plate": ["A", "A", "B", "B", "A"],
            "timestamp": [
                datetime(2023, 1, 1, 0, 0, 0),
                datetime(2023, 1, 1, 0, 0, 1),
                datetime(2023, 1, 1, 0, 0, 2),
                datetime(2023, 1, 1, 0, 0, 3),
                datetime(2023, 1, 1, 0, 0, 4),
            ],
            "occupancy_status": ["1", "1", "1", "1", "1"],
            "implied_speed_kph": [10, 12, 14, 16, 18],
            "time_diff_seconds": [1, 1, 1, 1, 1],
            "distance_km": [0.1, 0.1, 0.1, 0.1, 0.1],
        }
    )

    df_with_period = add_period_id(df)
    expected_period_ids = [1, 1, 2, 2, 3]
    assert df_with_period["period_id"].to_list() == expected_period_ids
