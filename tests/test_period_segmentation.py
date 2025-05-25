import os
import shutil
import tempfile
import polars as pl
import pytest
from utils.period_segmentation import segment_periods_across_parquet

@pytest.fixture
def temp_dirs():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    state_file = os.path.join(tempfile.mkdtemp(), "state.pkl")
    yield input_dir, output_dir, state_file
    shutil.rmtree(input_dir)
    shutil.rmtree(output_dir)
    shutil.rmtree(os.path.dirname(state_file))

def create_test_parquet(input_dir, filename, data):
    df = pl.DataFrame(data)
    df.write_parquet(os.path.join(input_dir, filename))

def test_segment_periods_across_parquet(temp_dirs):
    input_dir, output_dir, state_file = temp_dirs
    # Create two files with a period spanning both
    data1 = {
        "license_plate": ["A", "A", "A", "B", "B"],
        "occupancy_status": [1, 1, 0, 1, 1],
        "timestamp": [1, 2, 3, 1, 2],
    }
    data2 = {
        "license_plate": ["A", "A", "B", "B"],
        "occupancy_status": [0, 1, 1, 0],
        "timestamp": [4, 5, 3, 4],
    }
    create_test_parquet(input_dir, "part1.parquet", data1)
    create_test_parquet(input_dir, "part2.parquet", data2)

    segment_periods_across_parquet(
        parquet_dir=input_dir,
        output_dir=output_dir,
        state_file=state_file,
        license_plate_col="license_plate",
        occupancy_col="occupancy_status",
        timestamp_col="timestamp",
        output_period_col="period_id",
        verbose=False,
    )

    # Read results
    df1 = pl.read_parquet(os.path.join(output_dir, "part1.parquet"))
    df2 = pl.read_parquet(os.path.join(output_dir, "part2.parquet"))
    # Check period IDs
    # For license_plate A: [1,1,0,0,1] -> periods: A_0 (1,1), A_1 (0), A_2 (1)
    # For license_plate B: [1,1,1,0] -> periods: B_0 (all 1s), B_1 (0)
    a_periods = df1.filter(pl.col("license_plate") == "A")["period_id"].to_list() + \
                df2.filter(pl.col("license_plate") == "A")["period_id"].to_list()
    b_periods = df1.filter(pl.col("license_plate") == "B")["period_id"].to_list() + \
                df2.filter(pl.col("license_plate") == "B")["period_id"].to_list()
    # Check that period IDs are unique and consistent
    assert a_periods == ["A_0", "A_0", "A_1", "A_1", "A_2"]
    assert b_periods == ["B_0", "B_0", "B_0", "B_1"] 