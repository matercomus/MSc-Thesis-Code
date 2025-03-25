"""
CSV to Parquet Converter for Taxi Trajectory Data
Converts large CSV files to Parquet format efficiently using Polars
"""

from pathlib import Path
import time
import polars as pl
from typing import Optional, List, Dict, Any

# --------------------------
# Column Definitions
# --------------------------

# Full column headers in order
FULL_COLUMN_HEADERS = [
    "company_id",  # First column: taxi company identifier (unimportant)
    "taxi_id",  # Second column: unique taxi identifier (unimportant)
    "license_plate",  # Third column: license plate number
    "timestamp",  # Fourth column: time of trajectory point collection
    "longitude",  # Fifth column: longitude of trajectory point
    "latitude",  # Sixth column: latitude of trajectory point
    "instant_speed",  # Seventh column: instantaneous speed (inaccurate, needs calculation)
    "occupancy_status",  # Eighth column: occupancy status (0=empty, 1=occupied, 2=reserved, 3=out of service)
    "unknown_9",  # Ninth column: no information (unimportant)
    "unknown_10",  # Tenth column: no information (unimportant)
    "unknown_11",  # Eleventh column: no information (unimportant)
]

# Columns to focus on
TARGET_COLUMNS = [
    "license_plate",  # 车牌号 (3rd column)
    "timestamp",  # 轨迹点采集时间 (4th column)
    "longitude",  # 经度 (5th column)
    "latitude",  # 纬度 (6th column)
    "instant_speed",  # 瞬时速度 (7th column, but needs recalculation)
    "occupancy_status",  # 载客状态 (8th column, 0=空车, 1=载客, 2=预约, 3=暂停服务)
]

# --------------------------
# File Handling Functions
# --------------------------


def validate_input_file(csv_path: str) -> Path:
    """Check if input file exists and return Path object"""
    input_path = Path(csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return input_path


def determine_output_path(csv_path: Path, output_path: Optional[str]) -> Path:
    """Determine output path (defaults to same as input with .parquet extension)"""
    return Path(output_path) if output_path else csv_path.with_suffix(".parquet")


def clean_up_on_failure(out_path: Path) -> None:
    """Remove partial output file if conversion fails"""
    if out_path.exists():
        out_path.unlink()
        print(f"Removed partial output file: {out_path}")


# --------------------------
# Conversion Functions
# --------------------------


def get_column_definitions() -> Dict[str, Any]:
    """Define column names and their types with precise mapping"""
    return {
        "license_plate": pl.Categorical,
        "timestamp": pl.Datetime,
        "longitude": pl.Float64,
        "latitude": pl.Float64,
        "instant_speed": pl.Float64,
        "occupancy_status": pl.Int8,  # Using small integer type for status
    }


def perform_conversion(
    input_path: Path,
    out_path: Path,
    compression: str,
    columns_to_keep: Optional[List[str]] = None,
) -> pl.LazyFrame:
    """Core conversion logic using Polars"""
    # If no specific columns provided, use default target columns
    columns_to_keep = columns_to_keep or TARGET_COLUMNS

    # Validate requested columns
    invalid_columns = set(columns_to_keep) - set(TARGET_COLUMNS)
    if invalid_columns:
        raise ValueError(f"Invalid columns requested: {invalid_columns}")

    # Prepare column definitions for selected columns
    column_defs = get_column_definitions()
    selected_column_defs = {col: column_defs[col] for col in columns_to_keep}

    # Perform CSV scanning
    scan = pl.scan_csv(
        input_path,
        has_header=False,
        new_columns=FULL_COLUMN_HEADERS,
        infer_schema_length=10000,
        schema_overrides={
            FULL_COLUMN_HEADERS[i]: selected_column_defs.get(col, pl.Utf8)
            for i, col in enumerate(FULL_COLUMN_HEADERS)
        },
        null_values=["N", "n", ""],
        ignore_errors=True,
        # columns=column_indices, # Remove this line
    ).select(
        columns_to_keep
    )  # Select the desired columns after scanning

    return scan


def write_parquet(scan: pl.LazyFrame, out_path: Path, compression: str) -> None:
    """Write the final parquet file"""
    scan.sink_parquet(out_path, compression=compression)


# --------------------------
# Reporting Functions
# --------------------------


def print_conversion_stats(out_path: Path, compression: str, start_time: float) -> None:
    """Print summary statistics after successful conversion"""
    duration = time.time() - start_time
    file_size = out_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 50)
    print("Conversion Complete!")
    print("-" * 50)
    print(f"Output file: {out_path}")
    print(f"File size: {file_size:.2f} MB")
    print(f"Compression: {compression}")
    print(f"Time taken: {duration:.2f} seconds")
    print("=" * 50)


# --------------------------
# Main Function
# --------------------------


def convert_csv_to_parquet(
    csv_path: str,
    output_path: Optional[str] = None,
    compression: str = "zstd",
    keep_columns: Optional[str] = None,
) -> None:
    """
    Convert CSV to Parquet format

    Args:
        csv_path: Path to input CSV file
        output_path: Optional output path (defaults to same as input with .parquet extension)
        compression: Compression algorithm (zstd, snappy, gzip, lz4)
        keep_columns: Comma-separated list of columns to keep (None keeps default target columns)
    """
    start_time = time.time()

    try:
        print(f"\nStarting conversion of {csv_path}...")

        # Validate and prepare paths
        input_path = validate_input_file(csv_path)
        out_path = determine_output_path(input_path, output_path)

        # Determine columns to keep
        columns_to_keep = None
        if keep_columns:
            columns_to_keep = [col.strip() for col in keep_columns.split(",")]

        # Perform the conversion
        scan = perform_conversion(input_path, out_path, compression, columns_to_keep)
        write_parquet(scan, out_path, compression)

        # Show results
        print_conversion_stats(out_path, compression, start_time)

    except Exception as e:
        print(f"\nError: {str(e)}")
        if "out_path" in locals():
            clean_up_on_failure(out_path)
        raise


# --------------------------
# Example Usage
# --------------------------

if __name__ == "__main__":
    # Example configuration
    config = {
        "csv_path": "2019.11.25.csv",
        "output_path": "2019.11.25.parquet",
        "compression": "zstd",
        # Optionally specify columns, otherwise uses default
        # "keep_columns": "license_plate,timestamp,longitude,latitude,instant_speed,occupancy_status",
    }

    print("Starting CSV to Parquet Conversion")
    print("-" * 50)
    print(f"Input: {config['csv_path']}")
    print(f"Output: {config['output_path']}")
    print(f"Compression: {config['compression']}")
    print("-" * 50)

    convert_csv_to_parquet(**config)
