"""
CSV to Parquet Converter for Taxi Trajectory Data
Converts large CSV files to Parquet format efficiently using Polars
"""

from pathlib import Path
import time
import polars as pl
from typing import Optional, List, Dict, Any
import argparse
import glob
import os
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from tqdm import tqdm

# --------------------------
# Column Definitions
# --------------------------

REGION_CSV_HEADERS = [
    "company",        # Taxi Company
    "vehicle_num",    # VehicleNum
    "timestamp",      # Time
    "longitude",      # Lng
    "latitude",       # Lat
    "instant_speed",  # Speed
    "occupancy_status" # OpenStatus
]

# Columns to focus on
TARGET_COLUMNS = [
    "vehicle_num",      # VehicleNum (2nd column)
    "timestamp",        # Time (3rd column)
    "longitude",        # Lng (4th column)
    "latitude",         # Lat (5th column)
    "instant_speed",    # Speed (6th column)
    "occupancy_status", # OpenStatus (7th column)
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
        "vehicle_num": pl.Categorical,
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
    columns_to_keep = columns_to_keep or TARGET_COLUMNS
    invalid_columns = set(columns_to_keep) - set(TARGET_COLUMNS)
    if invalid_columns:
        raise ValueError(f"Invalid columns requested: {invalid_columns}")
    column_defs = get_column_definitions()
    selected_column_defs = {col: column_defs[col] for col in columns_to_keep}
    scan = pl.scan_csv(
        input_path,
        has_header=False,
        new_columns=REGION_CSV_HEADERS,
        infer_schema_length=10000,
        schema_overrides={
            REGION_CSV_HEADERS[i]: selected_column_defs.get(col, pl.Utf8)
            for i, col in enumerate(REGION_CSV_HEADERS)
        },
        null_values=["N", "n", ""],
        ignore_errors=True,
    ).select(
        columns_to_keep
    )
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


def get_system_resources():
    """Return number of CPU cores and available memory in GB."""
    try:
        import psutil
        n_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        mem_gb = psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        n_cores = os.cpu_count() or 1
        mem_gb = 4  # fallback
    return n_cores, mem_gb


def find_csv_files(inputs):
    """Expand globs, directories, or file lists into a list of CSV file paths."""
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            files.extend(sorted(glob.glob(os.path.join(inp, '*.csv'))))
        elif '*' in inp or '?' in inp or '[' in inp:
            files.extend(sorted(glob.glob(inp)))
        else:
            files.append(inp)
    # Remove duplicates and non-files
    files = [f for f in sorted(set(files)) if os.path.isfile(f)]
    return files


def get_file_size(path):
    try:
        return os.path.getsize(path)
    except Exception:
        return 0


def convert_one_file(args):
    csv_path, output_path, compression = args
    start_time = time.time()
    stats = {'csv': csv_path, 'parquet': output_path, 'ok': False, 'error': None}
    try:
        before_size = get_file_size(csv_path)
        scan = perform_conversion(Path(csv_path), Path(output_path), compression, TARGET_COLUMNS)
        write_parquet(scan, Path(output_path), compression)
        after_size = get_file_size(output_path)
        duration = time.time() - start_time
        stats.update({
            'ok': True,
            'before_size': before_size,
            'after_size': after_size,
            'ratio': after_size / before_size if before_size else 0,
            'duration': duration,
            'throughput': before_size / duration / (1024 ** 2) if duration > 0 else 0
        })
    except Exception as e:
        stats['error'] = str(e)
        clean_up_on_failure(Path(output_path))
    return stats


def main():
    parser = argparse.ArgumentParser(description="Batch CSV to Parquet converter for taxi data.")
    parser.add_argument('inputs', nargs='+', help="Input CSV files, globs, or directories.")
    parser.add_argument('--output-dir', '-o', default=None, help="Output directory for Parquet files.")
    parser.add_argument('--compression', default='zstd', help="Parquet compression (zstd, snappy, gzip, lz4). Default: zstd")
    parser.add_argument('--processes', type=int, default=None, help="Number of parallel processes to use. Default: auto-detect.")
    parser.add_argument('--batch-size', type=int, default=None, help="Number of files per batch. Default: auto-detect.")
    args = parser.parse_args()

    n_cores, mem_gb = get_system_resources()
    n_proc = args.processes or max(1, n_cores - 1)
    batch_size = args.batch_size or max(1, min(8, int(mem_gb // 2)))

    files = find_csv_files(args.inputs)
    if not files:
        print("No input files found.")
        return

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        outputs = [os.path.join(args.output_dir, os.path.splitext(os.path.basename(f))[0] + '.parquet') for f in files]
    else:
        outputs = [os.path.splitext(f)[0] + '.parquet' for f in files]

    print(f"Found {len(files)} files. Using {n_proc} processes, batch size {batch_size}.")
    print(f"Compression: {args.compression}")

    tasks = list(zip(files, outputs, [args.compression]*len(files)))
    results = []
    with ProcessPoolExecutor(max_workers=n_proc) as executor:
        futs = [executor.submit(convert_one_file, t) for t in tasks]
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            if res['ok']:
                print(f"[OK] {res['csv']} -> {res['parquet']} | {res['before_size']//1024**2}MB -> {res['after_size']//1024**2}MB | ratio: {res['ratio']:.2f} | {res['duration']:.1f}s | {res['throughput']:.1f} MB/s")
            else:
                print(f"[FAIL] {res['csv']} | {res['error']}")

    # Summary
    ok = [r for r in results if r['ok']]
    print("\nSummary:")
    print(f"Converted: {len(ok)}/{len(results)} files successfully.")
    if ok:
        total_in = sum(r['before_size'] for r in ok)
        total_out = sum(r['after_size'] for r in ok)
        total_time = sum(r['duration'] for r in ok)
        print(f"Total input: {total_in//1024**3} GB, output: {total_out//1024**3} GB, ratio: {total_out/total_in:.2f}, total time: {total_time:.1f}s, avg throughput: {total_in/total_time/1024**2:.1f} MB/s")

if __name__ == "__main__":
    main()
