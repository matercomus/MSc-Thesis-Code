"""
CSV to Parquet Converter for Taxi Trajectory Data
Converts large CSV files to Parquet format efficiently using Polars
"""

from pathlib import Path
import time
import polars as pl
from typing import Optional, List
import argparse
import glob
import os
import logging
from tqdm import tqdm
import csv

# --------------------------
# Column Definitions
# --------------------------

# Columns to focus on (target names)
TARGET_COLUMNS = [
    "company_code",      # QH (1st column)
    "vehicle_number",    # VehicleNum (2nd column)
    "plate_number",      # License plate (3rd column)
    "timestamp",         # Time (4th column)
    "longitude",         # Lng (5th column)
    "latitude",          # Lat (6th column)
    "instant_speed",     # Speed (7th column)
    "occupancy_status",  # OpenStatus (8th column)
    "status",           # Status (last column)
]

# Original column names in the CSV files (no headers)
CSV_COLUMNS = [
    "QH",           # Company code
    "VehicleNum",   # Vehicle number
    "PlateNum",     # License plate
    "Time",         # Timestamp
    "Lng",          # Longitude
    "Lat",          # Latitude
    "Speed",        # Speed
    "OpenStatus",   # Occupancy status
    "Status",       # Status (N or CZ)
]

# Mapping from CSV header to target column names
CSV_TO_TARGET = {
    "QH": "company_code",
    "VehicleNum": "vehicle_number",
    "PlateNum": "plate_number",
    "Time": "timestamp",
    "Lng": "longitude",
    "Lat": "latitude",
    "Speed": "instant_speed",
    "OpenStatus": "occupancy_status",
    "Status": "status"
}

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


def detect_csv_structure(csv_path: str) -> tuple:
    """Detect CSV structure using Polars"""
    try:
        # Try reading without headers first
        df = pl.read_csv(
            csv_path,
            n_rows=5,
            has_header=False,
            encoding='utf-8'
        )
        n_cols = len(df.columns)
        
        # If we have the expected number of columns, use our predefined names
        if n_cols == len(CSV_COLUMNS):
            return CSV_COLUMNS, False
            
        # Try reading with headers
        df_with_header = pl.read_csv(
            csv_path,
            n_rows=0,
            has_header=True,
            encoding='utf-8'
        )
        if len(df_with_header.columns) > 0:
            return list(df_with_header.columns), True
            
        raise ValueError(f"Could not detect CSV structure for {csv_path}")
    except Exception as e:
        logging.error(f"Error detecting CSV structure: {e}")
        raise


def perform_conversion(
    input_path: Path,
    out_path: Path,
    compression: str,
    columns_to_keep: Optional[List[str]] = None,
) -> pl.LazyFrame:
    """Core conversion logic using Polars"""
    columns_to_keep = columns_to_keep or TARGET_COLUMNS
    
    # Define dtypes for each column
    DTYPES = {
        "QH": pl.Utf8,  # Taxi company code
        "VehicleNum": pl.Utf8,
        "PlateNum": pl.Utf8,
        "Time": pl.Utf8,  # parse as string, can convert to datetime later if needed
        "Lng": pl.Float64,
        "Lat": pl.Float64,
        "Speed": pl.Float64,
        "OpenStatus": pl.Int64,
        "Status": pl.Utf8,
    }
    
    scan = pl.scan_csv(
        input_path,
        has_header=False,
        new_columns=CSV_COLUMNS,
        null_values=["N", "n", ""],
        ignore_errors=False,
        schema_overrides=DTYPES,
        encoding='utf8'  # Changed from utf-8 to utf8
    ).rename(CSV_TO_TARGET)
    
    scan = scan.select(columns_to_keep)
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
    logging.info(f"[START] {csv_path} -> {output_path}")
    start_time = time.time()
    stats = {'csv': csv_path, 'parquet': output_path, 'ok': False, 'error': None}
    try:
        # Check CSV header for required columns
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
        required = set(CSV_TO_TARGET.keys())
        missing = required - set(header)
        if missing:
            msg = f"Missing required columns in {csv_path}: {missing}"
            logging.error(msg)
            stats['error'] = msg
            return stats
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
        logging.info(f"[DONE] {csv_path} -> {output_path} in {duration:.1f}s")
    except Exception as e:
        stats['error'] = str(e)
        logging.error(f"[FAIL] {csv_path}: {e}")
        # Remove partial output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Batch CSV to Parquet converter for taxi data.")
    parser.add_argument('inputs', nargs='+', help="Input CSV files, globs, or directories.")
    parser.add_argument('--output-dir', '-o', default=None, help="Output directory for Parquet files.")
    parser.add_argument('--compression', default='zstd', help="Parquet compression (zstd, snappy, gzip, lz4). Default: zstd")
    args = parser.parse_args()

    files = find_csv_files(args.inputs)
    if not files:
        print("No input files found.")
        return

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        outputs = [os.path.join(args.output_dir, os.path.splitext(os.path.basename(f))[0] + '.parquet') for f in files]
    else:
        outputs = [os.path.splitext(f)[0] + '.parquet' for f in files]

    print(f"Found {len(files)} files. Processing sequentially.")
    print(f"Compression: {args.compression}")

    results = []
    for csv_path, output_path in tqdm(zip(files, outputs), total=len(files), desc="Files"):
        res = convert_one_file((csv_path, output_path, args.compression))
        results.append(res)
        if res['ok']:
            print(f"[OK] {res['csv']} -> {res['parquet']} | {res['before_size']//1024**2}MB -> {res['after_size']//1024**2}MB | ratio: {res['ratio']:.2f} | {res['duration']:.1f}s | {res['throughput']:.1f} MB/s")
        else:
            print(f"[FAIL] {res['csv']} | {res['error']}")

    # Summary
    ok = [r for r in results if r.get('ok')]
    print("\nSummary:")
    print(f"Converted: {len(ok)}/{len(results)} files successfully.")
    if ok:
        total_in = sum(r['before_size'] for r in ok)
        total_out = sum(r['after_size'] for r in ok)
        total_time = sum(r['duration'] for r in ok)
        print(f"Total input: {total_in//1024**3} GB, output: {total_out//1024**3} GB, ratio: {total_out/total_in:.2f}, total time: {total_time:.1f}s, avg throughput: {total_in/total_time/1024**2:.1f} MB/s")

if __name__ == "__main__":
    main()
