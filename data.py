"""
CSV to Parquet Converter for Taxi Trajectory Data
Converts large CSV files to Parquet format efficiently using Polars
"""

from pathlib import Path
import time
import polars as pl
from typing import Optional
import glob
import os
import logging
import csv

# --------------------------
# Column Definitions
# --------------------------

# Define schema with all columns as strings to avoid parsing errors
SCHEMA = {
    'company': pl.String,
    'phone': pl.String,
    'plate': pl.String,
    'timestamp': pl.String,
    'longitude': pl.String,
    'latitude': pl.String,
    'speed': pl.String,
    'direction': pl.String,
    'status': pl.String,
    'angle': pl.String,
    'status_code': pl.String
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


def determine_output_path(csv_path: Path) -> Path:
    """Determine output path in the parquet directory"""
    # Create output directory if it doesn't exist
    output_dir = Path(r"C:\Users\matt\Dev\MSc-Thesis-Code\data\parquet")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the same filename but with .parquet extension
    return output_dir / f"{csv_path.stem}.parquet"


def clean_up_on_failure(out_path: Path) -> None:
    """Remove partial output file if conversion fails"""
    if out_path.exists():
        out_path.unlink()
        print(f"Removed partial output file: {out_path}")


# --------------------------
# Conversion Functions
# --------------------------


def perform_conversion(
    input_path: Path,
    out_path: Path,
    compression: str = 'zstd'
) -> None:
    """Core conversion logic using Polars scanning"""
    try:
        # Scan CSV with error handling
        scan = pl.scan_csv(
            input_path,
            has_header=False,
            encoding='utf8',
            schema=SCHEMA,
            ignore_errors=True  # Skip rows with parsing errors
        )
        
        # Write to parquet
        scan.sink_parquet(
            out_path,
            compression=compression
        )
        
    except Exception as e:
        logging.error(f"Error converting {input_path}: {e}")
        clean_up_on_failure(out_path)
        raise


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
        required = set(SCHEMA.keys())
        missing = required - set(header)
        if missing:
            msg = f"Missing required columns in {csv_path}: {missing}"
            logging.error(msg)
            stats['error'] = msg
            return stats
        before_size = get_file_size(csv_path)
        perform_conversion(Path(csv_path), Path(output_path), compression)
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
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('conversion.log'),
            logging.StreamHandler()
        ]
    )
    
    # Process files one by one
    base_path = r"I:\北京市出租车GPS数据"
    files = sorted([
        f"{base_path}\\2019.11.25.csv",
        f"{base_path}\\2019.11.26.csv",
        f"{base_path}\\2019.11.27.csv",
        f"{base_path}\\2019.11.28.csv",
        f"{base_path}\\2019.11.29.csv",
        f"{base_path}\\2019.11.30.csv",
        f"{base_path}\\2019.12.01.csv"
    ])
    
    for csv_file in files:
        try:
            start_time = time.time()
            input_path = validate_input_file(csv_file)
            output_path = determine_output_path(input_path)
            
            logging.info(f"Converting {input_path} to {output_path}")
            perform_conversion(input_path, output_path)
            
            duration = time.time() - start_time
            logging.info(f"Conversion completed in {duration:.2f} seconds")
            
        except Exception as e:
            logging.error(f"Failed to process {csv_file}: {e}")
            continue

if __name__ == "__main__":
    main()
