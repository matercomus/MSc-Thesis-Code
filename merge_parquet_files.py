import polars as pl
import glob
import os
import argparse
import time
import sys


def log(msg, verbose=True):
    if verbose:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def get_file_size(path):
    try:
        return os.path.getsize(path) / (1024 ** 2)  # MB
    except Exception:
        return 0

def main():
    parser = argparse.ArgumentParser(description="Merge multiple Parquet files into one using Polars (streaming mode).")
    parser.add_argument(
        "--input-dir", "-i", default="data/parquet", help="Directory containing Parquet files (default: data/parquet)"
    )
    parser.add_argument(
        "--output", "-o", default="data/merged_all.parquet", help="Output merged Parquet file (default: data/merged_all.parquet)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    try:
        log("==== Starting Parquet merge (streaming) ====", args.verbose)
        start = time.time()
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
        if not files:
            log(f"No Parquet files found in {args.input_dir}", True)
            return
        log(f"Found {len(files)} Parquet files to merge.", args.verbose)

        # Safety: check if output file exists
        if os.path.exists(args.output):
            print(f"WARNING: Output file '{args.output}' already exists.")
            print("To proceed and overwrite, type OVERWRITE and press Enter.")
            user_input = input("Type OVERWRITE to continue: ")
            if user_input.strip() != "OVERWRITE":
                print("Aborting to prevent accidental overwrite.")
                sys.exit(1)

        # Streaming merge using scan_parquet and sink_parquet
        log(f"Preparing lazy scan of all files...", args.verbose)
        lazy_frames = []
        for idx, f in enumerate(files):
            log(f"[{idx+1}/{len(files)}] Including {os.path.basename(f)} in merge", args.verbose)
            lazy_frames.append(pl.scan_parquet(f))
        merged = pl.concat(lazy_frames)
        log("Writing merged Parquet file (streaming, low memory)...", args.verbose)
        merged.sink_parquet(args.output)
        log(f"Merged file written to {args.output}", args.verbose)
        duration = time.time() - start
        log(f"==== Merge complete in {duration:.1f} seconds ====", args.verbose)

        # Final file stats
        log("Reading merged file for stats...", args.verbose)
        df_final = pl.read_parquet(args.output)
        file_size_mb = get_file_size(args.output)
        log(f"Final merged file stats:", True)
        log(f"  Path: {args.output}", True)
        log(f"  Size: {file_size_mb:.2f} MB", True)
        log(f"  Rows: {df_final.height}", True)
        log(f"  Columns: {df_final.width}", True)
        log(f"  Column names: {df_final.columns}", args.verbose)
    except Exception as e:
        log(f"FATAL ERROR: {e}", True)
        sys.exit(1)

if __name__ == "__main__":
    main() 