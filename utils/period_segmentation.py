import polars as pl
import pickle
import glob
import os
from typing import Dict, Tuple, Optional
import logging


def clean_parquet_files(input_dir, cleaned_dir, license_plate_col, occupancy_col, timestamp_col):
    os.makedirs(cleaned_dir, exist_ok=True)
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    for fname in parquet_files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(cleaned_dir, fname)
        lazy_df = pl.scan_parquet(in_path)
        before = lazy_df.select(pl.count()).collect().item()
        lazy_df = lazy_df.drop_nulls()
        lazy_df = lazy_df.filter(~pl.any_horizontal([pl.col(col).is_nan() for col in lazy_df.columns]))
        after = lazy_df.select(pl.count()).collect().item()
        dropped = before - after
        if dropped > 0:
            logging.warning(f"[CLEAN] Dropped {dropped} rows with null or NaN values in {fname}.")
        else:
            logging.info(f"[CLEAN] No null or NaN rows dropped in {fname}.")
        lazy_df.sink_parquet(out_path)


def segment_periods_across_parquet(
    parquet_dir: str,
    output_dir: str,
    state_file: str = "last_state.pkl",
    license_plate_col: str = "license_plate",
    occupancy_col: str = "occupancy_status",
    timestamp_col: str = "timestamp",
    output_period_col: str = "period_id",
    verbose: bool = True,
):
    """
    Segments periods of consecutive occupancy status for each license plate across multiple parquet files.
    Assigns a globally unique period ID of the form '{license_plate}_{period_number}' for each period.
    Processes files one by one, assigns period IDs across files, and saves processed files and state.

    Args:
        parquet_dir (str): Directory containing input parquet files.
        output_dir (str): Directory to save processed parquet files with period IDs.
        state_file (str): Path to pickle file for storing last state between files.
        license_plate_col (str): Column name for license plate.
        occupancy_col (str): Column name for occupancy status.
        timestamp_col (str): Column name for timestamp.
        output_period_col (str): Name of the output period ID column.
        verbose (bool): If True, print progress info.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load last state if exists
    if os.path.exists(state_file):
        with open(state_file, "rb") as f:
            last_state: Dict[str, Tuple[int, int]] = pickle.load(f)
    else:
        last_state = {}  # {license_plate: (last_occupancy, last_period_number)}

    # List all parquet files in sorted order
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))

    for parquet_file in parquet_files:
        logging.info(f"Processing {parquet_file} ...")
        # Load file lazily
        lazy_df = pl.scan_parquet(parquet_file)
        # Sort by license_plate and timestamp
        lazy_df = lazy_df.sort([license_plate_col, timestamp_col])
        # Collect to eager for row-wise operation
        df = lazy_df.collect()

        # Prepare period_id column (globally unique string)
        period_ids = []
        for row in df.iter_rows(named=True):
            lp = row[license_plate_col]
            occ = row[occupancy_col]
            if lp not in last_state:
                period_number = 0
            else:
                last_occ, last_period_number = last_state[lp]
                if occ == last_occ:
                    period_number = last_period_number
                else:
                    period_number = last_period_number + 1
            period_id_str = f"{lp}_{period_number}"
            period_ids.append(period_id_str)
            last_state[lp] = (occ, period_number)

        df = df.with_columns(pl.Series(output_period_col, period_ids))

        # Save processed file
        out_path = os.path.join(output_dir, os.path.basename(parquet_file))
        df.write_parquet(out_path)
        logging.info(f"Saved {out_path} and updated state.")

        # Save last state after each file
        with open(state_file, "wb") as f:
            pickle.dump(last_state, f)
        logging.debug(f"State saved to {state_file}.") 