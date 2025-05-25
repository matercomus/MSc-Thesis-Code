import argparse
import logging
import os
import polars as pl
from utils.period_segmentation import segment_periods_across_parquet, clean_parquet_files
from utils.pipeline_helpers import configure_logging, StepMetadataLogger


def main():
    parser = argparse.ArgumentParser(description="Step 01: Segment periods of consecutive occupancy status for each license plate across parquet files.")
    parser.add_argument('--input-dir', default='data/original/', help='Input directory containing parquet files')
    parser.add_argument('--output-dir', default='data/steps_data/01_segment_periods/', help='Output directory for processed parquet files')
    parser.add_argument('--state-file', default='data/steps_data/01_segment_periods/last_state.pkl', help='Path to pickle file for storing last state between files')
    parser.add_argument('--license-plate-col', default='license_plate', help='Column name for license plate')
    parser.add_argument('--occupancy-col', default='occupancy_status', help='Column name for occupancy status')
    parser.add_argument('--timestamp-col', default='timestamp', help='Column name for timestamp')
    parser.add_argument('--output-period-col', default='period_id', help='Name of the output period ID column')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    configure_logging()
    logging.info(f"Running period segmentation: {args}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Metadata logger
    metadata_logger = StepMetadataLogger(output_dir=args.output_dir)

    # Clean input files first, collect stats (lazily)
    cleaned_dir = os.path.join(args.output_dir, 'cleaned')
    os.makedirs(cleaned_dir, exist_ok=True)
    parquet_files = [f for f in os.listdir(args.input_dir) if f.endswith('.parquet')]
    cleaning_stats = {"files": {}, "total_before": 0, "total_after": 0, "total_dropped": 0}
    for fname in parquet_files:
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(cleaned_dir, fname)
        lazy_df = pl.scan_parquet(in_path)
        before = lazy_df.select(pl.len()).collect().item()
        lazy_df = lazy_df.drop_nulls()
        float_cols = [name for name, dtype in lazy_df.collect_schema().items() if dtype in (pl.Float32, pl.Float64)]
        if float_cols:
            lazy_df = lazy_df.filter(~pl.any_horizontal([pl.col(col).is_nan() for col in float_cols]))
        after = lazy_df.select(pl.len()).collect().item()
        dropped = before - after
        cleaning_stats["files"][fname] = {"before": before, "after": after, "dropped": dropped}
        cleaning_stats["total_before"] += before
        cleaning_stats["total_after"] += after
        cleaning_stats["total_dropped"] += dropped
        lazy_df.sink_parquet(out_path)
    metadata_logger.add_stat("cleaning", cleaning_stats)
    metadata_logger.log_stats()

    # Segment periods, collect stats (lazily)
    segmented_stats = {"files": {}, "total_rows": 0}
    cleaned_files = [f for f in os.listdir(cleaned_dir) if f.endswith('.parquet')]
    for fname in cleaned_files:
        lazy_df = pl.scan_parquet(os.path.join(cleaned_dir, fname))
        rows = lazy_df.select(pl.len()).collect().item()
        segmented_stats["files"][fname] = {"rows": rows}
        segmented_stats["total_rows"] += rows
    metadata_logger.add_stat("segmentation", segmented_stats)
    metadata_logger.log_stats()

    # Run segmentation
    segment_periods_across_parquet(
        parquet_dir=cleaned_dir,
        output_dir=args.output_dir,
        state_file=args.state_file,
        license_plate_col=args.license_plate_col,
        occupancy_col=args.occupancy_col,
        timestamp_col=args.timestamp_col,
        output_period_col=args.output_period_col,
        verbose=args.verbose,
    )

    # Save metadata
    metadata_logger.save()

if __name__ == "__main__":
    main() 