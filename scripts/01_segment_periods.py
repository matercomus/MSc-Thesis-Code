import argparse
import logging
from utils.period_segmentation import segment_periods_across_parquet


def main():
    parser = argparse.ArgumentParser(description="Step 01: Segment periods of consecutive occupancy status for each license plate across parquet files.")
    parser.add_argument('--input-dir', required=True, help='Input directory containing parquet files')
    parser.add_argument('--output-dir', required=True, help='Output directory for processed parquet files')
    parser.add_argument('--state-file', default='last_state.pkl', help='Path to pickle file for storing last state between files')
    parser.add_argument('--license-plate-col', default='license_plate', help='Column name for license plate')
    parser.add_argument('--occupancy-col', default='occupancy_status', help='Column name for occupancy status')
    parser.add_argument('--timestamp-col', default='timestamp', help='Column name for timestamp')
    parser.add_argument('--output-period-col', default='period_id', help='Name of the output period ID column')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    logging.info(f"Running period segmentation: {args}")

    segment_periods_across_parquet(
        parquet_dir=args.input_dir,
        output_dir=args.output_dir,
        state_file=args.state_file,
        license_plate_col=args.license_plate_col,
        occupancy_col=args.occupancy_col,
        timestamp_col=args.timestamp_col,
        output_period_col=args.output_period_col,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main() 