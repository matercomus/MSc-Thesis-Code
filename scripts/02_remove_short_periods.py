import argparse
import logging
from utils.pipeline_helpers import configure_logging

def main():
    parser = argparse.ArgumentParser(description="Step 02: Remove periods with total distance below a threshold.")
    parser.add_argument('--input-dir', required=True, help='Input directory containing segmented parquet files')
    parser.add_argument('--output-dir', required=True, help='Output directory for filtered parquet files')
    parser.add_argument('--distance-threshold', type=float, default=1.0, help='Minimum total distance (km) for a period to be kept')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    configure_logging()
    logging.info(f"[TODO] Remove short periods: {args}")
    # TODO: Implement logic to remove periods with total distance < threshold

if __name__ == "__main__":
    main() 