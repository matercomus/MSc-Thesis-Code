import argparse
import logging
from utils.pipeline_helpers import configure_logging

def main():
    parser = argparse.ArgumentParser(description="Step 04: Remove periods with temporal or speed outliers (IQR-based).")
    parser.add_argument('--input-dir', required=True, help='Input directory containing files with implied speeds')
    parser.add_argument('--output-dir', required=True, help='Output directory for filtered files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    configure_logging()
    logging.info(f"[TODO] Remove outlier periods: {args}")
    # TODO: Implement logic to remove periods with temporal or speed outliers

if __name__ == "__main__":
    main() 