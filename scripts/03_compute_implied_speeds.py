import argparse
import logging

def main():
    parser = argparse.ArgumentParser(description="Step 03: Compute implied speeds for each period using great-circle distance and time difference.")
    parser.add_argument('--input-dir', required=True, help='Input directory containing filtered parquet files')
    parser.add_argument('--output-dir', required=True, help='Output directory for files with implied speeds')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    logging.info(f"[TODO] Compute implied speeds: {args}")
    # TODO: Implement logic to compute implied speeds for each period

if __name__ == "__main__":
    main() 