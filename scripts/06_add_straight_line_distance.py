import argparse
import logging
from utils.pipeline_helpers import configure_logging

def main():
    parser = argparse.ArgumentParser(description="Step 06: Add straight-line distance from start to end point for each period.")
    parser.add_argument('--input-dir', required=True, help='Input directory containing simplified files')
    parser.add_argument('--output-dir', required=True, help='Output directory for files with straight-line distance')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    configure_logging()
    logging.info(f"[TODO] Add straight-line distance: {args}")
    # TODO: Implement logic to add straight-line distance

if __name__ == "__main__":
    main() 