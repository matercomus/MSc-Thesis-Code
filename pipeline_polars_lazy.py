import polars as pl
import logging
import argparse
from utils.pipeline_helpers import configure_logging


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="Polars Lazy Pipeline")
    parser.add_argument(
        "--input", "-i",
        default="data/parquet/2019.11.25_区域内.parquet",
        help="Input Parquet file (default: data/parquet/2019.11.25_区域内.parquet)"
    )
    args = parser.parse_args()
    input_path = args.input

    logging.info(f"Loading parquet file lazily: {input_path}")
    lazy_df = pl.scan_parquet(input_path)
    logging.info(f"Schema: {lazy_df.collect_schema()}")
    logging.info(f"Number of rows: {lazy_df.height}")
    logging.info("Removing rows containing NANs and nulls")
    lazy_df = lazy_df.drop_nans().drop_nulls()
    logging.info(f"Number of rows after removing NANs and nulls: {lazy_df.height}")


if __name__ == "__main__":
    main()