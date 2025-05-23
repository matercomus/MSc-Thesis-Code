import argparse
import os
import sys
import polars as pl
import plotly.express as px
from plotly_resampler import FigureResampler
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
import logging
import time
from tqdm import tqdm
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import polars.selectors as cs
from prettytable import PrettyTable
import re
import json

OUTPUT_DIR = "explore_outputs"
REMOVE_NULL_ROWS = False  # Set to True to drop all rows containing nulls before analysis

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# --- Utility Functions ---
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def is_nonempty_dir(path):
    return os.path.isdir(path) and any(os.scandir(path))

@contextmanager
def log_timing(task_name):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.info(f"[{task_name}] Completed in {elapsed:.2f} seconds.")

def save_datashader_image(agg, out_path):
    img = tf.shade(agg)
    img.to_pil().save(out_path)

def save_plotly(fig, out_base):
    fig.write_image(f"{out_base}.png")
    fig.write_html(f"{out_base}.html")

# --- Parallel Datashader helpers ---
def datashader_hist_helper(args):
    arr, col, out_path = args
    cvs = ds.Canvas(plot_width=400, plot_height=300)
    df_pd = pd.DataFrame({col: arr})
    agg = cvs.histogram(df_pd, col, bins=50)
    save_datashader_image(agg, out_path)
    return col

def datashader_pair_helper(args):
    df, x, y, out_path = args
    cvs = ds.Canvas(plot_width=300, plot_height=300)
    agg = cvs.points(df, x, y)
    save_datashader_image(agg, out_path)
    return (x, y)

# --- Tasks ---
def task_basic_stats(ldf):
    with log_timing("basic_stats"):
        outdir = os.path.join(OUTPUT_DIR, "basic_stats")
        ensure_dir(outdir)
        schema = dict(ldf.collect_schema().items())
        num_cols = ldf.select(cs.numeric()).collect_schema().names()
        all_cols = list(schema.keys())
        stats_exprs = [
            *(pl.col(c).mean().alias(f"{c}_mean") for c in num_cols),
            *(pl.col(c).std().alias(f"{c}_std") for c in num_cols),
            *(pl.col(c).min().alias(f"{c}_min") for c in num_cols),
            *(pl.col(c).max().alias(f"{c}_max") for c in num_cols),
            *(pl.col(c).median().alias(f"{c}_median") for c in num_cols),
            *(pl.col(c).null_count().alias(f"{c}_nulls") for c in all_cols),
            *(pl.col(c).n_unique().alias(f"{c}_unique") for c in all_cols),
        ]
        stats = ldf.select(stats_exprs).collect()
        # Parse stats into {stat: {col: value}}
        stat_names = ["mean", "std", "min", "max", "median", "nulls", "unique"]
        col_stats = {}
        for col in all_cols:
            col_stats[col] = {"dtype": str(schema[col])}
        for cname, val in zip(stats.columns, stats.rows()[0]):
            m = re.match(r"(.+)_([a-z]+)$", cname)
            if m:
                col, stat = m.groups()
                if col in col_stats and stat in stat_names:
                    col_stats[col][stat] = val
        # Build PrettyTable with two header rows: col name, dtype
        table = PrettyTable()
        table.field_names = ["statistic"] + all_cols
        dtype_row = [" "] + [str(schema[col]) for col in all_cols]
        table.add_row(dtype_row)
        for stat in stat_names:
            row = [stat]
            for col in all_cols:
                row.append(col_stats[col].get(stat, ""))
            table.add_row(row)
        with open(os.path.join(outdir, "describe.txt"), "w") as f:
            f.write(str(table) + "\n")
        # Write JSON output
        with open(os.path.join(outdir, "describe.json"), "w") as f:
            json.dump(col_stats, f, indent=2)

def task_histograms(ldf):
    with log_timing("histograms"):
        outdir = os.path.join(OUTPUT_DIR, "histograms")
        ensure_dir(outdir)
        schema = ldf.collect().schema
        cols = [col for col, dtype in schema.items() if pl.datatypes.is_numeric_dtype(dtype)]
        datashader_jobs = []
        plotly_cols = []
        for col in cols:
            col_df = ldf.select(col).collect()
            n_points = len(col_df)
            if n_points > 200_000:
                arr = col_df[col].to_numpy()
                out_path = os.path.join(outdir, f"{col}_hist_datashader.png")
                datashader_jobs.append((arr, col, out_path))
            else:
                plotly_cols.append((col, col_df))
        # Parallel Datashader
        if datashader_jobs:
            with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
                futures = {executor.submit(datashader_hist_helper, job): job[1] for job in datashader_jobs}
                for f in tqdm(as_completed(futures), total=len(futures), desc="Histograms (Datashader)"):
                    col = futures[f]
                    try:
                        f.result()
                        logging.info(f"[histograms] Used Datashader for {col}")
                    except Exception as e:
                        logging.error(f"[histograms] Datashader failed for {col}: {e}")
        # Serial Plotly
        for col, col_df in tqdm(plotly_cols, desc="Histograms (Plotly)"):
            fig = px.histogram(col_df, x=col, nbins=50, title=f"Histogram of {col}")
            save_plotly(fig, os.path.join(outdir, f"{col}_hist"))
            logging.info(f"[histograms] Used Plotly for {col} ({len(col_df)} points)")

def task_pairplot(ldf):
    with log_timing("pairplot"):
        outdir = os.path.join(OUTPUT_DIR, "pairplot")
        ensure_dir(outdir)
        schema = ldf.collect().schema
        num_cols = [col for col, dtype in schema.items() if pl.datatypes.is_numeric_dtype(dtype)]
        if 2 <= len(num_cols) <= 10:
            df = ldf.select(num_cols).collect().to_pandas()
            n_points = len(df)
            if n_points > 100_000:
                pairs = [(df, x, y, os.path.join(outdir, f"pair_{x}_{y}_datashader.png"))
                         for i, x in enumerate(num_cols) for j, y in enumerate(num_cols) if i < j]
                with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
                    futures = {executor.submit(datashader_pair_helper, job): (job[1], job[2]) for job in pairs}
                    for f in tqdm(as_completed(futures), total=len(futures), desc="Pairplot (Datashader)"):
                        x, y = futures[f]
                        try:
                            f.result()
                            logging.info(f"[pairplot] Used Datashader for pair {x}, {y} ({n_points} points)")
                        except Exception as e:
                            logging.error(f"[pairplot] Datashader failed for pair {x}, {y}: {e}")
            else:
                fig = px.scatter_matrix(df, dimensions=num_cols, title="Pairplot (Scatter Matrix)")
                fig.update_traces(diagonal_visible=False)
                save_plotly(fig, os.path.join(outdir, "pairplot"))
                logging.info(f"[pairplot] Used Plotly for {n_points} points")

def task_timeseries(ldf, time_col=None, value_col=None):
    with log_timing("timeseries"):
        if not time_col or not value_col:
            logging.warning("[timeseries] Skipping: time_col and value_col must be specified.")
            return
        outdir = os.path.join(OUTPUT_DIR, "timeseries")
        ensure_dir(outdir)
        df = ldf.select([time_col, value_col]).collect().to_pandas()
        n_points = len(df)
        if n_points > 100_000:
            fig = FigureResampler()
            fig.add_trace(
                dict(type="scattergl", x=df[time_col], y=df[value_col], mode="lines"),
                name=value_col
            )
            fig.update_layout(title=f"Time Series: {value_col}")
            fig.write_html(os.path.join(outdir, f"{value_col}_resampler.html"))
            logging.info(f"[timeseries] Used Plotly Resampler for {value_col} ({n_points} points)")
        else:
            fig = px.line(df, x=time_col, y=value_col, title=f"Time Series: {value_col}")
            save_plotly(fig, os.path.join(outdir, f"{value_col}_line"))
            logging.info(f"[timeseries] Used Plotly for {value_col} ({n_points} points)")

TASKS = {
    "basic_stats": task_basic_stats,
    "histograms": task_histograms,
    "pairplot": task_pairplot,
    # To use timeseries, pass --tasks timeseries --time-col <col> --value-col <col>
}

# --- Main ---
def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Data Exploration Script")
    parser.add_argument("--input", "-i", default="data/merged_all.parquet", help="Input Parquet file, comma-separated list of files, or directory")
    parser.add_argument("--output-dir", "-o", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--tasks", "-t", nargs="*", help="Tasks to run (default: all)")
    parser.add_argument("--time-col", type=str, help="Time column for timeseries task")
    parser.add_argument("--value-col", type=str, help="Value column for timeseries task")
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir

    if is_nonempty_dir(OUTPUT_DIR):
        logging.warning(f"WARNING: Output directory '{OUTPUT_DIR}' already exists and is not empty.")
        logging.warning("To proceed and overwrite previous outputs, type OVERWRITE and press Enter.")
        user_input = input("Type OVERWRITE to continue: ")
        if user_input.strip() != "OVERWRITE":
            logging.warning("Aborting to prevent accidental overwrite.")
            sys.exit(1)

    # Determine input files
    input_arg = args.input
    input_files = []
    if os.path.isdir(input_arg):
        # Directory: all .parquet files
        input_files = sorted([os.path.join(input_arg, f) for f in os.listdir(input_arg) if f.endswith('.parquet')])
    elif "," in input_arg:
        # Comma-separated list
        input_files = [f.strip() for f in input_arg.split(",") if f.strip()]
    else:
        # Single file
        input_files = [input_arg]

    to_run = args.tasks if args.tasks else TASKS.keys()

    for file_path in input_files:
        file_base = os.path.splitext(os.path.basename(file_path))[0]
        logging.info(f"Loading data from {file_path} (lazy mode)...")
        load_start = time.perf_counter()
        ldf = pl.scan_parquet(file_path)
        load_elapsed = time.perf_counter() - load_start
        logging.info(f"[data loading] Completed in {load_elapsed:.2f} seconds.")
        if REMOVE_NULL_ROWS:
            logging.info("[data cleaning] Removing all rows containing nulls before analysis.")
            ldf = ldf.drop_nulls()
        for task_name in to_run:
            # For each file, output to a subdirectory named after the file
            prev_output_dir = OUTPUT_DIR
            OUTPUT_DIR = os.path.join(prev_output_dir, file_base)
            if task_name == "timeseries":
                task_timeseries(ldf, time_col=args.time_col, value_col=args.value_col)
            elif task_name in TASKS:
                logging.info(f"Running task: {task_name} for {file_path}")
                TASKS[task_name](ldf)
            else:
                logging.warning(f"Unknown task: {task_name}")
            OUTPUT_DIR = prev_output_dir

if __name__ == "__main__":
    main()