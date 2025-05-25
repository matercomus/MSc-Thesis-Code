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
from utils.pipeline_helpers import StepMetadataLogger, configure_logging
from utils.stats_and_plotting import (
    basic_stats, period_length_stats, period_length_histogram, periods_per_license_plate, period_speed_stats, period_speed_histogram,
    period_start_end_time_distribution, period_duration_vs_speed_scatter, occupancy_status_transitions, periods_per_day_hour,
    period_length_by_license_plate, period_start_end_map, period_start_time_vs_duration_heatmap, idle_vs_occupied_distribution,
    speed_outlier_boxplot
)
import yaml
import inspect
from dataclasses import dataclass
from typing import Optional, Dict, Any

OUTPUT_DIR = "explore_outputs"
REMOVE_NULL_ROWS = False  # Set to True to drop all rows containing nulls before analysis

@dataclass
class AnalysisContext:
    """Context object containing all data needed for analysis functions"""
    ldf: pl.LazyFrame
    output_dir: str
    metadata_logger: StepMetadataLogger
    columns: Dict[str, str]
    results: Dict[str, Any]
    
    def get_df(self) -> pl.DataFrame:
        """Get eager DataFrame when needed"""
        return self.ldf.collect()
    
    def get_column(self, key: str, default: str = None) -> str:
        """Get column name from config"""
        return self.columns.get(key, default or key)

# Registry of all available analysis functions
PLOT_REGISTRY = {
    "basic_stats": basic_stats,
    "period_length_stats": period_length_stats,
    "period_length_histogram": period_length_histogram,
    "periods_per_license_plate": periods_per_license_plate,
    "period_speed_stats": period_speed_stats,
    "period_speed_histogram": period_speed_histogram,
    "period_start_end_time_distribution": period_start_end_time_distribution,
    "period_duration_vs_speed_scatter": period_duration_vs_speed_scatter,
    "occupancy_status_transitions": occupancy_status_transitions,
    "periods_per_day_hour": periods_per_day_hour,
    "period_length_by_license_plate": period_length_by_license_plate,
    "period_start_end_map": period_start_end_map,
    "period_start_time_vs_duration_heatmap": period_start_time_vs_duration_heatmap,
    "idle_vs_occupied_distribution": idle_vs_occupied_distribution,
    "speed_outlier_boxplot": speed_outlier_boxplot,
}

# Which functions require a DataFrame (not LazyFrame)
FUNC_NEEDS_DF = {
    "period_length_histogram": True,
    "period_speed_histogram": True,
}

# Which functions require the output of a previous function
FUNC_DEPENDENCIES = {
    "period_length_histogram": "period_length_stats",
    "period_speed_histogram": "period_speed_stats",
}

configure_logging()

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

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Data Exploration Script (YAML-configurable, file-by-file)")
    parser.add_argument("--config", type=str, default="period_analysis.yaml", help="YAML config file for period analysis")
    parser.add_argument("--output-dir", "-o", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--period-dir", type=str, default="data/steps_data/01_segment_periods/", help="Directory with period-segmented parquet files")
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    ensure_dir(OUTPUT_DIR)

    # Load config
    logging.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    columns = config.get('columns', {})
    plots = config.get('plots', [])
    logging.info(f"Loaded config: columns={columns}, plots={[p['name'] if isinstance(p, dict) else p for p in plots]}")

    # List all period-segmented files
    period_files = [os.path.join(args.period_dir, f) for f in os.listdir(args.period_dir) if f.endswith('.parquet')]
    if not period_files:
        logging.error(f"No parquet files found in {args.period_dir}")
        sys.exit(1)
    logging.info(f"Found {len(period_files)} parquet files in {args.period_dir}")

    for file_idx, period_file in enumerate(period_files):
        file_base = os.path.splitext(os.path.basename(period_file))[0]
        file_outdir = os.path.join(OUTPUT_DIR, file_base)
        ensure_dir(file_outdir)
        logging.info(f"[{file_idx+1}/{len(period_files)}] Processing file: {period_file} -> {file_outdir}")
        try:
            ldf = pl.scan_parquet(period_file)
            if REMOVE_NULL_ROWS:
                logging.info(f"Dropping null rows for {period_file}")
                ldf = ldf.drop_nulls()
            
            metadata_logger = StepMetadataLogger(output_dir=file_outdir)
            results = {}
            
            # Create analysis context
            context = AnalysisContext(
                ldf=ldf,
                output_dir=file_outdir,
                metadata_logger=metadata_logger,
                columns=columns,
                results=results
            )
            
            for plot_idx, plot in enumerate(plots):
                if isinstance(plot, dict):
                    name = plot['name']
                    params = plot.get('params', {})
                else:
                    name = plot
                    params = {}
                func = PLOT_REGISTRY.get(name)
                if not func:
                    logging.warning(f"Unknown plot/stat function: {name}")
                    continue
                
                # Handle dependencies
                if name in FUNC_DEPENDENCIES:
                    dep_name = FUNC_DEPENDENCIES[name]
                    dep_result = results.get(dep_name)
                    if dep_result is not None:
                        if name == "period_length_histogram":
                            context.results["period_lengths_df"] = dep_result
                        elif name == "period_speed_histogram":
                            context.results["period_speeds_df"] = dep_result
                
                logging.info(f"Running [{plot_idx+1}/{len(plots)}] {name} on {period_file}")
                try:
                    # Simply pass the context - let each function extract what it needs
                    result = func(context)
                    results[name] = result
                    context.results[name] = result
                    logging.info(f"Completed {name} for {period_file}")
                except Exception as e:
                    logging.error(f"Failed to run {name} on {period_file}: {e}")
            
            metadata_logger.save()
            logging.info(f"Finished all plots/stats for {period_file}. Metadata saved to {os.path.join(file_outdir, 'step_metadata.json')}")
        except Exception as e:
            logging.error(f"Failed to process file {period_file}: {e}")

    logging.info(f"All files processed. Outputs in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()