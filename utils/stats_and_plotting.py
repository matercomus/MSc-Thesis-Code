import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from typing import Optional
from utils.pipeline_helpers import StepMetadataLogger
import polars.selectors as cs
from prettytable import PrettyTable
import numpy as np
import pandas as pd

def basic_stats(ldf, output_dir, metadata_logger: Optional[StepMetadataLogger] = None):
    os.makedirs(output_dir, exist_ok=True)
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
    stat_names = ["mean", "std", "min", "max", "median", "nulls", "unique"]
    col_stats = {}
    for col in all_cols:
        col_stats[col] = {"dtype": str(schema[col])}
    for cname, val in zip(stats.columns, stats.rows()[0]):
        import re
        m = re.match(r"(.+)_([a-z]+)$", cname)
        if m:
            col, stat = m.groups()
            if col in col_stats and stat in stat_names:
                col_stats[col][stat] = val
    table = PrettyTable()
    table.field_names = ["statistic"] + all_cols
    dtype_row = [" "] + [str(schema[col]) for col in all_cols]
    table.add_row(dtype_row)
    for stat in stat_names:
        row = [stat]
        for col in all_cols:
            row.append(col_stats[col].get(stat, ""))
        table.add_row(row)
    with open(os.path.join(output_dir, "describe.txt"), "w") as f:
        f.write(str(table) + "\n")
    with open(os.path.join(output_dir, "describe.json"), "w") as f:
        json.dump(col_stats, f, indent=2)
    if metadata_logger:
        metadata_logger.add_stat("basic_stats", col_stats)
        metadata_logger.add_stat("basic_stats_table_txt", os.path.join(output_dir, "describe.txt"))
        metadata_logger.add_stat("basic_stats_json", os.path.join(output_dir, "describe.json"))

def period_length_stats(ldf, period_col, output_dir, metadata_logger: Optional[StepMetadataLogger] = None):
    os.makedirs(output_dir, exist_ok=True)
    # Compute period lengths
    period_lengths = ldf.groupby(period_col).agg(pl.len().alias("period_length"))
    period_lengths_df = period_lengths.collect()
    stats = {
        "mean": float(period_lengths_df["period_length"].mean()),
        "median": float(period_lengths_df["period_length"].median()),
        "min": int(period_lengths_df["period_length"].min()),
        "max": int(period_lengths_df["period_length"].max()),
        "std": float(period_lengths_df["period_length"].std()),
        "count": int(period_lengths_df.height),
    }
    # Save stats
    with open(os.path.join(output_dir, "period_length_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    if metadata_logger:
        metadata_logger.add_stat("period_length_stats", stats)
        metadata_logger.add_stat("period_length_stats_json", os.path.join(output_dir, "period_length_stats.json"))
    return period_lengths_df

def period_length_histogram(period_lengths_df, output_dir, metadata_logger: Optional[StepMetadataLogger] = None):
    import plotly.express as px
    fig = px.histogram(period_lengths_df.to_pandas(), x="period_length", nbins=50, title="Histogram of Period Lengths")
    out_path = os.path.join(output_dir, "period_length_hist.png")
    fig.write_image(out_path)
    fig.write_html(out_path.replace(".png", ".html"))
    if metadata_logger:
        metadata_logger.add_stat("period_length_histogram_plot", out_path)
        metadata_logger.add_stat("period_length_histogram_html", out_path.replace(".png", ".html"))
    return out_path

def periods_per_license_plate(ldf, license_plate_col, period_col, output_dir, metadata_logger: Optional[StepMetadataLogger] = None):
    os.makedirs(output_dir, exist_ok=True)
    periods_per_lp = ldf.groupby(license_plate_col).agg(pl.col(period_col).n_unique().alias("n_periods"))
    periods_per_lp_df = periods_per_lp.collect()
    stats = {
        "mean": float(periods_per_lp_df["n_periods"].mean()),
        "median": float(periods_per_lp_df["n_periods"].median()),
        "min": int(periods_per_lp_df["n_periods"].min()),
        "max": int(periods_per_lp_df["n_periods"].max()),
        "std": float(periods_per_lp_df["n_periods"].std()),
        "count": int(periods_per_lp_df.height),
    }
    with open(os.path.join(output_dir, "periods_per_license_plate_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    if metadata_logger:
        metadata_logger.add_stat("periods_per_license_plate_stats", stats)
        metadata_logger.add_stat("periods_per_license_plate_stats_json", os.path.join(output_dir, "periods_per_license_plate_stats.json"))
    # Plot
    fig = px.histogram(periods_per_lp_df.to_pandas(), x="n_periods", nbins=50, title="Histogram of Periods per License Plate")
    out_path = os.path.join(output_dir, "periods_per_license_plate_hist.png")
    fig.write_image(out_path)
    fig.write_html(out_path.replace(".png", ".html"))
    if metadata_logger:
        metadata_logger.add_stat("periods_per_license_plate_histogram_plot", out_path)
        metadata_logger.add_stat("periods_per_license_plate_histogram_html", out_path.replace(".png", ".html"))
    return periods_per_lp_df

def period_speed_stats(ldf, period_col, timestamp_col, distance_col, output_dir, metadata_logger: Optional[StepMetadataLogger] = None):
    os.makedirs(output_dir, exist_ok=True)
    # Compute period speeds: (max(distance) - min(distance)) / (max(timestamp) - min(timestamp)) per period
    period_speeds = ldf.groupby(period_col).agg([
        (pl.col(distance_col).max() - pl.col(distance_col).min()).alias("distance_delta"),
        (pl.col(timestamp_col).max() - pl.col(timestamp_col).min()).alias("time_delta"),
    ])
    # Avoid division by zero
    period_speeds = period_speeds.with_columns(
        (pl.col("distance_delta") / pl.col("time_delta")).alias("speed")
    )
    period_speeds_df = period_speeds.collect()
    # Remove inf/nan speeds
    period_speeds_df = period_speeds_df.filter(
        (pl.col("speed").is_finite()) & (pl.col("speed") >= 0)
    )
    stats = {
        "mean": float(period_speeds_df["speed"].mean()),
        "median": float(period_speeds_df["speed"].median()),
        "min": float(period_speeds_df["speed"].min()),
        "max": float(period_speeds_df["speed"].max()),
        "std": float(period_speeds_df["speed"].std()),
        "count": int(period_speeds_df.height),
    }
    with open(os.path.join(output_dir, "period_speed_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    if metadata_logger:
        metadata_logger.add_stat("period_speed_stats", stats)
        metadata_logger.add_stat("period_speed_stats_json", os.path.join(output_dir, "period_speed_stats.json"))
    return period_speeds_df

def period_speed_histogram(period_speeds_df, output_dir, metadata_logger: Optional[StepMetadataLogger] = None):
    fig = px.histogram(period_speeds_df.to_pandas(), x="speed", nbins=50, title="Histogram of Period Speeds")
    out_path = os.path.join(output_dir, "period_speed_hist.png")
    fig.write_image(out_path)
    fig.write_html(out_path.replace(".png", ".html"))
    if metadata_logger:
        metadata_logger.add_stat("period_speed_histogram_plot", out_path)
        metadata_logger.add_stat("period_speed_histogram_html", out_path.replace(".png", ".html"))
    return out_path

def period_start_end_time_distribution(ldf, timestamp_col, output_dir, metadata_logger=None):
    os.makedirs(output_dir, exist_ok=True)
    # Extract start and end times per period
    df = ldf.groupby("period_id").agg([
        pl.col(timestamp_col).min().alias("start_time"),
        pl.col(timestamp_col).max().alias("end_time"),
    ]).collect().to_pandas()
    # Convert to datetime if not already
    for col in ["start_time", "end_time"]:
        if not np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')
    # Hour of day
    for col in ["start_time", "end_time"]:
        if df[col].notnull().any():
            fig = px.histogram(df, x=df[col].dt.hour, nbins=24, title=f"{col.replace('_', ' ').title()} Hour Distribution")
            out_path = os.path.join(output_dir, f"{col}_hour_hist.png")
            fig.write_image(out_path)
            fig.write_html(out_path.replace(".png", ".html"))
            if metadata_logger:
                metadata_logger.add_stat(f"{col}_hour_histogram_plot", out_path)
                metadata_logger.add_stat(f"{col}_hour_histogram_html", out_path.replace(".png", ".html"))
    # Day of week
    for col in ["start_time", "end_time"]:
        if df[col].notnull().any():
            fig = px.histogram(df, x=df[col].dt.dayofweek, nbins=7, title=f"{col.replace('_', ' ').title()} Day of Week Distribution")
            out_path = os.path.join(output_dir, f"{col}_dow_hist.png")
            fig.write_image(out_path)
            fig.write_html(out_path.replace(".png", ".html"))
            if metadata_logger:
                metadata_logger.add_stat(f"{col}_dow_histogram_plot", out_path)
                metadata_logger.add_stat(f"{col}_dow_histogram_html", out_path.replace(".png", ".html"))

def period_duration_vs_speed_scatter(ldf, period_col, timestamp_col, distance_col, output_dir, metadata_logger=None):
    os.makedirs(output_dir, exist_ok=True)
    df = ldf.groupby(period_col).agg([
        (pl.col(distance_col).max() - pl.col(distance_col).min()).alias("distance_delta"),
        (pl.col(timestamp_col).max() - pl.col(timestamp_col).min()).alias("time_delta"),
    ]).with_columns([
        (pl.col("distance_delta") / pl.col("time_delta")).alias("speed"),
        pl.col("time_delta").alias("duration")
    ]).collect().to_pandas()
    df = df[(df["speed"].notnull()) & (df["duration"] > 0) & (df["speed"] >= 0)]
    fig = px.scatter(df, x="duration", y="speed", title="Period Duration vs. Speed", trendline="ols")
    out_path = os.path.join(output_dir, "duration_vs_speed_scatter.png")
    fig.write_image(out_path)
    fig.write_html(out_path.replace(".png", ".html"))
    if metadata_logger:
        metadata_logger.add_stat("duration_vs_speed_scatter_plot", out_path)
        metadata_logger.add_stat("duration_vs_speed_scatter_html", out_path.replace(".png", ".html"))

def occupancy_status_transitions(ldf, period_col, occupancy_col, output_dir, metadata_logger=None):
    os.makedirs(output_dir, exist_ok=True)
    # For each period, get the first and last occupancy status
    df = ldf.groupby(period_col).agg([
        pl.col(occupancy_col).first().alias("start_occ"),
        pl.col(occupancy_col).last().alias("end_occ"),
    ]).collect().to_pandas()
    df["transition"] = df["start_occ"].astype(str) + "→" + df["end_occ"].astype(str)
    counts = df["transition"].value_counts().sort_index()
    fig = px.bar(x=counts.index, y=counts.values, title="Occupancy Status Transitions", labels={"x": "Transition", "y": "Count"})
    out_path = os.path.join(output_dir, "occupancy_status_transitions.png")
    fig.write_image(out_path)
    fig.write_html(out_path.replace(".png", ".html"))
    if metadata_logger:
        metadata_logger.add_stat("occupancy_status_transitions_plot", out_path)
        metadata_logger.add_stat("occupancy_status_transitions_html", out_path.replace(".png", ".html"))

def periods_per_day_hour(ldf, period_col, timestamp_col, output_dir, metadata_logger=None):
    os.makedirs(output_dir, exist_ok=True)
    df = ldf.groupby(period_col).agg([
        pl.col(timestamp_col).min().alias("start_time"),
    ]).collect().to_pandas()
    df["start_time"] = pd.to_datetime(df["start_time"], unit='s', errors='coerce')
    # Per day
    per_day = df["start_time"].dt.date.value_counts().sort_index()
    fig = px.bar(x=per_day.index, y=per_day.values, title="Periods Started per Day", labels={"x": "Date", "y": "Count"})
    out_path = os.path.join(output_dir, "periods_per_day.png")
    fig.write_image(out_path)
    fig.write_html(out_path.replace(".png", ".html"))
    if metadata_logger:
        metadata_logger.add_stat("periods_per_day_plot", out_path)
        metadata_logger.add_stat("periods_per_day_html", out_path.replace(".png", ".html"))
    # Per hour
    per_hour = df["start_time"].dt.hour.value_counts().sort_index()
    fig = px.bar(x=per_hour.index, y=per_hour.values, title="Periods Started per Hour", labels={"x": "Hour", "y": "Count"})
    out_path = os.path.join(output_dir, "periods_per_hour.png")
    fig.write_image(out_path)
    fig.write_html(out_path.replace(".png", ".html"))
    if metadata_logger:
        metadata_logger.add_stat("periods_per_hour_plot", out_path)
        metadata_logger.add_stat("periods_per_hour_html", out_path.replace(".png", ".html"))

def period_length_by_license_plate(ldf, license_plate_col, period_col, output_dir, metadata_logger=None):
    os.makedirs(output_dir, exist_ok=True)
    df = ldf.groupby([license_plate_col, period_col]).agg(pl.len().alias("period_length")).collect().to_pandas()
    # Only plot for license plates with many periods
    counts = df[license_plate_col].value_counts()
    top_plates = counts[counts > 10].index[:10]
    df_top = df[df[license_plate_col].isin(top_plates)]
    if not df_top.empty:
        fig = px.box(df_top, x=license_plate_col, y="period_length", points="all", title="Period Length by License Plate (Top 10)")
        out_path = os.path.join(output_dir, "period_length_by_license_plate.png")
        fig.write_image(out_path)
        fig.write_html(out_path.replace(".png", ".html"))
        if metadata_logger:
            metadata_logger.add_stat("period_length_by_license_plate_plot", out_path)
            metadata_logger.add_stat("period_length_by_license_plate_html", out_path.replace(".png", ".html"))

def period_start_end_map(ldf, period_col, start_lat_col, start_lon_col, end_lat_col, end_lon_col, output_dir, metadata_logger=None):
    os.makedirs(output_dir, exist_ok=True)
    # Only run if all columns exist
    schema = ldf.collect_schema()
    needed = [start_lat_col, start_lon_col, end_lat_col, end_lon_col]
    if not all(col in schema for col in needed):
        return
    df = ldf.groupby(period_col).agg([
        pl.col(start_lat_col).first().alias("start_lat"),
        pl.col(start_lon_col).first().alias("start_lon"),
        pl.col(end_lat_col).last().alias("end_lat"),
        pl.col(end_lon_col).last().alias("end_lon"),
    ]).collect().to_pandas()
    fig = px.scatter_mapbox(df, lat="start_lat", lon="start_lon", zoom=10, mapbox_style="carto-positron", title="Period Start Locations")
    out_path = os.path.join(output_dir, "period_start_map.png")
    fig.write_image(out_path)
    fig.write_html(out_path.replace(".png", ".html"))
    if metadata_logger:
        metadata_logger.add_stat("period_start_map_plot", out_path)
        metadata_logger.add_stat("period_start_map_html", out_path.replace(".png", ".html"))
    fig = px.scatter_mapbox(df, lat="end_lat", lon="end_lon", zoom=10, mapbox_style="carto-positron", title="Period End Locations")
    out_path = os.path.join(output_dir, "period_end_map.png")
    fig.write_image(out_path)
    fig.write_html(out_path.replace(".png", ".html"))
    if metadata_logger:
        metadata_logger.add_stat("period_end_map_plot", out_path)
        metadata_logger.add_stat("period_end_map_html", out_path.replace(".png", ".html"))

def period_start_time_vs_duration_heatmap(ldf, period_col, timestamp_col, output_dir, metadata_logger=None):
    os.makedirs(output_dir, exist_ok=True)
    df = ldf.groupby(period_col).agg([
        pl.col(timestamp_col).min().alias("start_time"),
        (pl.col(timestamp_col).max() - pl.col(timestamp_col).min()).alias("duration"),
    ]).collect().to_pandas()
    df["start_time"] = pd.to_datetime(df["start_time"], unit='s', errors='coerce')
    if not df.empty:
        fig = px.density_heatmap(df, x=df["start_time"].dt.hour, y="duration", nbinsx=24, title="Start Time vs. Duration Heatmap")
        out_path = os.path.join(output_dir, "start_time_vs_duration_heatmap.png")
        fig.write_image(out_path)
        fig.write_html(out_path.replace(".png", ".html"))
        if metadata_logger:
            metadata_logger.add_stat("start_time_vs_duration_heatmap_plot", out_path)
            metadata_logger.add_stat("start_time_vs_duration_heatmap_html", out_path.replace(".png", ".html"))

def idle_vs_occupied_distribution(ldf, period_col, occupancy_col, output_dir, metadata_logger=None):
    os.makedirs(output_dir, exist_ok=True)
    df = ldf.groupby(period_col).agg([
        pl.col(occupancy_col).first().alias("start_occ")
    ]).collect().to_pandas()
    counts = df["start_occ"].value_counts()
    fig = px.pie(values=counts.values, names=counts.index.astype(str), title="Idle vs. Occupied Periods")
    out_path = os.path.join(output_dir, "idle_vs_occupied_pie.png")
    fig.write_image(out_path)
    fig.write_html(out_path.replace(".png", ".html"))
    if metadata_logger:
        metadata_logger.add_stat("idle_vs_occupied_pie_plot", out_path)
        metadata_logger.add_stat("idle_vs_occupied_pie_html", out_path.replace(".png", ".html"))

def cumulative_distance_per_period(ldf, period_col, distance_col, output_dir, metadata_logger=None):
    os.makedirs(output_dir, exist_ok=True)
    df = ldf.groupby(period_col).agg([
        (pl.col(distance_col).max() - pl.col(distance_col).min()).alias("distance")
    ]).collect().to_pandas()
    fig = px.histogram(df, x="distance", nbins=50, title="Cumulative Distance per Period")
    out_path = os.path.join(output_dir, "cumulative_distance_per_period_hist.png")
    fig.write_image(out_path)
    fig.write_html(out_path.replace(".png", ".html"))
    if metadata_logger:
        metadata_logger.add_stat("cumulative_distance_per_period_hist_plot", out_path)
        metadata_logger.add_stat("cumulative_distance_per_period_hist_html", out_path.replace(".png", ".html"))

def speed_outlier_boxplot(ldf, period_col, timestamp_col, distance_col, output_dir, metadata_logger=None):
    os.makedirs(output_dir, exist_ok=True)
    df = ldf.groupby(period_col).agg([
        (pl.col(distance_col).max() - pl.col(distance_col).min()).alias("distance_delta"),
        (pl.col(timestamp_col).max() - pl.col(timestamp_col).min()).alias("time_delta"),
    ]).with_columns([
        (pl.col("distance_delta") / pl.col("time_delta")).alias("speed")
    ]).collect().to_pandas()
    df = df[(df["speed"].notnull()) & (df["speed"] >= 0)]
    fig = px.box(df, y="speed", points="all", title="Period Speed Outlier Boxplot")
    out_path = os.path.join(output_dir, "speed_outlier_boxplot.png")
    fig.write_image(out_path)
    fig.write_html(out_path.replace(".png", ".html"))
    if metadata_logger:
        metadata_logger.add_stat("speed_outlier_boxplot_plot", out_path)
        metadata_logger.add_stat("speed_outlier_boxplot_html", out_path.replace(".png", ".html")) 