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

def basic_stats(context):
    ldf = context.ldf
    output_dir = context.output_dir
    metadata_logger = context.metadata_logger
    
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

def period_length_stats(context):
    ldf = context.ldf
    output_dir = context.output_dir
    metadata_logger = context.metadata_logger
    period_col = context.get_column('period_col', 'period_id')
    
    os.makedirs(output_dir, exist_ok=True)
    # Compute period lengths
    period_lengths = ldf.group_by(period_col).agg(pl.len().alias("period_length"))
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

def period_length_histogram(context):
    output_dir = context.output_dir
    metadata_logger = context.metadata_logger
    period_lengths_df = context.results.get("period_lengths_df")
    
    if period_lengths_df is None:
        raise ValueError("period_length_histogram requires period_lengths_df from period_length_stats")
    
    import plotly.express as px
    fig = px.histogram(period_lengths_df.to_pandas(), x="period_length", nbins=50, title="Histogram of Period Lengths")
    out_path = os.path.join(output_dir, "period_length_hist.png")
    fig.write_image(out_path)
    fig.write_html(out_path.replace(".png", ".html"))
    if metadata_logger:
        metadata_logger.add_stat("period_length_histogram_plot", out_path)
        metadata_logger.add_stat("period_length_histogram_html", out_path.replace(".png", ".html"))
    return out_path

def periods_per_license_plate(context):
    ldf = context.ldf
    output_dir = context.output_dir
    metadata_logger = context.metadata_logger
    license_plate_col = context.get_column('license_plate_col', 'license_plate')
    period_col = context.get_column('period_col', 'period_id')
    
    os.makedirs(output_dir, exist_ok=True)
    periods_per_lp = ldf.group_by(license_plate_col).agg(pl.col(period_col).n_unique().alias("n_periods"))
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

def period_speed_stats(context):
    ldf = context.ldf
    output_dir = context.output_dir
    metadata_logger = context.metadata_logger
    period_col = context.get_column('period_col', 'period_id')
    speed_col = context.get_column('speed_col', 'instant_speed')
    
    os.makedirs(output_dir, exist_ok=True)
    # Check if speed column exists
    schema = ldf.collect_schema()
    if speed_col not in schema:
        raise ValueError(f"Speed column '{speed_col}' not found in data")
    
    # Compute period speed statistics using instant_speed
    period_speeds = ldf.group_by(period_col).agg([
        pl.col(speed_col).mean().alias("avg_speed"),
        pl.col(speed_col).max().alias("max_speed"),
        pl.col(speed_col).min().alias("min_speed"),
        pl.col(speed_col).std().alias("std_speed"),
    ])
    period_speeds_df = period_speeds.collect()
    # Remove null/nan speeds
    period_speeds_df = period_speeds_df.filter(
        pl.col("avg_speed").is_not_null() & pl.col("avg_speed").is_finite()
    )
    stats = {
        "mean": float(period_speeds_df["avg_speed"].mean()),
        "median": float(period_speeds_df["avg_speed"].median()),
        "min": float(period_speeds_df["avg_speed"].min()),
        "max": float(period_speeds_df["avg_speed"].max()),
        "std": float(period_speeds_df["avg_speed"].std()),
        "count": int(period_speeds_df.height),
    }
    with open(os.path.join(output_dir, "period_speed_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    if metadata_logger:
        metadata_logger.add_stat("period_speed_stats", stats)
        metadata_logger.add_stat("period_speed_stats_json", os.path.join(output_dir, "period_speed_stats.json"))
    return period_speeds_df

def period_speed_histogram(context):
    output_dir = context.output_dir
    metadata_logger = context.metadata_logger
    period_speeds_df = context.results.get("period_speeds_df")
    
    if period_speeds_df is None:
        raise ValueError("period_speed_histogram requires period_speeds_df from period_speed_stats")
    
    fig = px.histogram(period_speeds_df.to_pandas(), x="avg_speed", nbins=50, title="Histogram of Average Period Speeds")
    out_path = os.path.join(output_dir, "period_speed_hist.png")
    fig.write_image(out_path)
    fig.write_html(out_path.replace(".png", ".html"))
    if metadata_logger:
        metadata_logger.add_stat("period_speed_histogram_plot", out_path)
        metadata_logger.add_stat("period_speed_histogram_html", out_path.replace(".png", ".html"))
    return out_path

def period_start_end_time_distribution(context):
    ldf = context.ldf
    output_dir = context.output_dir
    metadata_logger = context.metadata_logger
    timestamp_col = context.get_column('timestamp_col', 'timestamp')
    
    os.makedirs(output_dir, exist_ok=True)
    # Extract start and end times per period
    df = ldf.group_by("period_id").agg([
        pl.col(timestamp_col).min().alias("start_time"),
        pl.col(timestamp_col).max().alias("end_time"),
    ]).collect().to_pandas()
    
    # Convert timestamp strings to datetime if needed
    for col in ["start_time", "end_time"]:
        if df[col].dtype == 'object':  # String column
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                # If parsing fails, skip this analysis
                return
    
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

def period_duration_vs_speed_scatter(context):
    ldf = context.ldf
    output_dir = context.output_dir
    metadata_logger = context.metadata_logger
    period_col = context.get_column('period_col', 'period_id')
    timestamp_col = context.get_column('timestamp_col', 'timestamp')
    speed_col = context.get_column('speed_col', 'instant_speed')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get period durations and average speeds
    df = ldf.group_by(period_col).agg([
        pl.col(timestamp_col).min().alias("start_time"),
        pl.col(timestamp_col).max().alias("end_time"),
        pl.col(speed_col).mean().alias("avg_speed"),
    ]).collect()
    
    # Calculate duration - handle string timestamps
    try:
        if df["start_time"].dtype == pl.Utf8:
            # Parse string timestamps
            df = df.with_columns([
                pl.col("start_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
                pl.col("end_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
            ])
        
        df = df.with_columns(
            (pl.col("end_time") - pl.col("start_time")).dt.total_seconds().alias("duration_seconds")
        )
        
        df_pandas = df.filter(
            pl.col("duration_seconds").is_not_null() & 
            pl.col("avg_speed").is_not_null() & 
            (pl.col("duration_seconds") > 0)
        ).to_pandas()
        
        if not df_pandas.empty:
            fig = px.scatter(df_pandas, x="duration_seconds", y="avg_speed", 
                           title="Period Duration vs. Average Speed", 
                           labels={"duration_seconds": "Duration (seconds)", "avg_speed": "Average Speed"})
            out_path = os.path.join(output_dir, "duration_vs_speed_scatter.png")
            fig.write_image(out_path)
            fig.write_html(out_path.replace(".png", ".html"))
            if metadata_logger:
                metadata_logger.add_stat("duration_vs_speed_scatter_plot", out_path)
                metadata_logger.add_stat("duration_vs_speed_scatter_html", out_path.replace(".png", ".html"))
    except Exception as e:
        # Skip if timestamp parsing fails
        pass

def occupancy_status_transitions(context):
    ldf = context.ldf
    output_dir = context.output_dir
    metadata_logger = context.metadata_logger
    period_col = context.get_column('period_col', 'period_id')
    occupancy_col = context.get_column('occupancy_col', 'occupancy_status')
    
    os.makedirs(output_dir, exist_ok=True)
    # For each period, get the first and last occupancy status
    df = ldf.group_by(period_col).agg([
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

def periods_per_day_hour(context):
    ldf = context.ldf
    output_dir = context.output_dir
    metadata_logger = context.metadata_logger
    period_col = context.get_column('period_col', 'period_id')
    timestamp_col = context.get_column('timestamp_col', 'timestamp')
    
    os.makedirs(output_dir, exist_ok=True)
    df = ldf.group_by(period_col).agg([
        pl.col(timestamp_col).min().alias("start_time"),
    ]).collect()
    
    # Handle string timestamps
    try:
        if df["start_time"].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col("start_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
            )
        
        df_pandas = df.to_pandas()
        df_pandas["start_time"] = pd.to_datetime(df_pandas["start_time"], errors='coerce')
        
        # Per day
        per_day = df_pandas["start_time"].dt.date.value_counts().sort_index()
        fig = px.bar(x=per_day.index.astype(str), y=per_day.values, title="Periods Started per Day", labels={"x": "Date", "y": "Count"})
        out_path = os.path.join(output_dir, "periods_per_day.png")
        fig.write_image(out_path)
        fig.write_html(out_path.replace(".png", ".html"))
        if metadata_logger:
            metadata_logger.add_stat("periods_per_day_plot", out_path)
            metadata_logger.add_stat("periods_per_day_html", out_path.replace(".png", ".html"))
        
        # Per hour
        per_hour = df_pandas["start_time"].dt.hour.value_counts().sort_index()
        fig = px.bar(x=per_hour.index, y=per_hour.values, title="Periods Started per Hour", labels={"x": "Hour", "y": "Count"})
        out_path = os.path.join(output_dir, "periods_per_hour.png")
        fig.write_image(out_path)
        fig.write_html(out_path.replace(".png", ".html"))
        if metadata_logger:
            metadata_logger.add_stat("periods_per_hour_plot", out_path)
            metadata_logger.add_stat("periods_per_hour_html", out_path.replace(".png", ".html"))
    except Exception as e:
        # Skip if timestamp parsing fails
        pass

def period_length_by_license_plate(context):
    ldf = context.ldf
    output_dir = context.output_dir
    metadata_logger = context.metadata_logger
    license_plate_col = context.get_column('license_plate_col', 'license_plate')
    period_col = context.get_column('period_col', 'period_id')
    
    os.makedirs(output_dir, exist_ok=True)
    df = ldf.group_by([license_plate_col, period_col]).agg(pl.len().alias("period_length")).collect().to_pandas()
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

def period_start_end_map(context):
    ldf = context.ldf
    output_dir = context.output_dir
    metadata_logger = context.metadata_logger
    period_col = context.get_column('period_col', 'period_id')
    longitude_col = context.get_column('longitude_col', 'longitude')
    latitude_col = context.get_column('latitude_col', 'latitude')
    
    os.makedirs(output_dir, exist_ok=True)
    # Only run if all columns exist
    schema = ldf.collect_schema()
    needed = [longitude_col, latitude_col]
    if not all(col in schema for col in needed):
        return
    
    df = ldf.group_by(period_col).agg([
        pl.col(latitude_col).first().alias("start_lat"),
        pl.col(longitude_col).first().alias("start_lon"),
        pl.col(latitude_col).last().alias("end_lat"),
        pl.col(longitude_col).last().alias("end_lon"),
    ]).collect().to_pandas()
    
    # Filter out invalid coordinates
    df = df[(df["start_lat"].notnull()) & (df["start_lon"].notnull()) & 
            (df["end_lat"].notnull()) & (df["end_lon"].notnull())]
    
    if not df.empty:
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

def period_start_time_vs_duration_heatmap(context):
    ldf = context.ldf
    output_dir = context.output_dir
    metadata_logger = context.metadata_logger
    period_col = context.get_column('period_col', 'period_id')
    timestamp_col = context.get_column('timestamp_col', 'timestamp')
    
    os.makedirs(output_dir, exist_ok=True)
    df = ldf.group_by(period_col).agg([
        pl.col(timestamp_col).min().alias("start_time"),
        pl.col(timestamp_col).max().alias("end_time"),
    ]).collect()
    
    try:
        # Handle string timestamps
        if df["start_time"].dtype == pl.Utf8:
            df = df.with_columns([
                pl.col("start_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
                pl.col("end_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
            ])
        
        df = df.with_columns(
            (pl.col("end_time") - pl.col("start_time")).dt.total_seconds().alias("duration")
        )
        
        df_pandas = df.filter(
            pl.col("start_time").is_not_null() & pl.col("duration").is_not_null()
        ).to_pandas()
        
        if not df_pandas.empty:
            df_pandas["start_time"] = pd.to_datetime(df_pandas["start_time"], errors='coerce')
            fig = px.density_heatmap(df_pandas, x=df_pandas["start_time"].dt.hour, y="duration", nbinsx=24, title="Start Time vs. Duration Heatmap")
            out_path = os.path.join(output_dir, "start_time_vs_duration_heatmap.png")
            fig.write_image(out_path)
            fig.write_html(out_path.replace(".png", ".html"))
            if metadata_logger:
                metadata_logger.add_stat("start_time_vs_duration_heatmap_plot", out_path)
                metadata_logger.add_stat("start_time_vs_duration_heatmap_html", out_path.replace(".png", ".html"))
    except Exception as e:
        # Skip if timestamp parsing fails
        pass

def idle_vs_occupied_distribution(context):
    ldf = context.ldf
    output_dir = context.output_dir
    metadata_logger = context.metadata_logger
    period_col = context.get_column('period_col', 'period_id')
    occupancy_col = context.get_column('occupancy_col', 'occupancy_status')
    
    os.makedirs(output_dir, exist_ok=True)
    df = ldf.group_by(period_col).agg([
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

def speed_outlier_boxplot(context):
    ldf = context.ldf
    output_dir = context.output_dir
    metadata_logger = context.metadata_logger
    period_col = context.get_column('period_col', 'period_id')
    speed_col = context.get_column('speed_col', 'instant_speed')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if speed column exists
    schema = ldf.collect_schema()
    if speed_col not in schema:
        raise ValueError(f"Speed column '{speed_col}' not found in data")
    
    df = ldf.group_by(period_col).agg([
        pl.col(speed_col).mean().alias("avg_speed"),
        pl.col(speed_col).max().alias("max_speed"),
    ]).collect().to_pandas()
    
    df = df[(df["avg_speed"].notnull()) & (df["avg_speed"] >= 0)]
    if not df.empty:
        fig = px.box(df, y="avg_speed", points="all", title="Period Average Speed Outlier Boxplot")
        out_path = os.path.join(output_dir, "speed_outlier_boxplot.png")
        fig.write_image(out_path)
        fig.write_html(out_path.replace(".png", ".html"))
        if metadata_logger:
            metadata_logger.add_stat("speed_outlier_boxplot_plot", out_path)
            metadata_logger.add_stat("speed_outlier_boxplot_html", out_path.replace(".png", ".html")) 