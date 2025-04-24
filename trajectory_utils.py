"""
Utility functions for loading and plotting trajectories based on period IDs.
"""
import polars as pl
from utils import add_period_id
import plotly.express as px
import matplotlib.pyplot as plt

def load_points_with_periods(parquet_path: str = "cleaned_points_in_beijing.parquet") -> pl.DataFrame:
    """
    Load cleaned points and assign period IDs per license_plate and occupancy_status.
    Returns a Polars DataFrame with a new 'period_id' column.
    """
    df = pl.scan_parquet(parquet_path).collect()
    return add_period_id(df)

def plotly_trajectory(
    df: pl.DataFrame, license_plate: str, period_id: int
) -> "plotly.graph_objs._figure.Figure":
    """
    Plot the trajectory for a given license_plate and period_id using Plotly.
    Shows a red line on a white background connecting (longitude, latitude) points.
    """
    subset = (
        df.filter(
            (pl.col("license_plate") == license_plate)
            & (pl.col("period_id") == period_id)
        )
        .sort("timestamp")
    )
    if subset.is_empty():
        raise ValueError(f"No data for plate {license_plate}, period {period_id}")
    pd_df = subset.select(["longitude", "latitude"]).to_pandas()
    fig = px.line(
        pd_df,
        x="longitude",
        y="latitude",
        title=f"Trajectory: {license_plate} Period {period_id}",
        line_shape="linear",
    )
    fig.update_traces(line=dict(color="red"))
    fig.update_layout(
        plot_bgcolor="white",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
    )
    return fig

def matplotlib_trajectory(
    df: pl.DataFrame, license_plate: str, period_id: int
) -> "tuple[plt.Figure, plt.Axes]":
    """
    Plot the trajectory for a given license_plate and period_id using Matplotlib.
    Shows a red line on a white background connecting (longitude, latitude) points.
    """
    subset = (
        df.filter(
            (pl.col("license_plate") == license_plate)
            & (pl.col("period_id") == period_id)
        )
        .sort("timestamp")
    )
    if subset.is_empty():
        raise ValueError(f"No data for plate {license_plate}, period {period_id}")
    lons = subset["longitude"].to_list()
    lats = subset["latitude"].to_list()
    fig, ax = plt.subplots()
    ax.plot(lons, lats, color="red")
    ax.set_facecolor("white")
    ax.set_title(f"Trajectory: {license_plate} Period {period_id}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(min(lons), max(lons))
    ax.set_ylim(min(lats), max(lats))
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    return fig, ax