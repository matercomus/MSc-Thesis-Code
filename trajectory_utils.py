"""
Utility functions for the trajectory viewer application.
"""

import streamlit as st
import polars as pl
import hashlib
import plotly.express as px
from plotly.graph_objects import Figure

# Constants
PERIOD_FILE = "data/periods_with_sld_ratio.parquet"
DATA_FILE = "data/cleaned_with_period_id_in_beijing.parquet"

def anonymize_text(text: str) -> str:
    """Anonymize text by creating a deterministic hash."""
    return hashlib.sha256(str(text).encode()).hexdigest()[:8]

@st.cache_data
def get_period_ids(lp: str) -> list:
    """Return a sorted list of period_ids for the given license plate."""
    df = (
        pl.scan_parquet(PERIOD_FILE)
        .filter(pl.col("license_plate") == lp)
        .select("period_id")
        .unique()
        .sort("period_id")
        .collect()
    )
    return df["period_id"].to_list()

@st.cache_data
def get_period_info(lp: str, period_id) -> pl.DataFrame:
    """Return the metadata row for a given license plate and period_id."""
    return (
        pl.scan_parquet(PERIOD_FILE)
        .filter((pl.col("license_plate") == lp) & (pl.col("period_id") == period_id))
        .collect()
    )

@st.cache_data
def get_trajectory(lp: str, period_id) -> pl.DataFrame:
    """Return the trajectory DataFrame (latitude, longitude, timestamp) for the given license plate and period."""
    df = (
        pl.scan_parquet(DATA_FILE)
        .filter((pl.col("license_plate") == lp) & (pl.col("period_id") == period_id))
        .select([
            "timestamp",
            "latitude",
            "longitude",
            pl.col("is_outlier").fill_null(1).alias("is_outlier"),
        ])
        .sort("timestamp")
        .collect()
    )
    return df

@st.cache_data
def get_sample_license_plates(limit: int = 100) -> list:
    """Return a sorted list of up to `limit` unique license plates."""
    df = (
        pl.scan_parquet(PERIOD_FILE)
        .select("license_plate")
        .unique()
        .sort("license_plate")
        .limit(limit)
        .collect()
    )
    return df["license_plate"].to_list()

def create_trajectory_plot(traj_df: pl.DataFrame, bg_osm: bool = True) -> Figure:
    """Create a plotly figure for the trajectory."""
    lon_list = traj_df["longitude"].to_list()
    lat_list = traj_df["latitude"].to_list()

    if bg_osm:
        center_lat = sum(lat_list) / len(lat_list)
        center_lon = sum(lon_list) / len(lon_list)
        fig = px.line_mapbox(
            lat=lat_list,
            lon=lon_list,
        )
        fig.update_traces(line_color="red")
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_center={"lat": center_lat, "lon": center_lon},
            mapbox_zoom=12,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )
    else:
        fig = px.line(x=lon_list, y=lat_list)
        fig.update_traces(line_color="red")
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
    
    return fig

def get_filtered_periods(filter_sld: str, filter_if: str, filter_occ: list, hide_small: bool) -> pl.LazyFrame:
    """Get filtered periods based on the provided filters."""
    periods_lf = pl.scan_parquet(PERIOD_FILE)
    
    if filter_sld != "All":
        want_sld = (filter_sld == "Only outliers")
        periods_lf = periods_lf.filter(pl.col("is_sld_outlier") == want_sld)
    
    if filter_if != "All":
        want_if = (filter_if == "Only outliers")
        periods_lf = periods_lf.filter(pl.col("is_traj_outlier") == want_if)
    
    periods_lf = periods_lf.filter(pl.col("occupancy_status").is_in(filter_occ))
    
    if hide_small:
        periods_lf = periods_lf.filter(pl.col("sum_distance") >= 1.0)
    
    return periods_lf

def handle_license_plate_state(privacy_mode: bool, lp_list: list, display_lp_list: list, current_lp: str) -> str:
    """Handle license plate state transitions when privacy mode changes."""
    if privacy_mode:
        if current_lp in lp_list:
            return anonymize_text(current_lp)
        elif current_lp not in display_lp_list:
            return display_lp_list[0]
    else:
        if current_lp in display_lp_list:
            idx = display_lp_list.index(current_lp)
            return lp_list[idx]
        elif current_lp not in lp_list:
            return lp_list[0]
    
    return current_lp
