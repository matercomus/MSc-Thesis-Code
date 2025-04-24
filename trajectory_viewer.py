"""
Trajectory Previewer GUI

This script provides a simple web-based GUI for previewing trajectories from Parquet data.
It uses Streamlit for the interface, Polars for efficient data loading/filtering, and Plotly for plotting.

Requirements:
    pip install streamlit polars plotly

Usage:
    streamlit run trajectory_viewer.py

Ensure that the Parquet files 'periods_in_beijing.parquet' and 'cleaned_with_period_id_in_beijing.parquet'
are in the same directory as this script.
"""

import streamlit as st
st.set_page_config(page_title="Trajectory Previewer", layout="wide")
import polars as pl
import plotly.express as px

# Paths to Parquet files
PERIOD_FILE = "periods_in_beijing.parquet"
DATA_FILE = "cleaned_with_period_id_in_beijing.parquet"

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
    return (
        pl.scan_parquet(DATA_FILE)
        .filter((pl.col("license_plate") == lp) & (pl.col("period_id") == period_id))
        .select(["latitude", "longitude", "timestamp"])
        .sort("timestamp")
        .collect()
    )

def main():
    # Sidebar controls
    st.sidebar.header("Controls")
    lp = st.sidebar.text_input("License Plate", value="")
    if not lp:
        st.sidebar.info("Please enter a license plate to begin.")
        return
    # Reset period index when license plate changes
    if st.session_state.get("last_lp") != lp:
        st.session_state.period_index = 1
        st.session_state.last_lp = lp

    # Fetch periods
    try:
        period_ids = get_period_ids(lp)
    except Exception as e:
        st.sidebar.error(f"Error loading periods: {e}")
        return
    if not period_ids:
        st.sidebar.warning(f"No periods found for license plate '{lp}'.")
        return

    # Period selector
    default_idx = st.session_state.get("period_index", 1)
    period_idx = st.sidebar.slider("Select Period", 1, len(period_ids), value=default_idx)
    st.session_state.period_index = period_idx
    period_id = period_ids[period_idx - 1]
    st.sidebar.write(f"Period ID: {period_id}")

    # Main display: plot and metadata side by side
    st.subheader(f"Trajectory: {lp} | Period: {period_id}")

    # Load trajectory
    try:
        traj_df = get_trajectory(lp, period_id)
    except Exception as e:
        st.error(f"Error loading trajectory data: {e}")
        return
    if traj_df.is_empty():
        st.warning("No trajectory data available for this period.")
        return

    # Build plot
    lon_list = traj_df["longitude"].to_list()
    lat_list = traj_df["latitude"].to_list()
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

    # Load metadata
    try:
        info_df = get_period_info(lp, period_id)
    except Exception as e:
        st.error(f"Error loading period information: {e}")
        return

    # Display plot full-width
    st.plotly_chart(fig, use_container_width=True)

    # Display period metadata below
    st.subheader("Period Information")
    if info_df.is_empty():
        st.warning("No period information found.")
    else:
        st.dataframe(info_df, use_container_width=True)

if __name__ == "__main__":
    main()
