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
    st.title("Trajectory Previewer")
    st.write("Enter a license plate and navigate through its periods to preview trajectories.")

    # License plate input
    lp = st.text_input("License Plate", value="")
    if not lp:
        return

    # Fetch available period IDs
    try:
        period_ids = get_period_ids(lp)
    except Exception as e:
        st.error(f"Error loading periods: {e}")
        return

    if not period_ids:
        st.warning(f"No periods found for license plate '{lp}'.")
        return

    # Initialize or reset period index when lp changes
    if "period_index" not in st.session_state or st.session_state.get("lp") != lp:
        st.session_state.period_index = 0
        st.session_state.lp = lp

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Previous"):
            if st.session_state.period_index > 0:
                st.session_state.period_index -= 1
    with col2:
        if st.button("Next"):
            if st.session_state.period_index < len(period_ids) - 1:
                st.session_state.period_index += 1
    with col3:
        st.write(f"Period {st.session_state.period_index + 1} of {len(period_ids)}")

    # Current period selection
    period_id = period_ids[st.session_state.period_index]
    st.subheader(f"License Plate: {lp} | Period ID: {period_id}")

    # Load and plot trajectory
    try:
        traj_df = get_trajectory(lp, period_id)
    except Exception as e:
        st.error(f"Error loading trajectory data: {e}")
        return

    if traj_df.is_empty():
        st.warning("No trajectory data available for this period.")
    else:
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
        st.plotly_chart(fig, use_container_width=True)

    # Display period metadata
    try:
        info_df = get_period_info(lp, period_id)
    except Exception as e:
        st.error(f"Error loading period information: {e}")
        return

    if info_df.is_empty():
        st.warning("No period information found.")
    else:
        info_records = info_df.to_dicts()
        st.subheader("Period Information")
        st.table(info_records)

if __name__ == "__main__":
    main()