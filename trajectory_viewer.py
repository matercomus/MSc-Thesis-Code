"""
Trajectory Previewer GUI

This script provides a simple web-based GUI for previewing trajectories from Parquet data.
It uses Streamlit for the interface, Polars for efficient data loading/filtering, and Plotly for plotting.

Requirements:
    pip install streamlit polars plotly

Usage:
    streamlit run trajectory_viewer.py

Ensure that the Parquet files 'periods_with_sld_ratio.parquet' and 'cleaned_with_period_id_in_beijing.parquet'
are in the same directory as this script.
"""

import streamlit as st
import polars as pl
import plotly.express as px

st.set_page_config(page_title="Trajectory Previewer", layout="wide")

# Paths to Parquet files
# Use enriched periods file with straight-line to sum-distance ratio
PERIOD_FILE = "periods_with_sld_ratio.parquet"
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
    # Load the trajectory data, including outlier flags if available
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


def main():
    # Initialize navigation state
    SAMPLE_LP_LIMIT = 100
    lp_list = get_sample_license_plates(SAMPLE_LP_LIMIT)
    n_lp = len(lp_list)
    if "lp_index" not in st.session_state:
        st.session_state.lp_index = 0
    if "period_index" not in st.session_state:
        st.session_state.period_index = 0

    # Navigation callbacks
    def prev_plate():
        st.session_state.lp_index = max(0, st.session_state.lp_index - 1)
        st.session_state.period_index = 0

    def next_plate():
        st.session_state.lp_index = min(n_lp - 1, st.session_state.lp_index + 1)
        st.session_state.period_index = 0

    # Current license plate
    lp = lp_list[st.session_state.lp_index]

    # Fetch period IDs
    try:
        period_ids = get_period_ids(lp)
    except Exception as e:
        st.error(f"Error loading periods for plate '{lp}': {e}")
        return
    if not period_ids:
        st.warning(f"No periods found for license plate '{lp}'.")
        return
    # Clamp period index
    n_periods = len(period_ids)
    if st.session_state.period_index >= n_periods:
        st.session_state.period_index = n_periods - 1
    if st.session_state.period_index < 0:
        st.session_state.period_index = 0
    period_id = period_ids[st.session_state.period_index]
    n_periods = len(period_ids)
    if st.session_state.period_index >= n_periods:
        st.session_state.period_index = n_periods - 1
    if st.session_state.period_index < 0:
        st.session_state.period_index = 0
    period_id = period_ids[st.session_state.period_index]

    # Display trajectory header
    st.markdown(f"**Trajectory: {lp} | Period: {period_id} ({st.session_state.period_index+1}/{n_periods})**")

    # Toggle map background
    bg_osm = st.checkbox("Show map background (OSM)", value=True, key="bg_osm")

    # Compute SLD threshold for flagging abnormal trajectories
    from utils import compute_generic_iqr_threshold
    sld_th = compute_generic_iqr_threshold(
        pl.scan_parquet(PERIOD_FILE), "sld_ratio"
    )
    # Load period info once
    try:
        info_df = get_period_info(lp, period_id)
    except Exception as e:
        st.error(f"Error loading period information: {e}")
        return
    # Flag abnormal trajectories based on SLD threshold
    try:
        sld_ratio = info_df["sld_ratio"][0]
        if sld_ratio > sld_th:
            st.markdown(
                "<h3 style='color:red'>ABNORMAL (SLD)</h3>", unsafe_allow_html=True
            )
    except Exception:
        pass
    # Flag abnormal trajectories based on Isolation Forest indicator
    try:
        is_if_outlier = info_df["is_traj_outlier"][0]
        if is_if_outlier:
            st.markdown(
                "<h3 style='color:red'>ABNORMAL (IF)</h3>", unsafe_allow_html=True
            )
    except Exception:
        pass
    try:
        traj_df = get_trajectory(lp, period_id)
    except Exception as e:
        st.error(f"Error loading trajectory data: {e}")
        return
    if traj_df.is_empty():
        st.warning("No trajectory data available for this period.")
        return

    lon_list = traj_df["longitude"].to_list()
    lat_list = traj_df["latitude"].to_list()

    # Build plot
    if bg_osm:
        center_lat = sum(lat_list) / len(lat_list)
        center_lon = sum(lon_list) / len(lon_list)
        # Use Plotly Express line_mapbox for trajectories on OpenStreetMap
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

    # Display map
    st.plotly_chart(fig, use_container_width=True)

    # Combined navigation using a segmented control with Material icons
    nav_icons = {
        "prev_plate": ":material/arrow_upward:",
        "prev_period": ":material/arrow_back:",
        "next_period": ":material/arrow_forward:",
        "next_plate": ":material/arrow_downward:",
    }

    def handle_nav():
        choice = st.session_state.nav
        if choice == "prev_plate":
            st.session_state.lp_index = max(0, st.session_state.lp_index - 1)
            st.session_state.period_index = 0
        elif choice == "next_plate":
            st.session_state.lp_index = min(n_lp - 1, st.session_state.lp_index + 1)
            st.session_state.period_index = 0
        elif choice == "prev_period":
            st.session_state.period_index = max(0, st.session_state.period_index - 1)
        elif choice == "next_period":
            st.session_state.period_index = min(
                n_periods - 1, st.session_state.period_index + 1
            )
        # Reset the control for next use
        st.session_state.nav = None

    # Center the navigation control under the map
    nav_cols = st.columns([5, 2, 5])
    with nav_cols[1]:
        st.segmented_control(
            "Navigation",
            options=list(nav_icons.keys()),
            format_func=lambda key: nav_icons[key],
            key="nav",
            label_visibility="collapsed",
            on_change=handle_nav,
        )

    # Display period metadata
    st.subheader("Period Information")
    try:
        info_df = get_period_info(lp, period_id)
    except Exception as e:
        st.error(f"Error loading period information: {e}")
        return
    if info_df.is_empty():
        st.warning("No period information found.")
    else:
        st.dataframe(info_df, use_container_width=True)


if __name__ == "__main__":
    main()
