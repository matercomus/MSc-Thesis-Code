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
pl.enable_string_cache()
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
    # Sidebar: license plate selection controls
    st.sidebar.title("Trajectory Viewer")
    # Load all periods and compute counts per license plate
    # Load all periods lazily
    periods_all = pl.scan_parquet(PERIOD_FILE)
    lp_counts = (
        periods_all
        .group_by("license_plate")
        .agg(pl.count("period_id").alias("n_periods"))
        .collect()
    )
    # Filtering: show only plates with more than one period
    filter_multi = st.sidebar.checkbox(
        "Only plates with >1 period", value=False
    )
    if filter_multi:
        lp_counts = lp_counts.filter(pl.col("n_periods") > 1)
    # Sorting options
    sort_order = st.sidebar.radio(
        "Sort plates by",
        options=[
            "Plate (A-Z)",
            "Periods ascending",
            "Periods descending",
        ],
        index=0,
    )
    if sort_order == "Plate (A-Z)":
        lp_counts = lp_counts.sort("license_plate")
    elif sort_order == "Periods ascending":
        lp_counts = lp_counts.sort("n_periods")
    else:
        # Sort by number of periods descending
        lp_counts = lp_counts.sort("n_periods", descending=True)
    # License plate list
    lp_list = lp_counts.get_column("license_plate").to_list()
    # License plate selectbox bound to session_state 'lp'
    st.sidebar.selectbox("Select license plate", lp_list, key="lp")
    lp = st.session_state.lp
    # Sidebar: filters for period outlier indicators
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Period Filters")
    filter_sld = st.sidebar.selectbox(
        "SLD outlier status",
        ["All", "Only outliers", "Only non-outliers"],
        index=0,
        key="filter_sld"
    )
    filter_if = st.sidebar.selectbox(
        "IF (isolation forest) status",
        ["All", "Only outliers", "Only non-outliers"],
        index=0,
        key="filter_if"
    )
    # Load all periods for selected license plate
    periods_all = (
        pl.scan_parquet(PERIOD_FILE)
        .filter(pl.col("license_plate") == lp)
        .collect()
    )
    if periods_all.is_empty():
        st.sidebar.error(f"No periods for plate {lp}")
        return
    # Apply filters
    periods_df = periods_all
    if filter_sld != "All":
        want = (filter_sld == "Only outliers")
        periods_df = periods_df.filter(pl.col("is_sld_outlier") == want)
    if filter_if != "All":
        want_if = (filter_if == "Only outliers")
        periods_df = periods_df.filter(pl.col("is_traj_outlier") == want_if)
    if periods_df.is_empty():
        st.warning("No periods match selected outlier filters for this plate.")
        return
    # Display filtered periods and select period
    st.subheader(f"Periods for {lp}")
    st.dataframe(periods_df, use_container_width=True)
    # Period selectbox bound to session_state 'period_id'
    period_ids = periods_df.get_column("period_id").to_list()
    st.sidebar.selectbox("Select period ID", period_ids, key="period_id")
    period_id = st.session_state.period_id

    # Display trajectory header with selected period index
    # Compute position of selected period
    period_ids = periods_df.get_column("period_id").to_list()
    n_periods = len(period_ids)
    try:
        sel_idx = period_ids.index(period_id)
    except ValueError:
        sel_idx = 0
    st.markdown(f"**Trajectory: {lp} | Period: {period_id} ({sel_idx+1}/{n_periods})**")

    # Toggle map background
    bg_osm = st.sidebar.checkbox("Show map background (OSM)", value=True)

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
    is_sld_outlier = info_df.get_column("is_sld_outlier")[0]
    if is_sld_outlier:
        st.markdown("<h3 style='color:red'>ABNORMAL (SLD)</h3>", unsafe_allow_html=True)
    # Flag abnormal trajectories based on Isolation Forest indicator
    try:
        is_if_outlier = info_df["is_traj_outlier"][0]
        if is_if_outlier:
            st.markdown(
                "<h3 style='color:red'>ABNORMAL (IF)</h3>", unsafe_allow_html=True
            )
    except Exception:
        pass
    # Load trajectory for selected period
    try:
        traj_df = get_trajectory(lp, period_id)
    except Exception as e:
        st.error(f"Error loading trajectory data for period {period_id}: {e}")
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
    # Navigation controls (prev/next plate and period)
    nav_icons = {
        "prev_plate": ":material/arrow_upward:",
        "prev_period": ":material/arrow_back:",
        "next_period": ":material/arrow_forward:",
        "next_plate": ":material/arrow_downward:",
    }
    def handle_nav():
        choice = st.session_state.nav
        # Plate navigation
        idx_lp = lp_list.index(lp)
        if choice == "prev_plate":
            new_idx = max(0, idx_lp - 1)
            st.session_state.lp = lp_list[new_idx]
        elif choice == "next_plate":
            new_idx = min(len(lp_list) - 1, idx_lp + 1)
            st.session_state.lp = lp_list[new_idx]
        # Period navigation
        idx_p = period_ids.index(period_id)
        if choice == "prev_period":
            new_p = max(0, idx_p - 1)
            st.session_state.period_id = period_ids[new_p]
        elif choice == "next_period":
            new_p = min(len(period_ids) - 1, idx_p + 1)
            st.session_state.period_id = period_ids[new_p]
        # Reset nav
        st.session_state.nav = None
    cols_nav = st.columns([5, 2, 5])
    with cols_nav[1]:
        st.segmented_control(
            "",
            options=list(nav_icons.keys()),
            format_func=lambda k: nav_icons[k],
            key="nav",
            label_visibility="collapsed",
            on_change=handle_nav,
        )


    # (Removed redundant period metadata table)


if __name__ == "__main__":
    main()
