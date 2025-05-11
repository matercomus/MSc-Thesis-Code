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
import hashlib

st.set_page_config(page_title="Trajectory Previewer", layout="wide")

# Paths to Parquet files
# Use enriched periods file with straight-line to sum-distance ratio
PERIOD_FILE = "periods_with_sld_ratio.parquet"
DATA_FILE = "cleaned_with_period_id_in_beijing.parquet"

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
    # Sidebar: application title and global period filters
    st.sidebar.title("Trajectory Viewer")
    
    # Add privacy mode toggle at the top of sidebar
    privacy_mode = st.sidebar.toggle("Privacy Mode", value=False, 
                                   help="Hide sensitive information like license plates and timestamps")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Period Filters")
    filter_sld = st.sidebar.selectbox(
        "SLD outlier status",
        ["All", "Only outliers", "Only non-outliers"],
        index=0,
        key="filter_sld"
    )
    filter_if = st.sidebar.selectbox(
        "IF outlier status",
        ["All", "Only outliers", "Only non-outliers"],
        index=0,
        key="filter_if"
    )
    # Load all periods lazily and apply global filters
    periods_lf = pl.scan_parquet(PERIOD_FILE)
    if filter_sld != "All":
        want_sld = (filter_sld == "Only outliers")
        periods_lf = periods_lf.filter(pl.col("is_sld_outlier") == want_sld)
    if filter_if != "All":
        want_if = (filter_if == "Only outliers")
        periods_lf = periods_lf.filter(pl.col("is_traj_outlier") == want_if)
    # Filter by occupancy status (multi-select)
    occ_df = (
        periods_lf.select("occupancy_status")
        .unique()
        .sort("occupancy_status")
        .collect()
    )
    occ_options = occ_df.get_column("occupancy_status").to_list()
    filter_occ = st.sidebar.multiselect(
        "Occupancy status", options=occ_options, default=occ_options, key="filter_occ"
    )
    periods_lf = periods_lf.filter(pl.col("occupancy_status").is_in(filter_occ))
    # Optionally hide very short periods by total distance
    hide_small = st.sidebar.checkbox(
        "Hide periods with sum_distance < 1 km", value=False, key="hide_small"
    )
    if hide_small:
        periods_lf = periods_lf.filter(pl.col("sum_distance") >= 1.0)
    # Compute counts per license plate
    lp_counts = (
        periods_lf
        .group_by("license_plate")
        .agg(pl.count("period_id").alias("n_periods"))
        .collect()
    )
    # Optional: only plates with multiple filtered periods
    filter_multi = st.sidebar.checkbox(
        "Only plates with >1 period", value=False
    )
    if filter_multi:
        lp_counts = lp_counts.filter(pl.col("n_periods") > 1)
    # Sorting options for license plates
    sort_order = st.sidebar.radio(
        "Sort plates by",
        ["Plate (A-Z)", "Periods ascending", "Periods descending"],
        index=0,
    )
    if sort_order == "Plate (A-Z)":
        lp_counts = lp_counts.sort("license_plate")
    elif sort_order == "Periods ascending":
        lp_counts = lp_counts.sort("n_periods")
    else:
        lp_counts = lp_counts.sort("n_periods", descending=True)
    # License plate selection
    lp_list = lp_counts.get_column("license_plate").to_list()
    display_lp_list = [anonymize_text(lp) if privacy_mode else lp for lp in lp_list]
    
    # Handle state transition when privacy mode changes
    if "lp" in st.session_state:
        current_lp = st.session_state.lp
        if privacy_mode:
            # If we're in privacy mode, find the original plate and get its hash
            if current_lp in lp_list:
                st.session_state.lp = anonymize_text(current_lp)
            elif current_lp not in display_lp_list:
                # If the current value isn't in either list, reset to first plate
                st.session_state.lp = display_lp_list[0]
        else:
            # If we're in normal mode, find the original plate
            if current_lp in display_lp_list:
                idx = display_lp_list.index(current_lp)
                st.session_state.lp = lp_list[idx]
            elif current_lp not in lp_list:
                # If the current value isn't in either list, reset to first plate
                st.session_state.lp = lp_list[0]
    
    st.sidebar.selectbox("Select license plate", display_lp_list, key="lp")
    # Get the original license plate from the display value
    lp = lp_list[display_lp_list.index(st.session_state.lp)]
    # Load filtered periods for selected plate
    periods_df = (
        periods_lf.filter(pl.col("license_plate") == lp)
        .collect()
    )
    if periods_df.is_empty():
        st.warning(f"No periods match selected filters for plate {lp if not privacy_mode else 'selected'}.")
        return
        
    # Display periods and select period
    if privacy_mode:
        st.subheader("Periods")
        # Create a copy of the dataframe for display
        display_df = periods_df.clone()
        # Anonymize sensitive columns
        display_df = display_df.with_columns([
            pl.col("license_plate").map_elements(anonymize_text),
            pl.col("start_time").map_elements(lambda x: "***" if x is not None else None),
            pl.col("end_time").map_elements(lambda x: "***" if x is not None else None)
        ])
        st.dataframe(display_df, use_container_width=True)
    else:
        st.subheader(f"Periods for {lp}")
        st.dataframe(periods_df, use_container_width=True)
        
    period_ids = periods_df.get_column("period_id").to_list()
    st.sidebar.selectbox("Select period ID", period_ids, key="period_id")
    period_id = st.session_state.period_id

    # Display trajectory header with selected period index
    period_ids = periods_df.get_column("period_id").to_list()
    n_periods = len(period_ids)
    try:
        sel_idx = period_ids.index(period_id)
    except ValueError:
        sel_idx = 0
    
    if privacy_mode:
        st.markdown(f"**Trajectory: *** | Period: {period_id} ({sel_idx+1}/{n_periods})**")
    else:
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
