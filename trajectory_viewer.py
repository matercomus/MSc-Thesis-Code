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
from trajectory_utils import (
    get_period_info,
    get_trajectory,
    create_trajectory_plot,
    get_filtered_periods,
    handle_license_plate_state,
    anonymize_text,
)

st.set_page_config(page_title="Trajectory Previewer", layout="wide")

def setup_sidebar():
    """Setup the sidebar with filters and controls."""
    st.sidebar.title("Trajectory Viewer")
    
    # Privacy mode toggle
    privacy_mode = st.sidebar.toggle(
        "Privacy Mode", 
        value=False,
        help="Hide sensitive information like license plates and timestamps"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Period Filters")
    
    # Filter controls
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
    
    # Get occupancy status options
    periods_lf = pl.scan_parquet("periods_with_sld_ratio.parquet")
    occ_df = (
        periods_lf.select("occupancy_status")
        .unique()
        .sort("occupancy_status")
        .collect()
    )
    occ_options = occ_df.get_column("occupancy_status").to_list()
    
    filter_occ = st.sidebar.multiselect(
        "Occupancy status", 
        options=occ_options, 
        default=occ_options, 
        key="filter_occ"
    )
    
    hide_small = st.sidebar.checkbox(
        "Hide periods with sum_distance < 1 km", 
        value=False, 
        key="hide_small"
    )
    
    return privacy_mode, filter_sld, filter_if, filter_occ, hide_small

def display_trajectory_info(lp: str, period_id: int, privacy_mode: bool):
    """Display trajectory information and metadata."""
    try:
        info_df = get_period_info(lp, period_id)
    except Exception as e:
        st.error(f"Error loading period information: {e}")
        return
    
    # Display outlier warnings
    is_sld_outlier = info_df.get_column("is_sld_outlier")[0]
    if is_sld_outlier:
        st.markdown("<h3 style='color:red'>ABNORMAL (SLD)</h3>", unsafe_allow_html=True)
    
    try:
        is_if_outlier = info_df["is_traj_outlier"][0]
        if is_if_outlier:
            st.markdown("<h3 style='color:red'>ABNORMAL (IF)</h3>", unsafe_allow_html=True)
    except Exception:
        pass

def main():
    # Setup sidebar and get filter values
    privacy_mode, filter_sld, filter_if, filter_occ, hide_small = setup_sidebar()
    
    # Get filtered periods
    periods_lf = get_filtered_periods(filter_sld, filter_if, filter_occ, hide_small)
    
    # Compute counts per license plate
    lp_counts = (
        periods_lf
        .group_by("license_plate")
        .agg(pl.count("period_id").alias("n_periods"))
        .collect()
    )
    
    # Optional: only plates with multiple filtered periods
    filter_multi = st.sidebar.checkbox("Only plates with >1 period", value=False)
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
        new_lp = handle_license_plate_state(privacy_mode, lp_list, display_lp_list, current_lp)
        if new_lp != current_lp:
            st.session_state.lp = new_lp
    
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
        display_df = periods_df.clone()
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
    
    # Display trajectory header
    n_periods = len(period_ids)
    try:
        sel_idx = period_ids.index(period_id)
    except ValueError:
        sel_idx = 0
    
    if privacy_mode:
        st.markdown(f"**Trajectory: *** | Period: {period_id} ({sel_idx+1}/{n_periods})**")
    else:
        st.markdown(f"**Trajectory: {lp} | Period: {period_id} ({sel_idx+1}/{n_periods})**")
    
    # Display trajectory information
    display_trajectory_info(lp, period_id, privacy_mode)
    
    # Toggle map background
    bg_osm = st.sidebar.checkbox("Show map background (OSM)", value=True)
    
    # Load and display trajectory
    try:
        traj_df = get_trajectory(lp, period_id)
    except Exception as e:
        st.error(f"Error loading trajectory data for period {period_id}: {e}")
        return
    
    if traj_df.is_empty():
        st.warning("No trajectory data available for this period.")
        return
    
    # Create and display plot
    fig = create_trajectory_plot(traj_df, bg_osm)
    st.plotly_chart(fig, use_container_width=True)
    
    # Navigation controls
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

if __name__ == "__main__":
    main()
