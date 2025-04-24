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
    # Sidebar controls
    st.sidebar.header("Controls")
    # Map background selector
    bg_option = st.sidebar.selectbox(
        "Map Background", ["White", "OpenStreetMap"], index=1
    )

    # --------- License Plate & Period Navigation ---------
    # Initialize navigation state
    SAMPLE_LP_LIMIT = 100
    lp_list = get_sample_license_plates(SAMPLE_LP_LIMIT)
    n_lp = len(lp_list)
    if "lp_index" not in st.session_state:
        st.session_state.lp_index = 0
    if "period_index" not in st.session_state:
        st.session_state.period_index = 0

    # Callback functions
    def prev_lp():
        st.session_state.lp_index = max(0, st.session_state.lp_index - 1)
        st.session_state.period_index = 0

    def next_lp():
        st.session_state.lp_index = min(n_lp - 1, st.session_state.lp_index + 1)
        st.session_state.period_index = 0

    # License Plate controls
    st.sidebar.subheader("License Plate")
    lp_cols = st.sidebar.columns([1, 4, 1])
    with lp_cols[0]:
        lp_cols[0].button("⬆️ Prev", on_click=prev_lp, key="prev_lp_btn")
    with lp_cols[1]:
        st.sidebar.markdown(f"**{lp_list[st.session_state.lp_index]}**")
    with lp_cols[2]:
        lp_cols[2].button("Next ⬇️", on_click=next_lp, key="next_lp_btn")
    lp = lp_list[st.session_state.lp_index]

    # Fetch periods for selected license plate
    try:
        period_ids = get_period_ids(lp)
    except Exception as e:
        st.sidebar.error(f"Error loading periods: {e}")
        return
    if not period_ids:
        st.sidebar.warning(f"No periods found for license plate '{lp}'.")
        return
    n_periods = len(period_ids)

    # Callback for period nav
    def prev_period():
        st.session_state.period_index = max(0, st.session_state.period_index - 1)

    def next_period():
        st.session_state.period_index = min(n_periods - 1, st.session_state.period_index + 1)

    # Period controls
    st.sidebar.subheader("Period")
    period_cols = st.sidebar.columns([1, 4, 1])
    with period_cols[0]:
        period_cols[0].button("◀️ Prev", on_click=prev_period, key="prev_period_btn")
    with period_cols[1]:
        st.sidebar.markdown(f"**{period_ids[st.session_state.period_index]}**")
    with period_cols[2]:
        period_cols[2].button("Next ▶️", on_click=next_period, key="next_period_btn")
    period_id = period_ids[st.session_state.period_index]
    st.sidebar.markdown(f"_Total periods: {n_periods}_")

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

    # Build plot with optional map background
    lon_list = traj_df["longitude"].to_list()
    lat_list = traj_df["latitude"].to_list()
    if bg_option == "OpenStreetMap":
        import plotly.graph_objects as go
        # Center map on trajectory
        center_lat = sum(lat_list) / len(lat_list)
        center_lon = sum(lon_list) / len(lon_list)
        # Use Maplibre scattermap on 'map' subplot
        fig = go.Figure(
            go.Scattermap(
                lat=lat_list,
                lon=lon_list,
                mode="lines",
                line=dict(color="red", width=2),
                subplot="map",
            )
        )
        fig.update_layout(
            map_style="open-street-map",
            map_center={"lat": center_lat, "lon": center_lon},
            map_zoom=12,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )
    else:
        # Simple white-background plot
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
