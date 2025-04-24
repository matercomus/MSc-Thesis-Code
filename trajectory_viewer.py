"""
Marimo notebook for interactive visualization of vehicle trajectories by period.
"""

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


@app.cell
def _():
    import polars as pl
    # Enable global string cache to handle Categorical comparisons
    pl.enable_string_cache()
    import marimo
    from trajectory_utils import load_points_with_periods, plotly_trajectory
    return load_points_with_periods, marimo, plotly_trajectory


@app.cell
def viewer(load_points_with_periods):
    # Load raw point data and assign period IDs
    # Use cleaned points file (contains timestamp) rather than summary periods file
    df = load_points_with_periods()
    # Build table of available periods
    periods = (
        df.select(["license_plate", "period_id"])
        .unique()
        .sort(["license_plate", "period_id"])
    )
    return df, periods


@app.cell
def _(marimo, periods):
    # Display periods table
    marimo.ui.table(periods)
    # User input for plate and period
    inp = marimo.ui.text(
        label=(
            "Enter license_plate and period_id separated by space.\n"
            "Example: 京BV4164 1"
        )
    )
    inp
    return (inp,)


@app.cell
def _(df, inp, marimo, periods, plotly_trajectory):
    if inp.value:
        parts = inp.value.strip().split()
        lp = parts[0]
        pid = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        # Filter and show selection
        mask = periods["license_plate"] == lp
        if pid is not None:
            mask &= periods["period_id"] == pid
        sel = periods.filter(mask)
        marimo.ui.table(sel)
        # If full selection, plot trajectory
        if pid is not None:
            fig = plotly_trajectory(df, lp, pid)
            marimo.ui.plotly(fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
