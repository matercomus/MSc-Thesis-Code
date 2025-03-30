import marimo

__generated_with = "0.11.25"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Data Exploration I

        ## Todo:

        - [x] Missing values
        - [ ] Remove outliers (Timestamp and license plates)
        - [ ] Exclude data outside Beijing
        - [ ] Calc velocity
        - [x] Len, Min, Max, Mean
        - [ ] Plots
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from utils import profile_data, filter_chinese_license_plates
    return filter_chinese_license_plates, mo, pl, profile_data


@app.cell
def _(filter_chinese_license_plates, pl, profile_data):
    # File path to your Parquet file
    parquet_file_path = "2019.11.25.parquet"

    # Load the Parquet file lazily using Polars
    ldf = pl.scan_parquet(parquet_file_path)

    summary_df, _ = profile_data(ldf)

    ldf_lp_filtered = filter_chinese_license_plates(ldf)
    summary_df_lp_filtered, _ = profile_data(ldf_lp_filtered)

    del ldf
    return (
        ldf,
        ldf_lp_filtered,
        parquet_file_path,
        summary_df,
        summary_df_lp_filtered,
    )


@app.cell
def _(summary_df):
    summary_df
    return


@app.cell
def _(summary_df_lp_filtered):
    summary_df_lp_filtered
    return


if __name__ == "__main__":
    app.run()
