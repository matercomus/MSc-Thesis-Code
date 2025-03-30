import marimo

__generated_with = "0.11.31"
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


@app.cell
def _(ldf_lp_filtered):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Efficiently aggregate timestamp counts using Polars
    timestamp_counts = ldf_lp_filtered.group_by("timestamp").len().collect()

    # Convert the aggregated Polars DataFrame to Pandas for Seaborn
    df_timestamp_counts = timestamp_counts.to_pandas()
    return df_timestamp_counts, plt, sns, timestamp_counts


@app.cell
def _(df_timestamp_counts, plt, sns):
    sns.histplot(data=df_timestamp_counts, x='timestamp', weights='count')
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
