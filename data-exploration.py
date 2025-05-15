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
        - [ ] profile code
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
    parquet_file_path = "data/2019.11.25.parquet"

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
    sns.histplot(data=df_timestamp_counts, x='timestamp', weights='len')
    plt.gca()
    return


@app.cell
def _(ldf_lp_filtered, profile_data):
    from utils import filter_by_date

    ldf_lp_filtered_2019 = filter_by_date(ldf_lp_filtered, correct_year=2019)
    summary_df_lp_filtered_2019, _ = profile_data(ldf_lp_filtered_2019)
    summary_df_lp_filtered_2019
    return filter_by_date, ldf_lp_filtered_2019, summary_df_lp_filtered_2019


@app.cell
def _(ldf_lp_filtered_2019, plt, sns):
    # Efficiently aggregate timestamp counts using Polars
    timestamp_counts2 = ldf_lp_filtered_2019.group_by("timestamp").len().collect()

    # Convert the aggregated Polars DataFrame to Pandas for Seaborn
    df_timestamp_counts2 = timestamp_counts2.to_pandas()

    sns.histplot(data=df_timestamp_counts2, x='timestamp', weights='len', bins=12)
    plt.gca()
    return df_timestamp_counts2, timestamp_counts2


@app.cell
def _(df_timestamp_counts2):
    df_timestamp_counts2.describe()
    return


@app.cell
def _(df_timestamp_counts2, plt, sns):
    # If timestamps are datetime objects
    import matplotlib.dates as mdates

    ax = sns.histplot(data=df_timestamp_counts2, x='timestamp', weights='len', bins=50)

    # Show ticks every 7 days (approximately 1/4 of a month)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))  # Format as "Month Day"
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()
    return ax, mdates


@app.cell
def _(filter_by_date, ldf_lp_filtered_2019):
    from datetime import datetime

    ldf_lp_filtered_2019_11_25 = filter_by_date(
            ldf_lp_filtered_2019,
            start_date=datetime(2019, 11, 25, 0, 0),
            end_date=datetime(2019, 11, 25, 23, 59, 59, 999999),)
    return datetime, ldf_lp_filtered_2019_11_25


@app.cell
def _(ldf_lp_filtered_2019_11_25):
    timestamp_counts3 = ldf_lp_filtered_2019_11_25.group_by("timestamp").len().collect()
    df_timestamp_counts3 = timestamp_counts3.to_pandas()
    return df_timestamp_counts3, timestamp_counts3


@app.cell
def _(df_timestamp_counts3, plt, sns):
    # Plot
    sns.histplot(data=df_timestamp_counts3, x='timestamp', weights='len')
    plt.xticks(rotation=90)
    plt.gca()
    return


@app.cell
def _(df_timestamp_counts3):
    df_timestamp_counts3.describe()
    return


@app.cell
def _(ldf_lp_filtered_2019_11_25, profile_data):
    summary_ldf_lp_filtered_2019_11_25, _ = profile_data(ldf_lp_filtered_2019_11_25)
    return (summary_ldf_lp_filtered_2019_11_25,)


@app.cell
def _(summary_ldf_lp_filtered_2019_11_25):
    summary_ldf_lp_filtered_2019_11_25
    return


@app.cell
def _(ldf_lp_filtered_2019_11_25):
    ldf_lp_filtered_2019_11_25.sink_parquet("data/ldf_lp_filtered_2019_11_25.parquet")
    return


@app.cell
def _(df_timestamp_counts3, plt, sns):
    # Plot
    sns.histplot(data=df_timestamp_counts3, x='timestamp', weights='len', bins=24)
    plt.xticks(rotation=90)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
