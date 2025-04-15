import marimo

__generated_with = "0.12.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import geopandas as gpd
    import polars as pl
    from datetime import datetime, timedelta
    return datetime, gpd, mo, pl, timedelta


@app.cell
def _(pl):
    ldf = pl.scan_parquet("ldf_lp_filtered_2019_11_25.parquet")
    return (ldf,)


@app.cell
def _(datetime, ldf, pl, timedelta):
    def filter_last_n_hours(ldf, specific_day, n_hours=6):
        # Define the start and end times based on the selected number of hours
        last_n_hours_start = specific_day + timedelta(hours=24 - n_hours)
        last_n_hours_end = specific_day + timedelta(hours=24)

        # Filter the LazyFrame to only keep the last n hours of the specified day
        df_filtered_last_nh_lazy = ldf.filter(
            (pl.col("timestamp") >= last_n_hours_start) & 
            (pl.col("timestamp") < last_n_hours_end)
        )
        return df_filtered_last_nh_lazy

    df_filtered_last_6h_lazy = filter_last_n_hours(ldf=ldf,specific_day=datetime(2019, 11, 25),n_hours=3)
    return df_filtered_last_6h_lazy, filter_last_n_hours


@app.cell
def _(df_filtered_last_6h_lazy):
    df_pandas = df_filtered_last_6h_lazy.collect().to_pandas()  # Convert to Pandas
    return (df_pandas,)


@app.cell
def _(gpd):
    # Load the GADM China shapefile (adjust the path to your extracted files)
    china = gpd.read_file("/home/matt/Downloads/gadm41_CHN_shp/gadm41_CHN_1.shp")

    # Filter Beijing's boundary (NAME_1 = 'Beijing')
    beijing_boundary = china[china["NAME_1"] == "Beijing"]

    # Check CRS (should be EPSG:4326 for GPS compatibility)
    beijing_boundary = beijing_boundary.to_crs("EPSG:4326")
    return beijing_boundary, china


@app.cell
def _(df_pandas, gpd):
    # Create a GeoDataFrame from longitude/latitude
    gdf_points = gpd.GeoDataFrame(
        df_pandas,
        geometry=gpd.points_from_xy(df_pandas.longitude, df_pandas.latitude),
        crs="EPSG:4326"  # Match Beijing boundary's CRS
    )
    return (gdf_points,)


@app.cell
def _(beijing_boundary, gdf_points, gpd):
    # Perform a spatial join to find points within Beijing's boundary
    points_in_beijing = gpd.sjoin(
        gdf_points,
        beijing_boundary,
        predicate="within",  # Filter points inside Beijing
        how="inner"
    )

    # Drop unnecessary columns from the join (e.g., "index_right")
    points_in_beijing = points_in_beijing.drop(columns=["index_right", "geometry"])
    return (points_in_beijing,)


@app.cell
def _(pl, points_in_beijing):
    # Convert the filtered GeoDataFrame back to Polars
    df_filtered_polars = pl.from_pandas(points_in_beijing)

    # Use LazyFrame for further processing if needed
    df_filtered_lazy_2 = df_filtered_polars.lazy()
    return df_filtered_lazy_2, df_filtered_polars


@app.cell
def _(df_filtered_lazy_2):
    df_filtered_lazy_2.sink_parquet("filtered_points_in_beijing.parquet")
    return


if __name__ == "__main__":
    app.run()
