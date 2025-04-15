import marimo

__generated_with = "0.12.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    return gpd, mcolors, mo, pl, plt


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
def _(pl):
    filtered_points_in_beijing = pl.scan_parquet("filtered_points_in_beijing.parquet")
    return (filtered_points_in_beijing,)


@app.cell
def _(filtered_points_in_beijing):
    points_in_beijing = filtered_points_in_beijing.collect().to_pandas()  # Convert to Pandas
    return (points_in_beijing,)


@app.cell
def _(beijing_boundary, gpd, plt, points_in_beijing):
    # Plot Beijing's boundary and the filtered points
    fig, ax = plt.subplots(figsize=(10, 10))
    beijing_boundary.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=1, alpha=0.5)
    gdf_points_in_beijing = gpd.GeoDataFrame(
        points_in_beijing, 
        geometry=gpd.points_from_xy(points_in_beijing.longitude, points_in_beijing.latitude),
        crs="EPSG:4326"
    )
    gdf_points_in_beijing.plot(ax=ax, color='red', markersize=5, label='Filtered Points')
    plt.title('Filtered Points within Beijing')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.gca()
    return ax, fig, gdf_points_in_beijing


@app.cell
def _(fig):
    fig.savefig("filtered_points_in_beijing.png", dpi=300, bbox_inches='tight')
    return


@app.cell
def _(fig):
    fig.savefig("filtered_points_in_beijing.pdf", dpi=300, bbox_inches='tight')
    return


if __name__ == "__main__":
    app.run()
