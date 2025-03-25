import marimo

__generated_with = "0.11.25"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""# Large CSV to Parquet Converter""")
    return


@app.cell
def _(mo):
    # File inputs
    csv_path = mo.ui.text(
        label="CSV File Path",
        placeholder="/path/to/your/large_file.csv",
    )
    output_path = mo.ui.text(
        label="Output Parquet Path (optional)",
        placeholder="/path/to/output.parquet",
    )

    # Conversion options
    compression = mo.ui.dropdown(
        label="Compression",
        options=["zstd", "snappy", "gzip", "lz4"],
        value="zstd",
    )
    batch_size = mo.ui.number(
        label="Batch Size (rows)",
        value=100000,
        start=1000,
        stop=1000000,
        step=10000,
    )
    keep_columns = mo.ui.text(
        label="Columns to Keep (comma separated)",
        placeholder="col1,col2,col3",
    )

    # Conversion button
    run_button = mo.ui.button(
        label="Convert to Parquet",
    )

    # Layout components using vstack
    options = mo.vstack([
        mo.md("## Conversion Options").callout(kind="info"),
        csv_path,
        output_path,
        compression,
        batch_size,
        keep_columns,
        run_button
    ])

    options
    return (
        batch_size,
        compression,
        csv_path,
        keep_columns,
        options,
        output_path,
        run_button,
    )


@app.cell
def _():
    import polars as pl
    from pathlib import Path
    import time
    return Path, pl, time


@app.cell
def _(
    Path,
    batch_size,
    compression,
    csv_path,
    keep_columns,
    mo,
    output_path,
    pl,
    run_button,
    time,
):
    if run_button.value:
        start_time = time.time()

        input_path = Path(csv_path.value.strip())
        out_path = Path(output_path.value.strip()) if output_path.value.strip() else None

        columns = (
            [col.strip() for col in keep_columns.value.split(",")] 
            if keep_columns.value.strip() 
            else None
        )

        if not input_path.exists():
            mo.stop(f"Input file not found: {input_path}")

        if out_path is None:
            out_path = input_path.with_suffix('.parquet')

        with mo.status.spinner(title=f"Converting {input_path.name}..."):
            try:
                scan = pl.scan_csv(
                    input_path,
                    infer_schema_length=10000,
                    batch_size=batch_size.value,
                )

                if columns:
                    scan = scan.select(columns)

                scan.sink_parquet(
                    out_path,
                    compression=compression.value
                )

                duration = time.time() - start_time
                file_size = out_path.stat().st_size / (1024 * 1024)

                mo.md(f"""
                ## Conversion Complete!
                - **Output file:** `{out_path}`
                - **File size:** {file_size:.2f} MB
                - **Compression:** {compression.value}
                - **Time taken:** {duration:.2f} seconds
                """).callout(kind="success")

            except Exception as e:
                mo.stop(f"Conversion failed: {str(e)}").callout(kind="danger")
    return columns, duration, file_size, input_path, out_path, scan, start_time


@app.cell
def _(mo):
    mo.md("""
    ### Usage Notes:
    1. For very large files, use a smaller batch size if you have limited RAM
    2. Zstd compression offers the best balance of speed and compression ratio
    3. Specifying columns to keep can significantly reduce memory usage
    4. The conversion happens in streaming mode - the full file is never loaded into memory
    """).callout(kind="info")
    return


if __name__ == "__main__":
    app.run()
