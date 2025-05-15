"""
threshold_analysis.py

Generate scientific evidence (statistics and histograms) for automatic threshold selection
based on empirical distributions of time gaps and implied speeds.

Usage:
    python3 threshold_analysis.py

Outputs:
    - time_diff_histogram.png
    - speed_histogram.png
    - Prints percentile and summary statistics to console.
"""
import os
from datetime import datetime
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

from utils import add_time_distance_calcs, add_implied_speed

def main():
    # Create output directory named by timestamp
    out_dir = f"threshold_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Results will be saved to: {out_dir}")
    # Load filtered raw points (including occupancy status)
    print("Loading data/filtered_points_in_beijing.parquet...")
    lazy_full = pl.scan_parquet("data/filtered_points_in_beijing.parquet")

    # Compute time and speed features on full dataset, then restrict to contiguous occupied segments
    print("Computing time differences and implied speeds for contiguous occupancy=1 segments...")
    df = (
        lazy_full
        .sort(["license_plate", "timestamp"])
        .pipe(add_time_distance_calcs)
        .pipe(add_implied_speed)
        .with_columns(
            pl.col("occupancy_status"),
            pl.col("occupancy_status").shift(1).over("license_plate").alias("prev_occ")
        )
        .filter((pl.col("occupancy_status") == 1) & (pl.col("prev_occ") == 1))
        .select(["time_diff_seconds", "implied_speed_kph"])
        .collect()
    )

    # Convert to numpy arrays, dropping nulls
    td = df["time_diff_seconds"].drop_nulls().to_numpy()
    spd = df["implied_speed_kph"].drop_nulls().to_numpy()

    # Compute robust IQR-based thresholds (Q3 + 1.5*IQR)
    q1_td, q3_td = np.percentile(td, [25, 75])
    iqr_td = q3_td - q1_td
    thr_td_iqr = q3_td + 1.5 * iqr_td
    q1_spd, q3_spd = np.percentile(spd, [25, 75])
    iqr_spd = q3_spd - q1_spd
    thr_spd_iqr = q3_spd + 1.5 * iqr_spd
    # Report IQR-based thresholds
    print(f"\nIQR-based threshold for time_diff_seconds: {thr_td_iqr:.2f} s")
    print(f"IQR-based threshold for implied_speed_kph: {thr_spd_iqr:.2f} km/h")

    # Prepare arrays for cleaned data and outliers
    td_clean = td[td <= thr_td_iqr]
    td_out = td[td > thr_td_iqr]
    spd_clean = spd[spd <= thr_spd_iqr]
    spd_out = spd[spd > thr_spd_iqr]

    # Define percentile tiers (as floats for consistency)
    pct_raw = [90.0, 95.0, 97.5, 99.0, 99.5, 99.9]
    pct_clean = [90.0, 95.0, 97.5]
    pct_out = [90.0, 95.0]

    # Print detailed percentiles for time differences
    print("\nPercentiles for time_diff_seconds (raw):")
    for p in pct_raw:
        print(f"  {p}th: {np.percentile(td, p):.2f}s")
    print("\nPercentiles for time_diff_seconds (clean ≤ IQR):")
    for p in pct_clean:
        print(f"  {p}th: {np.percentile(td_clean, p):.2f}s")
    print("\nPercentiles for time_diff_seconds (outliers > IQR):")
    for p in pct_out:
        print(f"  {p}th: {np.percentile(td_out, p):.2f}s")

    # Print detailed percentiles for implied speeds
    print("\nPercentiles for implied_speed_kph (raw):")
    for p in pct_raw:
        print(f"  {p}th: {np.percentile(spd, p):.2f} km/h")
    print("\nPercentiles for implied_speed_kph (clean ≤ IQR):")
    for p in pct_clean:
        print(f"  {p}th: {np.percentile(spd_clean, p):.2f} km/h")
    print("\nPercentiles for implied_speed_kph (outliers > IQR):")
    for p in pct_out:
        print(f"  {p}th: {np.percentile(spd_out, p):.2f} km/h")

    # Summary statistics via Polars and save to CSV
    print("\nSaving summary statistics to CSV...")
    # Raw summaries
    df.select(["time_diff_seconds"]).describe().write_csv(os.path.join(out_dir, "summary_time_diff_raw.csv"))
    df.select(["implied_speed_kph"]).describe().write_csv(os.path.join(out_dir, "summary_speed_raw.csv"))
    # Clean summaries
    pl.DataFrame({"time_diff_seconds": td_clean}).describe().write_csv(os.path.join(out_dir, "summary_time_diff_clean.csv"))
    pl.DataFrame({"implied_speed_kph": spd_clean}).describe().write_csv(os.path.join(out_dir, "summary_speed_clean.csv"))

    # Save percentile tables to CSV
    print("Saving percentile tables to CSV...")
    pl.DataFrame({"percentile": pct_raw, "time_diff_raw": [np.percentile(td, p) for p in pct_raw]}).write_csv(os.path.join(out_dir, "percentiles_time_diff_raw.csv"))
    pl.DataFrame({"percentile": pct_clean, "time_diff_clean": [np.percentile(td_clean, p) for p in pct_clean]}).write_csv(os.path.join(out_dir, "percentiles_time_diff_clean.csv"))
    pl.DataFrame({"percentile": pct_out, "time_diff_outliers": [np.percentile(td_out, p) for p in pct_out]}).write_csv(os.path.join(out_dir, "percentiles_time_diff_outliers.csv"))
    pl.DataFrame({"percentile": pct_raw, "speed_raw": [np.percentile(spd, p) for p in pct_raw]}).write_csv(os.path.join(out_dir, "percentiles_speed_raw.csv"))
    pl.DataFrame({"percentile": pct_clean, "speed_clean": [np.percentile(spd_clean, p) for p in pct_clean]}).write_csv(os.path.join(out_dir, "percentiles_speed_clean.csv"))
    pl.DataFrame({"percentile": pct_out, "speed_outliers": [np.percentile(spd_out, p) for p in pct_out]}).write_csv(os.path.join(out_dir, "percentiles_speed_outliers.csv"))

    # Generate and save histograms
    print("\nGenerating and saving histograms...")
    # Time differences: full distribution (log scale)
    plt.figure(figsize=(8, 5))
    plt.hist(td, bins=500, log=True, color="steelblue", edgecolor="black")
    plt.title("Time Difference Distribution (raw, log scale)")
    plt.xlabel("time_diff_seconds")
    plt.ylabel("Count (log)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_time_diff_raw.png"), dpi=150)
    plt.close()
    # Time differences: clean distribution
    plt.figure(figsize=(8, 5))
    plt.hist(td_clean, bins=200, color="teal", edgecolor="black")
    plt.axvline(thr_td_iqr, color="red", linestyle="--", label=f"IQR thr: {thr_td_iqr:.2f}s")
    plt.title("Time Difference Distribution (clean ≤ IQR)")
    plt.xlabel("time_diff_seconds")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_time_diff_clean.png"), dpi=150)
    plt.close()
    # Time differences: outliers
    if td_out.size > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(td_out, bins=100, color="orange", edgecolor="black")
        plt.title("Time Difference Outliers (> IQR)")
        plt.xlabel("time_diff_seconds")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hist_time_diff_outliers.png"), dpi=150)
        plt.close()

    # Speeds: raw distribution (log scale)
    plt.figure(figsize=(8, 5))
    plt.hist(spd, bins=500, log=True, color="darkgreen", edgecolor="black")
    plt.title("Speed Distribution (raw, log scale)")
    plt.xlabel("implied_speed_kph")
    plt.ylabel("Count (log)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_speed_raw.png"), dpi=150)
    plt.close()
    # Speeds: clean distribution
    plt.figure(figsize=(8, 5))
    plt.hist(spd_clean, bins=200, color="green", edgecolor="black")
    plt.axvline(thr_spd_iqr, color="red", linestyle="--", label=f"IQR thr: {thr_spd_iqr:.2f} km/h")
    plt.title("Speed Distribution (clean ≤ IQR)")
    plt.xlabel("implied_speed_kph")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_speed_clean.png"), dpi=150)
    plt.close()
    # Speeds: outliers
    if spd_out.size > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(spd_out, bins=100, color="salmon", edgecolor="black")
        plt.title("Speed Outliers (> IQR)")
        plt.xlabel("implied_speed_kph")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hist_speed_outliers.png"), dpi=150)
        plt.close()

    print(f"\nAll investigation results have been saved in '{out_dir}'.")

if __name__ == "__main__":
    main()