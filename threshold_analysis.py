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
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

from utils import add_time_distance_calcs, add_implied_speed

def main():
    # Load filtered raw points and focus on occupied taxi trajectories (occupancy_status == 1)
    print("Loading filtered_points_in_beijing.parquet (occupancy_status=1)...")
    lazy = (
        pl.scan_parquet("filtered_points_in_beijing.parquet")
        .filter(pl.col("occupancy_status") == 1)
    )

    # Compute time and speed features
    print("Computing time differences and implied speeds...")
    df = (
        lazy
        .sort("license_plate", "timestamp")
        .pipe(add_time_distance_calcs)
        .pipe(add_implied_speed)
        .select(["time_diff_seconds", "implied_speed_kph"])
        .collect()
    )

    # Convert to numpy arrays, dropping nulls
    td = df["time_diff_seconds"].drop_nulls().to_numpy()
    spd = df["implied_speed_kph"].drop_nulls().to_numpy()

    # Define percentiles of interest
    percentiles = [90, 95, 97.5, 99, 99.5]
    print("\nPercentiles for time_diff_seconds (seconds):")
    for p in percentiles:
        print(f"  {p}th percentile: {np.percentile(td, p):.2f}")

    print("\nPercentiles for implied_speed_kph (km/h):")
    for p in percentiles:
        print(f"  {p}th percentile: {np.percentile(spd, p):.2f}")

    # Summary statistics via Polars
    print("\nSummary statistics for time_diff_seconds:")
    # Use DataFrame.describe on the selected column
    print(df.select(["time_diff_seconds"]).describe())
    print("\nSummary statistics for implied_speed_kph:")
    print(df.select(["implied_speed_kph"]).describe())

    # Compute robust IQR-based thresholds (Q3 + 1.5*IQR)
    q1_td, q3_td = np.percentile(td, [25, 75])
    iqr_td = q3_td - q1_td
    thr_td_iqr = q3_td + 1.5 * iqr_td
    q1_spd, q3_spd = np.percentile(spd, [25, 75])
    iqr_spd = q3_spd - q1_spd
    thr_spd_iqr = q3_spd + 1.5 * iqr_spd
    print(f"\nIQR-based threshold for time_diff_seconds: {thr_td_iqr:.2f} (Q3 + 1.5*IQR)")
    print(f"IQR-based threshold for implied_speed_kph: {thr_spd_iqr:.2f} (Q3 + 1.5*IQR)")

    # Plot histograms with 99th percentile and IQR lines
    thr_td = np.percentile(td, 99)
    thr_spd = np.percentile(spd, 99)

    print("\nGenerating histograms...")
    plt.figure(figsize=(10, 6))
    plt.hist(td, bins=500, log=True, color="steelblue", edgecolor="black")
    plt.axvline(thr_td, color="red", linestyle="--",
                label=f"99th percentile: {thr_td:.2f}s")
    plt.axvline(thr_td_iqr, color="orange", linestyle="-.",
                label=f"IQR threshold: {thr_td_iqr:.2f}s")
    plt.title("Distribution of time differences (s)")
    plt.xlabel("time_diff_seconds")
    plt.ylabel("Count (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("time_diff_histogram.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(spd, bins=500, log=True, color="darkgreen", edgecolor="black")
    plt.axvline(thr_spd, color="red", linestyle="--",
                label=f"99th percentile: {thr_spd:.2f} km/h")
    plt.axvline(thr_spd_iqr, color="orange", linestyle="-.",
                label=f"IQR threshold: {thr_spd_iqr:.2f} km/h")
    plt.title("Distribution of implied speeds (km/h)")
    plt.xlabel("implied_speed_kph")
    plt.ylabel("Count (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("speed_histogram.png", dpi=150)
    plt.close()

    print("\nHistograms saved: time_diff_histogram.png, speed_histogram.png")

if __name__ == "__main__":
    main()