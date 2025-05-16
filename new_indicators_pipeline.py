"""
new_indicators_pipeline.py

A pure Python script for the full Beijing taxi trajectory cleaning and analysis pipeline.
- No Marimo, no notebooks, just a clean script.
- Uses tqdm for progress bars on long-running steps.
- Uses logging (see utils.py) for status updates.
- Outputs: cleaned_points_in_beijing.parquet, cleaned_with_period_id_in_beijing.parquet, periods_with_sld_ratio.parquet

Requirements: polars, pandas, tqdm, scikit-learn
"""

import logging
from tqdm import tqdm
import polars as pl
import pandas as pd
import numpy as np
from utils import (
    configure_logging,
    add_time_distance_calcs,
    add_implied_speed,
    add_abnormality_flags,
    select_final_columns,
    add_period_id,
    summarize_periods,
    detect_outliers_pd,
    compute_iqr_thresholds,
    compute_generic_iqr_threshold,
)
import osmnx as ox
import networkx as nx
from pipeline_utils import file_hash, write_meta, is_up_to_date
from pathlib import Path
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import glob
import shutil
import json
from datetime import datetime
import random
import subprocess
import sys
import re


# Set OSMnx settings for debugging and reproducibility
ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 180
ox.settings.overpass_rate_limit = True
ox.settings.default_retries = 3
ox.settings.default_response_json = True
# ox.settings.default_access = 'all'  # Removed: causes Overpass query errors

def save_parquet(df: pl.DataFrame, path: str, label: str = None):
    df.write_parquet(path)
    msg = f"Saved {label or path} to {path} (shape: {df.shape})"
    print(msg)
    logging.info(msg)

def attach_period_id(cleaned_df: pl.DataFrame, period_df: pl.DataFrame) -> pl.DataFrame:
    period_meta = period_df.select(["license_plate", "period_id", "start_time", "end_time"])
    joined = cleaned_df.join(
        period_meta,
        on=["license_plate"],
        how="left",
    ).filter(
        (pl.col("timestamp") >= pl.col("start_time")) & (pl.col("timestamp") <= pl.col("end_time"))
    )
    return joined

def ensure_osm_graph(osm_graph_path, periods_path, buffer=None, output_dir="pipeline_stats"):
    """
    Ensure the OSMnx graph is downloaded and covers the data bounding box.
    Buffer can be set via argument or OSM_GRAPH_BUFFER env var (default 0.05).
    If the graph is too small, try larger buffers, then fallback to Beijing city graph.
    output_dir: directory to save stats (for testability)
    """
    meta_path = str(osm_graph_path) + '.meta.json'
    if Path(osm_graph_path).exists() and Path(meta_path).exists():
        print("OSM graph already exists and is up-to-date.")
        return
    print("Downloading OSM graph cropped to data bounding box...")
    periods = pd.read_parquet(periods_path)
    min_lat = min(periods['start_latitude'].min(), periods['end_latitude'].min())
    max_lat = max(periods['start_latitude'].max(), periods['end_latitude'].max())
    min_lon = min(periods['start_longitude'].min(), periods['end_longitude'].min())
    max_lon = max(periods['end_longitude'].max(), periods['end_longitude'].max())
    # Buffer: argument > env var > default
    tried_buffers = []
    if buffer is None:
        buffer = float(os.environ.get("OSM_GRAPH_BUFFER", 0.05))
    for try_buffer in [buffer, 0.5, 1.0]:
        tried_buffers.append(try_buffer)
        bbox = (max_lat + try_buffer, min_lat - try_buffer, max_lon + try_buffer, min_lon - try_buffer)
        print(f"OSMnx bbox (north, south, east, west): {bbox}")
        print(f"  Covers lat: {min_lat-try_buffer:.6f} to {max_lat+try_buffer:.6f}")
        print(f"  Covers lon: {min_lon-try_buffer:.6f} to {max_lon+try_buffer:.6f}")
        G = ox.graph_from_bbox(bbox=bbox, network_type='drive')
        print(f"Downloaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges (buffer={try_buffer})")
        if len(G.nodes) >= 10:
            break
    else:
        print("WARNING: All bbox attempts resulted in too few nodes. Falling back to Beijing city graph.")
        G = ox.graph_from_place('Beijing, China', network_type='drive')
        print(f"Fallback Beijing graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    ox.save_graphml(G, osm_graph_path)
    write_meta(meta_path, {
        "graphml_hash": file_hash(osm_graph_path),
        "bbox": [min_lat, max_lat, min_lon, max_lon],
        "tried_buffers": tried_buffers,
        "final_node_count": len(G.nodes),
        "final_edge_count": len(G.edges),
    })

def save_step_stats(step_name: str, stats: dict, run_id: str, output_dir: str = "pipeline_stats"):
    """Save statistics for a pipeline step to a JSON file in a per-run subfolder."""
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(run_dir, f"{step_name}_stats_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logging.info(f"Saved {step_name} statistics to {output_path}")

def compute_network_shortest_paths_batched(
    periods_path,
    osm_graph_path,
    output_path,
    batch_size=500,
    num_workers=8,
    checkpoint_dir="network_checkpoints",
    run_id=None,
    node_cache_batch_size=5000,
    node_cache_num_workers=4,
    output_dir="pipeline_stats"
):
    """
    Compute network shortest paths and save stats to output_dir.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("network_shortest_path")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load periods
    periods_df = pd.read_parquet(periods_path)
    G = ox.load_graphml(osm_graph_path)
    
    # --- Persistent node cache logic (vectorized, batched, parallel) ---
    node_cache_path = os.path.join(checkpoint_dir, "point_to_node.parquet")
    node_cache_meta = node_cache_path + ".meta.json"
    periods_hash = file_hash(periods_path)
    graph_hash = file_hash(osm_graph_path)
    cache_valid = False
    if os.path.exists(node_cache_path) and os.path.exists(node_cache_meta):
        with open(node_cache_meta) as f:
            meta = json.load(f)
        if meta.get("periods_hash") == periods_hash and meta.get("graph_hash") == graph_hash:
            cache_valid = True
    if cache_valid:
        logger.info(f"Loading persistent node cache from {node_cache_path}")
        node_df = pd.read_parquet(node_cache_path)
        point_to_node = {(row.lat, row.lon): row.node for row in node_df.itertuples(index=False)}
    else:
        # Precompute/caching nearest nodes for all unique points (vectorized, batched, parallel)
        all_points = list(set(
            list(zip(periods_df['start_latitude'], periods_df['start_longitude'])) +
            list(zip(periods_df['end_latitude'], periods_df['end_longitude']))
        ))
        logger.info(f"Caching nearest OSM nodes for {len(all_points)} unique points (batched, parallel)...")
        point_to_node = {}
        node_rows = []
        def process_batch(batch):
            lats, lons = zip(*batch)
            try:
                nodes = ox.nearest_nodes(G, X=lons, Y=lats)
                return list(zip(batch, nodes))
            except Exception as e:
                logger.warning(f"Failed batch: {e}")
                # Mark all as None
                return list(zip(batch, [None]*len(batch)))
        # Split into batches
        batches = [all_points[i:i+node_cache_batch_size] for i in range(0, len(all_points), node_cache_batch_size)]
        with ThreadPoolExecutor(max_workers=node_cache_num_workers) as executor:
            futures = {executor.submit(process_batch, batch): batch for batch in batches}
            for future in tqdm(as_completed(futures), total=len(batches), desc="Node cache batches"):
                results = future.result()
                for (latlon, node) in results:
                    lat, lon = latlon
                    point_to_node[(lat, lon)] = node
                    node_rows.append({"lat": lat, "lon": lon, "node": node})
        # Save cache
        node_df = pd.DataFrame(node_rows)
        node_df.to_parquet(node_cache_path)
        with open(node_cache_meta, "w") as f:
            json.dump({"periods_hash": periods_hash, "graph_hash": graph_hash}, f)
        logger.info(f"Saved persistent node cache to {node_cache_path}")

    # Initialize statistics
    stats = {
        "total_periods": len(periods_df),
        "graph_info": {
            "nodes": len(G.nodes),
            "edges": len(G.edges),
        },
        "node_assignment": {
            "total_points": 0,
            "successful_assignments": 0,
            "failed_assignments": 0,
        },
        "path_computation": {
            "total_pairs": 0,
            "successful_paths": 0,
            "failed_paths": 0,
            "same_node_pairs": 0,
        },
        "distance_stats": {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        },
        "ratio_stats": {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "nan_count": 0,
        }
    }

    # Unique key for checkpointing
    periods_df['unique_key'] = periods_df['license_plate'].astype(str) + '_' + periods_df['period_id'].astype(str)

    # Check for existing checkpoint
    checkpoint_file = os.path.join(checkpoint_dir, "network_paths_checkpoint.parquet")
    if os.path.exists(checkpoint_file):
        done_df = pd.read_parquet(checkpoint_file)
        done_keys = set(done_df['unique_key'])
        logger.info(f"Loaded checkpoint with {len(done_keys)} completed periods.")
    else:
        done_df = pd.DataFrame()
        done_keys = set()

    # Filter out already processed periods
    to_process = periods_df[~periods_df['unique_key'].isin(done_keys)].copy()
    logger.info(f"Processing {len(to_process)} new periods (skipping {len(done_keys)} already done).")

    # Group by unique start/end node pairs to avoid redundant shortest path calculations
    to_process['start_node'] = [point_to_node.get((row['start_latitude'], row['start_longitude'])) for _, row in to_process.iterrows()]
    to_process['end_node'] = [point_to_node.get((row['end_latitude'], row['end_longitude'])) for _, row in to_process.iterrows()]
    node_pairs = set(zip(to_process['start_node'], to_process['end_node']))
    node_pair_to_dist = {}

    def compute_pair(pair):
        orig, dest = pair
        if orig is None or dest is None:
            return pair, float('nan')
        if orig == dest:
            # For same node, use a small distance (1 meter = 0.001 km)
            return pair, 0.001
        try:
            dist = nx.shortest_path_length(G, orig, dest, weight='length') / 1000
            if dist < 0.001:  # Less than 1 meter
                dist = 0.001  # Set minimum distance to 1 meter
            return pair, dist
        except Exception as e:
            logger.warning(f"Failed to compute shortest path for {pair}: {e}")
            return pair, float('nan')

    # Use ThreadPoolExecutor to avoid pickling issues
    logger.info(f"Computing shortest paths for {len(node_pairs)} unique node pairs...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_pair = {executor.submit(compute_pair, pair): pair for pair in node_pairs}
        for future in tqdm(as_completed(future_to_pair), total=len(node_pairs), desc="Node pairs"):
            pair, dist = future.result()
            node_pair_to_dist[pair] = dist

    # Assign distances to periods
    def get_dist(row):
        # Skip if start/end node is None
        if row['start_node'] is None or row['end_node'] is None:
            return float('nan')
        # For very small actual distances, use minimum network distance
        if row['sum_distance'] < 0.001:
            return 0.001
        return node_pair_to_dist.get((row['start_node'], row['end_node']), float('nan'))

    # Process in batches
    all_results = []
    all_distances = []
    all_ratios = []
    
    for i in tqdm(range(0, len(to_process), batch_size), desc="Batches"):
        batch = to_process.iloc[i:i+batch_size]
        batch['network_shortest_distance'] = batch.apply(get_dist, axis=1)
        # Compute route deviation ratio, handling NaN and inf values
        batch['route_deviation_ratio'] = batch.apply(
            lambda row: row['sum_distance'] / row['network_shortest_distance']
            if not pd.isna(row['network_shortest_distance']) and row['network_shortest_distance'] > 0
            else float('nan'),
            axis=1
        )
        
        # Collect statistics
        valid_distances = batch['network_shortest_distance'].dropna()
        valid_ratios = batch['route_deviation_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
        all_distances.extend(valid_distances)
        all_ratios.extend(valid_ratios)
        stats["ratio_stats"]["nan_count"] += batch['route_deviation_ratio'].isna().sum()
        
        all_results.append(batch)
        # Save checkpoint
        if os.path.exists(checkpoint_file):
            prev = pd.read_parquet(checkpoint_file)
            pd.concat([prev, batch]).drop_duplicates('unique_key').to_parquet(checkpoint_file)
        else:
            batch.to_parquet(checkpoint_file)
        logger.info(f"Checkpointed batch {i//batch_size+1} ({len(batch)} periods).")

    # Combine with previous results
    if not done_df.empty:
        all_results.append(done_df)
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Compute final statistics
    if all_distances:
        stats["distance_stats"].update({
            "min": float(min(all_distances)),
            "max": float(max(all_distances)),
            "mean": float(np.mean(all_distances)),
            "median": float(np.median(all_distances)),
        })
    if all_ratios:
        stats["ratio_stats"].update({
            "min": float(min(all_ratios)),
            "max": float(max(all_ratios)),
            "mean": float(np.mean(all_ratios)),
            "median": float(np.median(all_ratios)),
        })
    
    # Save statistics
    if run_id is not None:
        save_step_stats("network_shortest_paths", stats, run_id, output_dir=output_dir)
    else:
        save_step_stats("network_shortest_paths", stats, "default", output_dir=output_dir)
    
    final_df.to_parquet(output_path)
    logger.info(f"Saved final results to {output_path}")

    return final_df

def compute_network_outlier_flag(input_path, output_path, iqr_multiplier=1.5, run_id=None):
    input_paths = {"network_ratio_hash": input_path}
    if is_up_to_date(output_path, input_paths):
        print(f"{output_path} is up-to-date. Skipping computation.")
        return pd.read_parquet(output_path)
    
    # Setup logging
    logger = logging.getLogger("network_outlier_flag")
    
    # Initialize statistics
    stats = {
        "input_shape": None,
        "valid_ratios": 0,
        "outliers": 0,
        "quantiles": {
            "q1": None,
            "q3": None,
            "iqr": None,
            "threshold": None,
        }
    }
    
    df = pd.read_parquet(input_path)
    stats["input_shape"] = list(df.shape)
    
    # Filter out NaN and infinite values before computing quantiles
    valid_ratios = df['route_deviation_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    stats["valid_ratios"] = len(valid_ratios)
    
    if len(valid_ratios) == 0:
        logger.warning("No valid route_deviation_ratio values found. Setting all outlier flags to NaN.")
        df['is_network_outlier'] = np.nan
    else:
        q1 = valid_ratios.quantile(0.25)
        q3 = valid_ratios.quantile(0.75)
        iqr = q3 - q1
        threshold = q3 + iqr_multiplier * iqr
        
        stats["quantiles"].update({
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "threshold": float(threshold),
        })
        
        # Set outlier flag, handling NaN and inf values
        df['is_network_outlier'] = df['route_deviation_ratio'].apply(
            lambda x: True if pd.notna(x) and not np.isinf(x) and x > threshold else False
        )
        
        stats["outliers"] = int(df['is_network_outlier'].sum())
        
        logger.info(f"Network ratio outlier threshold: {threshold:.2f} (Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f})")
        logger.info(f"Found {stats['outliers']} network outliers out of {stats['valid_ratios']} valid ratios")
    
    # Save statistics
    if run_id is not None:
        save_step_stats("network_outlier_flag", stats, run_id)
    else:
        save_step_stats("network_outlier_flag", stats, "default")
    
    df.to_parquet(output_path)
    write_meta(str(output_path) + '.meta.json', {
        "network_ratio_hash": file_hash(input_path)
    })
    return df

def clean_pipeline_outputs():
    print("Cleaning pipeline outputs and checkpoints...")
    # Remove all files in network_checkpoints/
    checkpoint_dir = Path("network_checkpoints")
    if checkpoint_dir.exists():
        for f in checkpoint_dir.glob("*"):
            try:
                f.unlink()
                print(f"Deleted checkpoint: {f}")
            except Exception as e:
                print(f"Could not delete {f}: {e}")
    # Remove all .meta.json files in data/
    for meta_file in glob.glob("data/*.meta.json"):
        try:
            os.remove(meta_file)
            print(f"Deleted meta: {meta_file}")
        except Exception as e:
            print(f"Could not delete {meta_file}: {e}")
    # Remove all intermediate Parquet files in data/ except filtered_points_in_beijing.parquet
    keep_files = {"filtered_points_in_beijing.parquet"}
    for pq_file in glob.glob("data/*.parquet"):
        if os.path.basename(pq_file) not in keep_files:
            try:
                os.remove(pq_file)
                print(f"Deleted parquet: {pq_file}")
            except Exception as e:
                print(f"Could not delete {pq_file}: {e}")
    print("Clean complete.")

def clean_step_outputs(step):
    """Clean only the outputs/meta/checkpoints for a specific pipeline step."""
    step_map = {
        "cleaned_points": [
            "data/cleaned_points_in_beijing.parquet",
            "data/cleaned_points_in_beijing.parquet.meta.json",
        ],
        "cleaned_with_period_id": [
            "data/cleaned_with_period_id_in_beijing.parquet",
            "data/cleaned_with_period_id_in_beijing.parquet.meta.json",
        ],
        "periods_with_sld_ratio": [
            "data/periods_with_sld_ratio.parquet",
            "data/periods_with_sld_ratio.parquet.meta.json",
        ],
        "network": [
            "data/periods_with_network_ratio.parquet",
            "data/periods_with_network_ratio.parquet.meta.json",
            "data/periods_with_network_ratio_flagged.parquet",
            "data/periods_with_network_ratio_flagged.parquet.meta.json",
            "network_checkpoints/",
        ],
        "osm_graph": [
            "beijing_drive.graphml",
            "beijing_drive.graphml.meta.json",
        ],
    }
    if step not in step_map:
        print(f"Unknown step: {step}. Valid steps: {list(step_map.keys())}")
        return
    print(f"Cleaning outputs for step: {step}")
    for path in step_map[step]:
        if path.endswith("/"):
            # Directory: remove all files inside
            if os.path.exists(path):
                for f in os.listdir(path):
                    fpath = os.path.join(path, f)
                    try:
                        os.remove(fpath)
                        print(f"Deleted checkpoint: {fpath}")
                    except Exception as e:
                        print(f"Could not delete {fpath}: {e}")
        else:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Deleted: {path}")
                except Exception as e:
                    print(f"Could not delete {path}: {e}")
    print("Step clean complete.")

def find_latest_output(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    # Extract timestamp/run_id from filename
    def extract_runid(fname):
        m = re.search(r'_(\d{8}_\d{6})', fname)
        return m.group(1) if m else ''
    files = [(f, extract_runid(f)) for f in files]
    # Sort by run_id (timestamp)
    files = sorted(files, key=lambda x: x[1], reverse=True)
    return files[0][0] if files else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Remove all checkpoints, meta, and intermediate outputs before running pipeline.")
    parser.add_argument("--clean-step", type=str, help="Remove outputs/meta/checkpoints for specific steps (comma-separated). Valid steps: cleaned_points, cleaned_with_period_id, periods_with_sld_ratio, network, osm_graph.")
    parser.add_argument("--run-analysis", action="store_true", help="Automatically run the analysis tool after pipeline completes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--reuse-latest", action="store_true", help="Reuse latest available outputs for each step if possible.")
    args = parser.parse_args()
    if args.clean and args.clean_step:
        parser.error("--clean and --clean-step cannot be used together.")
    if args.clean and args.reuse_latest:
        parser.error("--clean and --reuse-latest cannot be used together.")
    if args.clean_step and args.reuse_latest:
        parser.error("--clean-step and --reuse-latest cannot be used together.")
    if args.clean:
        clean_pipeline_outputs()
    if args.clean_step:
        steps = [s.strip() for s in args.clean_step.split(",")]
        for step in steps:
            clean_step_outputs(step)

    configure_logging()
    logging.info("Starting indicators pipeline (pure Python version)")

    # --- Generate run_id and set up per-run stats folder ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_stats_dir = os.path.join("pipeline_stats", run_id)
    os.makedirs(run_stats_dir, exist_ok=True)

    # --- Set and log random seed for reproducibility ---
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    logging.info(f"Random seed set to {seed}")

    # --- Save pip freeze output for environment reproducibility ---
    env_txt = os.path.join(run_stats_dir, "environment.txt")
    with open(env_txt, "w") as f:
        subprocess.run(["pip", "freeze"], stdout=f)
    logging.info(f"Saved pip freeze to {env_txt}")

    # --- Step 1: Cleaned points ---
    cleaned_points_path = f"data/cleaned_points_in_beijing_{run_id}.parquet"
    cleaned_points_meta = cleaned_points_path + ".meta.json"
    filtered_points_path = "data/filtered_points_in_beijing.parquet"
    input_paths = {"filtered_points_hash": filtered_points_path}
    reuse_cleaned_points = False
    if args.reuse_latest and not os.path.exists(cleaned_points_path):
        latest = find_latest_output("data/cleaned_points_in_beijing_*.parquet")
        if latest and is_up_to_date(latest, input_paths):
            logging.info(f"[REUSE] Using {latest} for cleaned points step.")
            cleaned_points_path = latest
            cleaned_points_meta = cleaned_points_path + ".meta.json"
            reuse_cleaned_points = True
    if is_up_to_date(cleaned_points_path, input_paths):
        logging.info(f"{cleaned_points_path} is up-to-date. Skipping computation.")
        cleaned_df = pl.read_parquet(cleaned_points_path)
    else:
        # Load filtered points
        logging.info("Loading filtered_points_in_beijing.parquet...")
        lazy_df = pl.scan_parquet(filtered_points_path)

        # Compute robust thresholds using IQR on occupied trips only
        logging.info("Computing IQR-based thresholds for time gaps and speeds...")
        time_gap_th, speed_th = compute_iqr_thresholds(lazy_df)
        logging.info(f"Thresholds: time_gap={time_gap_th:.2f} sec, speed={speed_th:.2f} kph")

        # Compute base indicators and abnormality flags
        logging.info("Computing time, distance, speed, and abnormality flags...")
        base_df = (
            lazy_df.sort("license_plate", "timestamp")
            .pipe(add_time_distance_calcs)
            .pipe(add_implied_speed)
            .pipe(add_abnormality_flags, time_gap_th, speed_th)
            .pipe(select_final_columns)
        )
        logging.info("Collecting base DataFrame (this may take a while)...")
        base_df = base_df.collect()
        logging.info(f"Base DataFrame shape: {base_df.shape}")

        # Detect outliers using Isolation Forest (parallel, reproducible)
        logging.info("Detecting trajectory outliers (Isolation Forest, parallel, reproducible)...")
        needed_cols = ["license_plate", "timestamp", "implied_speed_kph", "time_diff_seconds", "latitude", "longitude"]
        pd_df = base_df.select(needed_cols).to_pandas()
        tqdm.pandas(desc="Outlier detection")
        pd_df["is_outlier"] = detect_outliers_pd(pd_df, n_jobs=-1, random_state=seed).values
        # Merge is_outlier back into base_df (Polars)
        base_df = base_df.with_columns(
            pl.Series("is_outlier", pd_df["is_outlier"].values)
        )
        results = base_df

        # Remove license plates with any temporal gap or position jump
        logging.info("Filtering out taxis with any temporal gap or position jump...")
        flagged_plates = results.filter(
            (pl.col("is_temporal_gap")) | (pl.col("is_position_jump"))
        ).get_column("license_plate").unique().to_list()
        cleaned_df = (
            results.filter(~pl.col("license_plate").is_in(flagged_plates))
            .drop(["is_temporal_gap", "is_position_jump"])
        )
        logging.info(f"Cleaned DataFrame shape: {cleaned_df.shape}")
        # Validation
        if cleaned_df.is_empty():
            logging.error("Cleaned DataFrame is empty after filtering! Aborting.")
            raise RuntimeError("Cleaned DataFrame is empty.")
        if cleaned_df.null_count().sum() > 0:
            logging.warning("Cleaned DataFrame contains nulls.")
        save_parquet(cleaned_df, cleaned_points_path, label="Cleaned points")
        write_meta(cleaned_points_meta, {"filtered_points_hash": file_hash(filtered_points_path), "seed": seed})

    # --- Step 2: Cleaned with period_id ---
    cleaned_with_pid_path = f"data/cleaned_with_period_id_in_beijing_{run_id}.parquet"
    cleaned_with_pid_meta = cleaned_with_pid_path + ".meta.json"
    input_paths = {"cleaned_points_hash": cleaned_points_path}
    reuse_cleaned_with_pid = False
    if args.reuse_latest and not os.path.exists(cleaned_with_pid_path):
        latest = find_latest_output("data/cleaned_with_period_id_in_beijing_*.parquet")
        if latest and is_up_to_date(latest, input_paths):
            logging.info(f"[REUSE] Using {latest} for cleaned_with_period_id step.")
            cleaned_with_pid_path = latest
            cleaned_with_pid_meta = cleaned_with_pid_path + ".meta.json"
            reuse_cleaned_with_pid = True
    if is_up_to_date(cleaned_with_pid_path, input_paths):
        logging.info(f"{cleaned_with_pid_path} is up-to-date. Skipping computation.")
        cleaned_with_period_id = pl.read_parquet(cleaned_with_pid_path)
    else:
        # Add period_id and summarize periods
        logging.info("Adding period_id and summarizing periods...")
        period_df = cleaned_df.pipe(add_period_id).pipe(summarize_periods)
        # Remove small periods (fewer than 3 points)
        MIN_PERIOD_POINTS = 3
        period_df = period_df.filter(pl.col("count_rows") >= MIN_PERIOD_POINTS)
        logging.info(f"Period summary shape: {period_df.shape}")

        # Attach period_id (and period time bounds) to cleaned points
        logging.info("Attaching period_id to cleaned points...")
        cleaned_with_period_id = attach_period_id(cleaned_df, period_df)
        logging.info(f"Cleaned with period_id shape: {cleaned_with_period_id.shape}")
        # Validation
        if cleaned_with_period_id.is_empty():
            logging.error("Cleaned with period_id DataFrame is empty! Aborting.")
            raise RuntimeError("Cleaned with period_id DataFrame is empty.")
        save_parquet(cleaned_with_period_id, cleaned_with_pid_path, label="Cleaned points with period_id")
        write_meta(cleaned_with_pid_meta, {"cleaned_points_hash": file_hash(cleaned_points_path), "seed": seed})

    # --- Step 3: Period summary with SLD ratio ---
    periods_sld_path = f"data/periods_with_sld_ratio_{run_id}.parquet"
    periods_sld_meta = periods_sld_path + ".meta.json"
    input_paths = {"cleaned_with_pid_hash": cleaned_with_pid_path}
    reuse_periods_sld = False
    if args.reuse_latest and not os.path.exists(periods_sld_path):
        latest = find_latest_output("data/periods_with_sld_ratio_*.parquet")
        if latest and is_up_to_date(latest, input_paths):
            logging.info(f"[REUSE] Using {latest} for periods_with_sld_ratio step.")
            periods_sld_path = latest
            periods_sld_meta = periods_sld_path + ".meta.json"
            reuse_periods_sld = True
    if is_up_to_date(periods_sld_path, input_paths):
        logging.info(f"{periods_sld_path} is up-to-date. Skipping computation.")
        period_df = pl.read_parquet(periods_sld_path)
    else:
        # Compute per-period outlier counts (Isolation Forest)
        logging.info("Computing per-period outlier counts...")
        outlier_stats = (
            cleaned_with_period_id.group_by(["license_plate", "period_id"])
            .agg((pl.col("is_outlier") == -1).sum().alias("traj_outlier_count"))
        )
        # Join outlier stats into period summary and compute ratio and flag
        period_df = (
            cleaned_with_period_id.pipe(add_period_id).pipe(summarize_periods)
            .join(outlier_stats, on=["license_plate", "period_id"], how="left")
            .with_columns([
                pl.col("traj_outlier_count").fill_null(0),
                (pl.col("traj_outlier_count") / pl.col("count_rows")).fill_null(0.0).alias("traj_outlier_ratio"),
                (pl.col("traj_outlier_count") > 0).alias("is_traj_outlier"),
            ])
        )
        # Flag SLD-based outliers using IQR threshold
        logging.info("Flagging SLD-based outliers...")
        sld_th = compute_generic_iqr_threshold(period_df.lazy(), "sld_ratio")
        period_df = period_df.with_columns(
            (pl.col("sld_ratio") > sld_th).alias("is_sld_outlier")
        )
        # Validation
        if period_df.is_empty():
            logging.error("Period summary DataFrame is empty! Aborting.")
            raise RuntimeError("Period summary DataFrame is empty.")
        save_parquet(period_df, periods_sld_path, label="Period summary with SLD ratio and outlier flags")
        write_meta(periods_sld_meta, {"cleaned_with_pid_hash": file_hash(cleaned_with_pid_path), "seed": seed})

    # After period_df is created and saved as data/periods_with_sld_ratio_{run_id}.parquet
    # 1. Ensure OSM graph
    ensure_osm_graph("beijing_drive.graphml", periods_sld_path)
    # 2. Compute network shortest paths and ratios (batched, threaded, checkpointed)
    network_ratio_path = f"data/periods_with_network_ratio_{run_id}.parquet"
    reuse_network_ratio = False
    input_paths = {"periods_sld_hash": periods_sld_path}
    if args.reuse_latest and not os.path.exists(network_ratio_path):
        latest = find_latest_output("data/periods_with_network_ratio_*.parquet")
        if latest and is_up_to_date(latest, input_paths):
            logging.info(f"[REUSE] Using {latest} for network_ratio step.")
            network_ratio_path = latest
            reuse_network_ratio = True
    compute_network_shortest_paths_batched(
        periods_path=periods_sld_path,
        osm_graph_path="beijing_drive.graphml",
        output_path=network_ratio_path,
        batch_size=500,
        num_workers=8,
        checkpoint_dir="network_checkpoints",
        run_id=run_id
    )
    # 3. Compute threshold and flag outliers
    final_periods_path = f"data/periods_with_network_ratio_flagged_{run_id}.parquet"
    reuse_final_periods = False
    input_paths = {"network_ratio_hash": network_ratio_path}
    if args.reuse_latest and not os.path.exists(final_periods_path):
        latest = find_latest_output("data/periods_with_network_ratio_flagged_*.parquet")
        if latest and is_up_to_date(latest, input_paths):
            logging.info(f"[REUSE] Using {latest} for network_ratio_flagged step.")
            final_periods_path = latest
            reuse_final_periods = True
    compute_network_outlier_flag(
        input_path=network_ratio_path,
        output_path=final_periods_path,
        run_id=run_id
    )

    # --- Write LAST_RUN_ID file ---
    with open(os.path.join("pipeline_stats", "LAST_RUN_ID"), "w") as f:
        f.write(run_id)

    logging.info("Pipeline completed: all parquet files written.")

    # --- Optionally run analysis tool ---
    if args.run_analysis:
        analysis_cmd = [
            sys.executable, "pipeline_network_analysis.py",
            "--graph", "beijing_drive.graphml",
            "--periods", final_periods_path,
            "--all"
        ]
        logging.info(f"Running analysis tool: {' '.join(analysis_cmd)}")
        subprocess.run(analysis_cmd)

if __name__ == "__main__":
    main()
