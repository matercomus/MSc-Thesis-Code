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
from pipeline_utils import file_hash, write_meta, is_up_to_date, PipelineStats
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
import git  # Add at the top for git commit hash


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
    Buffer can be set via argument or OSM_GRAPH_BUFFER env var (default 1.0).
    If the graph is too small, try larger buffers, then fallback to Beijing city graph.
    output_dir: directory to save stats (for testability)
    Returns: (action, output_file)
    """
    meta_path = str(osm_graph_path) + '.meta.json'
    input_paths = {"periods_hash": periods_path}
    if Path(osm_graph_path).exists() and Path(meta_path).exists():
        if is_up_to_date(osm_graph_path, input_paths):
            logging.info(f"[REUSE] OSM graph is up-to-date: {osm_graph_path}")
            return "reused", osm_graph_path
        else:
            logging.info(f"[RECOMPUTE] OSM graph exists but is not up-to-date: {osm_graph_path}")
    else:
        logging.info(f"[RECOMPUTE] OSM graph does not exist: {osm_graph_path}")
    print("Downloading OSM graph cropped to data bounding box...")
    periods = pd.read_parquet(periods_path)
    min_lat = min(periods['start_latitude'].min(), periods['end_latitude'].min())
    max_lat = max(periods['start_latitude'].max(), periods['end_latitude'].max())
    min_lon = min(periods['start_longitude'].min(), periods['end_longitude'].min())
    max_lon = max(periods['end_longitude'].max(), periods['end_longitude'].max())
    # Buffer: argument > env var > default
    tried_buffers = []
    if buffer is None:
        buffer = float(os.environ.get("OSM_GRAPH_BUFFER", 1.0))
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
        "periods_hash": file_hash(periods_path),
    })
    logging.info(f"[RECOMPUTE] OSM graph written: {osm_graph_path}")
    return "recomputed", osm_graph_path

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
    output_dir="pipeline_stats",
    clean_mode=False
):
    """
    Compute network shortest paths and save stats to output_dir.
    If output is up-to-date, recompute stats from file and save stats file.
    On reuse, if a previous stats file exists, merge in detailed stats.
    """
    logger = logging.getLogger("network_shortest_path")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load periods
    periods_df = pd.read_parquet(periods_path)
    G = ox.load_graphml(osm_graph_path)

    # --- Node cache handling ---
    node_cache_path = os.path.join(checkpoint_dir, "point_to_node.parquet")
    node_cache_meta = node_cache_path + ".meta.json"
    periods_hash = file_hash(periods_path)
    graph_hash = file_hash(osm_graph_path)
    
    if clean_mode:
        logger.info("[CLEAN] Skipping node cache in clean mode")
        cache_valid = False
    else:
        cache_valid = os.path.exists(node_cache_path) and os.path.exists(node_cache_meta)
        if cache_valid:
            with open(node_cache_meta) as f:
                meta = json.load(f)
            cache_valid = meta.get("periods_hash") == periods_hash and meta.get("graph_hash") == graph_hash

    if cache_valid:
        logger.info(f"[REUSE] Loading node cache from {node_cache_path}")
        node_df = pd.read_parquet(node_cache_path)
        point_to_node = {(row.lat, row.lon): row.node for row in node_df.itertuples(index=False)}
    else:
        logger.info("[COMPUTE] Building node cache from scratch")
        # Get unique points
        all_points = list(set(
            list(zip(periods_df['start_latitude'], periods_df['start_longitude'])) +
            list(zip(periods_df['end_latitude'], periods_df['end_longitude']))
        ))
        logger.info(f"Processing {len(all_points)} unique points in {node_cache_batch_size} point batches")
        
        point_to_node = {}
        node_rows = []
        
        # Process in batches
        batches = [all_points[i:i+node_cache_batch_size] for i in range(0, len(all_points), node_cache_batch_size)]
        
        # Define process_batch here so it can access G
        def process_batch(batch):
            lats, lons = zip(*batch)
            try:
                nodes = ox.nearest_nodes(G, lons, lats)  # vectorized
            except Exception:
                nodes = [None] * len(batch)
            return list(zip(batch, nodes))
        
        with ThreadPoolExecutor(max_workers=node_cache_num_workers) as executor:
            futures = {executor.submit(process_batch, batch): batch for batch in batches}
            for future in tqdm(as_completed(futures), total=len(batches), desc="Node cache batches"):
                results = future.result()
                for (latlon, node) in results:
                    lat, lon = latlon
                    point_to_node[(lat, lon)] = node
                    node_rows.append({"lat": lat, "lon": lon, "node": node})
        
        # Save cache unless in clean mode
        if not clean_mode:
            node_df = pd.DataFrame(node_rows)
            node_df.to_parquet(node_cache_path)
            with open(node_cache_meta, "w") as f:
                json.dump({"periods_hash": periods_hash, "graph_hash": graph_hash}, f)
            logger.info(f"[SAVE] Node cache written to {node_cache_path}")

    # Save node assignments for reproducibility
    node_assignment_path = os.path.join(output_dir, run_id, f"node_assignment_{run_id}.parquet") if run_id else "node_assignment.parquet"
    node_df = pd.DataFrame(node_rows)
    node_df.to_parquet(node_assignment_path)
    logger.info(f"[SAVE] Node assignments saved to {node_assignment_path}")

    # --- Node assignment stats ---
    total_points = len(all_points)
    assigned_nodes = [point_to_node.get((lat, lon)) for (lat, lon) in all_points]
    successful_assignments = sum(n is not None for n in assigned_nodes)
    failed_assignments = total_points - successful_assignments
    
    # More detailed stats
    stats = {
        "node_assignment": {
            "total_points": total_points,
            "successful_assignments": successful_assignments,
            "failed_assignments": failed_assignments,
            "success_pct": successful_assignments / total_points if total_points else None,
            "fail_pct": failed_assignments / total_points if total_points else None,
            "unique_nodes_used": len(set(n for n in assigned_nodes if n is not None)),
            "cache_status": "reused" if cache_valid else "computed",
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
    node_pair_rows = []

    # --- Path computation stats ---
    total_pairs = len(node_pairs)
    successful_paths = 0
    failed_paths = 0
    same_node_pairs = 0
    path_lengths = []
    # --- BATCHED NODE PAIR SHORTEST PATHS ---
    def compute_pair(pair):
        orig, dest = pair
        if orig is None or dest is None:
            return pair, float('nan')
        if orig == dest:
            # For same node, skip (will be filtered out later)
            return pair, float('nan')
        try:
            dist = nx.shortest_path_length(G, orig, dest, weight='length') / 1000
            return pair, dist
        except Exception as e:
            # Count unreachable node pairs in stats
            return pair, float('nan')

    logger.info(f"Computing shortest paths for {len(node_pairs)} unique node pairs (batched)...")
    node_pairs = list(node_pairs)
    batch_size_pairs = 1000
    for i in tqdm(range(0, len(node_pairs), batch_size_pairs), desc="Node pair batches"):
        batch_pairs = node_pairs[i:i+batch_size_pairs]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(compute_pair, pair): pair for pair in batch_pairs}
        for future in as_completed(futures):
            pair, dist = future.result()
            node_pair_to_dist[pair] = dist
            if pair[0] is not None and pair[1] is not None:
                node_pair_rows.append({"start_node": pair[0], "end_node": pair[1], "distance_km": dist})
            if pair[0] is None or pair[1] is None:
                continue
            if pair[0] == pair[1]:
                same_node_pairs += 1
            elif np.isnan(dist):
                failed_paths += 1
            else:
                successful_paths += 1
                path_lengths.append(dist)

    stats["path_computation"].update({
        "total_pairs": total_pairs,
        "successful_paths": successful_paths,
        "failed_paths": failed_paths,
        "same_node_pairs": same_node_pairs,
        "success_pct": successful_paths / total_pairs if total_pairs else None,
        "fail_pct": failed_paths / total_pairs if total_pairs else None,
        "same_node_pct": same_node_pairs / total_pairs if total_pairs else None,
    })
    # Path length distribution stats
    if path_lengths:
        arr = np.array(path_lengths)
        stats["path_computation"].update({
            "length_min": float(np.min(arr)),
            "length_max": float(np.max(arr)),
            "length_mean": float(np.mean(arr)),
            "length_median": float(np.median(arr)),
            "length_std": float(np.std(arr)),
            "length_percentiles": {p: float(np.percentile(arr, p)) for p in [5, 25, 50, 75, 95]},
            "length_hist": np.histogram(arr, bins=10)[0].tolist(),
            "length_bin_edges": np.histogram(arr, bins=10)[1].tolist(),
        })

    # Assign distances to periods
    def get_dist(row):
        if row['start_node'] is None or row['end_node'] is None:
            return float('nan')
        return node_pair_to_dist.get((row['start_node'], row['end_node']), float('nan'))

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
        all_results.append(batch)
        valid_distances = batch['network_shortest_distance'].dropna()
        valid_ratios = batch['route_deviation_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
        all_distances.extend(valid_distances)
        all_ratios.extend(valid_ratios)
        stats["ratio_stats"]["nan_count"] += batch['route_deviation_ratio'].isna().sum()
        # Save checkpoint
        if os.path.exists(checkpoint_file):
            prev = pd.read_parquet(checkpoint_file)
            pd.concat([prev, batch]).drop_duplicates('unique_key').to_parquet(checkpoint_file)
        else:
            batch.to_parquet(checkpoint_file)
        logger.info(f"Checkpointed batch {i//batch_size+1} ({len(batch)} periods).")
    # Route deviation ratio distribution stats
    if all_ratios:
        arr = np.array(all_ratios)
        stats["ratio_stats"].update({
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "percentiles": {p: float(np.percentile(arr, p)) for p in [5, 25, 50, 75, 95]},
            "hist": np.histogram(arr, bins=10)[0].tolist(),
            "bin_edges": np.histogram(arr, bins=10)[1].tolist(),
        })

    # Combine with previous results
    if not done_df.empty:
        all_results.append(done_df)
    final_df = pd.concat(all_results, ignore_index=True)

    # --- FILTER OUT FAILED/DEGENERATE PERIODS ---
    n_before_failed_filter = len(final_df)
    failed_mask = final_df['network_shortest_distance'].isna() | (final_df['network_shortest_distance'] < 1.0)
    n_failed_periods = failed_mask.sum()
    if n_failed_periods > 0:
        logger.info(f"Filtering out {n_failed_periods} periods with failed or degenerate network path computation (NaN or <1km).")
    final_df = final_df[~failed_mask].copy()
    n_final = len(final_df)

    # Update stats
    stats["filtered_failed_periods"] = int(n_failed_periods)
    stats["final_periods"] = int(n_final)

    # Compute final statistics
    if not final_df.empty:
        stats["distance_stats"].update({
            "min": float(final_df['network_shortest_distance'].min()),
            "max": float(final_df['network_shortest_distance'].max()),
            "mean": float(final_df['network_shortest_distance'].mean()),
            "median": float(final_df['network_shortest_distance'].median()),
        })
        stats["ratio_stats"].update({
            "min": float(final_df['route_deviation_ratio'].min()),
            "max": float(final_df['route_deviation_ratio'].max()),
            "mean": float(final_df['route_deviation_ratio'].mean()),
            "median": float(final_df['route_deviation_ratio'].median()),
        })

    # Save statistics
    if run_id is not None:
        save_step_stats("network_shortest_paths", stats, run_id, output_dir=output_dir)
    else:
        save_step_stats("network_shortest_paths", stats, "default", output_dir=output_dir)

    # Save node pair info for reproducibility
    node_pairs_path = os.path.join(output_dir, run_id, f"node_pairs_{run_id}.parquet") if run_id else "node_pairs.parquet"
    pd.DataFrame(node_pair_rows).to_parquet(node_pairs_path)

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

# Helper to extract run_id from a reused_from path
def extract_run_id_from_path(path):
    if not path:
        return None
    m = re.search(r'_(\d{8}_\d{6})', str(path))
    return m.group(1) if m else None

# Helper to copy stats fields from source run's metadata
def copy_stats_from_source(step_name, reused_from, depth=0):
    indent = '  ' * depth
    src_run_id = extract_run_id_from_path(reused_from)
    print(f"{indent}[DEBUG] copy_stats_from_source: step={step_name}, reused_from={reused_from}, extracted_run_id={src_run_id}")
    if not src_run_id:
        print(f"{indent}[DEBUG] No run id extracted from reused_from path.")
        return {}
    src_meta_path = os.path.join("pipeline_stats", src_run_id, "run_metadata.json")
    print(f"{indent}[DEBUG] Looking for source metadata at: {src_meta_path}")
    if not os.path.exists(src_meta_path):
        print(f"{indent}[DEBUG] Source metadata file does not exist.")
        return {}
    with open(src_meta_path) as f:
        src_meta = json.load(f)
    src_step = src_meta.get("steps", {}).get(step_name, {})
    fields = ["n_before", "n_after", "filtered", "pct_filtered", "criteria"]
    found = {k: src_step[k] for k in fields if k in src_step}
    print(f"{indent}[DEBUG] Found fields: {found}")
    # If not all fields found and this step is also reused, recurse
    if (len(found) < len(fields)) and src_step.get("reused") and src_step.get("reused_from") and depth < 10:
        print(f"{indent}[DEBUG] Not all fields found, recursing to next reused_from: {src_step.get('reused_from')}")
        next_found = copy_stats_from_source(step_name, src_step.get("reused_from"), depth+1)
        # Merge, prefer values from next_found if not present in found
        for k in fields:
            if k not in found and k in next_found:
                found[k] = next_found[k]
    return found

# --- Helper for cleaned_points step ---
def run_cleaned_points(filtered_points_path, cleaned_points_path, cleaned_points_meta, step_name, stats, args):
    import polars as pl
    from utils import add_time_distance_calcs, add_implied_speed, add_abnormality_flags, select_final_columns, compute_iqr_thresholds
    import logging
    from pipeline_utils import file_hash, write_meta
    # Compute thresholds using IQR on the filtered points
    lazy_df = pl.scan_parquet(filtered_points_path)
    gap_threshold_sec, speed_threshold_kph = compute_iqr_thresholds(lazy_df)
    logging.info(f"Using gap_threshold_sec={gap_threshold_sec:.2f}, speed_threshold_kph={speed_threshold_kph:.2f} for abnormality flags")
    cleaned_df = pl.read_parquet(filtered_points_path).pipe(
        add_time_distance_calcs
    ).pipe(
        add_implied_speed
    ).pipe(
        add_abnormality_flags, gap_threshold_sec, speed_threshold_kph
    ).pipe(
        select_final_columns
    )
    save_parquet(cleaned_df, cleaned_points_path, "cleaned points")
    write_meta(cleaned_points_meta, {"filtered_points_hash": file_hash(filtered_points_path)})
    stats.record_step_stats(step_name, cleaned_df)
    return cleaned_df

# --- Helper for cleaned_with_period_id step ---
def run_cleaned_with_period_id(cleaned_points_path, cleaned_with_pid_path, cleaned_with_pid_meta, step_name, stats, args):
    import polars as pl
    from utils import add_period_id
    from pipeline_utils import file_hash, write_meta
    import logging
    # Read cleaned points and add period_id
    cleaned_points_df = pl.read_parquet(cleaned_points_path)
    cleaned_with_pid = add_period_id(cleaned_points_df)
    # Save
    cleaned_with_pid.write_parquet(cleaned_with_pid_path)
    write_meta(cleaned_with_pid_meta, {"cleaned_points_hash": file_hash(cleaned_points_path)})
    stats.record_step_stats(step_name, cleaned_with_pid)
    return cleaned_with_pid

# --- Helper for periods_with_sld_ratio step ---
def run_periods_with_sld_ratio(cleaned_with_period_id, periods_sld_path, periods_sld_meta, step_name, stats, args):
    from utils import summarize_periods
    from pipeline_utils import file_hash, write_meta
    import logging
    # Summarize periods
    period_df = summarize_periods(cleaned_with_period_id)
    period_df.write_parquet(periods_sld_path)
    write_meta(periods_sld_meta, {"cleaned_with_pid_hash": file_hash(periods_sld_path)})
    stats.record_step_stats(step_name, period_df)
    return period_df

# --- Helper for network ratio step ---
def run_network_ratio_step(periods_sld_path, run_id, stats, args):
    import os
    import logging
    from pipeline_utils import file_hash, write_meta, is_up_to_date
    import pandas as pd
    network_ratio_path = f"data/periods_with_network_ratio_{run_id}.parquet"
    input_paths = {"periods_sld_hash": periods_sld_path}
    reused_from = None
    step_name = "network_ratio"
    step_info = {"timestamp": datetime.now().isoformat()}
    # 1. Ensure OSM graph
    action, osm_graph_path = ensure_osm_graph("beijing_drive.graphml", periods_sld_path)
    # 2. Compute or reuse
    if args.clean:
        logging.info(f"[CLEAN] Computing {step_name} from scratch")
        network_ratio_df = compute_network_shortest_paths_batched(
            periods_path=periods_sld_path,
            osm_graph_path=osm_graph_path,
            output_path=network_ratio_path,
            batch_size=500,
            num_workers=8,
            checkpoint_dir="network_checkpoints",
            run_id=run_id,
            clean_mode=args.clean
        )
        step_info["status"] = "computed"
        step_info["output_path"] = network_ratio_path
    else:
        if args.reuse_latest and not os.path.exists(network_ratio_path):
            latest = find_latest_output("data/periods_with_network_ratio_*.parquet")
            if latest and is_up_to_date(latest, input_paths):
                logging.info(f"[REUSE] Using {latest} for {step_name} step")
                network_ratio_path = latest
                reused_from = latest
        if is_up_to_date(network_ratio_path, input_paths):
            logging.info(f"[SKIP] {network_ratio_path} is up-to-date")
            network_ratio_df = pd.read_parquet(network_ratio_path)
            step_info["status"] = "reused"
            step_info["reused_from"] = reused_from
            step_info["output_path"] = network_ratio_path
        else:
            logging.info(f"[COMPUTE] Computing {step_name} from scratch")
            network_ratio_df = compute_network_shortest_paths_batched(
                periods_path=periods_sld_path,
                osm_graph_path=osm_graph_path,
                output_path=network_ratio_path,
                batch_size=500,
                num_workers=8,
                checkpoint_dir="network_checkpoints",
                run_id=run_id,
                clean_mode=args.clean
            )
            step_info["status"] = "computed"
            step_info["output_path"] = network_ratio_path
    stats.record_step_stats(step_name, network_ratio_df)
    return network_ratio_df, step_info

# --- Helper for network outlier flag step ---
def run_network_outlier_flag_step(network_ratio_path, run_id, stats, args):
    import logging
    from pipeline_utils import file_hash, write_meta, is_up_to_date
    import pandas as pd
    step_name = "network_outlier_flag"
    step_info = {"timestamp": datetime.now().isoformat()}
    output_path = f"data/periods_with_network_ratio_flagged_{run_id}.parquet"
    input_paths = {"network_ratio_hash": network_ratio_path}
    reused_from = None
    final_df = None
    if args.clean:
        logging.info(f"[CLEAN] Computing {step_name} from scratch")
        final_df = compute_network_outlier_flag(
            input_path=network_ratio_path,
            output_path=output_path,
            run_id=run_id
        )
        step_info["status"] = "computed"
        step_info["output_path"] = output_path
    else:
        if args.reuse_latest and not os.path.exists(output_path):
            latest = find_latest_output("data/periods_with_network_ratio_flagged_*.parquet")
            if latest and is_up_to_date(latest, input_paths):
                logging.info(f"[REUSE] Using {latest} for {step_name} step")
                final_df = pd.read_parquet(latest)
                step_info["status"] = "reused"
                step_info["reused_from"] = latest
                step_info["output_path"] = latest
        if final_df is None:
            logging.info(f"[COMPUTE] Computing {step_name} from scratch")
            final_df = compute_network_outlier_flag(
                input_path=network_ratio_path,
                output_path=output_path,
                run_id=run_id
            )
            step_info["status"] = "computed"
            step_info["output_path"] = output_path
    stats.record_step_stats(step_name, final_df)
    return final_df, step_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Remove all checkpoints, meta, and intermediate outputs before running pipeline.")
    parser.add_argument("--clean-step", type=str, help="Remove outputs/meta/checkpoints for specific steps (comma-separated). Valid steps: cleaned_points, cleaned_with_period_id, periods_with_sld_ratio, network, osm_graph.")
    parser.add_argument("--run-analysis", action="store_true", help="Automatically run the analysis tool after pipeline completes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--reuse-latest", action="store_true", help="Try to reuse latest available outputs for each step.")
    args = parser.parse_args()
    if args.clean and args.clean_step:
        parser.error("--clean and --clean-step cannot be used together.")
    if args.clean and args.reuse_latest:
        parser.error("--clean and --reuse-latest cannot be used together.")
    if args.clean_step and args.reuse_latest:
        parser.error("--clean-step and --reuse-latest cannot be used together.")
    if args.clean:
        logging.info("[CLEAN] Running pipeline in clean mode - all steps will be computed from scratch")
        clean_pipeline_outputs()
    elif args.reuse_latest:
        logging.info("[REUSE] Running pipeline with --reuse-latest - will try to reuse latest available outputs")
    else:
        logging.info("[NORMAL] Running pipeline in normal mode - will reuse up-to-date outputs")

    configure_logging()
    logging.info("Starting indicators pipeline (pure Python version)")

    # --- Generate run_id and set up per-run stats folder ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_stats_dir = os.path.join("pipeline_stats", run_id)
    os.makedirs(run_stats_dir, exist_ok=True)
    # --- Initialize pipeline stats ---
    stats = PipelineStats(run_id, output_dir="pipeline_stats")
    # --- Set up per-run logging ---
    log_path = os.path.join(run_stats_dir, "pipeline.log")
    # Remove all handlers associated with the root logger object (for idempotency)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # --- Save git commit hash ---
    try:
        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha
    except Exception:
        commit_hash = "unknown"
    with open(os.path.join(run_stats_dir, "git_commit.txt"), "w") as f:
        f.write(commit_hash)
    # --- Save pip freeze output for environment reproducibility ---
    env_txt = os.path.join(run_stats_dir, "environment.txt")
    with open(env_txt, "w") as f:
        subprocess.run(["pip", "freeze"], stdout=f)
    # --- Initialize run_metadata ---
    run_metadata = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(),
        "git_commit": commit_hash,
        "env_file": env_txt,
        "steps": {},
        "args": vars(args),
    }
    def save_metadata():
        with open(os.path.join(run_stats_dir, "run_metadata.json"), "w") as f:
            json.dump(run_metadata, f, indent=2, default=str)

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
    step_name = "cleaned_points"
    step_info = {"timestamp": datetime.now().isoformat()}
    cleaned_df = None  # Initialize to None
    cleaned_points_path = f"data/cleaned_points_in_beijing_{run_id}.parquet"
    cleaned_points_meta = cleaned_points_path + ".meta.json"
    filtered_points_path = "data/filtered_points_in_beijing.parquet"
    input_paths = {"filtered_points_hash": filtered_points_path}
    reused_from = None

    if args.clean:
        # Clean mode: always compute from scratch
        logging.info(f"[CLEAN] Computing {step_name} from scratch")
        cleaned_df = run_cleaned_points(filtered_points_path, cleaned_points_path, cleaned_points_meta, step_name, stats, args)
        step_info["status"] = "computed"
        step_info["output_path"] = cleaned_points_path
    else:
        # Try to reuse latest or existing
        if args.reuse_latest and not os.path.exists(cleaned_points_path):
            latest = find_latest_output("data/cleaned_points_in_beijing_*.parquet")
            if latest and is_up_to_date(latest, input_paths):
                logging.info(f"[REUSE] Using {latest} for {step_name} step")
                cleaned_points_path = latest
                cleaned_points_meta = cleaned_points_path + ".meta.json"
                reused_from = latest

        if is_up_to_date(cleaned_points_path, input_paths):
            logging.info(f"[SKIP] {cleaned_points_path} is up-to-date")
            cleaned_df = pl.read_parquet(cleaned_points_path)
            if cleaned_df.is_empty():
                warning = f"Reused {step_name} is empty. Recomputing."
                logging.warning(warning)
                step_info["status"] = "recomputed_due_to_empty"
                step_info["warning"] = warning
                cleaned_df = run_cleaned_points(filtered_points_path, cleaned_points_path, cleaned_points_meta, step_name, stats, args)
                step_info["status"] = "recomputed"
                step_info["output_path"] = cleaned_points_path
            else:
                step_info["status"] = "reused"
                step_info["reused_from"] = reused_from
                step_info["output_path"] = cleaned_points_path
        else:
            logging.info(f"[COMPUTE] Computing {step_name} from scratch")
            cleaned_df = run_cleaned_points(filtered_points_path, cleaned_points_path, cleaned_points_meta, step_name, stats, args)
            step_info["status"] = "computed"
            step_info["output_path"] = cleaned_points_path

    # Update metadata and record stats
    run_metadata["steps"][step_name] = step_info
    save_metadata()
    if cleaned_df is not None:
        stats.record_step_stats("cleaned_points", cleaned_df)

    # --- Step 2: Cleaned with period_id ---
    step_name = "cleaned_with_period_id"
    step_info = {"timestamp": datetime.now().isoformat()}
    cleaned_with_period_id = None  # Initialize to None
    cleaned_with_pid_path = f"data/cleaned_with_period_id_in_beijing_{run_id}.parquet"
    cleaned_with_pid_meta = cleaned_with_pid_path + ".meta.json"
    input_paths = {"cleaned_points_hash": cleaned_points_path}
    reused_from = None

    if args.clean:
        logging.info(f"[CLEAN] Computing {step_name} from scratch")
        cleaned_with_period_id = run_cleaned_with_period_id(cleaned_points_path, cleaned_with_pid_path, cleaned_with_pid_meta, step_name, stats, args)
        step_info["status"] = "computed"
        step_info["output_path"] = cleaned_with_pid_path
    else:
        if args.reuse_latest and not os.path.exists(cleaned_with_pid_path):
            latest = find_latest_output("data/cleaned_with_period_id_in_beijing_*.parquet")
            if latest and is_up_to_date(latest, input_paths):
                logging.info(f"[REUSE] Using {latest} for {step_name} step")
                cleaned_with_pid_path = latest
                cleaned_with_pid_meta = cleaned_with_pid_path + ".meta.json"
                reused_from = latest

        if is_up_to_date(cleaned_with_pid_path, input_paths):
            logging.info(f"[SKIP] {cleaned_with_pid_path} is up-to-date")
            cleaned_with_period_id = pl.read_parquet(cleaned_with_pid_path)
            step_info["status"] = "reused"
            step_info["reused_from"] = reused_from
            step_info["output_path"] = cleaned_with_pid_path
        else:
            logging.info(f"[COMPUTE] Computing {step_name} from scratch")
            cleaned_with_period_id = run_cleaned_with_period_id(cleaned_points_path, cleaned_with_pid_path, cleaned_with_pid_meta, step_name, stats, args)
            step_info["status"] = "computed"
            step_info["output_path"] = cleaned_with_pid_path

    run_metadata["steps"][step_name] = step_info
    save_metadata()

    # --- Step 3: Period summary with SLD ratio ---
    step_name = "periods_with_sld_ratio"
    step_info = {"timestamp": datetime.now().isoformat()}
    period_df = None  # Initialize to None
    periods_sld_path = f"data/periods_with_sld_ratio_{run_id}.parquet"
    periods_sld_meta = periods_sld_path + ".meta.json"
    input_paths = {"cleaned_with_pid_hash": cleaned_with_pid_path}
    reused_from = None

    if args.clean:
        logging.info(f"[CLEAN] Computing {step_name} from scratch")
        period_df = run_periods_with_sld_ratio(cleaned_with_period_id, periods_sld_path, periods_sld_meta, step_name, stats, args)
        step_info["status"] = "computed"
        step_info["output_path"] = periods_sld_path
    else:
        if args.reuse_latest and not os.path.exists(periods_sld_path):
            latest = find_latest_output("data/periods_with_sld_ratio_*.parquet")
            if latest and is_up_to_date(latest, input_paths):
                logging.info(f"[REUSE] Using {latest} for {step_name} step")
                periods_sld_path = latest
                reused_from = latest

        if is_up_to_date(periods_sld_path, input_paths):
            logging.info(f"[SKIP] {periods_sld_path} is up-to-date")
            period_df = pl.read_parquet(periods_sld_path)
            step_info["status"] = "reused"
            step_info["reused_from"] = reused_from
            step_info["output_path"] = periods_sld_path
        else:
            logging.info(f"[COMPUTE] Computing {step_name} from scratch")
            period_df = run_periods_with_sld_ratio(cleaned_with_period_id, periods_sld_path, periods_sld_meta, step_name, stats, args)
            step_info["status"] = "computed"
            step_info["output_path"] = periods_sld_path

    run_metadata["steps"][step_name] = step_info
    save_metadata()

    # --- Step 4: Network ratio ---
    network_ratio_df, network_ratio_step_info = run_network_ratio_step(periods_sld_path, run_id, stats, args)
    run_metadata["steps"]["network_ratio"] = network_ratio_step_info
    save_metadata()

    # --- Step 5: Network outlier flag ---
    network_ratio_path = network_ratio_step_info["output_path"]
    final_df, network_outlier_flag_step_info = run_network_outlier_flag_step(network_ratio_path, run_id, stats, args)
    run_metadata["steps"]["network_outlier_flag"] = network_outlier_flag_step_info
    save_metadata()

    # --- Write LAST_RUN_ID file ---
    with open(os.path.join("pipeline_stats", "LAST_RUN_ID"), "w") as f:
        f.write(run_id)

    logging.info("Pipeline completed: all parquet files written.")
    print(f"\nPipeline run complete. See {run_stats_dir} for logs, metadata, and outputs.")
    print(f"  - Log: {log_path}\n  - Metadata: {os.path.join(run_stats_dir, 'run_metadata.json')}\n  - Git commit: {os.path.join(run_stats_dir, 'git_commit.txt')}\n  - Environment: {env_txt}")

    # --- Save pipeline stats ---
    stats.record_meta("git_commit", commit_hash)
    stats.record_meta("env_file", env_txt)
    stats.save()

    # --- Optionally run analysis tool ---
    # Always generate Markdown summary report for this run
    analysis_cmd = [
        sys.executable, "pipeline_network_analysis.py",
        "--full-analysis",
        "--run-id", run_id
    ]
    logging.info(f"Generating Markdown summary report: {' '.join(analysis_cmd)}")
    subprocess.run(analysis_cmd)
    # If --run-analysis is set, also run the full CLI analysis tool with all options
    if args.run_analysis:
        extra_cmd = [
            sys.executable, "pipeline_network_analysis.py",
            "--graph", "beijing_drive.graphml",
            "--periods", f"data/periods_with_network_ratio_flagged_{run_id}.parquet",
            "--all"
        ]
        logging.info(f"Running extra analysis tool: {' '.join(extra_cmd)}")
        subprocess.run(extra_cmd)

if __name__ == "__main__":
    main()
