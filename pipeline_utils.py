import hashlib
import json
from pathlib import Path
import polars as pl
from itertools import combinations
from typing import List, Dict, Any, Optional
import logging
import os
import time
import psutil
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import (
    add_time_distance_calcs, add_implied_speed, add_abnormality_flags, select_final_columns, add_period_id, summarize_periods, compute_iqr_thresholds
)

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def write_meta(meta_path, meta_dict):
    with open(meta_path, 'w') as f:
        json.dump(meta_dict, f)

def read_meta(meta_path):
    if not Path(meta_path).exists():
        return None
    with open(meta_path, 'r') as f:
        return json.load(f)

def is_up_to_date(output_path, input_paths):
    meta_path = str(output_path) + '.meta.json'
    if not Path(output_path).exists() or not Path(meta_path).exists():
        return False
    meta = read_meta(meta_path)
    for key, path in input_paths.items():
        if meta.get(key) != file_hash(path):
            return False
    return True

class PipelineStats:
    def __init__(self, run_id: str, output_dir: str = "pipeline_stats"):
        self.run_id = run_id
        self.output_dir = output_dir
        self.stats = {
            "run_id": run_id,
            "steps": {},
            "indicator_overlaps": {},
            "filtering": {},
            "meta": {},
        }

    def record_filtering(self, step: str, before: int, after: int, criteria: str, extra: Optional[Dict[str, Any]] = None):
        pct = 100.0 * (before - after) / before if before > 0 else 0.0
        self.stats["filtering"][step] = {
            "before": before,
            "after": after,
            "filtered": before - after,
            "filtered_pct": pct,
            "criteria": criteria,
        }
        if extra:
            self.stats["filtering"][step].update(extra)

    def record_step_stats(self, step: str, df: pl.DataFrame, abnormal_col: Optional[str] = None):
        n_total = df.height
        step_stats = {"n_total": n_total}
        if abnormal_col and abnormal_col in df.columns:
            n_abnormal = df.filter(pl.col(abnormal_col)).height
            step_stats["n_abnormal"] = n_abnormal
            step_stats["abnormal_pct"] = 100.0 * n_abnormal / n_total if n_total > 0 else 0.0
        self.stats["steps"][step] = step_stats

    def record_indicator_flags(self, df: pl.DataFrame, indicator_cols: List[str]):
        # For each indicator, count flagged
        for col in indicator_cols:
            if col in df.columns:
                n_flagged = df.filter(pl.col(col)).height
                self.stats["indicator_overlaps"][col] = {
                    "n_flagged": n_flagged,
                    "flagged_pct": 100.0 * n_flagged / df.height if df.height > 0 else 0.0,
                }
        # For all combinations
        for r in range(2, len(indicator_cols) + 1):
            for combo in combinations(indicator_cols, r):
                mask = pl.lit(True)
                for col in combo:
                    mask = mask & pl.col(col)
                n_combo = df.filter(mask).height
                key = " & ".join(combo)
                self.stats["indicator_overlaps"][key] = {
                    "n_flagged": n_combo,
                    "flagged_pct": 100.0 * n_combo / df.height if df.height > 0 else 0.0,
                }

    def record_meta(self, key: str, value: Any):
        self.stats["meta"][key] = value

    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, self.run_id, "pipeline_stats.json")
        with open(out_path, "w") as f:
            json.dump(self.stats, f, indent=2, default=str)

    def get_stats(self):
        return self.stats 

# --- General pipeline utilities ---
def save_parquet(df: pl.DataFrame, path: str, label: str = None):
    df.write_parquet(path)
    msg = f"Saved {label or path} to {path} (shape: {df.shape})"
    print(msg)
    logging.info(msg)

def save_step_stats(step_name: str, stats: dict, run_id: str, output_dir: str = "pipeline_stats"):
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(run_dir, f"{step_name}_stats_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logging.info(f"Saved {step_name} statistics to {output_path}")

def profile_step(step_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024**2  # MB
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            t1 = time.perf_counter()
            mem_after = process.memory_info().rss / 1024**2  # MB
            elapsed = t1 - t0
            logging.info(f"[PROFILE] {step_name}: time={elapsed:.2f}s, mem_before={mem_before:.2f}MB, mem_after={mem_after:.2f}MB, delta={mem_after-mem_before:.2f}MB")
            return result
        return wrapper
    return decorator

# --- OSMnx/graph helpers ---
def ensure_osm_graph(osm_graph_path, periods_path, buffer=None, output_dir="pipeline_stats"):
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

# --- Step helpers ---
@profile_step("cleaned_points")
def run_cleaned_points(filtered_points_path, cleaned_points_path, cleaned_points_meta, step_name, stats, args):
    lazy_df = pl.scan_parquet(filtered_points_path)
    gap_threshold_sec, speed_threshold_kph = compute_iqr_thresholds(lazy_df)
    logging.info(f"Using gap_threshold_sec={gap_threshold_sec:.2f}, speed_threshold_kph={speed_threshold_kph:.2f} for abnormality flags")
    cleaned_df = lazy_df.pipe(
        add_time_distance_calcs
    ).pipe(
        add_implied_speed
    ).pipe(
        add_abnormality_flags, gap_threshold_sec, speed_threshold_kph
    ).pipe(
        select_final_columns
    ).collect(streaming=True)
    save_parquet(cleaned_df, cleaned_points_path, "cleaned points")
    write_meta(cleaned_points_meta, {"filtered_points_hash": file_hash(filtered_points_path)})
    stats.record_step_stats(step_name, cleaned_df)
    return cleaned_df

@profile_step("cleaned_with_period_id")
def run_cleaned_with_period_id(cleaned_points_path, cleaned_with_pid_path, cleaned_with_pid_meta, step_name, stats, args):
    lazy_df = pl.scan_parquet(cleaned_points_path)
    cleaned_points_df = lazy_df.collect(streaming=True)
    cleaned_with_pid = add_period_id(cleaned_points_df)
    cleaned_with_pid.write_parquet(cleaned_with_pid_path)
    write_meta(cleaned_with_pid_meta, {"cleaned_points_hash": file_hash(cleaned_points_path)})
    stats.record_step_stats(step_name, cleaned_with_pid)
    return cleaned_with_pid

@profile_step("periods_with_sld_ratio")
def run_periods_with_sld_ratio(cleaned_with_period_id, periods_sld_path, periods_sld_meta, step_name, stats, args):
    period_df = summarize_periods(cleaned_with_period_id)
    period_df.write_parquet(periods_sld_path)
    write_meta(periods_sld_meta, {"cleaned_with_pid_hash": file_hash(periods_sld_path)})
    stats.record_step_stats(step_name, period_df)
    return period_df

@profile_step("network_ratio")
def run_network_ratio_step(periods_sld_path, run_id, stats, args):
    network_ratio_path = f"data/periods_with_network_ratio_{run_id}.parquet"
    input_paths = {"periods_sld_hash": periods_sld_path}
    reused_from = None
    step_name = "network_ratio"
    step_info = {"timestamp": datetime.now().isoformat()}
    action, osm_graph_path = ensure_osm_graph("beijing_drive.graphml", periods_sld_path)
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

@profile_step("network_outlier_flag")
def run_network_outlier_flag_step(network_ratio_path, run_id, stats, args):
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

@profile_step("network_shortest_paths")
def compute_network_shortest_paths_batched(
    periods_path,
    osm_graph_path,
    output_path,
    batch_size=500,
    num_workers=8,
    checkpoint_dir="network_checkpoints",
    run_id=None,
    node_cache_batch_size=1000,
    node_cache_num_workers=4,
    output_dir="pipeline_stats",
    clean_mode=False
):
    logger = logging.getLogger("network_shortest_path")
    os.makedirs(checkpoint_dir, exist_ok=True)
    periods_df = pd.read_parquet(periods_path)
    G = ox.load_graphml(osm_graph_path)
    node_cache_path = os.path.join(checkpoint_dir, "point_to_node.parquet")
    node_cache_meta = node_cache_path + ".meta.json"
    periods_hash = file_hash(periods_path)
    graph_hash = file_hash(osm_graph_path)
    # --- Node cache handling ---
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
        all_points = list(point_to_node.keys())
    else:
        logger.info("[COMPUTE] Building node cache from scratch")
        all_points = list(set(
            list(zip(periods_df['start_latitude'], periods_df['start_longitude'])) +
            list(zip(periods_df['end_latitude'], periods_df['end_longitude']))
        ))
        logger.info(f"Processing {len(all_points)} unique points in {node_cache_batch_size} point batches")
        point_to_node = {}
        import tempfile
        import csv
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as tmpfile:
            writer = csv.writer(tmpfile)
            writer.writerow(['lat', 'lon', 'node'])
            def process_batch(batch):
                lats, lons = zip(*batch)
                try:
                    nodes = ox.nearest_nodes(G, lons, lats)
                except Exception:
                    nodes = [None] * len(batch)
                return list(zip(batch, nodes))
            batches = [all_points[i:i+node_cache_batch_size] for i in range(0, len(all_points), node_cache_batch_size)]
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from tqdm import tqdm
            with ThreadPoolExecutor(max_workers=node_cache_num_workers) as executor:
                futures = {executor.submit(process_batch, batch): batch for batch in batches}
                for future in tqdm(as_completed(futures), total=len(batches), desc="Node cache batches"):
                    results = future.result()
                    for (latlon, node) in results:
                        lat, lon = latlon
                        point_to_node[(lat, lon)] = node
                        writer.writerow([lat, lon, node])
            tmpfile_path = tmpfile.name
        if not clean_mode:
            node_df = pd.read_csv(tmpfile_path)
            node_df.to_parquet(node_cache_path)
            os.remove(tmpfile_path)
            with open(node_cache_meta, "w") as f:
                json.dump({"periods_hash": periods_hash, "graph_hash": graph_hash}, f)
            logger.info(f"[SAVE] Node cache written to {node_cache_path}")
    node_assignment_path = os.path.join(output_dir, run_id, f"node_assignment_{run_id}.parquet") if run_id else "node_assignment.parquet"
    if not cache_valid:
        node_df.to_parquet(node_assignment_path)
    else:
        node_df.to_parquet(node_assignment_path)
    logger.info(f"[SAVE] Node assignments saved to {node_assignment_path}")
    total_points = len(all_points)
    assigned_nodes = [point_to_node.get((lat, lon)) for (lat, lon) in all_points]
    successful_assignments = sum(n is not None for n in assigned_nodes)
    failed_assignments = total_points - successful_assignments
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
    periods_df['unique_key'] = periods_df['license_plate'].astype(str) + '_' + periods_df['period_id'].astype(str)
    checkpoint_file = os.path.join(checkpoint_dir, "network_paths_checkpoint.parquet")
    if os.path.exists(checkpoint_file):
        done_df = pd.read_parquet(checkpoint_file)
        done_keys = set(done_df['unique_key'])
        logger.info(f"Loaded checkpoint with {len(done_keys)} completed periods.")
    else:
        done_df = pd.DataFrame()
        done_keys = set()
    to_process = periods_df[~periods_df['unique_key'].isin(done_keys)].copy()
    logger.info(f"Processing {len(to_process)} new periods (skipping {len(done_keys)} already done).")
    # --- Path computation ---
    to_process['start_node'] = [point_to_node.get((row['start_latitude'], row['start_longitude'])) for _, row in to_process.iterrows()]
    to_process['end_node'] = [point_to_node.get((row['end_latitude'], row['end_longitude'])) for _, row in to_process.iterrows()]
    node_pairs = set(zip(to_process['start_node'], to_process['end_node']))
    node_pair_to_dist = {}
    node_pair_rows = []
    def compute_pair(pair):
        orig, dest = pair
        if orig is None or dest is None:
            return pair, float('nan')
        if orig == dest:
            return pair, float('nan')
        try:
            dist = nx.shortest_path_length(G, orig, dest, weight='length') / 1000
            return pair, dist
        except Exception:
            return pair, float('nan')
    node_pairs = list(node_pairs)
    batch_size_pairs = 1000
    import tempfile
    batch_files = []
    for i in tqdm(range(0, len(node_pairs), batch_size_pairs), desc="Node pair batches"):
        batch_pairs = node_pairs[i:i+batch_size_pairs]
        node_pair_rows = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(compute_pair, pair): pair for pair in batch_pairs}
            for future in tqdm(as_completed(futures), total=len(batch_pairs), desc=f"Batch {i//batch_size_pairs+1}", leave=False):
                pair, dist = future.result()
                if pair[0] is not None and pair[1] is not None:
                    node_pair_rows.append({"start_node": pair[0], "end_node": pair[1], "distance_km": dist})
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
        pd.DataFrame(node_pair_rows).to_parquet(tmp.name)
        batch_files.append(tmp.name)
        tmp.close()
    node_pairs_path = os.path.join(output_dir, run_id, f"node_pairs_{run_id}.parquet") if run_id else "node_pairs.parquet"
    all_node_pairs_df = pd.concat([pd.read_parquet(f) for f in batch_files], ignore_index=True)
    all_node_pairs_df.to_parquet(node_pairs_path)
    for f in batch_files:
        os.remove(f)
    node_pair_to_dist = {(row['start_node'], row['end_node']): row['distance_km'] for _, row in all_node_pairs_df.iterrows()}
    # --- Assign distances to periods in batches ---
    all_results_files = []
    all_distances = []
    all_ratios = []
    for i in tqdm(range(0, len(to_process), batch_size), desc="Batches"):
        batch = to_process.iloc[i:i+batch_size].copy()
        def get_dist(row):
            if row['start_node'] is None or row['end_node'] is None:
                return float('nan')
            return node_pair_to_dist.get((row['start_node'], row['end_node']), float('nan'))
        batch['network_shortest_distance'] = batch.apply(get_dist, axis=1)
        batch['route_deviation_ratio'] = batch.apply(
            lambda row: row['sum_distance'] / row['network_shortest_distance']
            if not pd.isna(row['network_shortest_distance']) and row['network_shortest_distance'] > 0
            else float('nan'),
            axis=1
        )
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
        batch.to_parquet(tmp.name)
        all_results_files.append(tmp.name)
        tmp.close()
        valid_distances = batch['network_shortest_distance'].dropna()
        valid_ratios = batch['route_deviation_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
        all_distances.extend(valid_distances)
        all_ratios.extend(valid_ratios)
        stats["ratio_stats"]["nan_count"] += batch['route_deviation_ratio'].isna().sum()
        if os.path.exists(checkpoint_file):
            prev = pd.read_parquet(checkpoint_file)
            pd.concat([prev, batch]).drop_duplicates('unique_key').to_parquet(checkpoint_file)
        else:
            batch.to_parquet(checkpoint_file)
        logger.info(f"Checkpointed batch {i//batch_size+1} ({len(batch)} periods).")
    final_df = pd.concat([pd.read_parquet(f) for f in all_results_files], ignore_index=True)
    for f in all_results_files:
        os.remove(f)
    # --- FILTER OUT FAILED/DEGENERATE PERIODS ---
    n_before_failed_filter = len(final_df)
    failed_mask = final_df['network_shortest_distance'].isna() | (final_df['network_shortest_distance'] < 1.0)
    n_failed_periods = failed_mask.sum()
    if n_failed_periods > 0:
        logger.info(f"Filtering out {n_failed_periods} periods with failed or degenerate network path computation (NaN or <1km).")
    final_df = final_df[~failed_mask].copy()
    n_final = len(final_df)
    stats["filtered_failed_periods"] = int(n_failed_periods)
    stats["final_periods"] = int(n_final)
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
    if run_id is not None:
        save_step_stats("network_shortest_paths", stats, run_id, output_dir=output_dir)
    else:
        save_step_stats("network_shortest_paths", stats, "default", output_dir=output_dir)
    node_pairs_path = os.path.join(output_dir, run_id, f"node_pairs_{run_id}.parquet") if run_id else "node_pairs.parquet"
    all_node_pairs_df.to_parquet(node_pairs_path)
    final_df.to_parquet(output_path)
    logger.info(f"Saved final results to {output_path}")
    return final_df

def clean_pipeline_outputs():
    import os
    import glob
    from pathlib import Path
    print("Cleaning pipeline outputs and checkpoints...")
    checkpoint_dir = Path("network_checkpoints")
    if checkpoint_dir.exists():
        for f in checkpoint_dir.glob("*"):
            try:
                f.unlink()
                print(f"Deleted checkpoint: {f}")
            except Exception as e:
                print(f"Could not delete {f}: {e}")
    for meta_file in glob.glob("data/*.meta.json"):
        try:
            os.remove(meta_file)
            print(f"Deleted meta: {meta_file}")
        except Exception as e:
            print(f"Could not delete {meta_file}: {e}")
    keep_files = {"filtered_points_in_beijing.parquet"}
    for pq_file in glob.glob("data/*.parquet"):
        if os.path.basename(pq_file) not in keep_files:
            try:
                os.remove(pq_file)
                print(f"Deleted parquet: {pq_file}")
            except Exception as e:
                print(f"Could not delete {pq_file}: {e}")
    print("Clean complete.")

def find_latest_output(pattern):
    import glob
    import re
    files = glob.glob(pattern)
    if not files:
        return None
    def extract_runid(fname):
        m = re.search(r'_(\d{8}_\d{6})', fname)
        return m.group(1) if m else ''
    files = [(f, extract_runid(f)) for f in files]
    files = sorted(files, key=lambda x: x[1], reverse=True)
    return files[0][0] if files else None 