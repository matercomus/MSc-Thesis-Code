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
import jsonlines

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

    def record_step_stats(self, step: str, df, abnormal_col: Optional[str] = None):
        # Support both Polars and pandas DataFrames
        if hasattr(df, 'height'):
            n_total = df.height
            columns = df.columns
            get_abnormal = lambda col: df.filter(pl.col(col)).height
        else:
            n_total = len(df)
            columns = df.columns
            get_abnormal = lambda col: df[df[col]].shape[0]
        step_stats = {"n_total": n_total}
        if abnormal_col and abnormal_col in columns:
            n_abnormal = get_abnormal(abnormal_col)
            step_stats["n_abnormal"] = n_abnormal
            step_stats["abnormal_pct"] = 100.0 * n_abnormal / n_total if n_total > 0 else 0.0
        self.stats["steps"][step] = step_stats

    def record_indicator_flags(self, df, indicator_cols: List[str]):
        # Support both Polars and pandas DataFrames
        if hasattr(df, 'height'):
            from functools import reduce
            import operator
            n_total = df.height
            columns = df.columns
            get_flagged = lambda col: df.filter(pl.col(col)).height
            get_combo = lambda combo: df.filter(
                reduce(operator.and_, (pl.col(col) for col in combo))
            ).height
        else:
            n_total = len(df)
            columns = df.columns
            get_flagged = lambda col: df[df[col]].shape[0]
            get_combo = lambda combo: df[df[list(combo)].all(axis=1)].shape[0]
        # For each indicator, count flagged
        indicator_flags = {}
        for col in indicator_cols:
            if col in columns:
                n_flagged = get_flagged(col)
                self.stats["indicator_overlaps"][col] = {
                    "n_flagged": n_flagged,
                    "flagged_pct": 100.0 * n_flagged / n_total if n_total > 0 else 0.0,
                }
                indicator_flags[col] = n_flagged
        self.stats["indicator_flags"] = indicator_flags
        # For all combinations
        for r in range(2, len(indicator_cols) + 1):
            for combo in combinations(indicator_cols, r):
                n_combo = get_combo(combo)
                key = " & ".join(combo)
                self.stats["indicator_overlaps"][key] = {
                    "n_flagged": n_combo,
                    "flagged_pct": 100.0 * n_combo / n_total if n_total > 0 else 0.0,
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
def run_periods_with_sld_ratio(cleaned_with_period_id, cleaned_with_pid_path, periods_sld_path, periods_sld_meta, step_name, stats, args):
    period_df = summarize_periods(cleaned_with_period_id)
    period_df.write_parquet(periods_sld_path)
    write_meta(periods_sld_meta, {"cleaned_with_pid_hash": file_hash(cleaned_with_pid_path)})
    stats.record_step_stats(step_name, period_df)
    return period_df

@profile_step("network_ratio")
def run_network_ratio_step(periods_sld_path, run_id, stats, args, resume=False):
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
            batch_size=2000,
            num_workers=8,
            checkpoint_dir="network_checkpoints",
            run_id=run_id,
            resume=resume
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
                batch_size=2000,
                num_workers=8,
                checkpoint_dir="network_checkpoints",
                run_id=run_id,
                resume=resume
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
    batch_size=2000,
    num_workers=8,
    checkpoint_dir="network_checkpoints",
    run_id=None,
    node_cache_batch_size=500,
    output_dir="pipeline_stats",
    resume=False
):
    import polars as pl
    import pandas as pd
    import numpy as np
    import tempfile
    logger = logging.getLogger("network_shortest_path")
    os.makedirs(checkpoint_dir, exist_ok=True)
    G = ox.load_graphml(osm_graph_path)
    node_cache_path = os.path.join(checkpoint_dir, "point_to_node.parquet")
    node_cache_meta = node_cache_path + ".meta.json"
    periods_hash = file_hash(periods_path)
    graph_hash = file_hash(osm_graph_path)
    # --- Batch profiling setup ---
    batch_profile_path = None
    batch_checkpoint_path = None
    last_completed_batch = -1
    if run_id:
        run_stats_dir = os.path.join(output_dir, run_id)
        os.makedirs(run_stats_dir, exist_ok=True)
        batch_profile_path = os.path.join(run_stats_dir, "network_batch_profile.jsonl")
        batch_checkpoint_path = os.path.join(run_stats_dir, "last_completed_batch.txt")
        if resume and os.path.exists(batch_checkpoint_path):
            with open(batch_checkpoint_path) as f:
                try:
                    last_completed_batch = int(f.read().strip())
                    logger.info(f"[RESUME] Resuming from batch {last_completed_batch + 1}")
                except Exception:
                    last_completed_batch = -1
    else:
        batch_profile_path = "network_batch_profile.jsonl"
        batch_checkpoint_path = "last_completed_batch.txt"
        if resume and os.path.exists(batch_checkpoint_path):
            with open(batch_checkpoint_path) as f:
                try:
                    last_completed_batch = int(f.read().strip())
                    logger.info(f"[RESUME] Resuming from batch {last_completed_batch + 1}")
                except Exception:
                    last_completed_batch = -1
    # --- Node cache handling ---
    # We'll build the node cache in streaming batches as well
    # First, collect all unique (lat, lon) pairs from the periods file in streaming batches
    logger.info("[NODE CACHE] Building/using node cache in streaming batches...")
    unique_points = set()
    periods_schema = pl.read_parquet(periods_path, n_rows=1).schema
    # Use Polars streaming to get all unique points
    scan = pl.scan_parquet(periods_path)
    for col in ["start_latitude", "start_longitude", "end_latitude", "end_longitude"]:
        if col not in periods_schema:
            raise ValueError(f"Column {col} not found in periods file!")
    # Collect unique points in batches
    batch_idx = 0
    for batch in scan.select([
        pl.col("start_latitude"), pl.col("start_longitude"),
        pl.col("end_latitude"), pl.col("end_longitude")
    ]).collect(streaming=True).iter_rows():
        s_lat, s_lon, e_lat, e_lon = batch
        unique_points.add((s_lat, s_lon))
        unique_points.add((e_lat, e_lon))
    unique_points = list(unique_points)
    logger.info(f"[NODE CACHE] Found {len(unique_points)} unique points for node assignment.")
    # Assign nodes in batches (parallelized)
    point_to_node = {}
    def assign_nodes_batch(batch_points):
        lats, lons = zip(*batch_points)
        try:
            nodes = ox.nearest_nodes(G, lons, lats)
        except Exception:
            nodes = [None] * len(batch_points)
        return list(zip(batch_points, nodes))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(0, len(unique_points), node_cache_batch_size):
            batch_points = unique_points[i:i+node_cache_batch_size]
            futures.append(executor.submit(assign_nodes_batch, batch_points))
        for future in tqdm(futures, desc="Node assignment batches (parallel)"):
            for (latlon, node) in future.result():
                point_to_node[latlon] = node
    logger.info(f"[NODE CACHE] Node assignment complete.")
    # --- Streaming batch processing of periods ---
    scan = pl.scan_parquet(periods_path)
    n_rows = scan.select([pl.count()]).collect().item()
    n_batches = (n_rows + batch_size - 1) // batch_size
    logger.info(f"[BATCH] Will process {n_rows} periods in {n_batches} batches of size {batch_size}.")
    batch_files = []
    for batch_idx in tqdm(range(n_batches), desc="Period batches"):
        if resume and batch_idx <= last_completed_batch:
            logger.info(f"[RESUME] Skipping batch {batch_idx} (already completed)")
            continue
        batch_start_time = time.perf_counter()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024**2  # MB
        logger.info(f"[BATCH] Starting batch {batch_idx+1}/{n_batches}")
        batch = scan.slice(batch_idx * batch_size, batch_size).collect(streaming=True).to_pandas()
        # Assign nodes
        batch['start_node'] = [point_to_node.get((row['start_latitude'], row['start_longitude'])) for _, row in batch.iterrows()]
        batch['end_node'] = [point_to_node.get((row['end_latitude'], row['end_longitude'])) for _, row in batch.iterrows()]
        # Compute shortest paths for all pairs in this batch (parallelized)
        def compute_pair(row):
            orig, dest = row['start_node'], row['end_node']
            if orig is None or dest is None:
                return float('nan')
            if orig == dest:
                return float('nan')
            try:
                dist = nx.shortest_path_length(G, orig, dest, weight='length') / 1000
                return dist
            except Exception:
                return float('nan')
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(compute_pair, [row for _, row in batch.iterrows()]))
        batch['network_shortest_distance'] = results
        batch['route_deviation_ratio'] = batch.apply(
            lambda row: row['sum_distance'] / row['network_shortest_distance']
            if not pd.isna(row['network_shortest_distance']) and row['network_shortest_distance'] > 0
            else float('nan'),
            axis=1
        )
        # Write batch to temp parquet
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
        batch.to_parquet(tmp.name)
        batch_files.append(tmp.name)
        tmp.close()
        mem_after = process.memory_info().rss / 1024**2  # MB
        batch_end_time = time.perf_counter()
        elapsed = batch_end_time - batch_start_time
        logger.info(f"[BATCH] Finished batch {batch_idx+1}/{n_batches} | Time: {elapsed:.2f}s | Mem: {mem_before:.2f}MB -> {mem_after:.2f}MB | Rows: {len(batch)}")
        # Profiling
        batch_profile = {
            "batch_index": batch_idx,
            "start_row": batch_idx * batch_size,
            "end_row": batch_idx * batch_size + len(batch),
            "n_periods": len(batch),
            "time_sec": elapsed,
            "mem_before_mb": mem_before,
            "mem_after_mb": mem_after,
            "delta_mb": mem_after - mem_before,
            "timestamp": datetime.now().isoformat(),
        }
        with open(batch_profile_path, "a") as f:
            f.write(json.dumps(batch_profile) + "\n")
        with open(batch_checkpoint_path, "w") as f:
            f.write(str(batch_idx))
    # Merge all batch files into final output (streaming)
    logger.info(f"[MERGE] Merging {len(batch_files)} batch files into {output_path}")
    final_df = pl.concat([pl.read_parquet(f) for f in batch_files], rechunk=True)
    final_df.write_parquet(output_path)
    # Write .meta.json for reuse logic
    meta_path = str(output_path) + '.meta.json'
    write_meta(meta_path, {
        "periods_hash": file_hash(periods_path),
        "osm_graph_hash": file_hash(osm_graph_path)
    })
    for f in batch_files:
        os.remove(f)
    logger.info(f"[DONE] All batches processed and merged. Output written to {output_path}")
    return final_df.to_pandas()

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

def find_latest_output(pattern, input_paths=None):
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
    if input_paths is None:
        return files[0][0] if files else None
    # Search for the most recent up-to-date file
    for f, _ in files:
        if is_up_to_date(f, input_paths):
            return f
    return None

def compute_network_outlier_flag(input_path, output_path, run_id):
    import pandas as pd
    import numpy as np
    import os
    import json

    df = pd.read_parquet(input_path)
    # Only consider valid ratios
    valid = df["route_deviation_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    df["is_network_outlier"] = df["route_deviation_ratio"] > threshold
    df.to_parquet(output_path)

    # Write .meta.json for reuse logic
    meta_path = str(output_path) + '.meta.json'
    write_meta(meta_path, {
        "network_ratio_hash": file_hash(input_path)
    })

    # Optionally save stats for reproducibility
    stats = {
        "input_shape": list(df.shape),
        "valid_ratios": int(valid.shape[0]),
        "outliers": int(df["is_network_outlier"].sum()),
        "quantiles": {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "threshold": float(threshold),
        }
    }
    stats_dir = os.path.join("pipeline_stats", run_id)
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"network_outlier_flag_stats_{run_id}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    return df 