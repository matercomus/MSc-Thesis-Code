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
    detect_outliers_parallel_by_group,
    compute_iqr_thresholds,
    compute_generic_iqr_threshold,
    attach_period_id,
    set_polars_threads,
)
import osmnx as ox
import networkx as nx
from pipeline_utils import (
    file_hash,
    write_meta,
    is_up_to_date,
    PipelineStats,
    save_parquet,
    save_step_stats,
    ensure_osm_graph,
    profile_step,
    run_cleaned_points,
    run_cleaned_with_period_id,
    run_periods_with_sld_ratio,
    run_network_ratio_step,
    run_network_outlier_flag_step,
    compute_network_shortest_paths_batched,
    clean_pipeline_outputs,
    find_latest_output,
)
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
import time
import psutil


# Set OSMnx settings for debugging and reproducibility
ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 180
ox.settings.overpass_rate_limit = True
ox.settings.default_retries = 3
ox.settings.default_response_json = True
# ox.settings.default_access = 'all'  # Removed: causes Overpass query errors

def main():
    set_polars_threads(num_threads=8)
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Remove all checkpoints, meta, and intermediate outputs before running pipeline.")
    parser.add_argument("--clean-step", type=str, help="Remove outputs/meta/checkpoints for specific steps (comma-separated). Valid steps: cleaned_points, cleaned_with_period_id, periods_with_sld_ratio, network, osm_graph.")
    parser.add_argument("--run-analysis", action="store_true", help="Automatically run the analysis tool after pipeline completes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--reuse-latest", action="store_true", help="Try to reuse latest available outputs for each step.")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed batch if possible (network_ratio step only)")
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
    # --- Write LAST_RUN_ID file early for crash-resistance ---
    with open(os.path.join("pipeline_stats", "LAST_RUN_ID"), "w") as f:
        f.write(run_id)
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
    # Update LAST_RUN_ID after step (extra robustness)
    with open(os.path.join("pipeline_stats", "LAST_RUN_ID"), "w") as f:
        f.write(run_id)

    # --- Outlier Detection (Isolation Forest, parallel by group) ---
    if cleaned_df is not None:
        pd_df = cleaned_df.to_pandas()
        pd_df["is_outlier"] = detect_outliers_parallel_by_group(pd_df, group_col="license_plate").values
        cleaned_df = pl.from_pandas(pd_df)

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
    # Update LAST_RUN_ID after step (extra robustness)
    with open(os.path.join("pipeline_stats", "LAST_RUN_ID"), "w") as f:
        f.write(run_id)

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
        period_df = run_periods_with_sld_ratio(cleaned_with_period_id, cleaned_with_pid_path, periods_sld_path, periods_sld_meta, step_name, stats, args)
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
            period_df = run_periods_with_sld_ratio(cleaned_with_period_id, cleaned_with_pid_path, periods_sld_path, periods_sld_meta, step_name, stats, args)
            step_info["status"] = "computed"
            step_info["output_path"] = periods_sld_path

    run_metadata["steps"][step_name] = step_info
    save_metadata()
    # Update LAST_RUN_ID after step (extra robustness)
    with open(os.path.join("pipeline_stats", "LAST_RUN_ID"), "w") as f:
        f.write(run_id)

    # --- Step 4: Network ratio ---
    resume_batches = args.resume or args.reuse_latest
    network_ratio_df, network_ratio_step_info = run_network_ratio_step(periods_sld_path, run_id, stats, args, resume=resume_batches)
    run_metadata["steps"]["network_ratio"] = network_ratio_step_info
    save_metadata()
    # Update LAST_RUN_ID after step (extra robustness)
    with open(os.path.join("pipeline_stats", "LAST_RUN_ID"), "w") as f:
        f.write(run_id)

    # --- Step 5: Network outlier flag ---
    network_ratio_path = network_ratio_step_info["output_path"]
    final_df, network_outlier_flag_step_info = run_network_outlier_flag_step(network_ratio_path, run_id, stats, args)
    run_metadata["steps"]["network_outlier_flag"] = network_outlier_flag_step_info
    save_metadata()
    # Update LAST_RUN_ID after step (extra robustness)
    with open(os.path.join("pipeline_stats", "LAST_RUN_ID"), "w") as f:
        f.write(run_id)

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
