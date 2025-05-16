import argparse
import os
import osmnx as ox
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import logging
from pathlib import Path
import seaborn as sns
from tabulate import tabulate


def get_analysis_dir():
    last_run_file = os.path.join("pipeline_stats", "LAST_RUN_ID")
    if not os.path.exists(last_run_file):
        raise RuntimeError("LAST_RUN_ID not found. Run the pipeline first.")
    with open(last_run_file) as f:
        run_id = f.read().strip()
    analysis_dir = os.path.join("pipeline_stats", run_id, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    return analysis_dir


def save_json_stats(filename, stats):
    analysis_dir = get_analysis_dir()
    out_path = os.path.join(analysis_dir, filename)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Saved analysis stats to {out_path}")


def print_graph_info(G, return_stats=False):
    info = {
        "nodes": len(G.nodes),
        "edges": len(G.edges),
        "lat_min": None,
        "lat_max": None,
        "lon_min": None,
        "lon_max": None,
    }
    lats = [data['y'] for _, data in G.nodes(data=True)]
    lons = [data['x'] for _, data in G.nodes(data=True)]
    info["lat_min"] = float(min(lats))
    info["lat_max"] = float(max(lats))
    info["lon_min"] = float(min(lons))
    info["lon_max"] = float(max(lons))
    print(f"Graph: {info['nodes']} nodes, {info['edges']} edges")
    print(f"Graph bounding box:")
    print(f"  lat: {info['lat_min']:.6f} to {info['lat_max']:.6f}")
    print(f"  lon: {info['lon_min']:.6f} to {info['lon_max']:.6f}")
    if return_stats:
        return info


def print_data_info(periods, return_stats=False):
    min_lat = min(periods['start_latitude'].min(), periods['end_latitude'].min())
    max_lat = max(periods['start_latitude'].max(), periods['end_latitude'].max())
    min_lon = min(periods['start_longitude'].min(), periods['end_longitude'].min())
    max_lon = max(periods['end_longitude'].max(), periods['end_longitude'].max())
    info = {
        "lat_min": float(min_lat),
        "lat_max": float(max_lat),
        "lon_min": float(min_lon),
        "lon_max": float(max_lon),
        "n_periods": int(len(periods)),
    }
    print(f"Data bounding box:")
    print(f"  lat: {min_lat:.6f} to {max_lat:.6f}")
    print(f"  lon: {min_lon:.6f} to {max_lon:.6f}")
    print(f"  #periods: {len(periods)}")
    if return_stats:
        return info


def sample_nodes(G, n=10):
    print(f"Sample of {n} nodes:")
    for i, (node, data) in enumerate(G.nodes(data=True)):
        print(f"  Node {node}: x={data['x']:.6f}, y={data['y']:.6f}")
        if i + 1 >= n:
            break


def plot_graph(G, ax=None, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8)) if ax is None else (None, ax)
    ox.plot_graph(G, ax=ax, show=False, close=False)
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved graph plot to {save_path}")


def plot_data_on_graph(G, periods, ax=None, sample=1000, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8)) if ax is None else (None, ax)
    ox.plot_graph(G, ax=ax, show=False, close=False, node_color='gray', edge_color='lightblue')
    # Plot start and end points
    starts = periods[['start_longitude', 'start_latitude']].dropna().sample(min(sample, len(periods)))
    ends = periods[['end_longitude', 'end_latitude']].dropna().sample(min(sample, len(periods)))
    ax.scatter(starts['start_longitude'], starts['start_latitude'], c='red', s=8, label='Start', alpha=0.6)
    ax.scatter(ends['end_longitude'], ends['end_latitude'], c='blue', s=8, label='End', alpha=0.6)
    ax.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved data-on-graph plot to {save_path}")


def check_node_assignment(G, periods, n=10):
    print(f"Checking node assignment for {n} random points:")
    sample = periods.sample(n=min(n, len(periods)))
    for idx, row in sample.iterrows():
        lat, lon = row['start_latitude'], row['start_longitude']
        try:
            node = ox.nearest_nodes(G, lon, lat)
            node_data = G.nodes[node]
            dist = ox.distance.great_circle(lat, lon, node_data['y'], node_data['x'])
            print(f"  ({lat:.6f}, {lon:.6f}) -> node {node} at ({node_data['y']:.6f}, {node_data['x']:.6f}), dist={dist:.2f}m")
        except Exception as e:
            print(f"  ({lat:.6f}, {lon:.6f}) -> ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline OSMnx Network Analysis Tool")
    parser.add_argument('--graph', type=str, required=True, help='Path to OSMnx GraphML file')
    parser.add_argument('--periods', type=str, required=True, help='Path to periods Parquet file')
    parser.add_argument('--info', action='store_true', help='Print graph and data info')
    parser.add_argument('--plot-graph', action='store_true', help='Plot the OSMnx network')
    parser.add_argument('--plot-data', action='store_true', help='Plot data points on the network')
    parser.add_argument('--sample-nodes', action='store_true', help='Print a sample of node coordinates')
    parser.add_argument('--check-nodes', action='store_true', help='Show node assignment for a sample of data points')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--sample', type=int, default=1000, help='Sample size for plotting/checks')
    args = parser.parse_args()

    G = ox.load_graphml(args.graph)

    # --- Robust fallback for reused periods file ---
    periods_path = args.periods
    if not os.path.exists(periods_path):
        # Try to find the actual file from run_metadata.json
        # Assume periods_path is like data/periods_with_network_ratio_flagged_{run_id}.parquet
        # Extract run_id
        import re
        m = re.search(r'_(\d{8}_\d{6})', periods_path)
        run_id = m.group(1) if m else None
        if run_id:
            meta_path = Path(f'pipeline_stats/{run_id}/run_metadata.json')
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                step = meta.get('steps', {}).get('network_outlier_flag', {})
                actual_path = step.get('output_path')
                if actual_path and os.path.exists(actual_path):
                    print(f"[INFO] Input file {periods_path} not found, using reused output {actual_path} from run_metadata.json.")
                    periods_path = actual_path
                else:
                    raise FileNotFoundError(f"Could not find fallback output_path in {meta_path}")
            else:
                raise FileNotFoundError(f"run_metadata.json not found for run_id {run_id}")
        else:
            raise FileNotFoundError(f"Input file {periods_path} not found and could not extract run_id.")
    periods = pd.read_parquet(periods_path)

    analysis_dir = get_analysis_dir()

    if args.all or args.info:
        print("--- Graph Info ---")
        ginfo = print_graph_info(G, return_stats=True)
        print("--- Data Info ---")
        dinfo = print_data_info(periods, return_stats=True)
        # Save as JSON
        save_json_stats("info_stats.json", {"graph": ginfo, "data": dinfo})
    if args.all or args.sample_nodes:
        print("--- Node Sample ---")
        sample_nodes(G, n=10)
    if args.all or args.plot_graph:
        print("--- Plotting Graph ---")
        plot_graph(G, save_path=os.path.join(analysis_dir, "graph.png"))
    if args.all or args.plot_data:
        print("--- Plotting Data Points on Graph ---")
        plot_data_on_graph(G, periods, sample=args.sample, save_path=os.path.join(analysis_dir, "data_on_graph.png"))
    if args.all or args.check_nodes:
        print("--- Node Assignment Check ---")
        check_node_assignment(G, periods, n=10)

# --- BEGIN: Comprehensive Markdown Analysis and Comparison ---

def load_run_stats(run_id):
    stats_dir = os.path.join("pipeline_stats", run_id)
    stats_path = os.path.join(stats_dir, "pipeline_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    with open(stats_path) as f:
        stats = json.load(f)
    return stats, stats_dir

def get_last_run_id():
    with open(os.path.join("pipeline_stats", "LAST_RUN_ID")) as f:
        return f.read().strip()

def get_previous_run_id(current_run_id):
    runs = sorted([d for d in os.listdir("pipeline_stats") if d.isdigit() or ("_" in d and d.replace("_","").isdigit())])
    if current_run_id in runs:
        idx = runs.index(current_run_id)
        if idx > 0:
            return runs[idx-1]
    return None

def plot_histogram(data, col, title, xlabel, save_path):
    plt.figure(figsize=(7,4))
    sns.histplot(data[col].dropna(), bins=20, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_bar(data, col, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(7,4))
    sns.barplot(x=data.index, y=data[col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def reused_from_link(reused_from):
    """Return a Markdown link to the reused run's analysis report if it exists, else just the run id."""
    if not reused_from:
        return ""
    rel_path = f"../../{reused_from}/analysis/analysis_report.md"
    abs_path = os.path.join("pipeline_stats", reused_from, "analysis", "analysis_report.md")
    if os.path.exists(abs_path):
        return f"[{reused_from}]({rel_path})"
    else:
        return reused_from

def make_markdown_report(run_id, stats, analysis_dir, plots, compare_stats=None, compare_run_id=None):
    md = f"# Pipeline Analysis Report\n\n"
    md += f"**Run ID:** `{run_id}`\n\n"
    if compare_stats:
        md += f"**Compared to Run ID:** `{compare_run_id}`\n\n"
    md += "## Summary Table\n"
    # Try to load run_metadata.json for step stats
    meta_path = os.path.join("pipeline_stats", run_id, "run_metadata.json")
    meta_steps = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
            meta_steps = meta.get("steps", {})
    summary_rows = []
    first_before = None
    if meta_steps:
        for step, step_stats in meta_steps.items():
            n_before = step_stats.get("n_before")
            n_after = step_stats.get("n_after")
            filtered = step_stats.get("filtered")
            pct_filtered = step_stats.get("pct_filtered")
            reused = step_stats.get("reused")
            reused_from = step_stats.get("reused_from")
            if first_before is None and n_before is not None:
                first_before = n_before
            cumulative_ret = (n_after / first_before * 100) if (first_before and n_after is not None) else None
            reused_from_display = reused_from_link(reused_from) if reused else ""
            summary_rows.append([
                step,
                n_before,
                n_after,
                filtered,
                f"{pct_filtered:.2f}%" if pct_filtered is not None else None,
                f"{cumulative_ret:.2f}%" if cumulative_ret is not None else None,
                "Yes" if reused else "No",
                reused_from_display
            ])
        headers = ["Step", "Before", "After", "Filtered", "% Filtered", "Cumulative % Retained", "Reused", "Reused From"]
    else:
        filtering = stats.get("filtering", {})
        for step, filt in filtering.items():
            n_before = filt.get("before")
            n_after = filt.get("after")
            filtered = filt.get("filtered")
            pct_filtered = filt.get("filtered_pct")
            if first_before is None and n_before is not None:
                first_before = n_before
            cumulative_ret = (n_after / first_before * 100) if (first_before and n_after is not None) else None
            summary_rows.append([
                step,
                n_before,
                n_after,
                filtered,
                f"{pct_filtered:.2f}%" if pct_filtered is not None else None,
                f"{cumulative_ret:.2f}%" if cumulative_ret is not None else None
            ])
        headers = ["Step", "Before", "After", "Filtered", "% Filtered", "Cumulative % Retained"]
    if summary_rows:
        md += tabulate(
            summary_rows,
            headers=headers,
            tablefmt="github"
        ) + "\n\n"
    # Indicator counts
    if "indicator_flags" in stats:
        md += "## Indicator Flag Counts\n"
        flag_rows = []
        for flag, count in stats["indicator_flags"].items():
            flag_rows.append([flag, count])
        md += tabulate(flag_rows, headers=["Indicator", "Count"], tablefmt="github") + "\n\n"
    # Overlaps
    if "indicator_overlaps" in stats:
        md += "## Indicator Overlaps\n"
        overlap_rows = []
        for combo, count in stats["indicator_overlaps"].items():
            overlap_rows.append([combo, count])
        md += tabulate(overlap_rows, headers=["Overlap", "Count"], tablefmt="github") + "\n\n"
    # Node/path stats
    if "node_assignment" in stats:
        md += "## Node Assignment Stats\n"
        node_rows = [[k, v] for k, v in stats["node_assignment"].items()]
        md += tabulate(node_rows, headers=["Metric", "Value"], tablefmt="github") + "\n\n"
    if "path_computation" in stats:
        md += "## Path Computation Stats\n"
        path_rows = [[k, v] for k, v in stats["path_computation"].items()]
        md += tabulate(path_rows, headers=["Metric", "Value"], tablefmt="github") + "\n\n"
    # Plots
    for desc, fname in plots:
        md += f"### {desc}\n\n![]({fname})\n\n"
    # Comparison
    if compare_stats:
        md += "# Comparison to Previous Run\n\n"
        # Compare indicator flags
        if "indicator_flags" in stats and "indicator_flags" in compare_stats:
            md += "## Indicator Flag Count Comparison\n"
            comp_rows = []
            for flag in set(stats["indicator_flags"]).union(compare_stats["indicator_flags"]):
                v1 = stats["indicator_flags"].get(flag, 0)
                v2 = compare_stats["indicator_flags"].get(flag, 0)
                comp_rows.append([flag, v1, v2, v1-v2])
            md += tabulate(comp_rows, headers=["Indicator", f"{run_id}", f"{compare_run_id}", "Diff"], tablefmt="github") + "\n\n"
        # Compare overlaps
        if "indicator_overlaps" in stats and "indicator_overlaps" in compare_stats:
            md += "## Indicator Overlap Comparison\n"
            comp_rows = []
            for combo in set(stats["indicator_overlaps"]).union(compare_stats["indicator_overlaps"]):
                v1 = stats["indicator_overlaps"].get(combo, 0)
                v2 = compare_stats["indicator_overlaps"].get(combo, 0)
                if isinstance(v1, dict) or isinstance(v2, dict):
                    comp_rows.append([combo, str(v1), str(v2), "N/A"])
                else:
                    comp_rows.append([combo, v1, v2, v1-v2])
            md += tabulate(comp_rows, headers=["Overlap", f"{run_id}", f"{compare_run_id}", "Diff"], tablefmt="github") + "\n\n"
    # Save
    md_path = os.path.join(analysis_dir, "analysis_report.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved Markdown report to {md_path}")

# --- CLI Entrypoint for Full Analysis ---
def run_full_analysis(run_id=None, compare_to=None):
    if run_id is None:
        run_id = get_last_run_id()
    stats, stats_dir = load_run_stats(run_id)
    analysis_dir = os.path.join(stats_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    # Load main periods file for histograms
    periods_path = os.path.join("data", f"periods_with_network_ratio_flagged_{run_id}.parquet")
    if not os.path.exists(periods_path):
        # Try to find in run_metadata
        meta_path = os.path.join(stats_dir, "run_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            step = meta.get("steps", {}).get("network_outlier_flag", {})
            periods_path = step.get("output_path", periods_path)
    periods = pd.read_parquet(periods_path)
    plots = []
    # Histograms
    for col, desc, xlabel in [
        ("route_deviation_ratio", "Route Deviation Ratio Histogram", "Route Deviation Ratio"),
        ("network_shortest_distance", "Network Shortest Distance Histogram", "Network Shortest Distance (km)")
    ]:
        fname = f"{col}_hist.png"
        save_path = os.path.join(analysis_dir, fname)
        plot_histogram(periods, col, desc, xlabel, save_path)
        plots.append((desc, fname))
    # Bar plot for indicator flags
    if "indicator_flags" in stats:
        flag_df = pd.DataFrame(list(stats["indicator_flags"].items()), columns=["flag", "count"]).set_index("flag")
        fname = "indicator_flags_bar.png"
        save_path = os.path.join(analysis_dir, fname)
        plot_bar(flag_df, "count", "Indicator Flag Counts", "Flag", "Count", save_path)
        plots.append(("Indicator Flag Counts", fname))
    # Comparison
    compare_stats = None
    compare_run_id = None
    if compare_to:
        compare_stats, _ = load_run_stats(compare_to)
        compare_run_id = compare_to
    elif compare_to is not False:
        prev_run = get_previous_run_id(run_id)
        if prev_run:
            compare_stats, _ = load_run_stats(prev_run)
            compare_run_id = prev_run
    make_markdown_report(run_id, stats, analysis_dir, plots, compare_stats, compare_run_id)

# Add CLI for full analysis
if __name__ == "__main__":
    import sys
    if "--full-analysis" in sys.argv:
        sys.argv.remove("--full-analysis")  # Remove before parsing
        parser = argparse.ArgumentParser()
        parser.add_argument("--run-id", type=str, default=None, help="Run ID to analyze (default: last run)")
        parser.add_argument("--compare-to", type=str, default=None, help="Run ID to compare to (default: previous run)")
        args = parser.parse_args()
        run_full_analysis(run_id=args.run_id, compare_to=args.compare_to)
    else:
        main() 