import argparse
import os
import osmnx as ox
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


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
    if fig:
        plt.show()


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
    if fig:
        plt.show()


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
    periods = pd.read_parquet(args.periods)
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

if __name__ == "__main__":
    main() 