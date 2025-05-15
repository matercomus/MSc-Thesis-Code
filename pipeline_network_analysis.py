import argparse
import osmnx as ox
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def print_graph_info(G):
    print(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    lats = [data['y'] for _, data in G.nodes(data=True)]
    lons = [data['x'] for _, data in G.nodes(data=True)]
    print(f"Graph bounding box:")
    print(f"  lat: {min(lats):.6f} to {max(lats):.6f}")
    print(f"  lon: {min(lons):.6f} to {max(lons):.6f}")


def print_data_info(periods):
    min_lat = min(periods['start_latitude'].min(), periods['end_latitude'].min())
    max_lat = max(periods['start_latitude'].max(), periods['end_latitude'].max())
    min_lon = min(periods['start_longitude'].min(), periods['end_longitude'].min())
    max_lon = max(periods['end_longitude'].max(), periods['end_longitude'].max())
    print(f"Data bounding box:")
    print(f"  lat: {min_lat:.6f} to {max_lat:.6f}")
    print(f"  lon: {min_lon:.6f} to {max_lon:.6f}")
    print(f"  #periods: {len(periods)}")


def sample_nodes(G, n=10):
    print(f"Sample of {n} nodes:")
    for i, (node, data) in enumerate(G.nodes(data=True)):
        print(f"  Node {node}: x={data['x']:.6f}, y={data['y']:.6f}")
        if i + 1 >= n:
            break


def plot_graph(G, ax=None):
    fig, ax = plt.subplots(figsize=(8, 8)) if ax is None else (None, ax)
    ox.plot_graph(G, ax=ax, show=False, close=False)
    if fig:
        plt.show()


def plot_data_on_graph(G, periods, ax=None, sample=1000):
    fig, ax = plt.subplots(figsize=(8, 8)) if ax is None else (None, ax)
    ox.plot_graph(G, ax=ax, show=False, close=False, node_color='gray', edge_color='lightblue')
    # Plot start and end points
    starts = periods[['start_longitude', 'start_latitude']].dropna().sample(min(sample, len(periods)))
    ends = periods[['end_longitude', 'end_latitude']].dropna().sample(min(sample, len(periods)))
    ax.scatter(starts['start_longitude'], starts['start_latitude'], c='red', s=8, label='Start', alpha=0.6)
    ax.scatter(ends['end_longitude'], ends['end_latitude'], c='blue', s=8, label='End', alpha=0.6)
    ax.legend()
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
            dist = ox.distance.great_circle_vec(lat, lon, node_data['y'], node_data['x'])
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

    if args.all or args.info:
        print("--- Graph Info ---")
        print_graph_info(G)
        print("--- Data Info ---")
        print_data_info(periods)
    if args.all or args.sample_nodes:
        print("--- Node Sample ---")
        sample_nodes(G, n=10)
    if args.all or args.plot_graph:
        print("--- Plotting Graph ---")
        plot_graph(G)
    if args.all or args.plot_data:
        print("--- Plotting Data Points on Graph ---")
        plot_data_on_graph(G, periods, sample=args.sample)
    if args.all or args.check_nodes:
        print("--- Node Assignment Check ---")
        check_node_assignment(G, periods, n=10)

if __name__ == "__main__":
    main() 