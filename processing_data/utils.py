import json
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx


def inspect_dataset(data_dir):
    metadata_path = Path(data_dir) / 'metadata.json'
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("Dataset Overview:")
    for key, value in metadata.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


def visualize_graphs(graphs, num_samples=3, save_path=None):
    fig, axes = plt.subplots(1, min(num_samples, len(graphs)), figsize=(5*num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    colors = {0: 'blue', 1: 'red', 2: 'green'}
    labels = {0: 'Non-event', 1: 'Merge', 2: 'Split'}
    
    for idx, (ax, graph) in enumerate(zip(axes, graphs[:num_samples])):
        G = to_networkx(graph, to_undirected=True)
        pos = {i: (graph.pos[i, 0].item(), graph.pos[i, 1].item()) for i in range(graph.num_nodes)}
        
        node_colors = [colors.get(int(graph.y[i]), 'gray') for i in range(graph.num_nodes)]
        
        nx.draw(G, pos, ax=ax, node_size=30, node_color=node_colors, 
                edge_color='gray', alpha=0.6, linewidths=0.5)
        
        ax.set_title(f'Graph {idx+1}\n{graph.num_nodes} nodes, {graph.num_edges} edges')
        
        if idx == 0:
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color, markersize=8, label=label)
                      for label, color in colors.items() if label in labels]
            ax.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def check_data_quality(graphs):
    issues = []
    
    for i, g in enumerate(graphs):
        if g.num_edges == 0:
            issues.append(f"Graph {i}: No edges")
        
        if g.num_nodes < 2:
            issues.append(f"Graph {i}: Only {g.num_nodes} nodes")
        
        edge_density = g.num_edges / (g.num_nodes * (g.num_nodes - 1))
        if edge_density > 0.8:
            issues.append(f"Graph {i}: Very dense ({edge_density:.2f})")
    
    if issues:
        print("Data quality issues found:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("No data quality issues detected")
    
    return len(issues) == 0