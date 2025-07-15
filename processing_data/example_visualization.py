from visualize_3d import ParticleGraphVisualizer
from preprocessor import load_preprocessed_data


def visualize_particle_events():
    _, _, _, data = load_preprocessed_data("../data/processed", batch_size=1)
    graphs = data['graphs']
    
    graph = graphs[0]
    print(f"Graph has {graph.num_nodes} nodes across frames {graph.frame_numbers.min()}-{graph.frame_numbers.max()}")
    
    visualizer = ParticleGraphVisualizer(graph)
    
    # Static visualization
    fig = visualizer.create_3d_plot(
        start_frame=5,
        end_frame=15,
        show_temporal=True,
        show_proximity=True
    )
    fig.show()
    
    # Save to file
    fig.write_html("particle_graph_3d.html")
    
    # Create animation
    anim_fig = visualizer.create_animation(
        start_frame=0,
        end_frame=20,
        window_size=5
    )
    anim_fig.write_html("particle_animation.html")


def analyze_edge_distribution():
    _, _, _, data = load_preprocessed_data("../data/processed", batch_size=1)
    graph = data['graphs'][0]
    
    if hasattr(graph, 'edge_type'):
        temporal_edges = (graph.edge_type == 0).sum().item()
        proximity_edges = (graph.edge_type == 1).sum().item()
        
        print(f"Temporal edges: {temporal_edges}")
        print(f"Proximity edges: {proximity_edges}")
        print(f"Ratio: {temporal_edges/proximity_edges:.2f}")
    
    visualizer = ParticleGraphVisualizer(graph)
    
    # Visualize only temporal edges
    fig_temporal = visualizer.create_3d_plot(0, 10, show_temporal=True, show_proximity=False)
    fig_temporal.update_layout(title="Temporal Edges Only")
    fig_temporal.show()
    
    # Visualize only proximity edges
    fig_proximity = visualizer.create_3d_plot(0, 10, show_temporal=False, show_proximity=True)
    fig_proximity.update_layout(title="Proximity Edges Only")
    fig_proximity.show()


def visualize_event_regions():
    _, _, _, data = load_preprocessed_data("../data/processed", batch_size=1)
    
    # Find a graph with merge/split events
    for i, graph in enumerate(data['graphs']):
        merge_count = (graph.y == 1).sum()
        split_count = (graph.y == 2).sum()
        
        if merge_count > 0 or split_count > 0:
            print(f"Graph {i}: {merge_count} merges, {split_count} splits")
            
            visualizer = ParticleGraphVisualizer(graph)
            
            # Find frames with events
            merge_frames = graph.frame_numbers[graph.y == 1].unique()
            split_frames = graph.frame_numbers[graph.y == 2].unique()
            
            if len(merge_frames) > 0:
                event_frame = merge_frames[0].item()
                fig = visualizer.create_3d_plot(
                    start_frame=max(0, event_frame - 2),
                    end_frame=min(graph.frame_numbers.max().item(), event_frame + 2)
                )
                fig.update_layout(title=f"Merge Event at Frame {event_frame}")
                fig.show()
                break


if __name__ == "__main__":
    print("1. Basic visualization with all edges")
    visualize_particle_events()
    
    print("\n2. Analyzing edge types")
    analyze_edge_distribution()
    
    print("\n3. Finding and visualizing events")
    visualize_event_regions()