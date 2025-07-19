import torch
import numpy as np
from torch_geometric.data import Data
from scipy.spatial import cKDTree

def build_particle_graph(df, radius_buffer=0.0, sim_id=0, max_gap_frames=3):
    """
    Build particle graph with support for detection gaps.
    
    Args:
        df: DataFrame with particle data including detection gaps
        radius_buffer: Additional radius for proximity edges
        sim_id: Simulation ID
        max_gap_frames: Maximum frame gap to consider for temporal edges
    """
    positions = torch.tensor(df[['x', 'y', 'z']].values, dtype=torch.float)
    masses = torch.tensor(df[['mass']].values, dtype=torch.float)
    labels = torch.tensor(df['event_label'].values, dtype=torch.long)
    particle_ids = torch.tensor(df['particle'].values, dtype=torch.long)
    frame_ids = torch.tensor(df['frame'].values, dtype=torch.long)
    
    edge_list = []
    edge_features = []
    edge_types = []
    edge_gap_lengths = []  # New: track gap lengths
    
    # Add temporal edges (handles detection gaps)
    max_displacement = add_temporal_edges_with_gaps(
        df, edge_list, edge_features, edge_types, edge_gap_lengths, max_gap_frames
    )
    
    # Add proximity edges
    radius = max_displacement + radius_buffer
    # print("RADIUS VALUE --- >",radius)
    add_proximity_edges(df, radius, edge_list, edge_features, edge_types, edge_gap_lengths)
    
    # Convert to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float).unsqueeze(1)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_gaps = torch.tensor(edge_gap_lengths, dtype=torch.long)
    
    return Data(
        x=masses,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        edge_gaps=edge_gaps,  # New: gap length information
        y=labels,
        pos=positions,
        particle_ids=particle_ids,
        frame_numbers=frame_ids,
        simulation_id=torch.full((len(df),), sim_id, dtype=torch.long),
        # Additional metadata for gap analysis
        detection_stats=get_detection_statistics(df)
    )

def add_temporal_edges_with_gaps(df, edge_list, edge_features, edge_types, edge_gap_lengths, max_gap_frames,Verbose=False):
    """
    Add temporal edges between consecutive detections of the same particle.
    Handles detection gaps by connecting across missing frames.
    """
    max_displacement = 0.0
    gap_spanning_edges = 0
    direct_edges = 0
    
    for particle_id in df['particle'].unique():
        particle_df = df[df['particle'] == particle_id].sort_values('frame')
        indices = particle_df.index.values
        frames = particle_df['frame'].values
        positions = particle_df[['x', 'y', 'z']].values
        
        for i in range(len(indices) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            frame_gap = next_frame - current_frame
            
            # Only connect if gap is within acceptable range
            if frame_gap <= max_gap_frames:
                displacement = np.linalg.norm(positions[i+1] - positions[i])
                max_displacement = max(max_displacement, displacement)
                
                # Determine edge type based on gap
                if frame_gap == 1:
                    # Direct temporal connection (no gap)
                    edge_type = 0
                    direct_edges += 1
                else:
                    # Gap-spanning temporal connection
                    edge_type = 2  # New edge type for gap-spanning
                    gap_spanning_edges += 1
                
                # Add bidirectional edges
                edge_list.extend([[indices[i], indices[i+1]], [indices[i+1], indices[i]]])
                edge_features.extend([displacement, displacement])
                edge_types.extend([edge_type, edge_type])
                edge_gap_lengths.extend([frame_gap, frame_gap])
            else:
                # Gap too large - particle was lost and reappeared
                # Could optionally add a different edge type here
                pass
    if Verbose:
        print(f"Temporal edges: {direct_edges} direct, {gap_spanning_edges} gap-spanning")

    return max_displacement

def add_proximity_edges(df, radius, edge_list, edge_features, edge_types, edge_gap_lengths, Verbose=False):
    """
    Add proximity edges between particles in consecutive frames.
    Handles sparse data due to detection gaps.
    """
    frames = sorted(df['frame'].unique())
    proximity_edges = 0
    
    for i in range(len(frames) - 1):
        curr_frame = frames[i]
        next_frame = frames[i + 1]
        frame_gap = next_frame - curr_frame
        
        curr_df = df[df['frame'] == curr_frame]
        next_df = df[df['frame'] == next_frame]
        
        if len(curr_df) == 0 or len(next_df) == 0:
            continue
        
        curr_pos = curr_df[['x', 'y', 'z']].values
        next_pos = next_df[['x', 'y', 'z']].values
        
        # Build spatial tree for efficient neighbor search
        tree = cKDTree(next_pos)
        neighbors = tree.query_ball_point(curr_pos, radius)
        
        curr_indices = curr_df.index.values
        next_indices = next_df.index.values
        curr_particle_ids = curr_df['particle'].values
        next_particle_ids = next_df['particle'].values
        
        for idx, neighbor_list in enumerate(neighbors):
            curr_particle_id = curr_particle_ids[idx]
            
            for neighbor_idx in neighbor_list:
                next_particle_id = next_particle_ids[neighbor_idx]
                
                # Skip self-connections (already handled by temporal edges)
                if curr_particle_id == next_particle_id:
                    continue
                
                distance = np.linalg.norm(curr_pos[idx] - next_pos[neighbor_idx])
                
                # Add bidirectional proximity edges
                edge_list.extend([[curr_indices[idx], next_indices[neighbor_idx]],
                                [next_indices[neighbor_idx], curr_indices[idx]]])
                edge_features.extend([distance, distance])
                edge_types.extend([1, 1])  # 1 for proximity edges
                edge_gap_lengths.extend([frame_gap, frame_gap])  # Frame gap for proximity edges
                proximity_edges += 1
    if Verbose:
        print(f"Proximity edges: {proximity_edges}")

def get_detection_statistics(df):
    """
    Extract detection statistics for analysis.
    """
    stats = {}
    
    # Overall detection rate
    if 'detected_duration' in df.columns and 'duration' in df.columns:
        total_possible = df.groupby('particle')['duration'].first().sum()
        total_detected = df.groupby('particle')['detected_duration'].first().sum()
        stats['overall_detection_rate'] = total_detected / total_possible if total_possible > 0 else 1.0
    
    # Particles with gaps
    if 'undetectable_frames' in df.columns:
        particles_with_gaps = df[df['undetectable_frames'] != ''].groupby('particle').size()
        stats['particles_with_gaps'] = len(particles_with_gaps)
        stats['total_particles'] = df['particle'].nunique()
        stats['gap_rate'] = len(particles_with_gaps) / stats['total_particles']
    
    # Frame gap distribution
    frame_gaps = []
    for particle_id in df['particle'].unique():
        particle_df = df[df['particle'] == particle_id].sort_values('frame')
        frames = particle_df['frame'].values
        if len(frames) > 1:
            gaps = np.diff(frames)
            frame_gaps.extend(gaps[gaps > 1])  # Only gaps > 1 frame
    
    if frame_gaps:
        stats['avg_gap_length'] = np.mean(frame_gaps)
        stats['max_gap_length'] = np.max(frame_gaps)
        stats['num_gaps'] = len(frame_gaps)
    else:
        stats['avg_gap_length'] = 0
        stats['max_gap_length'] = 0
        stats['num_gaps'] = 0
    
    return stats

def build_unlabeled_graph(df, radius_buffer=0.0, max_gap_frames=3):
    """
    Build unlabeled graph for inference with detection gap support.
    """
    df = df.copy()
    df['event_label'] = 0
    return build_particle_graph(df, radius_buffer, sim_id=-1, max_gap_frames=max_gap_frames)

def analyze_graph_connectivity(graph_data):
    """
    Analyze the connectivity patterns in the graph to understand gap impact.
    """
    edge_types = graph_data.edge_type
    edge_gaps = graph_data.edge_gaps
    
    # Count edge types
    temporal_direct = (edge_types == 0).sum().item()
    proximity = (edge_types == 1).sum().item() 
    temporal_gap = (edge_types == 2).sum().item()
    
    # Analyze gap lengths
    gap_edges = edge_gaps[edge_types == 2]
    if len(gap_edges) > 0:
        avg_gap = gap_edges.float().mean().item()
        max_gap = gap_edges.max().item()
    else:
        avg_gap = max_gap = 0
    
    print(f"\n=== GRAPH CONNECTIVITY ANALYSIS ===")
    print(f"Direct temporal edges: {temporal_direct}")
    print(f"Gap-spanning temporal edges: {temporal_gap}")
    print(f"Proximity edges: {proximity}")
    print(f"Average gap length: {avg_gap:.2f} frames")
    print(f"Maximum gap length: {max_gap} frames")
    
    if hasattr(graph_data, 'detection_stats'):
        stats = graph_data.detection_stats
        print(f"Detection rate: {stats.get('overall_detection_rate', 0):.3f}")
        print(f"Particles with gaps: {stats.get('particles_with_gaps', 0)}/{stats.get('total_particles', 0)}")
    
    print("===================================\n")
    
    return {
        'temporal_direct': temporal_direct,
        'temporal_gap': temporal_gap, 
        'proximity': proximity,
        'avg_gap_length': avg_gap,
        'max_gap_length': max_gap
    }

# Example usage with detection gap handling
def process_simulation_with_gaps(csv_file, radius_buffer=5.0, max_gap_frames=3):
    """
    Process a simulation CSV file with detection gaps into a graph.
    """
    import pandas as pd
    
    df = pd.read_csv(csv_file)
    
    # Build graph with gap support
    graph = build_particle_graph(df, radius_buffer=radius_buffer, max_gap_frames=max_gap_frames)
    
    # Analyze connectivity
    connectivity_stats = analyze_graph_connectivity(graph)
    
    return graph, connectivity_stats

# Advanced: Gap-aware message passing preparation
def prepare_gap_aware_features(graph_data):
    """
    Prepare additional features for gap-aware GNN message passing.
    """
    # Add gap information as edge features
    edge_gaps_normalized = graph_data.edge_gaps.float() / graph_data.edge_gaps.max().float()
    
    # Create gap mask for different message passing schemes
    direct_temporal_mask = graph_data.edge_type == 0
    gap_temporal_mask = graph_data.edge_type == 2
    proximity_mask = graph_data.edge_type == 1
    
    # Combine edge attributes with gap information
    enhanced_edge_attr = torch.cat([
        graph_data.edge_attr,
        edge_gaps_normalized.unsqueeze(1),
        graph_data.edge_type.float().unsqueeze(1)
    ], dim=1)
    
    # Update graph data
    graph_data.edge_attr = enhanced_edge_attr
    graph_data.direct_temporal_mask = direct_temporal_mask
    graph_data.gap_temporal_mask = gap_temporal_mask  
    graph_data.proximity_mask = proximity_mask
    
    return graph_data

if __name__ == "__main__":
    # Test with a sample CSV file
    csv_file = "data/tracked_simdata_clean/tracked_particles_3d_00000.csv"
    
    try:
        graph, stats = process_simulation_with_gaps(csv_file, radius_buffer=5.0, max_gap_frames=3)
        enhanced_graph = prepare_gap_aware_features(graph)
        
        print("Graph construction successful!")
        print(f"Nodes: {graph.x.shape[0]}")
        print(f"Edges: {graph.edge_index.shape[1]}")
        print(f"Enhanced edge features: {enhanced_graph.edge_attr.shape}")
        
    except FileNotFoundError:
        print("CSV file not found. Make sure to run the simulation first.")
    except Exception as e:
        print(f"Error: {e}")