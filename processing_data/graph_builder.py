import torch
import numpy as np
from torch_geometric.data import Data
from scipy.spatial import cKDTree


def build_particle_graph(df, radius_buffer=0.0, sim_id=0):
    positions = torch.tensor(df[['x', 'y', 'z']].values, dtype=torch.float)
    masses = torch.tensor(df[['mass']].values, dtype=torch.float)
    labels = torch.tensor(df['event_label'].values, dtype=torch.long)
    particle_ids = torch.tensor(df['particle'].values, dtype=torch.long)
    frame_ids = torch.tensor(df['frame'].values, dtype=torch.long)
    
    edge_list = []
    edge_features = []
    edge_types = []
    
    max_displacement = _add_temporal_edges(df, edge_list, edge_features, edge_types)
    radius = max_displacement + radius_buffer
    _add_proximity_edges(df, radius, edge_list, edge_features, edge_types)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float).unsqueeze(1)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    
    return Data(
        x=masses,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        y=labels,
        pos=positions,
        particle_ids=particle_ids,
        frame_numbers=frame_ids,
        simulation_id=torch.full((len(df),), sim_id, dtype=torch.long)
    )


def _add_temporal_edges(df, edge_list, edge_features, edge_types):
    max_displacement = 0.0
    
    for particle_id in df['particle'].unique():
        particle_df = df[df['particle'] == particle_id].sort_values('frame')
        indices = particle_df.index.values
        positions = particle_df[['x', 'y', 'z']].values
        
        for i in range(len(indices) - 1):
            displacement = np.linalg.norm(positions[i+1] - positions[i])
            max_displacement = max(max_displacement, displacement)
            
            # Add bidirectional edges
            edge_list.extend([[indices[i], indices[i+1]], [indices[i+1], indices[i]]])
            edge_features.extend([displacement, displacement])
            edge_types.extend([0, 0])  # 0 for temporal edges
    
    return max_displacement


def _add_proximity_edges(df, radius, edge_list, edge_features, edge_types):
    frames = sorted(df['frame'].unique())
    
    for i in range(len(frames) - 1):
        curr_df = df[df['frame'] == frames[i]]
        next_df = df[df['frame'] == frames[i+1]]
        
        if len(curr_df) == 0 or len(next_df) == 0:
            continue
        
        curr_pos = curr_df[['x', 'y', 'z']].values
        next_pos = next_df[['x', 'y', 'z']].values
        
        tree = cKDTree(next_pos)
        neighbors = tree.query_ball_point(curr_pos, radius)
        
        curr_indices = curr_df.index.values
        next_indices = next_df.index.values
        
        for idx, neighbor_list in enumerate(neighbors):
            for neighbor_idx in neighbor_list:
                distance = np.linalg.norm(curr_pos[idx] - next_pos[neighbor_idx])
                
                # Add bidirectional edges
                edge_list.extend([[curr_indices[idx], next_indices[neighbor_idx]], 
                                [next_indices[neighbor_idx], curr_indices[idx]]])
                edge_features.extend([distance, distance])
                edge_types.extend([1, 1])  # 1 for proximity edges


def build_unlabeled_graph(df, radius_buffer=0.0):
    df = df.copy()
    df['event_label'] = 0
    return build_particle_graph(df, radius_buffer, sim_id=-1)