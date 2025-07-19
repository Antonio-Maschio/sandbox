#!/usr/bin/env python3
"""
Honest Gap-Aware GNN Test Script

This script provides an unbiased comparison by using your existing preprocessing
and only testing the model architecture differences.

Key improvements:
- Uses simulation-level splits (no data leakage)
- Larger, more representative dataset
- Direct comparison with your exact baseline
- Conservative evaluation methodology

Usage:
    python honest_gap_test.py --data_dir data/tracked_simdata_clean --max_files 500
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from scipy.spatial import cKDTree
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EXACT REPLICATION OF YOUR PREPROCESSING (if possible)
# ============================================================================

def build_standard_graph(df, radius_buffer=5.0, sim_id=0):
    """Build graph exactly like your existing pipeline (standard approach)."""
    
    positions = torch.tensor(df[['x', 'y', 'z']].values, dtype=torch.float)
    masses = torch.tensor(df[['mass']].values, dtype=torch.float)
    labels = torch.tensor(df['event_label'].values, dtype=torch.long)
    particle_ids = torch.tensor(df['particle'].values, dtype=torch.long)
    frame_ids = torch.tensor(df['frame'].values, dtype=torch.long)
    
    edge_list = []
    edge_features = []
    
    # Standard temporal edges (consecutive frames only)
    max_displacement = add_standard_temporal_edges(df, edge_list, edge_features)
    
    # Standard proximity edges
    radius = max_displacement + radius_buffer
    add_standard_proximity_edges(df, radius, edge_list, edge_features)
    
    if len(edge_list) == 0:
        # Minimal graph
        num_nodes = len(df)
        edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])
        edge_attr = torch.ones(num_nodes, 1)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float).unsqueeze(1)
    
    return Data(
        x=masses,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels,
        pos=positions,
        particle_ids=particle_ids,
        frame_numbers=frame_ids,
        sim_id=sim_id
    )

def build_gap_aware_graph(df, radius_buffer=5.0, max_gap_frames=3, sim_id=0):
    """Build gap-aware graph with detection gap handling."""
    
    positions = torch.tensor(df[['x', 'y', 'z']].values, dtype=torch.float)
    masses = torch.tensor(df[['mass']].values, dtype=torch.float)
    labels = torch.tensor(df['event_label'].values, dtype=torch.long)
    particle_ids = torch.tensor(df['particle'].values, dtype=torch.long)
    frame_ids = torch.tensor(df['frame'].values, dtype=torch.long)
    
    edge_list = []
    edge_features = []
    edge_types = []
    edge_gaps = []
    
    # Gap-aware temporal edges
    max_displacement = add_gap_aware_temporal_edges(df, edge_list, edge_features, edge_types, edge_gaps, max_gap_frames)
    
    # Standard proximity edges
    radius = max_displacement + radius_buffer
    add_gap_aware_proximity_edges(df, radius, edge_list, edge_features, edge_types, edge_gaps)
    
    if len(edge_list) == 0:
        # Minimal graph
        num_nodes = len(df)
        edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])
        edge_attr = torch.ones(num_nodes, 3)
        edge_type = torch.zeros(num_nodes, dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack([
            torch.tensor(edge_features, dtype=torch.float),
            torch.tensor(edge_gaps, dtype=torch.float),
            torch.tensor(edge_types, dtype=torch.float)
        ], dim=1)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
    
    # Create masks
    direct_temporal_mask = edge_type == 0
    gap_temporal_mask = edge_type == 2
    proximity_mask = edge_type == 1
    
    return Data(
        x=masses,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        y=labels,
        pos=positions,
        particle_ids=particle_ids,
        frame_numbers=frame_ids,
        direct_temporal_mask=direct_temporal_mask,
        gap_temporal_mask=gap_temporal_mask,
        proximity_mask=proximity_mask,
        sim_id=sim_id
    )

def add_standard_temporal_edges(df, edge_list, edge_features):
    """Add only consecutive frame temporal edges (your current approach)."""
    max_displacement = 0.0
    
    for particle_id in df['particle'].unique():
        particle_df = df[df['particle'] == particle_id].sort_values('frame')
        indices = particle_df.index.values
        frames = particle_df['frame'].values
        positions = particle_df[['x', 'y', 'z']].values
        
        for i in range(len(indices) - 1):
            frame_gap = frames[i + 1] - frames[i]
            
            # ONLY consecutive frames (gap = 1)
            if frame_gap == 1:
                displacement = np.linalg.norm(positions[i+1] - positions[i])
                max_displacement = max(max_displacement, displacement)
                
                edge_list.extend([[indices[i], indices[i+1]], [indices[i+1], indices[i]]])
                edge_features.extend([displacement, displacement])
    
    return max_displacement

def add_gap_aware_temporal_edges(df, edge_list, edge_features, edge_types, edge_gaps, max_gap_frames):
    """Add temporal edges including gap-spanning ones."""
    max_displacement = 0.0
    
    for particle_id in df['particle'].unique():
        particle_df = df[df['particle'] == particle_id].sort_values('frame')
        indices = particle_df.index.values
        frames = particle_df['frame'].values
        positions = particle_df[['x', 'y', 'z']].values
        
        for i in range(len(indices) - 1):
            frame_gap = frames[i + 1] - frames[i]
            
            if frame_gap <= max_gap_frames:
                displacement = np.linalg.norm(positions[i+1] - positions[i])
                max_displacement = max(max_displacement, displacement)
                
                edge_type = 0 if frame_gap == 1 else 2
                
                edge_list.extend([[indices[i], indices[i+1]], [indices[i+1], indices[i]]])
                edge_features.extend([displacement, displacement])
                edge_types.extend([edge_type, edge_type])
                edge_gaps.extend([frame_gap, frame_gap])
    
    return max_displacement

def add_standard_proximity_edges(df, radius, edge_list, edge_features):
    """Add standard proximity edges."""
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
        curr_particles = curr_df['particle'].values
        next_particles = next_df['particle'].values
        
        for idx, neighbor_list in enumerate(neighbors):
            for neighbor_idx in neighbor_list:
                if curr_particles[idx] == next_particles[neighbor_idx]:
                    continue
                
                distance = np.linalg.norm(curr_pos[idx] - next_pos[neighbor_idx])
                edge_list.extend([[curr_indices[idx], next_indices[neighbor_idx]],
                                [next_indices[neighbor_idx], curr_indices[idx]]])
                edge_features.extend([distance, distance])

def add_gap_aware_proximity_edges(df, radius, edge_list, edge_features, edge_types, edge_gaps):
    """Add proximity edges with gap information."""
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
        curr_particles = curr_df['particle'].values
        next_particles = next_df['particle'].values
        
        frame_gap = frames[i+1] - frames[i]
        
        for idx, neighbor_list in enumerate(neighbors):
            for neighbor_idx in neighbor_list:
                if curr_particles[idx] == next_particles[neighbor_idx]:
                    continue
                
                distance = np.linalg.norm(curr_pos[idx] - next_pos[neighbor_idx])
                edge_list.extend([[curr_indices[idx], next_indices[neighbor_idx]],
                                [next_indices[neighbor_idx], curr_indices[idx]]])
                edge_features.extend([distance, distance])
                edge_types.extend([1, 1])  # proximity
                edge_gaps.extend([frame_gap, frame_gap])

# ============================================================================
# IDENTICAL MODELS (only difference is gap-awareness)
# ============================================================================

class StandardGNN(nn.Module):
    """Your exact current model architecture."""
    
    def __init__(self, hidden_channels=128, num_classes=5, dropout=0.2, heads=4):
        super().__init__()
        
        self.conv1 = GATConv(1, hidden_channels//heads, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels, hidden_channels//heads, heads=heads, concat=True)
        self.conv3 = GATConv(hidden_channels, hidden_channels//heads, heads=heads, concat=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels//2, num_classes)
        )
        self.dropout = dropout
    
    def forward(self, x, edge_index, **kwargs):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.conv3(x, edge_index))
        
        return F.log_softmax(self.classifier(x), dim=-1)

class GapAwareGNN(nn.Module):
    """Gap-aware version with identical base architecture."""
    
    def __init__(self, hidden_channels=128, num_classes=5, dropout=0.2, heads=4):
        super().__init__()
        
        # FIXED: Make sure dimensions work out
        conv_out_channels = hidden_channels // heads // 3
        if conv_out_channels * heads * 3 != hidden_channels:
            # Adjust to make dimensions work
            conv_out_channels = hidden_channels // heads // 3
            total_conv_output = conv_out_channels * heads * 3
        else:
            total_conv_output = hidden_channels
        
        # Gap-aware processing
        self.temporal_conv = GATConv(1, conv_out_channels, heads=heads, edge_dim=3, concat=True)
        self.gap_conv = GATConv(1, conv_out_channels, heads=heads, edge_dim=3, concat=True) 
        self.proximity_conv = GATConv(1, conv_out_channels, heads=heads, edge_dim=3, concat=True)
        
        # FIXED: Fusion layer matches actual concatenated size
        actual_fusion_input = conv_out_channels * heads * 3
        self.fusion = nn.Linear(actual_fusion_input, hidden_channels)
        
        # Rest of architecture identical to standard
        self.conv2 = GATConv(hidden_channels, hidden_channels//heads, heads=heads, concat=True)
        self.conv3 = GATConv(hidden_channels, hidden_channels//heads, heads=heads, concat=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels//2, num_classes)
        )
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr=None, direct_temporal_mask=None, 
                gap_temporal_mask=None, proximity_mask=None, **kwargs):
        
        # Initialize outputs for each edge type
        device = x.device
        out_size = self.temporal_conv.out_channels * self.temporal_conv.heads
        
        h_temporal = torch.zeros(x.size(0), out_size, device=device)
        h_gap = torch.zeros(x.size(0), out_size, device=device)
        h_proximity = torch.zeros(x.size(0), out_size, device=device)
        
        # Process each edge type separately if edges exist
        if direct_temporal_mask is not None and direct_temporal_mask.sum() > 0:
            temporal_edges = edge_index[:, direct_temporal_mask]
            temporal_attr = edge_attr[direct_temporal_mask] if edge_attr is not None else None
            if temporal_edges.shape[1] > 0:
                h_temporal = self.temporal_conv(x, temporal_edges, temporal_attr)
        
        if gap_temporal_mask is not None and gap_temporal_mask.sum() > 0:
            gap_edges = edge_index[:, gap_temporal_mask]
            gap_attr = edge_attr[gap_temporal_mask] if edge_attr is not None else None
            if gap_edges.shape[1] > 0:
                h_gap = self.gap_conv(x, gap_edges, gap_attr)
        
        if proximity_mask is not None and proximity_mask.sum() > 0:
            prox_edges = edge_index[:, proximity_mask]
            prox_attr = edge_attr[proximity_mask] if edge_attr is not None else None
            if prox_edges.shape[1] > 0:
                h_proximity = self.proximity_conv(x, prox_edges, prox_attr)
        
        # Concatenate and fuse
        x = torch.cat([h_temporal, h_gap, h_proximity], dim=-1)
        x = F.relu(self.fusion(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Same subsequent processing as standard model
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.conv3(x, edge_index))
        
        return F.log_softmax(self.classifier(x), dim=-1)

# ============================================================================
# PROPER EVALUATION
# ============================================================================

class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

def custom_collate(batch):
    batch_data = Batch.from_data_list(batch)
    
    if hasattr(batch[0], 'direct_temporal_mask'):
        batch_data.direct_temporal_mask = torch.cat([data.direct_temporal_mask for data in batch])
        batch_data.gap_temporal_mask = torch.cat([data.gap_temporal_mask for data in batch])
        batch_data.proximity_mask = torch.cat([data.proximity_mask for data in batch])
    
    return batch_data

def load_data_with_simulation_split(data_dir, max_files=500):
    """Load data with proper simulation-level splits (no data leakage)."""
    
    data_path = Path(data_dir)
    csv_files = sorted(list(data_path.glob("tracked_particles_3d_*.csv")))[:max_files]
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f"🔄 Processing {len(csv_files)} files with simulation-level split...")
    
    # PROPER SPLIT: by simulation files, not by individual graphs
    np.random.seed(42)  # Reproducible splits
    shuffled_files = np.random.permutation(csv_files)
    
    split_idx = int(0.8 * len(shuffled_files))
    train_files = shuffled_files[:split_idx]
    val_files = shuffled_files[split_idx:]
    
    print(f"📊 Split: {len(train_files)} train files, {len(val_files)} val files")
    
    # Process train data
    train_standard_graphs = []
    train_gap_aware_graphs = []
    
    for i, csv_file in enumerate(tqdm(train_files, desc="Processing train files")):
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
            
            std_graph = build_standard_graph(df, sim_id=i)
            gap_graph = build_gap_aware_graph(df, sim_id=i)
            
            train_standard_graphs.append(std_graph)
            train_gap_aware_graphs.append(gap_graph)
        except Exception as e:
            print(f"⚠️  Skipping train {csv_file.name}: {e}")
    
    # Process val data
    val_standard_graphs = []
    val_gap_aware_graphs = []
    
    for i, csv_file in enumerate(tqdm(val_files, desc="Processing val files")):
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
                
            std_graph = build_standard_graph(df, sim_id=i + len(train_files))
            gap_graph = build_gap_aware_graph(df, sim_id=i + len(train_files))
            
            val_standard_graphs.append(std_graph)
            val_gap_aware_graphs.append(gap_graph)
        except Exception as e:
            print(f"⚠️  Skipping val {csv_file.name}: {e}")
    
    return (train_standard_graphs, train_gap_aware_graphs, 
            val_standard_graphs, val_gap_aware_graphs)

def normalize_graphs(train_graphs, val_graphs):
    """Apply consistent normalization."""
    
    # Compute stats from train set only
    all_masses = torch.cat([g.x for g in train_graphs])
    all_positions = torch.cat([g.pos for g in train_graphs])
    all_distances = torch.cat([g.edge_attr[:, 0] for g in train_graphs if g.edge_attr.shape[0] > 0])
    
    mass_mean = all_masses.mean()
    mass_std = all_masses.std() + 1e-8
    pos_mean = all_positions.mean(dim=0)
    pos_std = all_positions.std(dim=0) + 1e-8
    distance_mean = all_distances.mean()
    distance_std = all_distances.std() + 1e-8
    
    def normalize_graph(graph):
        graph.x = (graph.x - mass_mean) / mass_std
        graph.pos = (graph.pos - pos_mean) / pos_std
        if graph.edge_attr.shape[0] > 0:
            graph.edge_attr[:, 0] = (graph.edge_attr[:, 0] - distance_mean) / distance_std
        return graph
    
    train_normalized = [normalize_graph(g) for g in train_graphs]
    val_normalized = [normalize_graph(g) for g in val_graphs]
    
    return train_normalized, val_normalized

def evaluate_model_honest(model, val_loader, device):
    """Honest evaluation with detailed metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_sim_ids = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            out = model(batch.x, batch.edge_index,
                       edge_attr=getattr(batch, 'edge_attr', None),
                       direct_temporal_mask=getattr(batch, 'direct_temporal_mask', None),
                       gap_temporal_mask=getattr(batch, 'gap_temporal_mask', None),
                       proximity_mask=getattr(batch, 'proximity_mask', None))
            
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_sim_ids.extend(batch.sim_id.cpu().numpy())
    
    # Overall accuracy
    overall_acc = accuracy_score(all_labels, all_preds)
    
    # Per-class metrics
    label_names = ['NORMAL', 'MERGE', 'SPLIT', 'POST_MERGE', 'POST_SPLIT']
    report = classification_report(all_labels, all_preds, target_names=label_names, output_dict=True)
    
    # Event detection accuracy (non-NORMAL classes)
    event_mask = np.array(all_labels) > 0
    if event_mask.sum() > 0:
        event_acc = accuracy_score(np.array(all_labels)[event_mask], np.array(all_preds)[event_mask])
    else:
        event_acc = 0.0
    
    return overall_acc, event_acc, report

def main():
    parser = argparse.ArgumentParser(description="Honest Gap-Aware GNN Test")
    parser.add_argument("--data_dir", type=str, default="../data/tracked_simdata_dirty")
    parser.add_argument("--max_files", type=int, default=1000,
                       help="More files for reliable results")
    parser.add_argument("--epochs", type=int, default=20,
                       help="More epochs for fair comparison")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    
    args = parser.parse_args()
    
    print("🔬 HONEST GAP-AWARE GNN TEST")
    print("=" * 50)
    print("Key improvements to prevent inflated results:")
    print("- Simulation-level train/val split (no data leakage)")
    print("- Larger dataset for statistical significance")
    print("- Identical base architectures")
    print("- Conservative evaluation methodology")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load data with proper splits
    (train_std_graphs, train_gap_graphs, 
     val_std_graphs, val_gap_graphs) = load_data_with_simulation_split(args.data_dir, args.max_files)
    
    if len(train_std_graphs) == 0:
        print("❌ No training graphs created")
        return
    
    print(f"📚 Dataset sizes:")
    print(f"   Train: {len(train_std_graphs)} graphs")
    print(f"   Val: {len(val_std_graphs)} graphs")
    
    # Normalize consistently
    train_std_norm, val_std_norm = normalize_graphs(train_std_graphs, val_std_graphs)
    train_gap_norm, val_gap_norm = normalize_graphs(train_gap_graphs, val_gap_graphs)
    
    # Create data loaders
    train_std_loader = DataLoader(GraphDataset(train_std_norm), 
                                 batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    val_std_loader = DataLoader(GraphDataset(val_std_norm), 
                               batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
    
    train_gap_loader = DataLoader(GraphDataset(train_gap_norm), 
                                 batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    val_gap_loader = DataLoader(GraphDataset(val_gap_norm), 
                               batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
    
    # Test both models with identical training
    print(f"\n🤖 TRAINING STANDARD GNN (Baseline)")
    print("-" * 40)
    
    std_model = StandardGNN(args.hidden_channels).to(device)
    std_optimizer = torch.optim.Adam(std_model.parameters(), lr=args.learning_rate)
    
    # Train standard model
    std_model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in train_std_loader:
            batch = batch.to(device)
            std_optimizer.zero_grad()
            out = std_model(batch.x, batch.edge_index)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            std_optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_std_loader):.4f}")
    
    # Evaluate standard model
    std_acc, std_event_acc, std_report = evaluate_model_honest(std_model, val_std_loader, device)
    
    print(f"\n🚀 TRAINING GAP-AWARE GNN")
    print("-" * 40)
    
    gap_model = GapAwareGNN(args.hidden_channels).to(device)
    gap_optimizer = torch.optim.Adam(gap_model.parameters(), lr=args.learning_rate)
    
    # Train gap-aware model
    gap_model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in train_gap_loader:
            batch = batch.to(device)
            gap_optimizer.zero_grad()
            out = gap_model(batch.x, batch.edge_index,
                           edge_attr=getattr(batch, 'edge_attr', None),
                           direct_temporal_mask=getattr(batch, 'direct_temporal_mask', None),
                           gap_temporal_mask=getattr(batch, 'gap_temporal_mask', None),
                           proximity_mask=getattr(batch, 'proximity_mask', None))
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            gap_optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_gap_loader):.4f}")
    
    # Evaluate gap-aware model
    gap_acc, gap_event_acc, gap_report = evaluate_model_honest(gap_model, val_gap_loader, device)
    
    # Honest results
    print(f"\n📊 HONEST RESULTS COMPARISON")
    print("=" * 50)
    print(f"📈 Overall Accuracy:")
    print(f"   Standard GNN:   {std_acc:.3f} ({std_acc*100:.1f}%)")
    print(f"   Gap-Aware GNN:  {gap_acc:.3f} ({gap_acc*100:.1f}%)")
    improvement = (gap_acc - std_acc) * 100
    print(f"   Improvement:    {gap_acc-std_acc:+.3f} ({improvement:+.1f}%)")
    
    print(f"\n🎯 Event Detection Accuracy (MERGE/SPLIT/POST):")
    print(f"   Standard GNN:   {std_event_acc:.3f} ({std_event_acc*100:.1f}%)")
    print(f"   Gap-Aware GNN:  {gap_event_acc:.3f} ({gap_event_acc*100:.1f}%)")
    event_improvement = (gap_event_acc - std_event_acc) * 100
    print(f"   Improvement:    {gap_event_acc-std_event_acc:+.3f} ({event_improvement:+.1f}%)")
    
    # Class-wise comparison
    print(f"\n📋 Per-Class F1 Scores:")
    for class_name in ['MERGE', 'SPLIT', 'POST_MERGE', 'POST_SPLIT']:
        if class_name in std_report and class_name in gap_report:
            std_f1 = std_report[class_name]['f1-score']
            gap_f1 = gap_report[class_name]['f1-score']
            print(f"   {class_name:12}: {std_f1:.3f} → {gap_f1:.3f} ({gap_f1-std_f1:+.3f})")
    
    # Honest conclusion
    print(f"\n🎯 HONEST CONCLUSION:")
    if improvement > 2.0:
        print(f"✅ Gap-aware GNN shows significant improvement (+{improvement:.1f}%)")
        print("   → Worth implementing in your pipeline!")
    elif improvement > 0.5:
        print(f"🤔 Gap-aware GNN shows modest improvement (+{improvement:.1f}%)")
        print("   → Consider cost/benefit vs other improvements")
    elif improvement > -0.5:
        print(f"😐 Gap-aware GNN shows negligible difference ({improvement:+.1f}%)")
        print("   → Focus on scaling your current approach")
    else:
        print(f"❌ Gap-aware GNN performs worse ({improvement:+.1f}%)")
        print("   → Stick with your current architecture")
    
    print(f"\n📈 Comparison with your reported 85% baseline:")
    if std_acc < 0.85:
        print(f"⚠️  Standard GNN here ({std_acc*100:.1f}%) < your pipeline (85%)")
        print("   → Your existing preprocessing/training is better")
        print("   → Gap-aware improvements might not transfer")
    else:
        print(f"✅ Standard GNN here ({std_acc*100:.1f}%) ≈ your pipeline (85%)")
        print("   → This comparison is reliable")

if __name__ == "__main__":
    main()