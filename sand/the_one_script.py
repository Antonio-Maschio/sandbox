#!/usr/bin/env python3
"""
Single File Gap-Aware GNN Test Script

This script tests the potential of gap-aware GNN vs standard GNN on your data.
Everything is contained in this single file for easy testing.

Usage:
    python gap_aware_test.py --data_dir data/tracked_simdata_clean --max_files 100 --epochs 5

Results:
    - Trains both standard and gap-aware models
    - Compares accuracy on particles with vs without detection gaps
    - Shows potential improvement from gap-aware architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import random
from scipy.spatial import cKDTree
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GRAPH CONSTRUCTION (Gap-Aware)
# ============================================================================

def build_gap_aware_graph(df, radius_buffer=5.0, max_gap_frames=3, sim_id=0):
    """Build gap-aware graph from simulation DataFrame."""
    
    # Basic features
    positions = torch.tensor(df[['x', 'y', 'z']].values, dtype=torch.float)
    masses = torch.tensor(df[['mass']].values, dtype=torch.float)
    labels = torch.tensor(df['event_label'].values, dtype=torch.long)
    particle_ids = torch.tensor(df['particle'].values, dtype=torch.long)
    frame_ids = torch.tensor(df['frame'].values, dtype=torch.long)
    
    edge_list = []
    edge_features = []
    edge_types = []
    edge_gaps = []
    
    # Add temporal edges with gap handling
    max_displacement = add_temporal_edges(df, edge_list, edge_features, edge_types, edge_gaps, max_gap_frames)
    
    # Add proximity edges
    radius = max_displacement + radius_buffer
    add_proximity_edges(df, radius, edge_list, edge_features, edge_types, edge_gaps)
    
    if len(edge_list) == 0:
        # Create minimal graph with self-loops
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
    
    # Create masks for different edge types
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

def add_temporal_edges(df, edge_list, edge_features, edge_types, edge_gaps, max_gap_frames):
    """Add temporal edges handling detection gaps."""
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
                
                edge_type = 0 if frame_gap == 1 else 2  # 0=direct, 2=gap-spanning
                
                # Add bidirectional edges
                edge_list.extend([[indices[i], indices[i+1]], [indices[i+1], indices[i]]])
                edge_features.extend([displacement, displacement])
                edge_types.extend([edge_type, edge_type])
                edge_gaps.extend([frame_gap, frame_gap])
    
    return max_displacement

def add_proximity_edges(df, radius, edge_list, edge_features, edge_types, edge_gaps):
    """Add proximity edges between nearby particles."""
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
                    continue  # Skip temporal connections
                
                distance = np.linalg.norm(curr_pos[idx] - next_pos[neighbor_idx])
                
                edge_list.extend([[curr_indices[idx], next_indices[neighbor_idx]],
                                [next_indices[neighbor_idx], curr_indices[idx]]])
                edge_features.extend([distance, distance])
                edge_types.extend([1, 1])  # 1=proximity
                edge_gaps.extend([frame_gap, frame_gap])

# ============================================================================
# DATA NORMALIZATION
# ============================================================================

class DataNormalizer:
    def __init__(self):
        self.stats = {}
    
    def fit(self, graphs):
        """Compute normalization statistics from graphs."""
        all_masses = torch.cat([g.x for g in graphs])
        all_positions = torch.cat([g.pos for g in graphs])
        all_distances = torch.cat([g.edge_attr[:, 0] for g in graphs if g.edge_attr.shape[0] > 0])
        
        self.stats = {
            'mass_mean': all_masses.mean().item(),
            'mass_std': all_masses.std().item() + 1e-8,
            'pos_mean': all_positions.mean(dim=0),
            'pos_std': all_positions.std(dim=0) + 1e-8,
            'distance_mean': all_distances.mean().item(),
            'distance_std': all_distances.std().item() + 1e-8
        }
        
        print(f"📊 Normalization stats: Mass μ={self.stats['mass_mean']:.0f}, σ={self.stats['mass_std']:.0f}")
    
    def transform(self, graph):
        """Apply normalization to a graph."""
        graph.x = (graph.x - self.stats['mass_mean']) / self.stats['mass_std']
        graph.pos = (graph.pos - self.stats['pos_mean']) / self.stats['pos_std']
        if graph.edge_attr.shape[0] > 0:
            graph.edge_attr[:, 0] = (graph.edge_attr[:, 0] - self.stats['distance_mean']) / self.stats['distance_std']
        return graph

# ============================================================================
# MODELS
# ============================================================================

class StandardGNN(nn.Module):
    """Standard GNN (baseline) - doesn't use gap information."""
    
    def __init__(self, hidden_channels=128, num_classes=5):
        super().__init__()
        
        self.conv1 = GATConv(1, hidden_channels//4, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels, hidden_channels//4, heads=4, concat=True)
        self.conv3 = GATConv(hidden_channels, hidden_channels//4, heads=4, concat=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels//2, num_classes)
        )
    
    def forward(self, x, edge_index, **kwargs):
        # Standard GNN ignores edge attributes and gap information
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.conv3(x, edge_index))
        
        return F.log_softmax(self.classifier(x), dim=-1)

class GapAwareGNN(nn.Module):
    """Gap-aware GNN - uses gap information for better performance."""
    
    def __init__(self, hidden_channels=128, num_classes=5):
        super().__init__()
        
        # Separate convolutions for different edge types
        self.temporal_conv = GATConv(1, hidden_channels//8, heads=4, edge_dim=3, concat=True)
        self.gap_conv = GATConv(1, hidden_channels//8, heads=4, edge_dim=3, concat=True)
        self.proximity_conv = GATConv(1, hidden_channels//8, heads=4, edge_dim=3, concat=True)
        
        # Fusion layers
        self.fusion = nn.Linear(hidden_channels//2 * 3, hidden_channels)
        
        self.conv2 = GATConv(hidden_channels, hidden_channels//4, heads=4, concat=True)
        self.conv3 = GATConv(hidden_channels, hidden_channels//4, heads=4, concat=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels//2, num_classes)
        )
    
    def forward(self, x, edge_index, edge_attr=None, direct_temporal_mask=None, 
                gap_temporal_mask=None, proximity_mask=None, **kwargs):
        
        # Process different edge types separately
        h_temporal = h_gap = h_proximity = torch.zeros(x.size(0), self.temporal_conv.out_channels * self.temporal_conv.heads, device=x.device)
        
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
        
        # Fuse different edge type representations
        x = torch.cat([h_temporal, h_gap, h_proximity], dim=-1)
        x = F.relu(self.fusion(x))
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Standard processing
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.conv3(x, edge_index))
        
        return F.log_softmax(self.classifier(x), dim=-1)

# ============================================================================
# DATASET AND TRAINING
# ============================================================================

class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

def custom_collate(batch):
    """Custom collate function that handles gap-aware masks."""
    # Standard batching
    batch_data = Batch.from_data_list(batch)
    
    # Handle masks manually
    if hasattr(batch[0], 'direct_temporal_mask'):
        batch_data.direct_temporal_mask = torch.cat([data.direct_temporal_mask for data in batch])
        batch_data.gap_temporal_mask = torch.cat([data.gap_temporal_mask for data in batch])
        batch_data.proximity_mask = torch.cat([data.proximity_mask for data in batch])
    
    return batch_data

def train_model(model, train_loader, val_loader, epochs=5, device='cuda'):
    """Train a model and return final accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, 
                       edge_attr=getattr(batch, 'edge_attr', None),
                       direct_temporal_mask=getattr(batch, 'direct_temporal_mask', None),
                       gap_temporal_mask=getattr(batch, 'gap_temporal_mask', None),
                       proximity_mask=getattr(batch, 'proximity_mask', None))
            
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
    
    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    all_has_gaps = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            out = model(batch.x, batch.edge_index,
                       edge_attr=getattr(batch, 'edge_attr', None),
                       direct_temporal_mask=getattr(batch, 'direct_temporal_mask', None),
                       gap_temporal_mask=getattr(batch, 'gap_temporal_mask', None),
                       proximity_mask=getattr(batch, 'proximity_mask', None))
            
            preds = out.argmax(dim=1)
            
            # Determine which nodes have gap connections
            has_gaps = torch.zeros(batch.y.size(0), dtype=torch.bool)
            if hasattr(batch, 'gap_temporal_mask') and batch.gap_temporal_mask.sum() > 0:
                gap_edges = batch.edge_index[:, batch.gap_temporal_mask]
                has_gaps[gap_edges.flatten().unique()] = True
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_has_gaps.extend(has_gaps.cpu().numpy())
    
    # Calculate accuracies
    overall_acc = accuracy_score(all_labels, all_preds)
    
    # Accuracy on nodes with vs without gaps
    gap_indices = np.array(all_has_gaps)
    no_gap_indices = ~gap_indices
    
    gap_acc = accuracy_score(np.array(all_labels)[gap_indices], np.array(all_preds)[gap_indices]) if gap_indices.sum() > 0 else 0
    no_gap_acc = accuracy_score(np.array(all_labels)[no_gap_indices], np.array(all_preds)[no_gap_indices]) if no_gap_indices.sum() > 0 else 0
    
    return overall_acc, gap_acc, no_gap_acc, len(np.where(gap_indices)[0])

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def load_and_process_data(data_dir, max_files=100):
    """Load and process simulation data."""
    print(f"📂 Loading data from {data_dir}")
    
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("tracked_particles_3d_*.csv"))[:max_files]
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f"🔄 Processing {len(csv_files)} files...")
    
    graphs = []
    for i, csv_file in enumerate(tqdm(csv_files, desc="Creating graphs")):
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
                
            graph = build_gap_aware_graph(df, sim_id=i)
            graphs.append(graph)
        except Exception as e:
            print(f"⚠️  Skipping {csv_file.name}: {e}")
            continue
    
    print(f"✅ Created {len(graphs)} graphs")
    return graphs

def analyze_gap_distribution(graphs):
    """Analyze the distribution of detection gaps in the data."""
    total_edges = 0
    gap_edges = 0
    total_nodes = 0
    nodes_with_gaps = 0
    
    for graph in graphs:
        total_edges += graph.edge_index.shape[1]
        total_nodes += graph.x.shape[0]
        
        if hasattr(graph, 'gap_temporal_mask'):
            gap_edges += graph.gap_temporal_mask.sum().item()
            
            # Count nodes connected to gap edges
            if graph.gap_temporal_mask.sum() > 0:
                gap_edge_indices = graph.edge_index[:, graph.gap_temporal_mask]
                unique_gap_nodes = gap_edge_indices.flatten().unique()
                nodes_with_gaps += len(unique_gap_nodes)
    
    print(f"\n📊 Gap Analysis:")
    print(f"   Total edges: {total_edges:,}")
    print(f"   Gap-spanning edges: {gap_edges:,} ({gap_edges/total_edges*100:.1f}%)")
    print(f"   Total nodes: {total_nodes:,}")
    print(f"   Nodes with gaps: {nodes_with_gaps:,} ({nodes_with_gaps/total_nodes*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Test Gap-Aware GNN potential")
    parser.add_argument("--data_dir", type=str, default="../data/tracked_simdata_dirty",
                       help="Directory with simulation CSV files")
    parser.add_argument("--max_files", type=int, default=1000,
                       help="Maximum number of files to process")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--hidden_channels", type=int, default=128,
                       help="Hidden channels in models")
    
    args = parser.parse_args()
    
    print("🧪 GAP-AWARE GNN POTENTIAL TEST")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load and process data
    graphs = load_and_process_data(args.data_dir, args.max_files)
    
    if len(graphs) == 0:
        print("❌ No graphs created. Check your data directory.")
        return
    
    # Analyze gap distribution
    analyze_gap_distribution(graphs)
    
    # Normalize data
    normalizer = DataNormalizer()
    normalizer.fit(graphs)
    graphs = [normalizer.transform(g) for g in graphs]
    
    # Split data
    random.shuffle(graphs)
    split = int(0.8 * len(graphs))
    train_graphs = graphs[:split]
    val_graphs = graphs[split:]
    
    print(f"\n📚 Data split: {len(train_graphs)} train, {len(val_graphs)} val")
    
    # Create data loaders
    train_dataset = GraphDataset(train_graphs)
    val_dataset = GraphDataset(val_graphs)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
    
    # Test both models
    print(f"\n🤖 TESTING STANDARD GNN")
    print("-" * 30)
    standard_model = StandardGNN(args.hidden_channels).to(device)
    std_acc, std_gap_acc, std_no_gap_acc, gap_count = train_model(
        standard_model, train_loader, val_loader, args.epochs, device
    )
    
    print(f"\n🚀 TESTING GAP-AWARE GNN")
    print("-" * 30)
    gap_aware_model = GapAwareGNN(args.hidden_channels).to(device)
    gap_acc, gap_gap_acc, gap_no_gap_acc, _ = train_model(
        gap_aware_model, train_loader, val_loader, args.epochs, device
    )
    
    # Results comparison
    print(f"\n📊 RESULTS COMPARISON")
    print("=" * 50)
    print(f"📈 Overall Accuracy:")
    print(f"   Standard GNN:   {std_acc:.3f} ({std_acc*100:.1f}%)")
    print(f"   Gap-Aware GNN:  {gap_acc:.3f} ({gap_acc*100:.1f}%)")
    print(f"   Improvement:    {gap_acc-std_acc:+.3f} ({(gap_acc-std_acc)*100:+.1f}%)")
    
    print(f"\n🎯 Accuracy on Nodes WITHOUT Gaps:")
    print(f"   Standard GNN:   {std_no_gap_acc:.3f}")
    print(f"   Gap-Aware GNN:  {gap_no_gap_acc:.3f}")
    print(f"   Improvement:    {gap_no_gap_acc-std_no_gap_acc:+.3f}")
    
    if gap_count > 0:
        print(f"\n🔍 Accuracy on Nodes WITH Gaps ({gap_count} nodes):")
        print(f"   Standard GNN:   {std_gap_acc:.3f}")
        print(f"   Gap-Aware GNN:  {gap_gap_acc:.3f}")
        print(f"   Improvement:    {gap_gap_acc-std_gap_acc:+.3f}")
    else:
        print(f"\n⚠️  No nodes with detection gaps found in validation set")
    
    # Conclusion
    improvement = (gap_acc - std_acc) * 100
    print(f"\n🎯 CONCLUSION:")
    if improvement > 2:
        print(f"✅ Gap-aware GNN shows significant improvement (+{improvement:.1f}%)")
        print("   → Worth implementing in your full pipeline!")
    elif improvement > 0.5:
        print(f"🤔 Gap-aware GNN shows modest improvement (+{improvement:.1f}%)")
        print("   → Consider implementing if you need every % of accuracy")
    else:
        print(f"❌ Gap-aware GNN shows minimal improvement (+{improvement:.1f}%)")
        print("   → Focus on scaling up your current architecture instead")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Scale up on A5000: batch_size={args.batch_size*4}, hidden_channels={args.hidden_channels*2}")
    print(f"   2. Use more data: {args.max_files*10} files instead of {args.max_files}")
    print(f"   3. Train longer: {args.epochs*4} epochs instead of {args.epochs}")

if __name__ == "__main__":
    main()