import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, TransformerConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

class GapAwareParticleGNN(torch.nn.Module):
    """
    Advanced GNN optimized for A5000 with gap-aware processing for particle tracking.
    
    Key improvements:
    - Separate processing for different edge types (temporal, gap-spanning, proximity)
    - Efficient residual connections without projection layers
    - Detection gap attention mechanism
    - Larger capacity optimized for A5000
    - Mixed precision training ready
    """
    
    def __init__(self, 
                 num_node_features: int,
                 hidden_channels: int = 256,  # 4x bigger for A5000
                 num_classes: int = 5,
                 dropout: float = 0.2,  # Lower dropout with more data
                 heads: int = 8,  # More heads for A5000
                 num_layers: int = 8,  # Deeper network
                 edge_dim: int = 3):  # Enhanced edge features
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        
        # Input projection
        self.input_proj = nn.Linear(num_node_features, hidden_channels)
        
        # Separate convolutions for different edge types
        self.temporal_convs = nn.ModuleList()
        self.gap_convs = nn.ModuleList()
        self.proximity_convs = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Use different conv types for variety
            if i < num_layers // 2:
                # First half: GAT for attention
                temp_conv = GATConv(hidden_channels, hidden_channels // heads, 
                                  heads=heads, edge_dim=edge_dim, concat=True)
                gap_conv = GATConv(hidden_channels, hidden_channels // heads,
                                 heads=heads, edge_dim=edge_dim, concat=True)
                prox_conv = GATConv(hidden_channels, hidden_channels // heads,
                                  heads=heads, edge_dim=edge_dim, concat=True)
            else:
                # Second half: Transformer convs for long-range dependencies
                temp_conv = TransformerConv(hidden_channels, hidden_channels // heads,
                                          heads=heads, edge_dim=edge_dim, concat=True)
                gap_conv = TransformerConv(hidden_channels, hidden_channels // heads,
                                         heads=heads, edge_dim=edge_dim, concat=True)
                prox_conv = TransformerConv(hidden_channels, hidden_channels // heads,
                                          heads=heads, edge_dim=edge_dim, concat=True)
            
            self.temporal_convs.append(temp_conv)
            self.gap_convs.append(gap_conv)
            self.proximity_convs.append(prox_conv)
            
            # Fusion layer to combine different edge type outputs
            self.fusion_layers.append(nn.Sequential(
                nn.Linear(hidden_channels * 3, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            
            self.layer_norms.append(nn.LayerNorm(hidden_channels))
        
        # Gap-aware attention mechanism
        self.gap_attention = GapAwareAttention(hidden_channels, heads)
        
        # Classification head with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr, edge_type, batch=None, 
                direct_temporal_mask=None, gap_temporal_mask=None, proximity_mask=None):
        """
        Enhanced forward pass with gap-aware processing.
        """
        # Input projection
        x = self.input_proj(x)
        
        # Separate edge indices for different types
        if direct_temporal_mask is not None:
            temporal_edge_index = edge_index[:, direct_temporal_mask]
            temporal_edge_attr = edge_attr[direct_temporal_mask]
            
            gap_edge_index = edge_index[:, gap_temporal_mask] 
            gap_edge_attr = edge_attr[gap_temporal_mask]
            
            proximity_edge_index = edge_index[:, proximity_mask]
            proximity_edge_attr = edge_attr[proximity_mask]
        else:
            # Fallback: separate by edge_type
            temporal_mask = edge_type == 0
            gap_mask = edge_type == 2
            prox_mask = edge_type == 1
            
            temporal_edge_index = edge_index[:, temporal_mask]
            temporal_edge_attr = edge_attr[temporal_mask]
            gap_edge_index = edge_index[:, gap_mask]
            gap_edge_attr = edge_attr[gap_mask]
            proximity_edge_index = edge_index[:, prox_mask]
            proximity_edge_attr = edge_attr[prox_mask]
        
        # Layer-wise processing
        for i in range(self.num_layers):
            residual = x
            
            # Process each edge type separately
            h_temporal = h_gap = h_proximity = torch.zeros_like(x)
            
            if temporal_edge_index.size(1) > 0:
                h_temporal = self.temporal_convs[i](x, temporal_edge_index, temporal_edge_attr)
            
            if gap_edge_index.size(1) > 0:
                h_gap = self.gap_convs[i](x, gap_edge_index, gap_edge_attr)
                
            if proximity_edge_index.size(1) > 0:
                h_proximity = self.proximity_convs[i](x, proximity_edge_index, proximity_edge_attr)
            
            # Concatenate and fuse different edge type outputs
            h_combined = torch.cat([h_temporal, h_gap, h_proximity], dim=-1)
            h_fused = self.fusion_layers[i](h_combined)
            
            # Residual connection and normalization
            x = self.layer_norms[i](h_fused + residual)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Gap-aware attention for final representation
        x = self.gap_attention(x, gap_edge_index, batch)
        
        # Classification
        return F.log_softmax(self.classifier(x), dim=-1)

class GapAwareAttention(nn.Module):
    """
    Attention mechanism that specifically handles detection gaps.
    """
    def __init__(self, hidden_channels, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_channels // heads
        
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, hidden_channels)
        
    def forward(self, x, gap_edge_index, batch=None):
        if gap_edge_index.size(1) == 0:
            return x  # No gaps to attend to
            
        # Multi-head attention over gap connections
        q = self.q_proj(x).view(-1, self.heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.heads, self.head_dim)
        
        # Attention over gap edges
        src, dst = gap_edge_index
        attention_scores = (q[dst] * k[src]).sum(dim=-1) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # Apply attention
        attended = torch.zeros_like(v)
        for i, (s, d) in enumerate(gap_edge_index.t()):
            attended[d] += attention_weights[i].unsqueeze(-1) * v[s]
        
        attended = attended.view(-1, self.heads * self.head_dim)
        return x + self.out_proj(attended)

class ScaledUpOriginalGNN(torch.nn.Module):
    """
    Your original architecture scaled up for A5000 (fallback option).
    """
    def __init__(self, num_node_features: int, hidden_channels: int = 256,
                 num_classes: int = 5, dropout: float = 0.2, heads: int = 8):
        super().__init__()
        
        # Scaled up version of your original model
        self.conv1 = GATConv(num_node_features, hidden_channels, edge_dim=3, heads=heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, edge_dim=3, heads=heads, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_channels * heads)
        
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels * 2, edge_dim=3, heads=heads, concat=True)
        self.bn3 = nn.BatchNorm1d(hidden_channels * 2 * heads)
        
        self.conv4 = GATConv(hidden_channels * 2 * heads, hidden_channels * 2, edge_dim=3, heads=heads, concat=True)
        self.bn4 = nn.BatchNorm1d(hidden_channels * 2 * heads)
        
        self.conv5 = GATConv(hidden_channels * 2 * heads, hidden_channels, edge_dim=3, heads=heads, concat=True)
        self.bn5 = nn.BatchNorm1d(hidden_channels * heads)
        
        self.conv6 = GATConv(hidden_channels * heads, hidden_channels, edge_dim=3, heads=1, concat=False)
        self.bn6 = nn.BatchNorm1d(hidden_channels)
        
        # More efficient residual connections
        self.residual_layers = nn.ModuleList([
            nn.Linear(num_node_features, hidden_channels * heads),
            nn.Identity(),  # Same dimension
            nn.Linear(hidden_channels * heads, hidden_channels * 2 * heads),
            nn.Identity(),  # Same dimension  
            nn.Linear(hidden_channels * 2 * heads, hidden_channels * heads),
            nn.Linear(hidden_channels * heads, hidden_channels)
        ])
        
        # Bigger classifier
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr):
        layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]
        bns = [self.bn1, self.bn2, self.bn3, self.bn4, self.bn5, self.bn6]
        
        for i, (conv, bn, residual) in enumerate(zip(layers, bns, self.residual_layers)):
            identity = residual(x)
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x + identity)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return F.log_softmax(self.mlp(x), dim=1)

# Training configuration for A5000
def get_a5000_training_config():
    return {
        'batch_size': 32,  # 4-8x larger than RTX 2070
        'hidden_channels': 256,  # 4x larger
        'heads': 8,  # 2x more attention heads
        'num_layers': 8,  # Deeper network
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'use_amp': True,  # Mixed precision training
        'gradient_clip': 1.0
    }

if __name__ == "__main__":
    # Model comparison
    config = get_a5000_training_config()
    
    # Option 1: Gap-aware model (recommended)
    gap_aware_model = GapAwareParticleGNN(
        num_node_features=1,
        hidden_channels=config['hidden_channels'],
        heads=config['heads'],
        num_layers=config['num_layers']
    )
    
    # Option 2: Scaled up original
    scaled_original = ScaledUpOriginalGNN(
        num_node_features=1,
        hidden_channels=config['hidden_channels'],
        heads=config['heads']
    )
    
    print(f"Gap-aware model parameters: {sum(p.numel() for p in gap_aware_model.parameters()):,}")
    print(f"Scaled original parameters: {sum(p.numel() for p in scaled_original.parameters()):,}")