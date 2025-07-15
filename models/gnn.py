import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool, MessagePassing
from torch_geometric.utils import softmax, degree
class ParticleGNN(torch.nn.Module):
    """
    Graph Neural Network for particle event classification with 3-class support.
    
    Architecture:
        - Three GATConv layers with batch normalization
        - Configurable hidden dimensions and dropout
        - Outputs probabilities for three event classes:
          * Class 0: Non-event (normal movement)
          * Class 1: Merge event (particles combining)
          * Class 2: Split event (particle dividing)
    
    Args:
        num_node_features (int): Number of input features per node
        hidden_channels (int, optional): Size of hidden representations. Defaults to 16.
        num_classes (int, optional): Number of output classes. Defaults to 3.
        dropout (float, optional): Dropout probability. Defaults to 0.3.
    """
    def __init__(self, num_node_features: int, hidden_channels: int = 16, 
                 num_classes: int = 1, dropout: float = 0.3):
        super(ParticleGNN, self).__init__()
        
        # Graph convolutional layers with edge attributes
        self.conv1 = GATConv(num_node_features, hidden_channels, edge_dim=1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = GATConv(hidden_channels, hidden_channels*2, edge_dim=1)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels*2)
        
        self.conv3 = GATConv(hidden_channels*2, hidden_channels, edge_dim=1)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Output layer for classification
        self.out = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Graph connectivity [2, num_edges]
            edge_attr (torch.Tensor): Edge features [num_edges, edge_features]
        
        Returns:
            torch.Tensor: Log probabilities for each class [num_nodes, num_classes]
        """
        # First GNN layer with edge attributes
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GNN layer with edge attributes
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third GNN layer with edge attributes
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        # Output layer
        x = self.out(x)
        
        return F.log_softmax(x, dim=1)  # Use log_softmax for numerical stability


def load_particle_gnn_model(model_path: str, num_features: int, 
                            hidden_size: int = 16, n_classes: int = 3):
    """
    Load a pre-trained ParticleGNN model from disk.
    
    Reconstructs the model architecture and loads weights from a checkpoint file.
    
    Args:
        model_path (str): Path to model checkpoint file
        num_features (int): Number of input features per node
        hidden_size (int, optional): Dimension of hidden representations. Defaults to 16.
        n_classes (int, optional): Number of output classes. Defaults to 3.
        
    Returns:
        torch.nn.Module: Loaded model ready for inference
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If loading weights fails
    
    Example:
        >>> model = load_particle_gnn_model(
        ...     "models/particle_model.pt", 
        ...     num_features=4,
        ...     n_classes=3
        ... )
        >>> model.eval()  # Set to evaluation mode
    """
    # Initialize empty model
    model = ParticleGNN(
        num_node_features=num_features,
        hidden_channels=hidden_size,
        num_classes=n_classes
    )
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load saved weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
        
    return model

class ParticleGNNBigger(torch.nn.Module):
    """
    Larger Graph Neural Network for particle event classification with 3-class support.
    Architecture:
    - Six GATConv layers with batch normalization (doubled from 3)
    - Larger hidden dimensions (64 base instead of 16)
    - Multi-head attention for better feature extraction
    - Outputs probabilities for three event classes:
      * Class 0: Non-event (normal movement)
      * Class 1: Merge event (particles combining)
      * Class 2: Split event (particle dividing)
    
    Args:
        num_node_features (int): Number of input features per node
        hidden_channels (int, optional): Size of hidden representations. Defaults to 64.
        num_classes (int, optional): Number of output classes. Defaults to 3.
        dropout (float, optional): Dropout probability. Defaults to 0.3.
        heads (int, optional): Number of attention heads. Defaults to 4.
    """
    
    def __init__(self, num_node_features: int, hidden_channels: int = 64,
                 num_classes: int = 3, dropout: float = 0.3, heads: int = 4):
        super(ParticleGNNBigger, self).__init__()
        
        # Graph convolutional layers with edge attributes and multi-head attention
        self.conv1 = GATConv(num_node_features, hidden_channels, edge_dim=1, heads=heads, concat=True)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)
        
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, edge_dim=1, heads=heads, concat=True)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * heads)
        
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels * 2, edge_dim=1, heads=heads, concat=True)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels * 2 * heads)
        
        self.conv4 = GATConv(hidden_channels * 2 * heads, hidden_channels * 2, edge_dim=1, heads=heads, concat=True)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels * 2 * heads)
        
        self.conv5 = GATConv(hidden_channels * 2 * heads, hidden_channels, edge_dim=1, heads=heads, concat=True)
        self.bn5 = torch.nn.BatchNorm1d(hidden_channels * heads)
        
        self.conv6 = GATConv(hidden_channels * heads, hidden_channels, edge_dim=1, heads=1, concat=False)
        self.bn6 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Larger MLP head for classification
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels * 2),
            torch.nn.BatchNorm1d(hidden_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Graph connectivity [2, num_edges]
            edge_attr (torch.Tensor): Edge features [num_edges, edge_features]
            
        Returns:
            torch.Tensor: Log probabilities for each class [num_nodes, num_classes]
        """
        
        # First GNN layer
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GNN layer
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third GNN layer
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Fourth GNN layer
        x = self.conv4(x, edge_index, edge_attr)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Fifth GNN layer
        x = self.conv5(x, edge_index, edge_attr)
        x = self.bn5(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Sixth GNN layer (single head for final layer)
        x = self.conv6(x, edge_index, edge_attr)
        x = self.bn6(x)
        x = F.relu(x)
        
        # MLP head for classification
        x = self.mlp(x)
        
        return F.log_softmax(x, dim=1)
    

class ParticleGNNBiggerWithResidual(torch.nn.Module):
    """
    Larger Graph Neural Network for particle event classification with residual connections.
    This version adds residual connections to the original architecture to help with
    gradient flow in the deeper network.
    
    Args:
        num_node_features (int): Number of input features per node
        hidden_channels (int, optional): Size of hidden representations. Defaults to 64.
        num_classes (int, optional): Number of output classes. Defaults to 3.
        dropout (float, optional): Dropout probability. Defaults to 0.3.
        heads (int, optional): Number of attention heads. Defaults to 4.
    """
    
    def __init__(self, num_node_features: int, hidden_channels: int = 64,
                 num_classes: int = 5, dropout: float = 0.3, heads: int = 4):
        super(ParticleGNNBiggerWithResidual, self).__init__()
        
        # Graph convolutional layers with edge attributes and multi-head attention
        self.conv1 = GATConv(num_node_features, hidden_channels, edge_dim=1, heads=heads, concat=True)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)
        
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, edge_dim=1, heads=heads, concat=True)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * heads)
        
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels * 2, edge_dim=1, heads=heads, concat=True)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels * 2 * heads)
        
        self.conv4 = GATConv(hidden_channels * 2 * heads, hidden_channels * 2, edge_dim=1, heads=heads, concat=True)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels * 2 * heads)
        
        self.conv5 = GATConv(hidden_channels * 2 * heads, hidden_channels, edge_dim=1, heads=heads, concat=True)
        self.bn5 = torch.nn.BatchNorm1d(hidden_channels * heads)
        
        self.conv6 = GATConv(hidden_channels * heads, hidden_channels, edge_dim=1, heads=1, concat=False)
        self.bn6 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Projection layers residual
        self.proj1 = torch.nn.Linear(num_node_features, hidden_channels * heads)
        self.proj2 = torch.nn.Linear(hidden_channels * heads, hidden_channels * heads)
        self.proj3 = torch.nn.Linear(hidden_channels * heads, hidden_channels * 2 * heads)
        self.proj4 = torch.nn.Linear(hidden_channels * 2 * heads, hidden_channels * 2 * heads)
        self.proj5 = torch.nn.Linear(hidden_channels * 2 * heads, hidden_channels * heads)
        self.proj6 = torch.nn.Linear(hidden_channels * heads, hidden_channels)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels * 2),
            torch.nn.BatchNorm1d(hidden_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass through the network with residual connections.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Graph connectivity [2, num_edges]
            edge_attr (torch.Tensor): Edge features [num_edges, edge_features]
            
        Returns:
            torch.Tensor: Log probabilities for each class [num_nodes, num_classes]
        """
        
        identity = self.proj1(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x + identity)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        identity = self.proj2(x)
        x_conv = self.conv2(x, edge_index, edge_attr)
        x_conv = self.bn2(x_conv)
        x = F.relu(x_conv + identity)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        identity = self.proj3(x)
        x_conv = self.conv3(x, edge_index, edge_attr)
        x_conv = self.bn3(x_conv)
        x = F.relu(x_conv + identity)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        identity = self.proj4(x)
        x_conv = self.conv4(x, edge_index, edge_attr)
        x_conv = self.bn4(x_conv)
        x = F.relu(x_conv + identity)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        identity = self.proj5(x)
        x_conv = self.conv5(x, edge_index, edge_attr)
        x_conv = self.bn5(x_conv)
        x = F.relu(x_conv + identity)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        identity = self.proj6(x)
        x_conv = self.conv6(x, edge_index, edge_attr)
        x_conv = self.bn6(x_conv)
        x = F.relu(x_conv + identity)
        
        x = self.mlp(x)
        
        return F.log_softmax(x, dim=1)
    


class ParticleGNNImproved(torch.nn.Module):
    """
    Improved Graph Neural Network for particle event classification.
    
    Key improvements:
    - Reduced depth (4 layers) to prevent over-smoothing
    - Consistent architecture with principled dimension scaling
    - LayerNorm instead of BatchNorm for better attention compatibility
    - Proper residual connections without dimension explosion
    - Consistent attention head strategy
    - More efficient dimension management
    
    Architecture:
    - Four GATConv layers with residual connections
    - Gradual dimension scaling: 64 -> 128 -> 128 -> 64
    - Consistent 4-head attention with mean aggregation
    - LayerNorm for better gradient flow
    - Outputs probabilities for three event classes:
      * Class 0: Non-event (normal movement)
      * Class 1: Merge event (particles combining)  
      * Class 2: Split event (particle dividing)
    
    Args:
        num_node_features (int): Number of input features per node
        hidden_channels (int, optional): Base hidden dimension. Defaults to 64.
        num_classes (int, optional): Number of output classes. Defaults to 3.
        dropout (float, optional): Dropout probability. Defaults to 0.2.
        heads (int, optional): Number of attention heads. Defaults to 4.
    """
    
    def __init__(self, num_node_features: int, hidden_channels: int = 64,
                 num_classes: int = 3, dropout: float = 0.2, heads: int = 4):
        super(ParticleGNNImproved, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        
        # Input projection to get to hidden dimension
        self.input_proj = torch.nn.Linear(num_node_features, hidden_channels)
        
        # Graph convolutional layers with consistent architecture
        # Using concat=False and mean aggregation to control dimensions
        self.conv1 = GATConv(hidden_channels, hidden_channels, edge_dim=1, 
                            heads=heads, concat=False, dropout=dropout)
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        
        self.conv2 = GATConv(hidden_channels, hidden_channels * 2, edge_dim=1, 
                            heads=heads, concat=False, dropout=dropout)
        self.norm2 = torch.nn.LayerNorm(hidden_channels * 2)
        self.proj2 = torch.nn.Linear(hidden_channels, hidden_channels * 2)  # For residual
        
        self.conv3 = GATConv(hidden_channels * 2, hidden_channels * 2, edge_dim=1, 
                            heads=heads, concat=False, dropout=dropout)
        self.norm3 = torch.nn.LayerNorm(hidden_channels * 2)
        
        self.conv4 = GATConv(hidden_channels * 2, hidden_channels, edge_dim=1, 
                            heads=heads, concat=False, dropout=dropout)
        self.norm4 = torch.nn.LayerNorm(hidden_channels)
        self.proj4 = torch.nn.Linear(hidden_channels * 2, hidden_channels)  # For residual
        
        # Simplified but effective MLP head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels * 2),
            torch.nn.LayerNorm(hidden_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass through the improved network.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Graph connectivity [2, num_edges]
            edge_attr (torch.Tensor): Edge features [num_edges, edge_features]
            
        Returns:
            torch.Tensor: Log probabilities for each class [num_nodes, num_classes]
        """
        
        # Project input to hidden dimension
        x = self.input_proj(x)
        x = F.relu(x)
        
        # First GNN layer with residual connection
        identity = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.relu(x + identity)  # Simple residual (same dimensions)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GNN layer with residual connection (dimension change)
        identity = self.proj2(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.relu(x + identity)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third GNN layer with residual connection
        identity = x
        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)
        x = F.relu(x + identity)  # Simple residual (same dimensions)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Fourth GNN layer with residual connection (dimension change)
        identity = self.proj4(x)
        x = self.conv4(x, edge_index, edge_attr)
        x = self.norm4(x)
        x = F.relu(x + identity)
        
        # Final classification
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)


class ParticleGNNImprovedLite(torch.nn.Module):
    """
    Lightweight version of the improved ParticleGNN for faster training/inference.
    
    This version uses only 3 layers and smaller dimensions while maintaining
    the architectural improvements.
    """
    
    def __init__(self, num_node_features: int, hidden_channels: int = 32,
                 num_classes: int = 3, dropout: float = 0.2, heads: int = 4):
        super(ParticleGNNImprovedLite, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        
        # Input projection
        self.input_proj = torch.nn.Linear(num_node_features, hidden_channels)
        
        # Three-layer architecture: 32 -> 64 -> 32
        self.conv1 = GATConv(hidden_channels, hidden_channels, edge_dim=1, 
                            heads=heads, concat=False, dropout=dropout)
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        
        self.conv2 = GATConv(hidden_channels, hidden_channels * 2, edge_dim=1, 
                            heads=heads, concat=False, dropout=dropout)
        self.norm2 = torch.nn.LayerNorm(hidden_channels * 2)
        self.proj2 = torch.nn.Linear(hidden_channels, hidden_channels * 2)
        
        self.conv3 = GATConv(hidden_channels * 2, hidden_channels, edge_dim=1, 
                            heads=heads, concat=False, dropout=dropout)
        self.norm3 = torch.nn.LayerNorm(hidden_channels)
        self.proj3 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        
        # Simple classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, x, edge_index, edge_attr):
        # Input projection
        x = F.relu(self.input_proj(x))
        
        # Layer 1
        identity = x
        x = F.relu(self.norm1(self.conv1(x, edge_index, edge_attr)) + identity)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        identity = self.proj2(x)
        x = F.relu(self.norm2(self.conv2(x, edge_index, edge_attr)) + identity)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3
        identity = self.proj3(x)
        x = F.relu(self.norm3(self.conv3(x, edge_index, edge_attr)) + identity)
        
        # Classification
        return F.log_softmax(self.classifier(x), dim=1)

class EdgeGATConv(MessagePassing):
    """Custom GAT layer that better processes edge features"""
    def __init__(self, in_channels, out_channels, edge_dim, heads=1):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim
        
        # Separate transformations for source and target nodes
        self.lin_src = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_edge = torch.nn.Linear(edge_dim, heads * out_channels, bias=False)
        
        # Attention mechanism
        self.att_src = torch.nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = torch.nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_edge = torch.nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        self.bias = torch.nn.Parameter(torch.Tensor(heads * out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin_src.weight)
        torch.nn.init.xavier_uniform_(self.lin_dst.weight)
        torch.nn.init.xavier_uniform_(self.lin_edge.weight)
        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)
        torch.nn.init.xavier_uniform_(self.att_edge)
        torch.nn.init.zeros_(self.bias)
        
    def forward(self, x, edge_index, edge_attr):
        # Transform features
        x_src = self.lin_src(x).view(-1, self.heads, self.out_channels)
        x_dst = self.lin_dst(x).view(-1, self.heads, self.out_channels)
        
        # Add self-loops with zero edge features for stability
        edge_index, edge_attr = self.add_self_loops(edge_index, edge_attr, x.size(0))
        
        # Propagate messages
        out = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr)
        out = out.view(-1, self.heads * self.out_channels)
        out = out + self.bias
        
        return out
    
    def message(self, x_i, x_j, edge_attr, index):
        # Process edge features
        edge_feat = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        
        # Compute attention scores
        alpha = (x_i * self.att_src + x_j * self.att_dst + edge_feat * self.att_edge).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index)
        
        # Apply attention to messages
        out = alpha.unsqueeze(-1) * (x_j + edge_feat)
        return out.view(-1, self.heads * self.out_channels)
    
    def add_self_loops(self, edge_index, edge_attr, num_nodes):
        # Add self-loops
        loop_index = torch.arange(num_nodes, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        
        # Add zero edge features for self-loops
        loop_attr = torch.zeros((num_nodes, edge_attr.size(1)), device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
        
        return edge_index, edge_attr


class ParticleGNNAdvanced(torch.nn.Module):
    """
    Advanced Graph Neural Network for particle event classification.
    
    Key improvements:
    - Skip connections for better gradient flow
    - Edge feature processing network
    - Mixed global pooling strategies
    - Layer normalization for stability
    - Gated activation functions
    - Hierarchical feature aggregation
    
    Architecture designed for efficiency on memory-constrained GPUs.
    """
    
    def __init__(self, num_node_features: int, hidden_channels: int = 48,
                 num_classes: int = 3, dropout: float = 0.2, heads: int = 4,
                 edge_hidden: int = 16):
        super(ParticleGNNAdvanced, self).__init__()
        
        # Edge feature processing network
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, edge_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_hidden, edge_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_hidden, heads)
        )
        
        # Initial projection
        self.input_proj = torch.nn.Linear(num_node_features, hidden_channels)
        self.input_norm = torch.nn.LayerNorm(hidden_channels)
        
        # GNN blocks with skip connections
        self.conv1 = GATConv(hidden_channels, hidden_channels, edge_dim=heads, heads=heads, concat=False)
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        
        self.conv2 = EdgeGATConv(hidden_channels, hidden_channels, edge_dim=heads, heads=heads)
        self.norm2 = torch.nn.LayerNorm(hidden_channels * heads)
        self.proj2 = torch.nn.Linear(hidden_channels * heads, hidden_channels)
        
        self.conv3 = GATConv(hidden_channels, hidden_channels * 2, edge_dim=heads, heads=heads, concat=False)
        self.norm3 = torch.nn.LayerNorm(hidden_channels * 2)
        
        self.conv4 = EdgeGATConv(hidden_channels * 2, hidden_channels, edge_dim=heads, heads=heads)
        self.norm4 = torch.nn.LayerNorm(hidden_channels * heads)
        self.proj4 = torch.nn.Linear(hidden_channels * heads, hidden_channels * 2)
        
        # Gating mechanism for skip connections
        self.gate1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.gate2 = torch.nn.Linear(hidden_channels * 3, hidden_channels * 2)
        
        # Global context attention
        self.global_att = torch.nn.MultiheadAttention(hidden_channels * 2, num_heads=4, batch_first=True)
        
        # Final classifier with hierarchical features
        classifier_input = hidden_channels * 2 * 3  # 3 different pooling strategies
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(classifier_input, hidden_channels * 2),
            torch.nn.LayerNorm(hidden_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass with hierarchical feature processing.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, 1]
            batch: Batch assignment for nodes (optional)
        """
        
        # Process edge features
        edge_features = self.edge_encoder(edge_attr)
        
        # Initial projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.relu(x)
        identity = x
        
        # Block 1 with skip connection
        x1 = self.conv1(x, edge_index, edge_features)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        # Block 2 with gated skip
        x2 = self.conv2(x1, edge_index, edge_features)
        x2 = self.norm2(x2)
        x2 = F.relu(x2)
        x2 = self.proj2(x2)
        
        # Gated skip connection
        gate = torch.sigmoid(self.gate1(torch.cat([x2, identity], dim=-1)))
        x2 = gate * x2 + (1 - gate) * identity
        
        # Block 3
        x3 = self.conv3(x2, edge_index, edge_features)
        x3 = self.norm3(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        identity2 = x3
        
        # Block 4 with gated skip
        x4 = self.conv4(x3, edge_index, edge_features)
        x4 = self.norm4(x4)
        x4 = F.relu(x4)
        x4 = self.proj4(x4)
        
        # Second gated skip
        gate2 = torch.sigmoid(self.gate2(torch.cat([x4, identity2], dim=-1)))
        x4 = gate2 * x4 + (1 - gate2) * identity2
        
        # Global context attention (self-attention across all nodes)
        if batch is not None:
            # Handle batched graphs
            x_att = x4.unsqueeze(0) if batch.max() == 0 else x4
            x_att, _ = self.global_att(x_att, x_att, x_att)
            x4 = x4 + 0.1 * x_att.squeeze(0)  # Small weight for global context
        
        # Hierarchical pooling (handles both single graphs and batches)
        if batch is None:
            # Single graph - pool all nodes
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Three different pooling strategies
        x_mean = global_mean_pool(x4, batch)
        x_max = global_max_pool(x4, batch)
        x_sum = global_add_pool(x4, batch)
        
        # Concatenate pooled features
        x_pooled = torch.cat([x_mean, x_max, x_sum], dim=-1)
        
        # Final classification
        out = self.classifier(x_pooled)
        
        # Return per-node predictions by broadcasting
        if batch.max() == 0:  # Single graph
            out = out.repeat(x.size(0), 1)
        else:  # Batched graphs
            out = out[batch]
        
        return F.log_softmax(out, dim=1)
    
###################### Physics informed model caution with conservation of mass things #################################
class PositionalEncoding(torch.nn.Module):
    """Learnable positional encoding based on graph structure"""
    def __init__(self, d_model, max_nodes=5000):
        super().__init__()
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(max_nodes, d_model)
        
    def forward(self, batch_size, num_nodes, device):
        positions = torch.arange(num_nodes, device=device).unsqueeze(0)
        return self.embedding(positions).squeeze(0)


class PhysicsInformedLayer(torch.nn.Module):
    """Layer that respects mass conservation principles"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mass_gate = torch.nn.Linear(hidden_dim + 1, hidden_dim)
        self.transform = torch.nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, mass):
        # Gate based on mass to maintain physical constraints
        gate = torch.sigmoid(self.mass_gate(torch.cat([x, mass.unsqueeze(-1)], dim=-1)))
        transformed = self.transform(x)
        return gate * transformed + (1 - gate) * x


class PSFEdgeEncoder(torch.nn.Module):
    """Enhanced edge encoder for PSF spatial relationships"""
    def __init__(self, edge_dim_in=1, edge_dim_out=32):
        super().__init__()
        # Expecting edge features to be distances or spatial relationships
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(edge_dim_in, 64),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, edge_dim_out)
        )
        
        # Learnable distance encoding
        self.dist_embedding = torch.nn.Parameter(torch.randn(1, edge_dim_out))
        
    def forward(self, edge_attr):
        # Process edge features (distances)
        encoded = self.encoder(edge_attr)
        
        # Add distance-based modulation
        dist_factor = torch.exp(-edge_attr / 10.0)  # Exponential decay
        encoded = encoded * dist_factor + self.dist_embedding
        
        return encoded


class MassAwareGATConv(MessagePassing):
    """GAT layer that explicitly considers mass in message passing"""
    def __init__(self, in_channels, out_channels, edge_dim, heads=1):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim
        
        # Transformations
        self.lin_src = torch.nn.Linear(in_channels + 1, heads * out_channels)  # +1 for mass
        self.lin_dst = torch.nn.Linear(in_channels + 1, heads * out_channels)  # +1 for mass
        self.lin_edge = torch.nn.Linear(edge_dim, heads * out_channels)
        
        # Attention parameters
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 3 * out_channels))  # 3 for src, dst, edge
        
        # Mass-based attention modulation
        self.mass_att = torch.nn.Linear(2, heads)  # 2 mass values -> attention per head
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin_src.weight)
        torch.nn.init.xavier_uniform_(self.lin_dst.weight)
        torch.nn.init.xavier_uniform_(self.lin_edge.weight)
        torch.nn.init.xavier_uniform_(self.att)
        torch.nn.init.xavier_uniform_(self.mass_att.weight)
        
    def forward(self, x, mass, edge_index, edge_attr):
        # Concatenate mass to features
        x_with_mass = torch.cat([x, mass.unsqueeze(-1)], dim=-1)
        
        # Transform
        x_src = self.lin_src(x_with_mass).view(-1, self.heads, self.out_channels)
        x_dst = self.lin_dst(x_with_mass).view(-1, self.heads, self.out_channels)
        edge_feat = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        
        # Propagate
        out = self.propagate(edge_index, x=(x_src, x_dst), mass=mass, edge_feat=edge_feat)
        return out.view(-1, self.heads * self.out_channels)
    
    def message(self, x_i, x_j, mass_i, mass_j, edge_feat, index):
        # Concatenate all features
        x = torch.cat([x_i, x_j, edge_feat], dim=-1)
        
        # Compute attention
        alpha = (x * self.att).sum(dim=-1)
        
        # Mass-based attention modulation
        mass_ratio = torch.cat([mass_i.unsqueeze(-1), mass_j.unsqueeze(-1)], dim=-1)
        mass_mod = self.mass_att(mass_ratio)
        alpha = alpha + mass_mod
        
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index)
        
        # Weight messages by attention and mass
        messages = x_j + edge_feat
        messages = messages * alpha.unsqueeze(-1)
        
        return messages.view(-1, self.heads * self.out_channels)


class PSFGNNAdvanced(torch.nn.Module):
    """
    Specialized GNN for Point Spread Function tracking with mass-only features.
    
    Key adaptations for PSF tracking:
    - Enhanced edge processing for spatial relationships
    - Structural encodings to compensate for minimal node features
    - Physics-informed layers for mass conservation
    - Specialized attention for split/merge detection
    - Node degree features for connectivity patterns
    
    Classes:
    - 0: Normal movement
    - 1: Merge event
    - 2: Split event
    """
    
    def __init__(self, hidden_channels: int = 64, num_classes: int = 3, 
                 dropout: float = 0.2, heads: int = 6, edge_hidden: int = 32):
        super(PSFGNNAdvanced, self).__init__()
        
        # Edge encoder - critical for spatial relationships
        self.edge_encoder = PSFEdgeEncoder(edge_dim_in=1, edge_dim_out=edge_hidden)
        
        # Initial feature construction from mass + structural features
        # We'll add: mass, degree, positional encoding
        self.positional_encoding = PositionalEncoding(hidden_channels // 2)
        self.degree_encoder = torch.nn.Linear(1, hidden_channels // 4)
        self.mass_encoder = torch.nn.Linear(1, hidden_channels // 4)
        
        # Combine all initial features
        self.input_proj = torch.nn.Linear(hidden_channels, hidden_channels)
        self.input_norm = torch.nn.LayerNorm(hidden_channels)
        
        # Mass-aware GNN layers
        self.conv1 = MassAwareGATConv(hidden_channels, hidden_channels, edge_hidden, heads=heads)
        self.norm1 = torch.nn.LayerNorm(hidden_channels * heads)
        self.proj1 = torch.nn.Linear(hidden_channels * heads, hidden_channels)
        
        # Physics-informed transformation
        self.physics1 = PhysicsInformedLayer(hidden_channels)
        
        self.conv2 = MassAwareGATConv(hidden_channels, hidden_channels * 2, edge_hidden, heads=heads)
        self.norm2 = torch.nn.LayerNorm(hidden_channels * 2 * heads)
        self.proj2 = torch.nn.Linear(hidden_channels * 2 * heads, hidden_channels * 2)
        
        self.physics2 = PhysicsInformedLayer(hidden_channels * 2)
        
        # Specialized layers for split/merge detection
        self.split_detector = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2 + 1, hidden_channels),  # +1 for mass
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, heads)
        )
        
        self.merge_detector = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2 + 1, hidden_channels),  # +1 for mass
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, heads)
        )
        
        # Final convolution with split/merge awareness
        self.conv3 = GATConv(hidden_channels * 2, hidden_channels, edge_dim=edge_hidden + heads * 2, 
                            heads=heads, concat=True)
        self.norm3 = torch.nn.LayerNorm(hidden_channels * heads)
        self.proj3 = torch.nn.Linear(hidden_channels * heads, hidden_channels)
        
        # Global graph features for context
        self.global_gate = torch.nn.Linear(hidden_channels * 3, hidden_channels)
        
        # Classifier with mass conservation check
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels + 1, hidden_channels),  # +1 for original mass
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.LayerNorm(hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, num_classes)
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass for PSF tracking.
        
        Args:
            x: Node features (mass only) [num_nodes, 1]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features (distances) [num_edges, 1]
            batch: Batch assignment for nodes
        """
        # Extract mass (the only feature)
        mass = x.squeeze(-1) if x.dim() > 1 else x
        
        # Compute node degrees (important for split/merge detection)
        degrees = degree(edge_index[0], x.size(0))
        
        # Build initial feature representation
        pos_encoding = self.positional_encoding(1, x.size(0), x.device)
        degree_features = self.degree_encoder(degrees.unsqueeze(-1))
        mass_features = self.mass_encoder(mass.unsqueeze(-1))
        
        # Combine all features
        h = torch.cat([mass_features, degree_features, pos_encoding], dim=-1)
        h = self.input_proj(h)
        h = self.input_norm(h)
        h = F.relu(h)
        
        # Process edge features (critical for spatial relationships)
        edge_features = self.edge_encoder(edge_attr)
        
        # Layer 1: Mass-aware convolution
        h1 = self.conv1(h, mass, edge_index, edge_features)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)
        h1 = self.proj1(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        
        # Apply physics constraints
        h1 = self.physics1(h1, mass)
        
        # Layer 2: Expanded features
        h2 = self.conv2(h1, mass, edge_index, edge_features)
        h2 = self.norm2(h2)
        h2 = F.relu(h2)
        h2 = self.proj2(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        
        h2 = self.physics2(h2, mass)
        
        # Detect split/merge patterns
        h2_mass = torch.cat([h2, mass.unsqueeze(-1)], dim=-1)
        split_attention = self.split_detector(h2_mass)
        merge_attention = self.merge_detector(h2_mass)
        
        # Enhanced edge features with split/merge information
        edge_split_merge = torch.cat([split_attention[edge_index[0]], 
                                     merge_attention[edge_index[1]]], dim=-1)
        enhanced_edge_features = torch.cat([edge_features, edge_split_merge], dim=-1)
        
        # Layer 3: Final convolution with split/merge awareness
        h3 = self.conv3(h2, edge_index, enhanced_edge_features)
        h3 = self.norm3(h3)
        h3 = F.relu(h3)
        h3 = self.proj3(h3)
        
        # Global pooling for graph-level features
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        h_mean = global_mean_pool(h3, batch)
        h_max = global_max_pool(h3, batch)
        h_sum = global_add_pool(h3, batch)
        
        # Gate based on global features
        h_global = torch.cat([h_mean, h_max, h_sum], dim=-1)
        gate = torch.sigmoid(self.global_gate(h_global))
        
        # Apply gating and add mass information for final classification
        h_final = h3 * gate[batch]
        h_final = torch.cat([h_final, mass.unsqueeze(-1)], dim=-1)
        
        # Classify
        out = self.classifier(h_final)
        
        return F.log_softmax(out, dim=1)
    

######## flourophore model? @@@@@@@@@@@@@@@@@


class PositionalEncoding(torch.nn.Module):
    """Learnable positional encoding based on graph structure"""
    def __init__(self, d_model, max_nodes=5000):
        super().__init__()
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(max_nodes, d_model)
        
    def forward(self, batch_size, num_nodes, device):
        positions = torch.arange(num_nodes, device=device).unsqueeze(0)
        return self.embedding(positions).squeeze(0)


class IntensityAdaptiveLayer(torch.nn.Module):
    """
    Layer that adapts to fluorescence intensity changes.
    
    Handles common fluorophore behaviors:
    - Photobleaching: gradual intensity decrease
    - Blinking: temporary on/off states
    - Environmental fluctuations: pH, oxygen, temperature effects
    - Focus drift: apparent intensity changes from z-movement
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Learn intensity change patterns (photobleaching, blinking)
        self.intensity_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim // 2),  # Current intensity + relative intensity
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.transform = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        
    def forward(self, x, intensity, mean_intensity):
        # Encode intensity information
        relative_intensity = intensity / (mean_intensity + 1e-6)
        intensity_features = self.intensity_encoder(
            torch.cat([intensity.unsqueeze(-1), relative_intensity.unsqueeze(-1)], dim=-1)
        )
        
        # Combine with hidden features
        combined = torch.cat([x, intensity_features], dim=-1)
        transformed = self.transform(combined)
        return self.norm(transformed + x)  # Residual connection


class SpatialEdgeEncoder(torch.nn.Module):
    """Enhanced edge encoder for spatial relationships between fluorophores"""
    def __init__(self, edge_dim_in=1, edge_dim_out=32):
        super().__init__()
        # Expecting edge features to be distances or spatial relationships
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(edge_dim_in, 64),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, edge_dim_out)
        )
        
        # Learnable distance encoding
        self.dist_embedding = torch.nn.Parameter(torch.randn(1, edge_dim_out))
        
    def forward(self, edge_attr):
        # Process edge features (distances)
        encoded = self.encoder(edge_attr)
        
        # Add distance-based modulation
        dist_factor = torch.exp(-edge_attr / 10.0)  # Exponential decay
        encoded = encoded * dist_factor + self.dist_embedding
        
        return encoded


class IntensityAwareGATConv(MessagePassing):
    """GAT layer that considers fluorescence intensity patterns"""
    def __init__(self, in_channels, out_channels, edge_dim, heads=1):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim
        
        # Transformations
        self.lin_src = torch.nn.Linear(in_channels + 1, heads * out_channels)  # +1 for intensity
        self.lin_dst = torch.nn.Linear(in_channels + 1, heads * out_channels)  # +1 for intensity
        self.lin_edge = torch.nn.Linear(edge_dim, heads * out_channels)
        
        # Attention parameters
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 3 * out_channels))  # 3 for src, dst, edge
        
        # Intensity-based attention modulation (no conservation assumption)
        self.intensity_att = torch.nn.Sequential(
            torch.nn.Linear(3, heads * 2),  # intensity_i, intensity_j, ratio
            torch.nn.ReLU(),
            torch.nn.Linear(heads * 2, heads)
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin_src.weight)
        torch.nn.init.xavier_uniform_(self.lin_dst.weight)
        torch.nn.init.xavier_uniform_(self.lin_edge.weight)
        torch.nn.init.xavier_uniform_(self.att)
        for layer in self.intensity_att:
            if hasattr(layer, 'weight'):
                torch.nn.init.xavier_uniform_(layer.weight)
        
    def forward(self, x, intensity, edge_index, edge_attr):
        # Concatenate intensity to features
        x_with_intensity = torch.cat([x, intensity.unsqueeze(-1)], dim=-1)
        
        # Transform
        x_src = self.lin_src(x_with_intensity).view(-1, self.heads, self.out_channels)
        x_dst = self.lin_dst(x_with_intensity).view(-1, self.heads, self.out_channels)
        edge_feat = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        
        # Propagate - pass intensity as tuple for source/destination indexing
        out = self.propagate(edge_index, x=(x_src, x_dst), intensity=(intensity, intensity), edge_feat=edge_feat)
        return out.view(-1, self.heads * self.out_channels)
    
    def message(self, x_i, x_j, intensity_i, intensity_j, edge_feat, index):
        # Ensure all tensors have correct dimensions
        if intensity_i.dim() == 1:
            intensity_i = intensity_i.unsqueeze(-1)
        if intensity_j.dim() == 1:
            intensity_j = intensity_j.unsqueeze(-1)
            
        # Concatenate all features
        x = torch.cat([x_i, x_j, edge_feat], dim=-1)
        
        # Compute attention
        alpha = (x * self.att).sum(dim=-1)
        
        # Intensity-based attention modulation (no conservation)
        intensity_ratio = intensity_j / (intensity_i + 1e-6)
        intensity_features = torch.cat([
            intensity_i, 
            intensity_j, 
            intensity_ratio
        ], dim=-1)
        intensity_mod = self.intensity_att(intensity_features)
        alpha = alpha + intensity_mod
        
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index)
        
        # Weight messages by attention
        messages = x_j + edge_feat
        messages = messages * alpha.unsqueeze(-1)
        
        return messages.view(-1, self.heads * self.out_channels)


class FluorophoreGNN(torch.nn.Module):
    """
    Specialized GNN for single-particle fluorophore tracking.
    
    Designed for tracking fluorophores where the only node feature is intensity.
    The model builds rich representations from this single feature combined with
    graph structure and spatial relationships (edge distances).
    
    Key features for fluorophore tracking:
    - Handles intensity fluctuations (photobleaching, blinking)
    - No mass conservation assumptions
    - Enhanced edge processing for spatial relationships
    - Structural encodings to compensate for minimal node features
    - Specialized attention for split/merge detection
    
    Classes:
    - 0: Normal movement
    - 1: Merge event (fluorophores coming together)
    - 2: Split event (fluorophore appearing to split)
    
    Input format:
    - x: Fluorescence intensity values (can be [num_nodes, 1] or [num_nodes])
    - edge_index: Graph connectivity [2, num_edges]
    - edge_attr: Distances between connected nodes (can be [num_edges, 1] or [num_edges])
    - batch: Optional batch assignment for multiple graphs
    """
    
    def __init__(self, hidden_channels: int = 64, num_classes: int = 3, 
                 dropout: float = 0.2, heads: int = 6, edge_hidden: int = 32):
        super(FluorophoreGNN, self).__init__()
        
        # Edge encoder - critical for spatial relationships
        self.edge_encoder = SpatialEdgeEncoder(edge_dim_in=1, edge_dim_out=edge_hidden)
        
        # Initial feature construction from intensity + structural features
        self.positional_encoding = PositionalEncoding(hidden_channels // 2)
        self.degree_encoder = torch.nn.Linear(1, hidden_channels // 4)
        self.intensity_encoder = torch.nn.Linear(1, hidden_channels // 4)
        
        # Combine all initial features
        self.input_proj = torch.nn.Linear(hidden_channels, hidden_channels)
        self.input_norm = torch.nn.LayerNorm(hidden_channels)
        
        # Intensity-aware GNN layers
        self.conv1 = IntensityAwareGATConv(hidden_channels, hidden_channels, edge_hidden, heads=heads)
        self.norm1 = torch.nn.LayerNorm(hidden_channels * heads)
        self.proj1 = torch.nn.Linear(hidden_channels * heads, hidden_channels)
        
        # Intensity adaptive transformation
        self.adapt1 = IntensityAdaptiveLayer(hidden_channels)
        
        self.conv2 = IntensityAwareGATConv(hidden_channels, hidden_channels * 2, edge_hidden, heads=heads)
        self.norm2 = torch.nn.LayerNorm(hidden_channels * 2 * heads)
        self.proj2 = torch.nn.Linear(hidden_channels * 2 * heads, hidden_channels * 2)
        
        self.adapt2 = IntensityAdaptiveLayer(hidden_channels * 2)
        
        # Specialized layers for split/merge detection
        self.split_detector = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2 + 1, hidden_channels),  # +1 for intensity
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, heads)
        )
        
        self.merge_detector = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2 + 1, hidden_channels),  # +1 for intensity
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, heads)
        )
        
        # Final convolution with split/merge awareness
        self.conv3 = GATConv(hidden_channels * 2, hidden_channels, edge_dim=edge_hidden + heads * 2, 
                            heads=heads, concat=True)
        self.norm3 = torch.nn.LayerNorm(hidden_channels * heads)
        self.proj3 = torch.nn.Linear(hidden_channels * heads, hidden_channels)
        
        # Global graph features for context
        self.global_gate = torch.nn.Linear(hidden_channels * 3, hidden_channels)
        
        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels + 1, hidden_channels),  # +1 for original intensity
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.LayerNorm(hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, num_classes)
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass for fluorophore tracking.
        
        Args:
            x: Node features (intensity only) [num_nodes, 1] or [num_nodes]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features (distances) [num_edges, 1] or [num_edges]
            batch: Batch assignment for nodes (optional)
            
        Note: When using PyTorch Geometric's DataLoader, batch.x might be 1D or 2D
              depending on your data preprocessing. This method handles both cases.
        """
        # Handle different input formats
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            
        # Extract intensity
        intensity = x.squeeze(-1) if x.size(-1) == 1 else x[:, 0]
        
        # Ensure intensity is 1D
        if intensity.dim() > 1:
            intensity = intensity.view(-1)
            
        mean_intensity = intensity.mean()
        
        # Compute node degrees (important for split/merge detection)
        degrees = degree(edge_index[0], x.size(0))
        
        # Build initial feature representation
        pos_encoding = self.positional_encoding(1, x.size(0), x.device)
        degree_features = self.degree_encoder(degrees.unsqueeze(-1))
        intensity_features = self.intensity_encoder(intensity.unsqueeze(-1))
        
        # Combine all features
        h = torch.cat([intensity_features, degree_features, pos_encoding], dim=-1)
        h = self.input_proj(h)
        h = self.input_norm(h)
        h = F.relu(h)
        
        # Process edge features (critical for spatial relationships)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        edge_features = self.edge_encoder(edge_attr)
        
        # Layer 1: Intensity-aware convolution
        h1 = self.conv1(h, intensity, edge_index, edge_features)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)
        h1 = self.proj1(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        
        # Adapt to intensity patterns
        h1 = self.adapt1(h1, intensity, mean_intensity)
        
        # Layer 2: Expanded features
        h2 = self.conv2(h1, intensity, edge_index, edge_features)
        h2 = self.norm2(h2)
        h2 = F.relu(h2)
        h2 = self.proj2(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        
        h2 = self.adapt2(h2, intensity, mean_intensity)
        
        # Detect split/merge patterns
        h2_intensity = torch.cat([h2, intensity.unsqueeze(-1)], dim=-1)
        split_attention = self.split_detector(h2_intensity)
        merge_attention = self.merge_detector(h2_intensity)
        
        # Enhanced edge features with split/merge information
        edge_split_merge = torch.cat([split_attention[edge_index[0]], 
                                     merge_attention[edge_index[1]]], dim=-1)
        enhanced_edge_features = torch.cat([edge_features, edge_split_merge], dim=-1)
        
        # Layer 3: Final convolution with split/merge awareness
        h3 = self.conv3(h2, edge_index, enhanced_edge_features)
        h3 = self.norm3(h3)
        h3 = F.relu(h3)
        h3 = self.proj3(h3)
        
        # Global pooling for graph-level features
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # Ensure batch is on the same device as the features
        batch = batch.to(h3.device)
            
        h_mean = global_mean_pool(h3, batch)
        h_max = global_max_pool(h3, batch)
        h_sum = global_add_pool(h3, batch)
        
        # Gate based on global features
        h_global = torch.cat([h_mean, h_max, h_sum], dim=-1)
        gate = torch.sigmoid(self.global_gate(h_global))
        
        # Apply gating and add intensity information for final classification
        h_final = h3 * gate[batch]
        h_final = torch.cat([h_final, intensity.unsqueeze(-1)], dim=-1)
        
        # Classify
        out = self.classifier(h_final)
        
        return F.log_softmax(out, dim=1)