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
    
# Just to recap. I will not proceed with the gap aware GNN for now. I will be using an version of my old model attached. what should I be changing on my model to take advantage of the a5000? Just param or actual architecture 

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
    