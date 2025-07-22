import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from training.compute_class_wheights import compute_class_weights_harsh

class FocalLoss(torch.nn.Module):
    """
    Focal Loss implementation for addressing class imbalance.
    
    Focal Loss down-weights easy examples and focuses learning on hard examples.
    Particularly effective for extreme class imbalance scenarios.
    
    Args:
        alpha: Class weights tensor or float. If None, no weighting applied.
        gamma: Focusing parameter. Higher gamma reduces relative loss for well-classified examples.
        reduction: Specifies the reduction to apply to the output.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (raw logits or log probabilities)
            targets: Ground truth class indices
        """
        # Convert log probabilities to probabilities if needed
        if inputs.dim() > 1 and inputs.size(1) > 1:
            # Assume inputs are log probabilities from log_softmax
            log_pt = inputs
            pt = torch.exp(log_pt)
        else:
            # Handle case where inputs might be raw logits
            log_pt = F.log_softmax(inputs, dim=1)
            pt = torch.exp(log_pt)
        
        # Gather log probabilities for target classes
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight
        
        # Compute focal loss
        focal_loss = -focal_weight * log_pt
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def validate_model(
    model: torch.nn.Module, 
    val_loader: DataLoader, 
    criterion: torch.nn.Module, 
    device: torch.device
) -> Tuple[float, float, Dict[int, float]]:
    """
    Validate a graph neural network model on a validation dataset.
    """
    model.eval()
    total_loss = 0
    correct = 0
    
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            
            for class_idx in torch.unique(batch.y):
                class_idx = class_idx.item()
                if class_idx not in class_correct:
                    class_correct[class_idx] = 0
                    class_total[class_idx] = 0
                
                class_mask = (batch.y == class_idx)
                class_correct[class_idx] += int((pred[class_mask] == class_idx).sum())
                class_total[class_idx] += int(class_mask.sum())
    
    num_samples = sum(data.num_nodes for data in val_loader.dataset)
    val_loss = total_loss / len(val_loader.dataset)
    val_acc = correct / num_samples
    
    class_acc = {cls: (class_correct.get(cls, 0) / class_total.get(cls, 1)) 
                 for cls in range(5)}
    
    return val_loss, val_acc, class_acc


def compute_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequency.
    """
    y_train = torch.cat([data.y for data in train_loader.dataset])
    class_counts = torch.bincount(y_train)
    print("Class counts:", class_counts)
    
    if len(class_counts) < 5:
        padded_counts = torch.zeros(5, dtype=torch.float)
        padded_counts[:len(class_counts)] = class_counts
        class_counts = padded_counts
    
    total_samples = len(y_train)
    num_classes = 5
    
    class_weights = total_samples / (num_classes * class_counts.float())
    class_weights = torch.clamp(class_weights, min=0.1, max=200.0)
    
    return class_weights


def train_model_focal(
    model: torch.nn.Module, 
    train_loader: DataLoader, 
    val_loader: Optional[DataLoader] = None, 
    epochs: int = 100, 
    lr: float = 0.01, 
    weight_decay: float = 5e-4, 
    patience: int = 10,
    focal_gamma: float = 3.0,
    beta: float = 0.99999999999
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Train a GNN model using Focal Loss for extreme class imbalance.
    
    Args:
        model: The GNN model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Maximum number of training epochs
        lr: Initial learning rate
        weight_decay: Weight decay for regularization
        patience: Patience for early stopping
        focal_gamma: Focusing parameter for Focal Loss (higher = more focus on hard examples)
        beta: Controls effective number weighting strength (higher = more focus on minorities)
        
    Returns:
        Trained model and training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Compute class weights using effective number of samples and create Focal Loss
    class_weights = compute_class_weights_harsh(train_loader)
    class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma, reduction='mean')
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_acc': [],
        'class_acc': []
    }
    
    print(f"Training on device: {device}")
    print(f"Effective class weights: {class_weights.cpu().numpy()}")
    print(f"Focal Loss gamma: {focal_gamma}")
    print(f"Beta parameter: {beta}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if val_loader:
            val_loss, val_acc, class_acc = validate_model(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['class_acc'].append(class_acc)
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                weighted_acc = sum(class_acc[i] * class_weights[i] for i in range(5)) / sum(class_weights)

                print(f'Epoch {epoch:03d}: Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                print(f'  Class Acc: Non-event: {class_acc[0]:.4f}, '
                      f'Merge: {class_acc[1]:.4f}, Split: {class_acc[2]:.4f}, '
                      f'Post-merge: {class_acc[3]:.4f}, Post-split: {class_acc[4]:.4f}')
                print(f'  Weighted Acc: {weighted_acc:.4f}')
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
        else:
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch:03d}: Train Loss: {avg_train_loss:.4f}')
    
    if val_loader and best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation loss: {best_val_loss:.4f}")
    
    return model, history

def plot_training_history(
    history: Dict[str, List[float]], 
    figsize: Tuple[int, int] = (18, 5)
) -> None:
    """
    Visualize training progress for a 5-class particle event classification model.
    """
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if 'val_acc' in history:
        plt.subplot(1, 3, 2)
        plt.plot(history['val_acc'], label='Validation Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Overall Validation Accuracy')
        plt.grid(alpha=0.3)
    
    if 'class_acc' in history and history['class_acc']:
        plt.subplot(1, 3, 3)
        epochs = range(len(history['class_acc']))
        
        class0_acc = [epoch_acc[0] for epoch_acc in history['class_acc']]
        class1_acc = [epoch_acc[1] for epoch_acc in history['class_acc']]
        class2_acc = [epoch_acc[2] for epoch_acc in history['class_acc']]
        class3_acc = [epoch_acc[3] for epoch_acc in history['class_acc']]
        class4_acc = [epoch_acc[4] for epoch_acc in history['class_acc']]
        
        plt.plot(epochs, class0_acc, label='Non-event', color='blue')
        plt.plot(epochs, class1_acc, label='Merge', color='red')
        plt.plot(epochs, class2_acc, label='Split', color='green')
        plt.plot(epochs, class3_acc, label='Post merge', color='orange')
        plt.plot(epochs, class4_acc, label='Post split', color='purple')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Validation Accuracy')
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Usage example to replace your existing train_model call:
"""
# In your main training script, replace:
# trained_model, history = train_model(...)

# With:
trained_model, history = train_model_focal(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=config.epochs,
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
    patience=config.patience,
    focal_gamma=2.0  # Start with 2.0, try 3.0-5.0 if needed
)
"""