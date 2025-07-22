import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from training.compute_class_wheights import *


def validate_model(
    model: torch.nn.Module, 
    val_loader: DataLoader, 
    criterion: torch.nn.Module, 
    device: torch.device
) -> Tuple[float, float, Dict[int, float]]:
    """
    Validate a graph neural network model on a validation dataset.
    
    Computes loss and accuracy metrics across the entire validation set, with support
    for multi-class classification (specifically the 5-class particle event system).
    
    Args:
        model (torch.nn.Module): The GNN model to validate
        val_loader (DataLoader): DataLoader containing validation graphs
        criterion (torch.nn.Module): Loss function (typically NLLLoss with class weights)
        device (torch.device): Device to run validation on (CPU or GPU)
    
    Returns:
        Tuple[float, float, Dict[int, float]]: Contains:
            - val_loss (float): Average validation loss
            - val_acc (float): Overall validation accuracy
            - class_acc (Dict[int, float]): Per-class accuracy for each event type:
                - 0: Non-event accuracy
                - 1: Merge event accuracy
                - 2: Split event accuracy
                - 3: Post merge event accuracy
                - 4: Post split event accuracy
    
    Example:
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> model = load_model("model.pt").to(device)
        >>> val_loss, val_acc, class_acc = validate_model(model, val_loader, criterion, device)
        >>> print(f"Validation accuracy: {val_acc:.4f}")
        >>> print(f"Class accuracies: Non-event: {class_acc[0]:.4f}, Merge: {class_acc[1]:.4f}, Split: {class_acc[2]:.4f}")
    """
    model.eval()
    total_loss = 0
    correct = 0
    
    # Track per-class prediction accuracy
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
            
            # Track per-class accuracy
            for class_idx in torch.unique(batch.y):
                class_idx = class_idx.item()
                if class_idx not in class_correct:
                    class_correct[class_idx] = 0
                    class_total[class_idx] = 0
                
                class_mask = (batch.y == class_idx)
                class_correct[class_idx] += int((pred[class_mask] == class_idx).sum())
                class_total[class_idx] += int(class_mask.sum())
    
    # Calculate metrics
    num_samples = sum(data.num_nodes for data in val_loader.dataset)
    val_loss = total_loss / len(val_loader.dataset)
    val_acc = correct / num_samples
    
    # Calculate per-class accuracy - updated for 5 classes
    class_acc = {cls: (class_correct.get(cls, 0) / class_total.get(cls, 1)) 
                 for cls in range(5)}  # Ensure all 5 classes are represented
    
    return val_loss, val_acc, class_acc



def train_model(
    model: torch.nn.Module, 
    train_loader: DataLoader, 
    val_loader: Optional[DataLoader] = None, 
    epochs: int = 100, 
    lr: float = 0.01, 
    weight_decay: float = 5e-4, 
    patience: int = 10
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Train a GNN model for 5-class particle event classification with validation and early stopping.
    
    Implements a complete training pipeline with:
    - Class-weighted loss to handle imbalanced data
    - Learning rate scheduling for improved convergence
    - Early stopping to prevent overfitting
    - Detailed tracking of training and validation metrics
    
    The model is trained to classify particle events into five classes:
    - Class 0: Non-event (normal particle movement)
    - Class 1: Merge event (particles combining)
    - Class 2: Split event (particle dividing)
    - Class 3: Post merge event (after particles have merged)
    - Class 4: Post split event (after particle has split)
    
    Args:
        model (torch.nn.Module): The GNN model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (Optional[DataLoader]): DataLoader for validation data. Defaults to None.
        epochs (int): Maximum number of training epochs. Defaults to 100.
        lr (float): Initial learning rate. Defaults to 0.01.
        weight_decay (float): Weight decay for regularization. Defaults to 5e-4.
        patience (int): Patience for early stopping. Defaults to 10.
        
    Returns:
        Tuple[torch.nn.Module, Dict[str, List[float]]]: Contains:
            - Trained model with best validation performance
            - Training history dictionary with keys:
                - 'train_loss': List of training losses per epoch
                - 'val_loss': List of validation losses per epoch
                - 'val_acc': List of validation accuracies per epoch
                - 'class_acc': List of dictionaries with per-class accuracies
    
    Example:
        >>> model = ParticleGNN(num_node_features=4, num_classes=5)
        >>> model, history = train_model(
        ...     model, 
        ...     train_loader, 
        ...     val_loader, 
        ...     epochs=100,
        ...     lr=0.005,
        ...     patience=15
        ... )
        >>> # Plotting training progress
        >>> plt.figure(figsize=(12, 4))
        >>> plt.subplot(1, 2, 1)
        >>> plt.plot(history['train_loss'], label='Train Loss')
        >>> plt.plot(history['val_loss'], label='Validation Loss')
        >>> plt.legend()
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weights_harsh(train_loader)
    class_weights = class_weights.to(device)
    criterion = torch.nn.NLLLoss(weight=class_weights)  # Weighted loss
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_acc': [],
        'class_acc': []  # List of dictionaries with per-class accuracies
    }
    
    print(f"Training on device: {device}")
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.edge_attr)

            # out = model(batch.x, batch.edge_index, batch.edge_attr, 
            #        batch.edge_type, batch.batch,
            #        batch.direct_temporal_mask,
            #        batch.gap_temporal_mask, 
            #        batch.proximity_mask)

            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase (if provided)
        if val_loader:
            val_loss, val_acc, class_acc = validate_model(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['class_acc'].append(class_acc)
            
            # Print progress at regular intervals
            if epoch % 5 == 0 or epoch == epochs - 1:
                weighted_acc = sum(class_acc[i] * class_weights[i] for i in range(5)) / sum(class_weights)

                print(f'Epoch {epoch:03d}: Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                print(f'  Class Acc: Non-event: {class_acc[0]:.4f}, '
                      f'Merge: {class_acc[1]:.4f}, Split: {class_acc[2]:.4f}, '
                      f'Post-merge: {class_acc[3]:.4f}, Post-split: {class_acc[4]:.4f}')
                print(f'  Weighted Acc: {weighted_acc:.4f}')
            
            # Learning rate scheduling based on validation loss
            scheduler.step(val_loss)
            
            # Early stopping logic
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
            # If no validation set is provided, just report training loss
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch:03d}: Train Loss: {avg_train_loss:.4f}')
    
    # Restore best model if validation was used
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
    
    Creates a comprehensive visualization of training metrics with three panels:
    1. Loss curves (training and validation)
    2. Overall validation accuracy
    3. Per-class validation accuracy
    
    Args:
        history (Dict[str, List[float]]): Training history from train_model
        figsize (Tuple[int, int], optional): Figure dimensions. Defaults to (18, 5).
    
    Example:
        >>> model, history = train_model(model, train_loader, val_loader)
        >>> plot_training_history(history)
        >>> plt.savefig('training_progress.png', dpi=300)
    """
    plt.figure(figsize=figsize)
    
    # Plot loss curves
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot overall accuracy
    if 'val_acc' in history:
        plt.subplot(1, 3, 2)
        plt.plot(history['val_acc'], label='Validation Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Overall Validation Accuracy')
        plt.grid(alpha=0.3)
    
    # Plot per-class accuracy - updated for 5 classes
    if 'class_acc' in history and history['class_acc']:
        plt.subplot(1, 3, 3)
        epochs = range(len(history['class_acc']))
        
        class0_acc = [epoch_acc[0] for epoch_acc in history['class_acc']]
        class1_acc = [epoch_acc[1] for epoch_acc in history['class_acc']]
        class2_acc = [epoch_acc[2] for epoch_acc in history['class_acc']]
        class3_acc = [epoch_acc[3] for epoch_acc in history['class_acc']]  # Post merge
        class4_acc = [epoch_acc[4] for epoch_acc in history['class_acc']]  # Post split
        
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