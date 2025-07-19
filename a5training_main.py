import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch.cuda.amp import autocast, GradScaler
import time


def validate_model(
    model: torch.nn.Module, 
    val_loader: DataLoader, 
    criterion: torch.nn.Module, 
    device: torch.device,
    use_amp: bool = True
) -> Tuple[float, float, Dict[int, float]]:
    """
    Validate a graph neural network model with A5000 optimizations.
    
    Enhanced with mixed precision support for faster validation on A5000.
    """
    model.eval()
    total_loss = 0
    correct = 0
    
    # Track per-class prediction accuracy
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device, non_blocking=True)
            
            if use_amp:
                with autocast():
                    # Handle both old and new model signatures
                    if hasattr(model, 'use_global_attention'):
                        # New A5000 optimized model
                        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    else:
                        # Original gap-aware model
                        out = model(batch.x, batch.edge_index, batch.edge_attr, 
                                   batch.edge_type, batch.batch,
                                   batch.direct_temporal_mask,
                                   batch.gap_temporal_mask, 
                                   batch.proximity_mask)
                    loss = criterion(out, batch.y)
            else:
                if hasattr(model, 'use_global_attention'):
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    out = model(batch.x, batch.edge_index, batch.edge_attr, 
                               batch.edge_type, batch.batch,
                               batch.direct_temporal_mask,
                               batch.gap_temporal_mask, 
                               batch.proximity_mask)
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
    
    # Calculate per-class accuracy
    class_acc = {cls: (class_correct.get(cls, 0) / class_total.get(cls, 1)) 
                 for cls in range(5)}
    
    return val_loss, val_acc, class_acc


def compute_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """
    Compute class weights with A5000 memory optimization.
    """
    # More efficient collection for large datasets
    y_train = []
    for batch in train_loader:
        y_train.append(batch.y.cpu())
    
    y_train = torch.cat(y_train)
    class_counts = torch.bincount(y_train)
    
    # Handle potential edge case if some classes are missing
    if len(class_counts) < 5:
        padded_counts = torch.zeros(5, dtype=torch.float)
        padded_counts[:len(class_counts)] = class_counts
        class_counts = padded_counts
    
    total_samples = len(y_train)
    num_classes = 5
    
    # Compute inverse frequency weights
    class_weights = total_samples / (num_classes * class_counts.float())
    
    # Prevent extreme weights
    class_weights = torch.clamp(class_weights, min=0.1, max=10.0)
    
    return class_weights


def train_model_a5000(
    model: torch.nn.Module, 
    train_loader: DataLoader, 
    val_loader: Optional[DataLoader] = None, 
    epochs: int = 100, 
    lr: float = 0.001, 
    weight_decay: float = 1e-4, 
    patience: int = 15,
    use_amp: bool = True,
    gradient_accumulation_steps: int = 4,
    grad_clip_norm: float = 1.0,
    scheduler_type: str = "cosine_with_restarts"
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    A5000-optimized training function with mixed precision and gradient accumulation.
    
    Key A5000 optimizations:
    - Mixed precision training for 1.5-2x speedup
    - Gradient accumulation for larger effective batch sizes
    - Advanced learning rate scheduling
    - Memory-efficient data loading
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # A5000 optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Optimizer with A5000-tuned parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Advanced scheduler
    if scheduler_type == "cosine_with_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=epochs // 4, T_mult=2
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=10
        )
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None
    
    # Compute class weights
    class_weights = compute_class_weights(train_loader)
    class_weights = class_weights.to(device)
    criterion = torch.nn.NLLLoss(weight=class_weights)
    
    # Initialize tracking
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_acc': [],
        'class_acc': [],
        'learning_rate': []
    }
    
    print(f"A5000 Training Configuration:")
    print(f"- Device: {device}")
    print(f"- Mixed precision: {use_amp}")
    print(f"- Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"- Effective batch size: {train_loader.batch_size * gradient_accumulation_steps}")
    print(f"- Class weights: {class_weights.cpu().numpy()}")
    print(f"- Scheduler: {scheduler_type}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)
            
            if use_amp:
                with autocast():
                    # Handle both model types
                    if hasattr(model, 'use_global_attention'):
                        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    else:
                        out = model(batch.x, batch.edge_index, batch.edge_attr, 
                                   batch.edge_type, batch.batch,
                                   batch.direct_temporal_mask,
                                   batch.gap_temporal_mask, 
                                   batch.proximity_mask)
                    
                    loss = criterion(out, batch.y) / gradient_accumulation_steps
            else:
                if hasattr(model, 'use_global_attention'):
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    out = model(batch.x, batch.edge_index, batch.edge_attr, 
                               batch.edge_type, batch.batch,
                               batch.direct_temporal_mask,
                               batch.gap_temporal_mask, 
                               batch.proximity_mask)
                
                loss = criterion(out, batch.y) / gradient_accumulation_steps
            
            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Update scheduler for cosine annealing
                if scheduler_type == "cosine_with_restarts":
                    scheduler.step()
            
            total_loss += loss.item() * gradient_accumulation_steps
        
        # Handle remaining gradients if batch doesn't divide evenly
        if len(train_loader) % gradient_accumulation_steps != 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - epoch_start_time
        
        # Validation phase
        if val_loader:
            val_loss, val_acc, class_acc = validate_model(
                model, val_loader, criterion, device, use_amp
            )
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['class_acc'].append(class_acc)
            
            # Progress reporting
            if epoch % 5 == 0 or epoch == epochs - 1:
                weighted_acc = sum(class_acc[i] * class_weights[i] for i in range(5)) / sum(class_weights)
                
                print(f'Epoch {epoch:03d} ({epoch_time:.1f}s): '
                      f'Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                      f'Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
                print(f'  Class Acc: Non-event: {class_acc[0]:.4f}, '
                      f'Merge: {class_acc[1]:.4f}, Split: {class_acc[2]:.4f}, '
                      f'Post-merge: {class_acc[3]:.4f}, Post-split: {class_acc[4]:.4f}')
                print(f'  Weighted Acc: {weighted_acc:.4f}')
            
            # Scheduler step for ReduceLROnPlateau
            if scheduler_type != "cosine_with_restarts":
                scheduler.step(val_loss)
            
            # Early stopping
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
                print(f'Epoch {epoch:03d} ({epoch_time:.1f}s): '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Restore best model
    if val_loader and best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation loss: {best_val_loss:.4f}")
    
    return model, history


def plot_training_history_a5000(
    history: Dict[str, List[float]], 
    figsize: Tuple[int, int] = (20, 8)
) -> None:
    """
    Enhanced training visualization for A5000 training with learning rate tracking.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Plot loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.8)
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot overall accuracy
    if 'val_acc' in history:
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color='green', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Overall Validation Accuracy')
        axes[0, 1].grid(alpha=0.3)
    
    # Plot learning rate
    if 'learning_rate' in history:
        axes[0, 2].plot(history['learning_rate'], color='red', alpha=0.8)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(alpha=0.3)
    
    # Plot per-class accuracy
    if 'class_acc' in history and history['class_acc']:
        epochs = range(len(history['class_acc']))
        
        class_names = ['Non-event', 'Merge', 'Split', 'Post merge', 'Post split']
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for class_idx in range(5):
            class_acc = [epoch_acc[class_idx] for epoch_acc in history['class_acc']]
            axes[1, 0].plot(epochs, class_acc, label=class_names[class_idx], 
                           color=colors[class_idx], alpha=0.8)
        
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Per-Class Validation Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Plot rare events (Merge/Split) accuracy
        merge_acc = [epoch_acc[1] for epoch_acc in history['class_acc']]
        split_acc = [epoch_acc[2] for epoch_acc in history['class_acc']]
        
        axes[1, 1].plot(epochs, merge_acc, label='Merge Events', color='red', linewidth=2)
        axes[1, 1].plot(epochs, split_acc, label='Split Events', color='green', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Rare Events Accuracy (Key Target)')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        # Average accuracy progression
        avg_acc = [np.mean(list(epoch_acc.values())) for epoch_acc in history['class_acc']]
        axes[1, 2].plot(epochs, avg_acc, color='purple', linewidth=2, alpha=0.8)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Average Accuracy')
        axes[1, 2].set_title('Average Class Accuracy')
        axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Backward compatibility - alias the original function name
train_model = train_model_a5000
plot_training_history = plot_training_history_a5000