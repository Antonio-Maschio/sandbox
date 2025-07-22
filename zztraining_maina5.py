#!/usr/bin/env python3
"""Train a Graph Neural Network for particle event classification - A5000 Optimized Version."""

import torch
import time
import psutil
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import DataLoader

# Your existing imports
from processing_data.preprocessor import load_preprocessed_data
from models.gnn import ParticleGNNBiggerWithResidual
from models.gnn_A5000 import *
from models.model_util import save_model
from training.trainer import train_model, compute_class_weights, validate_model
from training.utils import plot_training_history
from training.roc_utils import compute_roc_curves, plot_roc_curves


@dataclass
class A5000TrainingConfig:
    """A5000-optimized training configuration."""
    # Data
    processed_dir: str = "data/dirty_processed90detec/"
    batch_size: int = None  # Will be auto-determined
    num_workers: int = None  # Will be auto-determined
    
    # Model
    hidden_channels: int = 32
    num_classes: int = 5
    dropout: float = 0.0
    
    # Training
    epochs: int = 1000
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    patience: int = 10
    
    # A5000 Optimizations
    mixed_precision: bool = True
    compile_model: bool = True
    auto_batch_size: bool = True
    
    # Paths
    model_save_path: str = "saved_models/a5000_optimized"
    
    # Visualization
    plot_roc_curves: bool = True
    
    # Metadata
    class_names: list = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["Non-event", "Merge", "Split", "Post merge", "Post split"]
        
        if self.num_workers is None:
            cpu_count = psutil.cpu_count()
            self.num_workers = min(cpu_count, 16)


def quick_a5000_batch_finder(dataset, model, device):
    """Quick batch size finder optimized for A5000."""
    print("=== A5000 Batch Size Optimization ===")
    
    # A5000 can handle much larger batches - start higher
    batch_sizes = [8,16,32, 64, 128, 256, 512, 768, 1024]
    working_batch = 16
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {torch.cuda.get_device_properties(0).name}")
        print(f"GPU Memory: {gpu_memory:.2f}GB")
    
    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            batch = next(iter(test_loader))
            batch = batch.to(device, non_blocking=True)
            
            # Test forward and backward pass
            model.train()
            output = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = torch.nn.functional.cross_entropy(output, batch.y)
            loss.backward()
            model.zero_grad()
            
            memory_used = torch.cuda.max_memory_allocated(0) / (1024**3)
            working_batch = batch_size
            print(f"✓ Batch size {batch_size} works - Memory: {memory_used:.2f}GB")
            
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            print(f"✗ Batch size {batch_size} failed - using {working_batch}")
            break
        finally:
            torch.cuda.empty_cache()
            if 'batch' in locals():
                del batch
    
    print(f"Selected batch size: {working_batch}")
    return working_batch


def get_a5000_data_loaders(config: A5000TrainingConfig):
    """A5000-optimized data loading."""
    return load_preprocessed_data(
        data_dir=config.processed_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers_=True if config.num_workers > 0 else False,
        prefetch_factor_=4 if config.num_workers > 0 else 2,
        drop_last_=True
    )


def initialize_a5000_model(num_features: int, config: A5000TrainingConfig) -> torch.nn.Module:
    """Initialize and optimize model for A5000."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ParticleGNNBiggerWithResidual(
        num_node_features=num_features,
        hidden_channels=config.hidden_channels,
        num_classes=config.num_classes
    ).to(device)
    
    # A5000 optimizations
    if config.compile_model and hasattr(torch, 'compile'):
        print("✓ Compiling model for A5000...")
        model = torch.compile(model, mode='max-autotune')
    
    return model


def train_model_a5000(
    model: torch.nn.Module, 
    train_loader, 
    val_loader = None, 
    epochs: int = 100, 
    lr: float = 0.01, 
    weight_decay: float = 5e-4, 
    patience: int = 10,
    use_mixed_precision: bool = True
):
    """A5000-optimized training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # A5000 optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ A5000 CUDA optimizations enabled")

    # Use AdamW with fused option for A5000
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            fused=True
        )
        print("✓ Using fused AdamW optimizer")
    except:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        print("✓ Using standard Adam optimizer")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Mixed precision setup
    scaler = GradScaler() if use_mixed_precision else None
    if use_mixed_precision:
        print("✓ Mixed precision enabled")

    # Compute class weights
    class_weights = compute_class_weights(train_loader).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Training tracking
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_acc': [],
        'class_acc': []
    }
    
    print(f"A5000 Training Configuration:")
    print(f"  Mixed Precision: {use_mixed_precision}")
    print(f"  Class weights: {class_weights.cpu().numpy()}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            if use_mixed_precision and scaler:
                with autocast():
                    out = model(batch.x, batch.edge_index, batch.edge_attr)
                    loss = criterion(out, batch.y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
        
        epoch_time = time.time() - epoch_start
        avg_train_loss = total_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if val_loader:
            val_loss, val_acc, class_acc = validate_model(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['class_acc'].append(class_acc)
            
            # Performance metrics
            samples_per_sec = len(train_loader.dataset) / epoch_time
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch:03d} ({epoch_time:.1f}s): Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                print(f'  Throughput: {samples_per_sec:.1f} samples/sec')
                print(f'  Class Acc: Non-event: {class_acc[0]:.4f}, '
                      f'Merge: {class_acc[1]:.4f}, Split: {class_acc[2]:.4f}, '
                      f'Post-merge: {class_acc[3]:.4f}, Post-split: {class_acc[4]:.4f}')
            
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
            samples_per_sec = len(train_loader.dataset) / epoch_time
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch:03d} ({epoch_time:.1f}s): Train Loss: {avg_train_loss:.4f}')
                print(f'  Throughput: {samples_per_sec:.1f} samples/sec')
    
    # Restore best model
    if val_loader and best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation loss: {best_val_loss:.4f}")
    
    return model, history


def print_data_stats(data_loader):
    """Print statistics about the data features."""
    sample = next(iter(data_loader))
    x = sample.x
    
    print(f"\nData shape: {x.shape}")
    print(f"Feature stats - Mean: {x.mean(dim=0).tolist()}")
    print(f"Feature stats - Std: {x.std(dim=0).tolist()}")
    print(f"Feature stats - Range: [{x.min():.3f}, {x.max():.3f}]")
    
    return x.size(1)


def main_a5000(config: Optional[A5000TrainingConfig] = None):
    """Main A5000-optimized training pipeline."""
    if config is None:
        config = A5000TrainingConfig()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print hardware info
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_props.name}")
        print(f"GPU Memory: {gpu_props.total_memory/(1024**3):.2f}GB")
    
    system_memory = psutil.virtual_memory().total / (1024**3)
    print(f"System RAM: {system_memory:.2f}GB")
    print(f"CPU Cores: {psutil.cpu_count()}")
    
    torch.cuda.empty_cache()
    
    # Auto batch size optimization
    if config.auto_batch_size and config.batch_size is None:
        print("\nLoading data for batch size optimization...")
        temp_config = A5000TrainingConfig()
        temp_config.batch_size = 32
        temp_config.num_workers = 4
        temp_config.auto_batch_size = False
        
        train_loader, val_loader, test_loader, data = get_a5000_data_loaders(temp_config)
        num_features = print_data_stats(train_loader)
        
        # Initialize model for testing
        model = initialize_a5000_model(num_features, config)
        
        # Find optimal batch size
        config.batch_size = quick_a5000_batch_finder(train_loader.dataset, model, device)
        
        # Adjust workers based on batch size
        if config.batch_size >= 256:
            config.num_workers = min(psutil.cpu_count(), 16)
        elif config.batch_size >= 128:
            config.num_workers = min(psutil.cpu_count(), 12)
        else:
            config.num_workers = min(psutil.cpu_count(), 8)
    
    elif config.batch_size is None:
        # Simple fallback for A5000
        config.batch_size = 128
        config.num_workers = 12
    
    print(f"\nFinal configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Workers: {config.num_workers}")
    print(f"  Mixed precision: {config.mixed_precision}")
    print(f"  Model compilation: {config.compile_model}")
    
    # Load data with optimized settings
    print("\nLoading data with A5000-optimized settings...")
    train_loader, val_loader, test_loader, data = get_a5000_data_loaders(config)
    
    if 'num_features' not in locals():
        num_features = print_data_stats(train_loader)
    
    if 'model' not in locals():
        model = initialize_a5000_model(num_features, config)
    
    # Train
    print(f"\nStarting A5000-optimized training for {config.epochs} epochs...")
    start_time = time.time()
    
    trained_model, history = train_model_a5000(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        patience=config.patience,
        use_mixed_precision=config.mixed_precision
    )
    
    training_time = time.time() - start_time
    total_samples = len(train_loader.dataset) * len(history['train_loss'])
    throughput = total_samples / training_time
    
    print(f"\n=== A5000 Training Results ===")
    print(f"Training completed in {training_time/60:.1f} minutes")
    print(f"Average throughput: {throughput:.1f} samples/second")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated(0)/(1024**3):.2f}GB")
    
    # Visualize results
    plot_training_history(history)
    
    # Compute and plot ROC curves if enabled
    roc_auc_scores = {}
    if config.plot_roc_curves:
        print("\nComputing ROC curves...")
        roc_data = compute_roc_curves(
            model=trained_model,
            data_loader=val_loader,
            device=device,
            num_classes=config.num_classes
        )
        
        model_name = Path(config.model_save_path).stem
        save_dir = Path(config.model_save_path).parent.parent
        
        plot_roc_curves(
            roc_data=roc_data,
            class_names=config.class_names,
            save_dir=save_dir,
            model_name=model_name
        )
        
        roc_auc_scores = {config.class_names[i]: roc_data[i][2] for i in range(config.num_classes)}
    
    # Save model
    Path(config.model_save_path).parent.mkdir(parents=True, exist_ok=True)
    save_model(
        model=trained_model,
        filepath=config.model_save_path,
        metadata={
            'training_time': training_time,
            'throughput': throughput,
            'peak_memory_gb': torch.cuda.max_memory_allocated(0)/(1024**3),
            'final_val_acc': history['val_acc'][-1] if history['val_acc'] else None,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'class_names': config.class_names,
            'num_epochs_trained': len(history['train_loss']),
            'config': config.__dict__,
            'roc_auc_scores': roc_auc_scores,
            'hardware': f"A5000-{config.batch_size}batch-{config.num_workers}workers"
        }
    )
    
    return trained_model, history


if __name__ == "__main__":
    # A5000 configurations for different scenarios
    
    # Quick test configuration
    quick_test_config = A5000TrainingConfig(
        batch_size=None,  # Auto-optimize
        epochs=10,
        learning_rate=0.005,
        mixed_precision=True,
        compile_model=True,
        model_save_path="saved_models/a5000_quick_test.pt"
    )
    
    # Full training configuration
    full_training_config = A5000TrainingConfig(
        batch_size=None,  # Auto-optimize
        epochs=500,
        learning_rate=0.005,
        mixed_precision=True,
        compile_model=True,
        model_save_path="saved_models/a5000_full_training.pt"
    )
    
    # Conservative configuration (if you encounter issues)
    conservative_config = A5000TrainingConfig(
        batch_size=128,  # Fixed smaller batch
        epochs=200,
        learning_rate=0.01,
        mixed_precision=False,
        compile_model=False,
        auto_batch_size=False,
        num_workers=8,
        model_save_path="saved_models/a5000_conservative.pt"
    )
    
    # Choose your configuration here:
    selected_config = full_training_config  # Change this line to switch configs
    
    print(f"Running A5000 training with: {Path(selected_config.model_save_path).stem}")
    model, history = main_a5000(selected_config)