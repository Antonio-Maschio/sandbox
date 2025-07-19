#!/usr/bin/env python3
"""Train a Graph Neural Network for particle event classification."""

import torch
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# from data_processing.data_2step import load_processed_data
from processing_data.preprocessor import load_preprocessed_data


from models.gnn import ParticleGNNBiggerWithResidual
from models.gnn_A5000 import *
from models.model_util import save_model
from training.trainer import train_model
from training.utils import plot_training_history
from training.roc_utils import compute_roc_curves, plot_roc_curves


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Data
    processed_dir: str = "data/dirty_processed90detec/"
    batch_size: int = 8
    num_workers: int = 2
    
    # Model
    hidden_channels: int = 32
    num_classes: int = 5
    dropout: float = 0.0
    
    # Training
    epochs: int = 1000
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    patience: int = 10
    
    # Paths
    model_save_path: str = "saved_models/dirty_processed"
    
    # Visualization
    plot_roc_curves: bool = True
    
    # Metadata
    class_names: list = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["Non-event", "Merge", "Split", "Post merge", "Post split"]


def get_data_loaders(config: TrainingConfig):
    """Load and prepare data loaders."""
    return load_preprocessed_data(
        data_dir=config.processed_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )


def initialize_model(num_features: int, config: TrainingConfig) -> torch.nn.Module:
    """Initialize the GNN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ParticleGNNBiggerWithResidual(
        num_node_features=num_features,
        hidden_channels=config.hidden_channels,
        num_classes=config.num_classes
    ).to(device)


    return model


def print_data_stats(data_loader):
    """Print statistics about the data features."""
    sample = next(iter(data_loader))
    x = sample.x
    
    print(f"\nData shape: {x.shape}")
    print(f"Feature stats - Mean: {x.mean(dim=0).tolist()}")
    print(f"Feature stats - Std: {x.std(dim=0).tolist()}")
    print(f"Feature stats - Range: [{x.min():.3f}, {x.max():.3f}]")
    
    return x.size(1)


def main(config: Optional[TrainingConfig] = None):
    """Main training pipeline."""
    if config is None:
        config = TrainingConfig()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.cuda.empty_cache()
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader,data = get_data_loaders(config)
    num_features = print_data_stats(train_loader)
    
    # Initialize model
    print(f"\nInitializing model with {config.num_classes} classes...")
    model = initialize_model(num_features, config)
    
    # Train
    print(f"\nStarting training for {config.epochs} epochs...")
    start_time = time.time()
    
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        patience=config.patience
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    
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
        
        # Extract model name from save path
        model_name = Path(config.model_save_path).stem
        save_dir = Path(config.model_save_path).parent.parent  # Go up to project root
        
        plot_roc_curves(
            roc_data=roc_data,
            class_names=config.class_names,
            save_dir=save_dir,
            model_name=model_name
        )
        
        # Store AUC scores
        roc_auc_scores = {config.class_names[i]: roc_data[i][2] for i in range(config.num_classes)}
    
    # Save model
    Path(config.model_save_path).parent.mkdir(parents=True, exist_ok=True)
    save_model(
        model=trained_model,
        filepath=config.model_save_path,
        metadata={
            'training_time': training_time,
            'final_val_acc': history['val_acc'][-1] if history['val_acc'] else None,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'class_names': config.class_names,
            'num_epochs_trained': len(history['train_loss']),
            'config': config.__dict__,
            'roc_auc_scores': roc_auc_scores
        }
    )
    
    return trained_model, history


if __name__ == "__main__":
    # Example with custom configuration
    custom_config = TrainingConfig(
        batch_size=8,
        epochs=500,#500
        learning_rate=0.005,
        model_save_path="saved_models/detectio90_w40.pt"
    )
    
    model, history = main(custom_config)
    