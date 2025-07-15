"""Model utilities for saving and loading PyTorch models."""

import torch
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple


def save_model(
    model: torch.nn.Module, 
    filepath: Union[str, Path], 
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save a PyTorch model with configuration and metadata.
    
    Args:
        model: Model to save
        filepath: Path where model will be saved
        metadata: Additional information to save with model
    """
    save_dict = {
        'state_dict': model.state_dict(),
        'model_config': extract_model_config(model)
    }
    
    if metadata:
        save_dict['metadata'] = metadata
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")


def load_model(
    model_class: type,
    model_path: Union[str, Path], 
    device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a model from checkpoint with metadata.
    
    Args:
        model_class: Class of the model to instantiate
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model and metadata dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)    
    # Extract configuration
    config = checkpoint.get('model_config', {})
    metadata = checkpoint.get('metadata', {})
    
    # Initialize model
    model = model_class(**config)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    if metadata:
        print(f"Metadata: epochs={metadata.get('num_epochs_trained')}, "
              f"val_acc={metadata.get('final_val_acc', 'N/A'):.4f}")
    
    return model, metadata


def extract_model_config(model: torch.nn.Module) -> Dict[str, Any]:
    """Extract configuration from model architecture.
    
    Args:
        model: Model to extract config from
        
    Returns:
        Dictionary of model configuration parameters
    """
    config = {}
    
    # Try to extract common parameters
    if hasattr(model, 'conv1'):
        config['num_node_features'] = model.conv1.in_channels
        config['hidden_channels'] = model.conv1.out_channels
    
    if hasattr(model, 'out'):
        config['num_classes'] = model.out.out_features
    
    if hasattr(model, 'dropout'):
        config['dropout'] = model.dropout
    
    # Add any model-specific attributes
    for attr in ['num_node_features', 'hidden_channels', 'num_classes']:
        if hasattr(model, attr):
            config[attr] = getattr(model, attr)
    
    return config


def load_model_for_inference(
    model_class: type,
    model_path: Union[str, Path],
    device: Optional[torch.device] = None
) -> torch.nn.Module:
    """Load a model ready for inference (simplified interface).
    
    Args:
        model_class: Class of the model
        model_path: Path to checkpoint
        device: Device for inference
        
    Returns:
        Model ready for inference
    """
    model, _ = load_model(model_class, model_path, device)
    return model