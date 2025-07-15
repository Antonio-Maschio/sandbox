"""ROC curve plotting utilities for multi-class classification."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def compute_roc_curves(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int
) -> Dict[int, Tuple[np.ndarray, np.ndarray, float]]:
    """Compute ROC curves for all classes.
    
    Args:
        model: Trained model
        data_loader: Data loader (typically validation or test)
        device: Computation device
        num_classes: Number of classes
        
    Returns:
        Dictionary mapping class index to (fpr, tpr, auc_score)
    """
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            probs = torch.softmax(out, dim=1)
            
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Binarize labels for one-vs-rest ROC calculation
    labels_binarized = label_binarize(all_labels, classes=range(num_classes))
    
    roc_data = {}
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_binarized[:, i], all_probs[:, i])
        auc_score = auc(fpr, tpr)
        roc_data[i] = (fpr, tpr, auc_score)
    
    return roc_data


def plot_roc_curves(
    roc_data: Dict[int, Tuple[np.ndarray, np.ndarray, float]],
    class_names: List[str],
    save_dir: Path,
    model_name: str
) -> None:
    """Plot and save ROC curves for each class.
    
    Args:
        roc_data: Dictionary with ROC curve data
        class_names: List of class names
        save_dir: Base directory for saving plots
        model_name: Model name for directory structure
    """
    # Create directory structure
    roc_dir = save_dir / "ROC_Curves" / model_name
    roc_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot individual ROC curves for each class
    for class_idx, (fpr, tpr, auc_score) in roc_data.items():
        plt.figure(figsize=(8, 6))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        
        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {class_names[class_idx]}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        # Save plot
        filename = f"{class_names[class_idx].replace(' ', '_').lower()}_ROC.png"
        save_path = roc_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ROC curve for {class_names[class_idx]} to {save_path}")
    
    # Create combined plot with all classes
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for class_idx, (fpr, tpr, auc_score) in roc_data.items():
        plt.plot(fpr, tpr, color=colors[class_idx % len(colors)], lw=2,
                label=f'{class_names[class_idx]} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Classes')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Save combined plot
    combined_path = roc_dir / "all_classes_ROC.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined ROC curve to {combined_path}")


def plot_micro_macro_roc(
    roc_data: Dict[int, Tuple[np.ndarray, np.ndarray, float]],
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    class_names: List[str],
    save_dir: Path,
    model_name: str
) -> None:
    """Plot micro and macro-averaged ROC curves.
    
    Args:
        roc_data: Dictionary with per-class ROC data
        all_labels: True labels
        all_probs: Predicted probabilities
        class_names: List of class names
        save_dir: Base directory for saving
        model_name: Model name for directory
    """
    roc_dir = save_dir / "ROC_Curves" / model_name
    
    # Compute macro-averaged ROC
    all_fpr = np.unique(np.concatenate([roc_data[i][0] for i in range(len(class_names))]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(len(class_names)):
        mean_tpr += np.interp(all_fpr, roc_data[i][0], roc_data[i][1])
    
    mean_tpr /= len(class_names)
    macro_auc = auc(all_fpr, mean_tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(all_fpr, mean_tpr, color='navy', lw=2,
            label=f'Macro-average ROC (AUC = {macro_auc:.3f})')
    
    # Add shaded area for standard deviation
    tprs = []
    for i in range(len(class_names)):
        tprs.append(np.interp(all_fpr, roc_data[i][0], roc_data[i][1]))
    
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
    plt.fill_between(all_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                    label='± 1 std. dev.')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-averaged ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    save_path = roc_dir / "macro_average_ROC.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved macro-averaged ROC curve to {save_path}")