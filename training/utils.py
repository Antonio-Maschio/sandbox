import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any

def evaluate_model(
    model: torch.nn.Module, 
    test_loader: torch.utils.data.DataLoader, 
    class_names: List[str] = ['Non-event', 'Merge', 'Split'],
    visualize_cm: bool = True,
    figsize: Tuple[int, int] = (10, 8)
) -> Dict[str, Any]:
    """
    Evaluate a 3-class particle event detection model with comprehensive metrics.
    
    Performs inference on test data and computes performance metrics for the 3-class
    particle event classification task (non-event, merge, split). Generates detailed
    reports, confusion matrices, and class-specific metrics.
    
    Args:
        model (torch.nn.Module): Trained GNN model for event classification
        test_loader (torch.utils.data.DataLoader): DataLoader containing test graphs
        class_names (List[str], optional): Names of the three classes for visualization.
            Defaults to ['Non-event', 'Merge', 'Split'].
        visualize_cm (bool, optional): Whether to display confusion matrix visualization.
            Defaults to True.
        figsize (Tuple[int, int], optional): Figure size for confusion matrix plot.
            Defaults to (10, 8).
    
    Returns:
        Dict[str, Any]: Evaluation metrics dictionary including:
            - 'accuracy': Overall classification accuracy
            - 'class_precision': Precision for each class [non-event, merge, split]
            - 'class_recall': Recall for each class
            - 'class_f1': F1 scores for each class
            - 'confusion_matrix': Full confusion matrix as numpy array
            - 'classification_report': Detailed sklearn classification report
    
    Example:
        >>> model = load_model('particle_gnn_3class.pt')
        >>> metrics = evaluate_model(model, test_loader)
        >>> print(f"Merge event F1 score: {metrics['class_f1'][1]:.4f}")
        >>> print(f"Split event recall: {metrics['class_recall'][2]:.4f}")
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    y_true = []
    y_pred = []
    
    # Collect predictions
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = out.argmax(dim=1)
            
            # Collect true and predicted labels
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    # Convert to numpy arrays for metric calculation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Class-specific metrics for all three classes
    prec, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, 
        labels=[0, 1, 2],  # Explicitly specify all three classes
        average=None
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    # Get detailed classification report as string
    class_report = classification_report(
        y_true, y_pred, 
        labels=[0, 1, 2], 
        target_names=class_names,
        digits=4
    )
    
    # Calculate class-specific statistics
    class_counts = {
        'true': {i: np.sum(y_true == i) for i in range(3)},
        'pred': {i: np.sum(y_pred == i) for i in range(3)}
    }
    
    false_positives = {
        i: np.sum((y_pred == i) & (y_true != i)) for i in range(3)
    }
    
    false_negatives = {
        i: np.sum((y_true == i) & (y_pred != i)) for i in range(3)
    }
    
    # Print detailed evaluation report
    print("\n===== 3-CLASS PARTICLE EVENT DETECTION EVALUATION =====")
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    print("\nClass-Specific Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name} (Class {i}):")
        print(f"  Precision:      {prec[i]:.4f}")
        print(f"  Recall:         {recall[i]:.4f}")
        print(f"  F1 Score:       {f1[i]:.4f}")
        print(f"  Support:        {support[i]} samples")
        print(f"  False Positives: {false_positives[i]} instances")
        print(f"  False Negatives: {false_negatives[i]} instances")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    header = "True \\ Pred |" + "".join(f" {name:10} |" for name in class_names)
    print(header)
    print("-" * len(header))
    
    for i, class_name in enumerate(class_names):
        row = f"{class_name:10} |"
        for j in range(3):
            row += f" {cm[i, j]:10d} |"
        print(row)
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(class_report)
    
    # Visualization of confusion matrix
    if visualize_cm:
        plt.figure(figsize=figsize)
        
        # Plot normalized confusion matrix as heatmap
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap with both raw counts and percentages
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False
        )
        plt.title('Confusion Matrix (Raw Counts)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
        
        # Additional normalized visualization
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix (Normalized by Row)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    # Return comprehensive metrics dictionary
    return {
        'accuracy': accuracy,
        'class_precision': prec,
        'class_recall': recall,
        'class_f1': f1,
        'class_support': support,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def plot_training_history(
    history: Dict[str, List[float]], 
    save_path: Optional[str] = None, 
    figsize: Tuple[int, int] = (16, 8)
) -> None:
    """
    Visualize training progress for a 3-class particle event detection model.
    
    Creates a comprehensive multi-panel visualization showing:
    1. Training and validation loss curves
    2. Overall validation accuracy
    3. Class-specific validation metrics for Non-event/Merge/Split classes
    
    Args:
        history (Dict[str, List[float]]): Training history dictionary with keys:
            - 'train_loss': List of training losses
            - 'val_loss': List of validation losses
            - 'val_acc': List of validation accuracies
            - 'class_acc': List of dictionaries with per-class accuracies
        save_path (Optional[str]): Path to save figure. If None, figure is displayed instead.
            Defaults to None.
        figsize (Tuple[int, int]): Figure dimensions (width, height). Defaults to (16, 8).
            
    Example:
        >>> model, history = train_model(model, train_loader, val_loader)
        >>> plot_training_history(history, save_path='results/training_history.png')
    """
    plt.figure(figsize=figsize)
    
    # Panel 1: Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    
    if 'val_loss' in history:
        plt.plot(history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Loss During Training', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    
    # Panel 2: Validation accuracy
    if 'val_acc' in history:
        plt.subplot(1, 3, 2)
        plt.plot(history['val_acc'], 'g-', label='Validation Accuracy', linewidth=2)
        
        # Add zoomed-in view of later epochs if training is long enough
        if len(history['val_acc']) > 30:
            ax = plt.gca()
            # Create inset axes for zoomed region (last 30% of training)
            start_idx = int(0.7 * len(history['val_acc']))
            axins = ax.inset_axes([0.5, 0.1, 0.45, 0.4])
            axins.plot(range(start_idx, len(history['val_acc'])), 
                      history['val_acc'][start_idx:], 'g-', linewidth=2)
            ax.indicate_inset_zoom(axins)
        
        plt.title('Overall Validation Accuracy', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(alpha=0.3)
    
    # Panel 3: Class-specific metrics
    if 'class_acc' in history and history['class_acc']:
        plt.subplot(1, 3, 3)
        epochs = range(len(history['class_acc']))
        
        # Extract class-specific accuracies
        non_event_acc = [epoch_acc[0] for epoch_acc in history['class_acc']]
        merge_acc = [epoch_acc[1] for epoch_acc in history['class_acc']]
        split_acc = [epoch_acc[2] for epoch_acc in history['class_acc']]
        
        plt.plot(epochs, non_event_acc, 'b-', label='Non-event', linewidth=2)
        plt.plot(epochs, merge_acc, 'r-', label='Merge', linewidth=2)
        plt.plot(epochs, split_acc, 'g-', label='Split', linewidth=2)
        
        plt.title('Class-Specific Validation Accuracy', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()