"""Simple script for applying trained model to data with ROC curve analysis."""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
from models.gnn import ParticleGNNBiggerWithResidual
from processing_data.graph_builder import build_particle_graph
from torch_geometric.loader import DataLoader
from models.model_util import load_model
from processing_data.preprocessor import DataPreprocessor


def norm_data_with_preprocessor(file_input, radius_buff=0.0, normalize_method='standard', sim_id=0, verbose=False):
    """Normalize data using the preprocessor approach."""
    df = pd.read_csv(file_input)
    
    graph = build_particle_graph(
        df, 
        radius_buffer=radius_buff,
        sim_id=sim_id
    )
    
    graph.original_data = df
    
    class SimpleDataset:
        def __init__(self, graphs):
            self.graphs = graphs
    
    dataset = SimpleDataset([graph])
    
    preprocessor = DataPreprocessor(dataset)
    preprocessor.normalize_features(method=normalize_method)
    preprocessor.normalize_edges(method=normalize_method)
    
    normalized_graph = dataset.graphs[0]
    
    node_features = normalized_graph.x.numpy()
    df_normalized = df.copy()
    
    feature_columns = ['x', 'y', 'z', 'mass']
    available_features = [f for f in feature_columns if f in df.columns]
    
    if len(available_features) == node_features.shape[1]:
        df_normalized[available_features] = node_features
    elif len(available_features) < node_features.shape[1]:
        df_normalized[available_features] = node_features[:, :len(available_features)]
    
    return df_normalized, preprocessor.normalization_params, normalized_graph


def obtain_graph(data, radius_buff=0.0, sim_id=0, as_loader=False, verbose=False):
    """Create graph from dataframe."""
    graph = build_particle_graph(
        data, 
        radius_buffer=radius_buff,
        sim_id=sim_id
    )
    
    graph.original_data = data
    
    if as_loader:
        return DataLoader([graph], batch_size=1, shuffle=False)
    
    return graph


def generate_model_predictions(model, graph):
    """Generate predictions for a graph."""
    device = next(model.parameters()).device
    graph = graph.to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index, graph.edge_attr)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    results = {
        'predictions': predictions.cpu().numpy(),
        'probabilities': probabilities.cpu().numpy(),
        'confidence': probabilities.max(dim=1)[0].cpu().numpy(),
        'particle_ids': graph.particle_ids.cpu().numpy() if hasattr(graph, 'particle_ids') else np.arange(graph.num_nodes),
        'frame_numbers': graph.frame_numbers.cpu().numpy() if hasattr(graph, 'frame_numbers') else np.zeros(graph.num_nodes),
        'positions': graph.pos.cpu().numpy() if hasattr(graph, 'pos') else None
    }
    
    if hasattr(graph, 'y'):
        results['event_labels'] = graph.y.cpu().numpy()
    
    return results


def plot_roc_curves_multiclass(y_true, y_prob, class_names, save_path='roc_curves.png', figsize=(12, 8)):
    """
    Plot ROC curves for multiclass classification using one-vs-rest approach.
    
    Args:
        y_true: True labels (1D array)
        y_prob: Prediction probabilities (2D array, shape: [n_samples, n_classes])
        class_names: List of class names
        save_path: Path to save the plot
        figsize: Figure size tuple
    
    Returns:
        Dictionary containing AUC scores for each class
    """
    n_classes = len(class_names)
    
    # Binarize the output labels for multiclass ROC
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Handle case where not all classes are present in y_true
    if y_true_bin.shape[1] != n_classes:
        # Create a proper binary matrix
        y_true_bin_full = np.zeros((len(y_true), n_classes))
        unique_classes = np.unique(y_true)
        for i, class_idx in enumerate(unique_classes):
            y_true_bin_full[:, class_idx] = (y_true == class_idx).astype(int)
        y_true_bin = y_true_bin_full
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=figsize)
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:  # Only compute if class is present
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
        else:
            print(f"Warning: No samples found for class '{class_names[i]}' in true labels")
            roc_auc[i] = None
    
    # Plot random classifier line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - One vs Rest Classification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")
    plt.show()
    
    return roc_auc


def plot_individual_roc_curves(y_true, y_prob, class_names, save_path='individual_roc_curves.png'):
    """
    Plot individual ROC curves for each class in separate subplots.
    
    Args:
        y_true: True labels (1D array)
        y_prob: Prediction probabilities (2D array)
        class_names: List of class names
        save_path: Path to save the plot
    
    Returns:
        Dictionary containing AUC scores for each class
    """
    n_classes = len(class_names)
    
    # Calculate number of rows and columns for subplots
    n_cols = 3
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_classes > 1 else [axes]
    
    # Binarize the output labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if y_true_bin.shape[1] != n_classes:
        y_true_bin_full = np.zeros((len(y_true), n_classes))
        unique_classes = np.unique(y_true)
        for i, class_idx in enumerate(unique_classes):
            y_true_bin_full[:, class_idx] = (y_true == class_idx).astype(int)
        y_true_bin = y_true_bin_full
    
    roc_auc = dict()
    
    for i in range(n_classes):
        ax = axes[i]
        
        if np.sum(y_true_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                   label=f'ROC curve (AUC = {roc_auc[i]:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{class_names[i]}')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No samples\nfor {class_names[i]}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{class_names[i]} (No samples)')
            roc_auc[i] = None
    
    # Hide empty subplots
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Individual ROC curves saved to {save_path}")
    plt.show()
    
    return roc_auc


def plot_class_distribution(y_true, class_names, save_path='class_distribution.png'):
    """Plot the distribution of true classes."""
    plt.figure(figsize=(10, 6))
    
    unique, counts = np.unique(y_true, return_counts=True)
    class_labels = [class_names[i] for i in unique]
    
    bars = plt.bar(class_labels, counts, color=plt.cm.Set1(np.linspace(0, 1, len(unique))))
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Distribution of True Classes', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Class distribution plot saved to {save_path}")
    plt.show()


def export_nodepredictions_csv(model, graph, output_path='node_predictions.csv', class_names=None):
    """Export predictions to CSV."""
    if class_names is None:
        class_names = ['non-event', 'merge', 'split', 'post-merge', 'post-split']
    
    results = generate_model_predictions(model, graph)
    
    if hasattr(graph, 'original_data'):
        df_export = graph.original_data.copy()
    else:
        df_export = pd.DataFrame({
            'frame': results['frame_numbers'].astype(float),
            'particle': results['particle_ids'].astype(float)
        })
        if results['positions'] is not None:
            df_export['x'] = results['positions'][:, 0]
            df_export['y'] = results['positions'][:, 1]
            df_export['z'] = results['positions'][:, 2]
    
    df_export['predicted_class'] = results['predictions']
    df_export['predicted_label'] = [class_names[p] for p in results['predictions']]
    df_export['confidence'] = results['confidence']
    
    for i, class_name in enumerate(class_names):
        col_name = f'prob_{class_name.replace("-", "_")}'
        df_export[col_name] = results['probabilities'][:, i]
    
    if 'event_label' in df_export.columns:
        df_export['correct_prediction'] = (df_export['predicted_class'] == df_export['event_label']).astype(int)
        accuracy = df_export['correct_prediction'].mean()
        print(f"Accuracy: {accuracy:.4f}")
    
    df_export.to_csv(output_path, index=False)
    print(f"Predictions exported to {output_path}")
    
    return df_export


def analyze_model_performance_with_roc(model, graph, class_names, output_dir='./'):
    """
    Comprehensive analysis including ROC curves, class distribution, and performance metrics.
    
    Args:
        model: Trained model
        graph: Input graph
        class_names: List of class names
        output_dir: Directory to save plots and results
    
    Returns:
        Dictionary containing all results and metrics
    """
    results = generate_model_predictions(model, graph)
    
    # Check if we have true labels for ROC analysis
    if 'event_labels' not in results:
        print("Warning: No true labels found. ROC curves cannot be computed.")
        print("Make sure your graph has 'y' attribute containing true labels.")
        return None
    
    y_true = results['event_labels']
    y_prob = results['probabilities']
    
    print(f"\n=== MODEL PERFORMANCE ANALYSIS ===")
    print(f"Total samples: {len(y_true)}")
    print(f"Number of classes: {len(class_names)}")
    
    # Plot class distribution
    plot_class_distribution(y_true, class_names, 
                          save_path=f'{output_dir}class_distribution.png')
    
    # Plot combined ROC curves
    print("\nComputing ROC curves...")
    roc_auc_combined = plot_roc_curves_multiclass(
        y_true, y_prob, class_names,
        save_path=f'{output_dir}roc_curves_combined.png'
    )
    
    # Plot individual ROC curves
    roc_auc_individual = plot_individual_roc_curves(
        y_true, y_prob, class_names,
        save_path=f'{output_dir}roc_curves_individual.png'
    )
    
    # Print AUC summary
    print(f"\n=== AUC SCORES SUMMARY ===")
    valid_aucs = []
    for i, class_name in enumerate(class_names):
        auc_score = roc_auc_combined.get(i)
        if auc_score is not None:
            print(f"  {class_name}: {auc_score:.3f}")
            valid_aucs.append(auc_score)
        else:
            print(f"  {class_name}: No samples (cannot compute AUC)")
    
    if valid_aucs:
        mean_auc = np.mean(valid_aucs)
        print(f"\nMean AUC (classes with samples): {mean_auc:.3f}")
    
    return {
        'predictions': results['predictions'],
        'probabilities': results['probabilities'],
        'true_labels': y_true,
        'roc_auc_scores': roc_auc_combined,
        'class_names': class_names,
        'mean_auc': mean_auc if valid_aucs else None
    }


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading and normalizing data...")
    df_norm, norm_params, graph = norm_data_with_preprocessor(
        "data/tracked_simdata_3d_temporal_vfinalllll/tracked_particles_3d_00000.csv",
        radius_buff=-22,
        normalize_method='standard'
    )
    
    print(f"Graph created with {graph.num_nodes} nodes")
    
    print("Loading model...")
    model, metadata = load_model(
        ParticleGNNBiggerWithResidual,
        "saved_models/experiment_v4.pt",
        device
    )
    
    class_names = metadata.get('class_names', ['non-event', 'merge', 'split', 'post-merge', 'post-split'])
    print(f"Model loaded. Classes: {class_names}")
    
    print("Generating predictions...")
    df_with_predictions = export_nodepredictions_csv(
        model, 
        graph, 
        output_path="predictions.csv",
        class_names=class_names
    )
    
    print("\nPrediction summary:")
    for class_name in class_names:
        count = (df_with_predictions['predicted_label'] == class_name).sum()
        percentage = count / len(df_with_predictions) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")

    # ROC CURVE ANALYSIS
    print("\n" + "="*50)
    print("PERFORMING ROC CURVE ANALYSIS")
    print("="*50)
    
    # Perform comprehensive analysis with ROC curves
    analysis_results = analyze_model_performance_with_roc(
        model, graph, class_names, output_dir='./'
    )
    
    if analysis_results:
        print(f"\nROC analysis completed successfully!")
        print(f"Check the generated plots:")
        print(f"  - class_distribution.png")
        print(f"  - roc_curves_combined.png") 
        print(f"  - roc_curves_individual.png")

    # Original weighted accuracy calculation
    print("\nCalculating weighted accuracy...")
    if 'event_label' in df_with_predictions.columns:
        label_map = {i: name for i, name in enumerate(class_names)}
        event_labels_named = df_with_predictions['event_label'].map(label_map)
        pred_labels_named = df_with_predictions['predicted_label']

        class_counts = event_labels_named.value_counts().to_dict()
        total_count = len(event_labels_named)
        class_weights = {cls: count / total_count for cls, count in class_counts.items()}

        weighted_acc = 0.0
        for cls in class_names:
            cls_true = (event_labels_named == cls)
            cls_total = cls_true.sum()

            if cls_total > 0:
                cls_correct = ((pred_labels_named == cls) & cls_true).sum()
                cls_acc = cls_correct / cls_total
                weight = class_weights.get(cls, 0)
                weighted_acc += cls_acc * weight
                print(f"  {cls}: acc={cls_acc:.3f}, weight={weight:.3f}, weighted={cls_acc * weight:.3f}")
            else:
                print(f"  {cls}: no true samples, skipped in weighted accuracy")

        print(f"Weighted accuracy: {weighted_acc:.4f}")
    else:
        print("True labels not found in dataframe. Cannot compute weighted accuracy.")