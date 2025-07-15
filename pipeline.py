"""Simple script for applying trained model to data."""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
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

    print("\nCalculating weighted accuracy...")

    if 'event_label' in df_with_predictions.columns:
        label_map = {i: name for i, name in enumerate(class_names)}
        event_labels_named = df_with_predictions['event_label'].map(label_map)
        pred_labels_named = df_with_predictions['predicted_label']

        class_counts = event_labels_named.value_counts().to_dict()
        total_count = len(event_labels_named)
        class_weights = {cls: count / total_count for cls, count in class_counts.items()}

        print("\nCalculating weighted accuracy...")
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