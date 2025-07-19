"""Simple script for applying trained model to data."""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


class PredictionVisualizer:
    def __init__(self, graph, predictions_df):
        self.graph = graph
        self.predictions_df = predictions_df
        self.event_colors = {
            0: 'blue',
            1: 'red', 
            2: 'green',
            3: 'darkred',
            4: 'darkgreen',
        }
        self.event_names = {
            0: 'Non-event',
            1: 'Merge',
            2: 'Split',
            3: 'Post-merge',
            4: 'Post-split'
        }
    
    def create_prediction_plot(self, start_frame, end_frame, show_temporal=True, show_proximity=True, highlight_errors=True):
        frame_mask = (self.predictions_df['frame'] >= start_frame) & (self.predictions_df['frame'] <= end_frame)
        filtered_df = self.predictions_df[frame_mask]
        
        if len(filtered_df) == 0:
            print(f"No nodes found in frame range {start_frame}-{end_frame}")
            return None
        
        traces = []
        
        has_labels = 'event_label' in filtered_df.columns
        
        for event_type in [0, 1, 2, 3, 4]:
            mask = filtered_df['predicted_class'] == event_type
            if not mask.any():
                continue
            
            event_df = filtered_df[mask]
            
            if has_labels and highlight_errors:
                correct_mask = event_df['predicted_class'] == event_df['event_label']
                correct_df = event_df[correct_mask]
                wrong_df = event_df[~correct_mask]
                
                if len(correct_df) > 0:
                    hover_text_correct = [
                        f"Particle: {row['particle']}<br>Frame: {row['frame']}<br>Predicted: {self.event_names[event_type]}<br>Confidence: {row['confidence']:.3f}<br>✓ Correct"
                        for _, row in correct_df.iterrows()
                    ]
                    
                    traces.append(go.Scatter3d(
                        x=correct_df['x'],
                        y=correct_df['y'],
                        z=correct_df['z'],
                        mode='markers',
                        name=f"{self.event_names[event_type]} (correct)",
                        marker=dict(
                            size=6,
                            color=self.event_colors[event_type],
                            symbol='circle',
                            line=dict(width=1, color='white')
                        ),
                        text=hover_text_correct,
                        hoverinfo='text',
                        showlegend=True
                    ))
                
                if len(wrong_df) > 0:
                    hover_text_wrong = [
                        f"Particle: {row['particle']}<br>Frame: {row['frame']}<br>Predicted: {self.event_names[event_type]}<br>True: {self.event_names[int(row['event_label'])]}<br>Confidence: {row['confidence']:.3f}<br>✗ Wrong"
                        for _, row in wrong_df.iterrows()
                    ]
                    
                    traces.append(go.Scatter3d(
                        x=wrong_df['x'],
                        y=wrong_df['y'],
                        z=wrong_df['z'],
                        mode='markers',
                        name=f"{self.event_names[event_type]} (wrong)",
                        marker=dict(
                            size=10,
                            color=self.event_colors[event_type],
                            symbol='square',
                            line=dict(width=2, color='black')
                        ),
                        text=hover_text_wrong,
                        hoverinfo='text',
                        showlegend=True
                    ))
            else:
                hover_text = [
                    f"Particle: {row['particle']}<br>Frame: {row['frame']}<br>Type: {self.event_names[event_type]}<br>Confidence: {row['confidence']:.3f}"
                    for _, row in event_df.iterrows()
                ]
                
                traces.append(go.Scatter3d(
                    x=event_df['x'],
                    y=event_df['y'],
                    z=event_df['z'],
                    mode='markers',
                    name=self.event_names[event_type],
                    marker=dict(
                        size=6,
                        color=self.event_colors[event_type],
                        line=dict(width=1, color='white')
                    ),
                    text=hover_text,
                    hoverinfo='text'
                ))
        
        edge_traces = self._create_edge_traces(filtered_df, show_temporal, show_proximity)
        traces.extend(edge_traces)
        
        trajectory_traces = self._create_trajectory_traces(filtered_df)
        traces.extend(trajectory_traces)
        
        layout = go.Layout(
            title=f'Particle Predictions (Frames {start_frame}-{end_frame})',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return go.Figure(data=traces, layout=layout)
    
    def _create_edge_traces(self, filtered_df, show_temporal, show_proximity):
        traces = []
        
        node_positions = {}
        for _, row in filtered_df.iterrows():
            key = (int(row['frame']), int(row['particle']))
            node_positions[key] = np.array([row['x'], row['y'], row['z']])
        
        temporal_edges = []
        proximity_edges = []
        
        if hasattr(self.graph, 'edge_index'):
            edge_index = self.graph.edge_index.cpu().numpy()
            frames = self.graph.frame_numbers.cpu().numpy()
            particles = self.graph.particle_ids.cpu().numpy()
            
            for i in range(edge_index.shape[1]):
                src_idx, dst_idx = edge_index[:, i]
                src_key = (int(frames[src_idx]), int(particles[src_idx]))
                dst_key = (int(frames[dst_idx]), int(particles[dst_idx]))
                
                if src_key in node_positions and dst_key in node_positions:
                    src_pos = node_positions[src_key]
                    dst_pos = node_positions[dst_key]
                    
                    if hasattr(self.graph, 'edge_type'):
                        edge_type = self.graph.edge_type[i].item()
                    else:
                        edge_type = 1 if frames[src_idx] == frames[dst_idx] else 0
                    
                    edge_trace = [src_pos, dst_pos, [None, None, None]]
                    
                    if edge_type == 0:
                        temporal_edges.extend(edge_trace)
                    else:
                        proximity_edges.extend(edge_trace)
        
        if show_temporal and temporal_edges:
            temporal_array = np.array(temporal_edges)
            traces.append(go.Scatter3d(
                x=temporal_array[:, 0],
                y=temporal_array[:, 1],
                z=temporal_array[:, 2],
                mode='lines',
                name='Temporal edges',
                line=dict(color='orange', width=3),
                hoverinfo='skip'
            ))
        
        if show_proximity and proximity_edges:
            proximity_array = np.array(proximity_edges)
            traces.append(go.Scatter3d(
                x=proximity_array[:, 0],
                y=proximity_array[:, 1],
                z=proximity_array[:, 2],
                mode='lines',
                name='Proximity edges',
                line=dict(color='black', width=1),
                opacity=0.7,
                hoverinfo='skip'
            ))
        
        return traces
    
    def _create_trajectory_traces(self, filtered_df):
        traces = []
        
        for pid in filtered_df['particle'].unique():
            particle_df = filtered_df[filtered_df['particle'] == pid].sort_values('frame')
            
            if len(particle_df) > 1:
                traces.append(go.Scatter3d(
                    x=particle_df['x'],
                    y=particle_df['y'],
                    z=particle_df['z'],
                    mode='lines',
                    line=dict(color='lightblue', width=2),
                    opacity=0.5,
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        return traces
    
    def create_animation(self, start_frame, end_frame, window_size=5):
        frames = []
        
        for frame_start in range(start_frame, end_frame - window_size + 1):
            frame_end = frame_start + window_size - 1
            fig = self.create_prediction_plot(frame_start, frame_end, highlight_errors=True)
            
            if fig:
                frames.append(go.Frame(
                    data=fig.data,
                    name=str(frame_start),
                    layout=go.Layout(title_text=f"Frames {frame_start}-{frame_end}")
                ))
        
        if not frames:
            return None
        
        initial_fig = self.create_prediction_plot(start_frame, start_frame + window_size - 1, highlight_errors=True)
        initial_fig.frames = frames
        
        initial_fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': 'Play', 'method': 'animate', 
                     'args': [None, {'frame': {'duration': 500}}]},
                    {'label': 'Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
                ]
            }],
            sliders=[{
                'steps': [
                    {'args': [[f.name], {'frame': {'duration': 0}, 'mode': 'immediate'}],
                     'label': f"Frame {f.name}", 'method': 'animate'}
                    for f in frames
                ],
                'active': 0,
                'x': 0.1,
                'len': 0.9
            }]
        )
        
        return initial_fig
    
    def create_error_plot(self, start_frame, end_frame):
        frame_mask = (self.predictions_df['frame'] >= start_frame) & (self.predictions_df['frame'] <= end_frame)
        filtered_df = self.predictions_df[frame_mask]
        
        if 'event_label' not in filtered_df.columns:
            print("No ground truth labels available for error visualization")
            return None
        
        wrong_mask = filtered_df['predicted_class'] != filtered_df['event_label']
        wrong_df = filtered_df[wrong_mask]
        
        if len(wrong_df) == 0:
            print(f"No prediction errors found in frame range {start_frame}-{end_frame}")
            return None
        
        traces = []
        
        hover_text = [
            f"Particle: {row['particle']}<br>Frame: {row['frame']}<br>Predicted: {self.event_names[int(row['predicted_class'])]}<br>True: {self.event_names[int(row['event_label'])]}<br>Confidence: {row['confidence']:.3f}"
            for _, row in wrong_df.iterrows()
        ]
        
        traces.append(go.Scatter3d(
            x=wrong_df['x'],
            y=wrong_df['y'],
            z=wrong_df['z'],
            mode='markers',
            name='Prediction errors',
            marker=dict(
                size=12,
                color=wrong_df['confidence'],
                colorscale='Reds_r',
                symbol='square',
                line=dict(width=2, color='black'),
                colorbar=dict(title="Confidence", thickness=15, x=1.02)
            ),
            text=hover_text,
            hoverinfo='text'
        ))
        
        trajectory_traces = self._create_trajectory_traces(wrong_df)
        traces.extend(trajectory_traces)
        
        layout = go.Layout(
            title=f'Prediction Errors (Frames {start_frame}-{end_frame})<br>{len(wrong_df)} errors out of {len(filtered_df)} predictions',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            hovermode='closest',
            showlegend=True
        )
        
        return go.Figure(data=traces, layout=layout)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading and normalizing data...")
    df_norm, norm_params, graph = norm_data_with_preprocessor(
        "data/tracked_simdata_dirty/tracked_particles_3d_00000.csv",
        radius_buff=0,
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
    
    print("\nCreating visualization...")
    visualizer = PredictionVisualizer(graph, df_with_predictions)
    
    frame_min = int(df_with_predictions['frame'].min())
    frame_max = int(df_with_predictions['frame'].max())
    
    fig = visualizer.create_prediction_plot(
        frame_min, 
        min(frame_min + 10, frame_max),
        show_temporal=True,
        show_proximity=True,
        highlight_errors=True
    )
    
    if fig:
        fig.write_html("predictions_visualization.html")
        print("Saved visualization to predictions_visualization.html")
        
        anim_fig = visualizer.create_animation(
            frame_min,
            min(frame_min + 20, frame_max),
            window_size=5
        )
        
        if anim_fig:
            anim_fig.write_html("predictions_animation.html")
            print("Saved animation to predictions_animation.html")
        
        error_fig = visualizer.create_error_plot(
            frame_min,
            min(frame_min + 20, frame_max)
        )
        
        if error_fig:
            error_fig.write_html("prediction_errors.html")
            print("Saved error visualization to prediction_errors.html")