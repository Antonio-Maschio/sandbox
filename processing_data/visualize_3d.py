import argparse
import plotly.graph_objects as go
import numpy as np
import torch
from pathlib import Path

from preprocessor import load_preprocessed_data


class ParticleGraphVisualizer:
    def __init__(self, graph):
        self.graph = graph
        self.event_colors = {
            0: 'blue',      # Non-event
            1: 'red',       # Merge
            2: 'green',      # Split
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
    
    def extract_frame_subgraph(self, start_frame, end_frame):
        frame_mask = (self.graph.frame_numbers >= start_frame) & (self.graph.frame_numbers <= end_frame)
        node_indices = torch.where(frame_mask)[0]
        node_mapping = {old: new for new, old in enumerate(node_indices.tolist())}
        
        filtered_edges = []
        filtered_edge_types = []
        filtered_edge_attrs = []
        
        for i in range(self.graph.edge_index.shape[1]):
            src, dst = self.graph.edge_index[:, i].tolist()
            if src in node_mapping and dst in node_mapping:
                filtered_edges.append([node_mapping[src], node_mapping[dst]])
                if hasattr(self.graph, 'edge_type'):
                    filtered_edge_types.append(self.graph.edge_type[i].item())
                else:
                    filtered_edge_types.append(0)
                filtered_edge_attrs.append(self.graph.edge_attr[i].item())
        
        return {
            'positions': self.graph.pos[node_indices],
            'labels': self.graph.y[node_indices],
            'frames': self.graph.frame_numbers[node_indices],
            'particle_ids': self.graph.particle_ids[node_indices],
            'edges': filtered_edges,
            'edge_types': filtered_edge_types,
            'edge_attrs': filtered_edge_attrs,
            'node_mapping': node_mapping
        }
    
    def create_3d_plot(self, start_frame, end_frame, show_temporal=True, show_proximity=True):
        subgraph = self.extract_frame_subgraph(start_frame, end_frame)
        
        if len(subgraph['positions']) == 0:
            print(f"No nodes found in frame range {start_frame}-{end_frame}")
            return None
        
        pos = subgraph['positions'].numpy()
        labels = subgraph['labels'].numpy()
        frames = subgraph['frames'].numpy()
        particle_ids = subgraph['particle_ids'].numpy()
        
        traces = []
        
        # Node traces by event type
        for event_type in [0, 1, 2, 3, 4]:
            mask = labels == event_type
            if not mask.any():
                continue
            
            hover_text = [
                f"Particle: {pid}<br>Frame: {f}<br>Type: {self.event_names[event_type]}"
                for pid, f in zip(particle_ids[mask], frames[mask])
            ]
            
            traces.append(go.Scatter3d(
                x=pos[mask, 0],
                y=pos[mask, 1],
                z=pos[mask, 2],
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
        
        # Edge traces
        temporal_edges = []
        proximity_edges = []
        
        for i, (edge, edge_type) in enumerate(zip(subgraph['edges'], subgraph['edge_types'])):
            src_pos = pos[edge[0]]
            dst_pos = pos[edge[1]]
            
            edge_trace = [src_pos, dst_pos, [None, None, None]]
            
            if edge_type == 0:  # Temporal
                temporal_edges.extend(edge_trace)
            else:  # Proximity
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
        
        # Particle trajectories
        trajectory_traces = self._create_trajectory_traces(subgraph, pos)
        traces.extend(trajectory_traces)
        
        # Layout
        layout = go.Layout(
            title=f'Particle Graph Visualization (Frames {start_frame}-{end_frame})',
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
    
    def _create_trajectory_traces(self, subgraph, positions):
        traces = []
        particle_ids = subgraph['particle_ids'].numpy()
        frames = subgraph['frames'].numpy()
        
        for pid in np.unique(particle_ids):
            mask = particle_ids == pid
            particle_frames = frames[mask]
            particle_pos = positions[mask]
            
            sort_idx = np.argsort(particle_frames)
            sorted_pos = particle_pos[sort_idx]
            
            if len(sorted_pos) > 1:
                traces.append(go.Scatter3d(
                    x=sorted_pos[:, 0],
                    y=sorted_pos[:, 1],
                    z=sorted_pos[:, 2],
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
            fig = self.create_3d_plot(frame_start, frame_end)
            
            if fig:
                frames.append(go.Frame(
                    data=fig.data,
                    name=str(frame_start),
                    layout=go.Layout(title_text=f"Frames {frame_start}-{frame_end}")
                ))
        
        if not frames:
            return None
        
        initial_fig = self.create_3d_plot(start_frame, start_frame + window_size - 1)
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


def main():
    parser = argparse.ArgumentParser(description='Visualize particle graphs in 3D')
    parser.add_argument('--data-dir', required=True, help='Directory with processed data')
    parser.add_argument('--graph-idx', type=int, default=0, help='Graph index to visualize')
    parser.add_argument('--start-frame', type=int, required=True, help='Start frame')
    parser.add_argument('--end-frame', type=int, required=True, help='End frame')
    parser.add_argument('--no-temporal', action='store_true', help='Hide temporal edges')
    parser.add_argument('--no-proximity', action='store_true', help='Hide proximity edges')
    parser.add_argument('--animate', action='store_true', help='Create animation')
    parser.add_argument('--window-size', type=int, default=5, help='Window size for animation')
    parser.add_argument('--output', help='Output HTML file')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_dir}")
    _, _, _, data = load_preprocessed_data(args.data_dir, batch_size=1)
    graphs = data['graphs']
    
    if args.graph_idx >= len(graphs):
        print(f"Error: Graph index {args.graph_idx} out of range (0-{len(graphs)-1})")
        return
    
    graph = graphs[args.graph_idx]
    visualizer = ParticleGraphVisualizer(graph)
    
    if args.animate:
        fig = visualizer.create_animation(args.start_frame, args.end_frame, args.window_size)
    else:
        fig = visualizer.create_3d_plot(
            args.start_frame, 
            args.end_frame,
            show_temporal=not args.no_temporal,
            show_proximity=not args.no_proximity
        )
    
    if fig:
        if args.output:
            fig.write_html(args.output)
            print(f"Saved visualization to {args.output}")
        else:
            fig.show()
    else:
        print("Failed to create visualization")


if __name__ == '__main__':
    main()