import plotly.graph_objects as go
import numpy as np
import ast
import pandas as pd

class TrackVisualizer:
    def __init__(self, df_tracks):
        self.df = df_tracks
        self.event_colors = {
            0: '#3498db',
            1: '#e74c3c',
            2: '#2ecc71',
            3: '#c0392b',
            4: '#27ae60'
        }
        self.event_names = {
            0: 'Non-event',
            1: 'Merge',
            2: 'Split',
            3: 'Post-merge',
            4: 'Post-split'
        }
    
    def parse_parent_ids(self, parent_str):
        if pd.isna(parent_str) or parent_str == "[]":
            return []
        try:
            return ast.literal_eval(parent_str)
        except:
            return []
    
    def create_track_visualization(self, start_frame, end_frame, show_lineage=True):
        mask = (self.df['frame'] >= start_frame) & (self.df['frame'] <= end_frame)
        filtered_df = self.df[mask].copy()
        
        if len(filtered_df) == 0:
            return None
        
        traces = []
        
        for event_type in range(5):
            event_df = filtered_df[filtered_df['predicted_class'] == event_type]
            if len(event_df) == 0:
                continue
            
            hover_text = []
            for _, row in event_df.iterrows():
                parents = self.parse_parent_ids(row['parent_ids_str'])
                parent_text = f"Parents: {parents}" if parents else "No parents"
                hover_text.append(
                    f"Particle: {row['particle']}<br>"
                    f"Frame: {row['frame']}<br>"
                    f"Type: {self.event_names[event_type]}<br>"
                    f"{parent_text}"
                )
            
            traces.append(go.Scatter3d(
                x=event_df['x'],
                y=event_df['y'],
                z=event_df['z'],
                mode='markers',
                name=self.event_names[event_type],
                marker=dict(
                    size=8,
                    color=self.event_colors[event_type],
                    line=dict(width=1, color='white')
                ),
                text=hover_text,
                hoverinfo='text'
            ))
        
        if show_lineage:
            lineage_traces = self._create_lineage_connections(filtered_df)
            traces.extend(lineage_traces)
        
        particle_traces = self._create_particle_trajectories(filtered_df)
        traces.extend(particle_traces)
        
        layout = go.Layout(
            title=f'Reconstructed Tracks (Frames {start_frame}-{end_frame})',
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
    
    def _create_lineage_connections(self, filtered_df):
        traces = []
        
        all_frames = sorted(filtered_df['frame'].unique())
        frame_data = {f: filtered_df[filtered_df['frame'] == f] for f in all_frames}
        
        for _, row in filtered_df.iterrows():
            parent_ids = self.parse_parent_ids(row['parent_ids_str'])
            
            for parent_id in parent_ids:
                parent_data = None
                for frame in range(int(row['frame'])-5, int(row['frame'])):
                    if frame in frame_data:
                        parent_row = frame_data[frame][frame_data[frame]['particle'] == parent_id]
                        if not parent_row.empty:
                            parent_data = parent_row.iloc[0]
                            break
                
                if parent_data is not None:
                    line_color = '#ff7f0e' if row['predicted_class'] in [1, 3] else '#1f77b4'
                    
                    traces.append(go.Scatter3d(
                        x=[parent_data['x'], row['x']],
                        y=[parent_data['y'], row['y']],
                        z=[parent_data['z'], row['z']],
                        mode='lines',
                        line=dict(color=line_color, width=3),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        return traces
    
    def _create_particle_trajectories(self, filtered_df):
        traces = []
        
        for particle_id in filtered_df['particle'].unique():
            particle_data = filtered_df[filtered_df['particle'] == particle_id].sort_values('frame')
            
            if len(particle_data) > 1:
                traces.append(go.Scatter3d(
                    x=particle_data['x'],
                    y=particle_data['y'],
                    z=particle_data['z'],
                    mode='lines',
                    line=dict(color='lightgray', width=1),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        return traces
    
    def create_lineage_graph(self, max_particles=50):
        import networkx as nx
        from plotly.subplots import make_subplots
        
        G = nx.DiGraph()
        
        sampled_df = self.df.sample(n=min(max_particles, len(self.df)))
        
        for _, row in sampled_df.iterrows():
            particle_id = row['particle']
            parent_ids = self.parse_parent_ids(row['parent_ids_str'])
            
            G.add_node(particle_id, 
                      frame=row['frame'],
                      event_type=row['predicted_class'])
            
            for parent_id in parent_ids:
                G.add_edge(parent_id, particle_id)
        
        if len(G.nodes()) == 0:
            return None
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        edge_trace = go.Scatter(
            x=[], y=[], 
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
        
        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                size=10,
                line_width=2
            )
        )
        
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
        
        node_colors = []
        node_text = []
        for node in G.nodes():
            event_type = G.nodes[node]['event_type']
            frame = G.nodes[node]['frame']
            node_colors.append(self.event_colors[event_type])
            node_text.append(f"Particle {node}<br>Frame {frame}<br>{self.event_names[event_type]}")
        
        node_trace['marker']['color'] = node_colors
        node_trace['text'] = node_text
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title='Particle Lineage Network',
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig


def visualize_reconstructed_tracks(reconstructed_csv, output_prefix="track_viz"):
    df = pd.read_csv(reconstructed_csv)
    
    visualizer = TrackVisualizer(df)
    
    frame_min = int(df['frame'].min())
    frame_max = int(df['frame'].max())
    
    fig_3d = visualizer.create_track_visualization(
        frame_min,
        min(frame_min + 20, frame_max),
        show_lineage=True
    )
    
    if fig_3d:
        fig_3d.write_html(f"{output_prefix}_3d.html")
        print(f"Saved 3D track visualization to {output_prefix}_3d.html")
    
    fig_network = visualizer.create_lineage_graph(max_particles=100)
    if fig_network:
        fig_network.write_html(f"{output_prefix}_network.html")
        print(f"Saved lineage network to {output_prefix}_network.html")


if __name__ == "__main__":
    visualize_reconstructed_tracks("reconstructed_tracks.csv", "track_viz")