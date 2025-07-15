import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

from preprocessor import load_preprocessed_data


class MatplotlibVisualizer:
    def __init__(self, graph):
        self.graph = graph
        self.event_colors = {0: 'blue', 1: 'red', 2: 'green'}
        self.event_names = {0: 'Non-event', 1: 'Merge', 2: 'Split'}
    
    def plot_frame_range(self, start_frame, end_frame, show_temporal=True, show_proximity=True):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        frame_mask = (self.graph.frame_numbers >= start_frame) & (self.graph.frame_numbers <= end_frame)
        node_indices = torch.where(frame_mask)[0]
        
        if len(node_indices) == 0:
            print(f"No nodes in frame range {start_frame}-{end_frame}")
            return fig, ax
        
        positions = self.graph.pos[node_indices].numpy()
        labels = self.graph.y[node_indices].numpy()
        frames = self.graph.frame_numbers[node_indices].numpy()
        particle_ids = self.graph.particle_ids[node_indices].numpy()
        
        node_mapping = {old.item(): new for new, old in enumerate(node_indices)}
        
        # Plot nodes by event type
        for event_type, color in self.event_colors.items():
            mask = labels == event_type
            if mask.any():
                ax.scatter(
                    positions[mask, 0],
                    positions[mask, 1],
                    positions[mask, 2],
                    c=color,
                    s=50,
                    alpha=0.8,
                    edgecolors='white',
                    linewidth=1,
                    label=self.event_names[event_type]
                )
        
        # Plot edges
        temporal_count = 0
        proximity_count = 0
        
        for i in range(self.graph.edge_index.shape[1]):
            src, dst = self.graph.edge_index[:, i].tolist()
            
            if src in node_mapping and dst in node_mapping:
                src_idx = node_mapping[src]
                dst_idx = node_mapping[dst]
                
                edge_type = self.graph.edge_type[i].item() if hasattr(self.graph, 'edge_type') else 0
                
                if edge_type == 0 and show_temporal:
                    ax.plot(
                        [positions[src_idx, 0], positions[dst_idx, 0]],
                        [positions[src_idx, 1], positions[dst_idx, 1]],
                        [positions[src_idx, 2], positions[dst_idx, 2]],
                        'orange', linewidth=2, alpha=0.8
                    )
                    temporal_count += 1
                elif edge_type == 1 and show_proximity:
                    ax.plot(
                        [positions[src_idx, 0], positions[dst_idx, 0]],
                        [positions[src_idx, 1], positions[dst_idx, 1]],
                        [positions[src_idx, 2], positions[dst_idx, 2]],
                        'gray', linewidth=0.5, alpha=0.3
                    )
                    proximity_count += 1
        
        # Plot particle trajectories
        for pid in np.unique(particle_ids):
            mask = particle_ids == pid
            particle_positions = positions[mask]
            particle_frames = frames[mask]
            
            if len(particle_positions) > 1:
                sort_idx = np.argsort(particle_frames)
                sorted_pos = particle_positions[sort_idx]
                
                ax.plot(
                    sorted_pos[:, 0],
                    sorted_pos[:, 1],
                    sorted_pos[:, 2],
                    'lightblue', linewidth=1.5, alpha=0.5
                )
        
        # Legend
        legend_elements = ax.get_legend_handles_labels()[0]
        if show_temporal:
            legend_elements.append(Line2D([0], [0], color='orange', lw=2, label=f'Temporal ({temporal_count})'))
        if show_proximity:
            legend_elements.append(Line2D([0], [0], color='gray', lw=2, label=f'Proximity ({proximity_count})'))
        legend_elements.append(Line2D([0], [0], color='lightblue', lw=2, label='Trajectories'))
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Particle Graph (Frames {start_frame}-{end_frame})')
        
        return fig, ax
    
    def create_multi_view(self, start_frame, end_frame):
        fig = plt.figure(figsize=(15, 5))
        
        # Get data
        frame_mask = (self.graph.frame_numbers >= start_frame) & (self.graph.frame_numbers <= end_frame)
        node_indices = torch.where(frame_mask)[0]
        
        if len(node_indices) == 0:
            return fig
        
        positions = self.graph.pos[node_indices].numpy()
        labels = self.graph.y[node_indices].numpy()
        
        # XY view
        ax1 = fig.add_subplot(131)
        for event_type, color in self.event_colors.items():
            mask = labels == event_type
            if mask.any():
                ax1.scatter(positions[mask, 0], positions[mask, 1], 
                          c=color, s=30, alpha=0.7, label=self.event_names[event_type])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('XY Projection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # XZ view
        ax2 = fig.add_subplot(132)
        for event_type, color in self.event_colors.items():
            mask = labels == event_type
            if mask.any():
                ax2.scatter(positions[mask, 0], positions[mask, 2], 
                          c=color, s=30, alpha=0.7)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_title('XZ Projection')
        ax2.grid(True, alpha=0.3)
        
        # YZ view
        ax3 = fig.add_subplot(133)
        for event_type, color in self.event_colors.items():
            mask = labels == event_type
            if mask.any():
                ax3.scatter(positions[mask, 1], positions[mask, 2], 
                          c=color, s=30, alpha=0.7)
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        ax3.set_title('YZ Projection')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f'Multi-view Projections (Frames {start_frame}-{end_frame})')
        plt.tight_layout()
        
        return fig
    
    def animate_frames(self, start_frame, end_frame, window_size=5, interval=500):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame_idx):
            ax.clear()
            current_start = start_frame + frame_idx
            current_end = min(current_start + window_size - 1, end_frame)
            
            self._plot_single_frame(ax, current_start, current_end)
            ax.set_title(f'Frames {current_start}-{current_end}')
        
        frames = end_frame - start_frame - window_size + 2
        anim = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=True)
        
        return fig, anim
    
    def _plot_single_frame(self, ax, start_frame, end_frame):
        frame_mask = (self.graph.frame_numbers >= start_frame) & (self.graph.frame_numbers <= end_frame)
        node_indices = torch.where(frame_mask)[0]
        
        if len(node_indices) == 0:
            return
        
        positions = self.graph.pos[node_indices].numpy()
        labels = self.graph.y[node_indices].numpy()
        
        for event_type, color in self.event_colors.items():
            mask = labels == event_type
            if mask.any():
                ax.scatter(
                    positions[mask, 0],
                    positions[mask, 1],
                    positions[mask, 2],
                    c=color, s=50, alpha=0.8, edgecolors='white', linewidth=1
                )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Directory with processed data')
    parser.add_argument('--graph-idx', type=int, default=0, help='Graph index')
    parser.add_argument('--start-frame', type=int, required=True, help='Start frame')
    parser.add_argument('--end-frame', type=int, required=True, help='End frame')
    parser.add_argument('--no-temporal', action='store_true', help='Hide temporal edges')
    parser.add_argument('--no-proximity', action='store_true', help='Hide proximity edges')
    parser.add_argument('--multi-view', action='store_true', help='Show 2D projections')
    parser.add_argument('--animate', action='store_true', help='Create animation')
    parser.add_argument('--save', help='Save figure to file')
    
    args = parser.parse_args()
    
    _, _, _, data = load_preprocessed_data(args.data_dir, batch_size=1)
    graph = data['graphs'][args.graph_idx]
    
    viz = MatplotlibVisualizer(graph)
    
    if args.multi_view:
        fig = viz.create_multi_view(args.start_frame, args.end_frame)
    elif args.animate:
        fig, anim = viz.animate_frames(args.start_frame, args.end_frame)
        if args.save:
            anim.save(args.save, writer='pillow')
    else:
        fig, ax = viz.plot_frame_range(
            args.start_frame, 
            args.end_frame,
            show_temporal=not args.no_temporal,
            show_proximity=not args.no_proximity
        )
    
    if args.save and not args.animate:
                    plt.savefig(args.save, dpi=150, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    main()