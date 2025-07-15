import re
import glob
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from processing_data.graph_builder import build_particle_graph, build_unlabeled_graph
# from graph_builder import build_particle_graph, build_unlabeled_graph


class ParticleDataset(Dataset):
    def __init__(self, pattern, radius_buffer=0.0, max_workers=4, labeled=True):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise ValueError(f"No files found: {pattern}")
        
        processor = partial(
            _process_file,
            radius_buffer=radius_buffer,
            labeled=labeled
        )
        
        graphs = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(processor, f): f for f in self.files}
            
            for future in tqdm(futures, desc="Loading graphs", colour="green"):
                graph = future.result()
                if graph is not None and graph.num_edges > 0:
                    graphs.append(graph)
        
        self.graphs = graphs
        self._compute_statistics()
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]
    
    def _compute_statistics(self):
        self.stats = {
            'num_graphs': len(self.graphs),
            'total_nodes': sum(g.num_nodes for g in self.graphs),
            'total_edges': sum(g.num_edges for g in self.graphs),
            'avg_nodes': sum(g.num_nodes for g in self.graphs) / len(self.graphs),
            'avg_edges': sum(g.num_edges for g in self.graphs) / len(self.graphs)
        }
        
        if hasattr(self.graphs[0], 'y'):
            class_counts = torch.zeros(3, dtype=torch.long)
            for g in self.graphs:
                for c in range(3):
                    class_counts[c] += (g.y == c).sum()
            self.stats['class_distribution'] = class_counts


def _process_file(filepath, radius_buffer, labeled):
    try:
        df = pd.read_csv(filepath)
        required_cols = ['particle', 'frame', 'x', 'y', 'z', 'mass']
        if labeled:
            required_cols.append('event_label')
        
        if not all(col in df.columns for col in required_cols):
            return None
        
        sim_id = int(re.search(r'(\d+)', Path(filepath).stem).group(1))
        
        if labeled:
            return build_particle_graph(df, radius_buffer, sim_id)
        else:
            return build_unlabeled_graph(df, radius_buffer)
            
    except Exception:
        return None