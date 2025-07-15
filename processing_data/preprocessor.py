import pickle
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch

from processing_data.dataset import ParticleDataset
# from dataset import ParticleDataset


class DataPreprocessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.normalization_params = {}
    
    def normalize_features(self, method='standard'):
        node_features = torch.cat([g.x for g in self.dataset.graphs])
        
        if method == 'standard':
            mean = node_features.mean(dim=0)
            std = node_features.std(dim=0)
            std[std == 0] = 1.0
            
            self.normalization_params = {'method': 'standard', 'mean': mean, 'std': std}
            
            for graph in self.dataset.graphs:
                graph.x = (graph.x - mean) / std
        
        elif method == 'minmax':
            min_val = node_features.min(dim=0)[0]
            max_val = node_features.max(dim=0)[0]
            range_val = max_val - min_val
            range_val[range_val == 0] = 1.0
            
            self.normalization_params = {'method': 'minmax', 'min': min_val, 'max': max_val}
            
            for graph in self.dataset.graphs:
                graph.x = (graph.x - min_val) / range_val
    
    def normalize_edges(self, method='standard'):
        edge_features = torch.cat([g.edge_attr for g in self.dataset.graphs])
        
        if method == 'standard':
            mean = edge_features.mean()
            std = edge_features.std()
            if std == 0:
                std = 1.0
            
            self.normalization_params['edge_mean'] = mean
            self.normalization_params['edge_std'] = std
            
            for graph in self.dataset.graphs:
                graph.edge_attr = (graph.edge_attr - mean) / std
        
        elif method == 'minmax':
            min_val = edge_features.min()
            max_val = edge_features.max()
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1.0
            
            self.normalization_params['edge_min'] = min_val
            self.normalization_params['edge_max'] = max_val
            
            for graph in self.dataset.graphs:
                graph.edge_attr = (graph.edge_attr - min_val) / range_val
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, seed=42):
        torch.manual_seed(seed)
        n = len(self.dataset)
        indices = torch.randperm(n).tolist()
        
        train_size = int(train_ratio * n)
        val_size = int(val_ratio * n)
        
        return {
            'train': indices[:train_size],
            'val': indices[train_size:train_size + val_size],
            'test': indices[train_size + val_size:]
        }


def save_preprocessed_data(save_dir, dataset, preprocessor, splits):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    data = {
        'graphs': dataset.graphs,
        'splits': splits,
        'normalization': preprocessor.normalization_params,
        'statistics': dataset.stats
    }
    
    with open(save_path / 'processed_data.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    metadata = {
        'num_graphs': len(dataset),
        'train_size': len(splits['train']),
        'val_size': len(splits['val']),
        'test_size': len(splits['test']),
        'statistics': {k: v.tolist() if isinstance(v, torch.Tensor) else v 
                      for k, v in dataset.stats.items()}
    }
    
    with open(save_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return save_path


def load_preprocessed_data(data_dir, batch_size=32, num_workers=0):
    data_path = Path(data_dir)
    
    with open(data_path / 'processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    graphs = data['graphs']
    splits = data['splits']
    
    def collate_fn(batch):
        return Batch.from_data_list(batch)
    
    loaders = {}
    for split_name, indices in splits.items():
        subset = Subset(graphs, indices)
        loaders[split_name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    return loaders['train'], loaders['val'], loaders['test'], data