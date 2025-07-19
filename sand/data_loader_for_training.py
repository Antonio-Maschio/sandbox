#!/usr/bin/env python3
"""
Data Loader for Preprocessed Particle Data

This module provides efficient data loading for the preprocessed particle graphs.
Supports train/validation splits, efficient batching, and memory management.

Usage:
    from data_loader_for_training import PreprocessedDataset, create_data_loaders
    
    train_loader, val_loader = create_data_loaders(
        data_dir="model/processed_data_vf",
        batch_size=32,
        val_split=0.2
    )
"""

import torch
import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from typing import List, Tuple, Optional
import random
from tqdm import tqdm

class PreprocessedDataset(Dataset):
    """
    Dataset class for preprocessed particle graphs.
    Efficiently loads chunks on-demand to manage memory.
    """
    
    def __init__(self, data_dir: str, chunk_indices: Optional[List[int]] = None, 
                 preload_chunks: bool = False):
        """
        Initialize dataset from preprocessed data directory.
        
        Args:
            data_dir: Directory containing processed graph chunks
            chunk_indices: Specific chunk indices to load (for train/val split)
            preload_chunks: Whether to load all chunks into memory immediately
        """
        self.data_dir = Path(data_dir)
        
        # Load metadata
        metadata_file = self.data_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.chunk_size = self.metadata['chunk_size']
        self.num_chunks = self.metadata['num_chunks']
        self.total_graphs = self.metadata['num_graphs']
        
        # Determine which chunks to use
        if chunk_indices is not None:
            self.chunk_indices = chunk_indices
        else:
            self.chunk_indices = list(range(self.num_chunks))
        
        # Build index mapping
        self._build_index_mapping()
        
        # Cache for loaded chunks
        self.chunk_cache = {}
        
        if preload_chunks:
            self._preload_chunks()
        
        print(f"📚 Dataset initialized:")
        print(f"   Total graphs available: {self.total_graphs}")
        print(f"   Using chunks: {len(self.chunk_indices)}")
        print(f"   Accessible graphs: {len(self)}")
    
    def _build_index_mapping(self):
        """Build mapping from dataset index to (chunk_index, graph_index_in_chunk)."""
        self.index_mapping = []
        
        for chunk_idx in self.chunk_indices:
            # Calculate how many graphs are in this chunk
            start_graph = chunk_idx * self.chunk_size
            end_graph = min((chunk_idx + 1) * self.chunk_size, self.total_graphs)
            graphs_in_chunk = end_graph - start_graph
            
            for graph_idx in range(graphs_in_chunk):
                self.index_mapping.append((chunk_idx, graph_idx))
    
    def _preload_chunks(self):
        """Load all chunks into memory."""
        print("🔄 Preloading chunks into memory...")
        for chunk_idx in tqdm(self.chunk_indices, desc="Loading chunks"):
            self._load_chunk(chunk_idx)
        print("✅ All chunks preloaded!")
    
    def _load_chunk(self, chunk_idx: int) -> List:
        """Load a specific chunk from disk."""
        if chunk_idx in self.chunk_cache:
            return self.chunk_cache[chunk_idx]
        
        chunk_file = self.data_dir / f"processed_graphs_chunk_{chunk_idx:04d}.pt"
        if not chunk_file.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_file}")
        
        chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
        self.chunk_cache[chunk_idx] = chunk_data
        
        # Memory management: limit cache size
        if len(self.chunk_cache) > 10:  # Keep max 10 chunks in memory
            # Remove oldest chunk (simple FIFO)
            oldest_chunk = min(self.chunk_cache.keys())
            if oldest_chunk != chunk_idx:  # Don't remove the chunk we just loaded
                del self.chunk_cache[oldest_chunk]
        
        return chunk_data
    
    def __len__(self) -> int:
        return len(self.index_mapping)
    
    def __getitem__(self, idx: int):
        """Get a graph by index."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        chunk_idx, graph_idx = self.index_mapping[idx]
        chunk_data = self._load_chunk(chunk_idx)
        
        return chunk_data[graph_idx]
    
    def get_normalization_stats(self) -> dict:
        """Load normalization statistics."""
        stats_file = self.data_dir / "normalization_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_data_analysis(self) -> str:
        """Get data analysis if available."""
        analysis_file = self.data_dir / "data_analysis.txt"
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                return f.read()
        return "No analysis available"

class ParticleBatch(Batch):
    """
    Custom batch class that properly handles boolean masks for gap-aware training.
    (Same as before but included for completeness)
    """
    @classmethod
    def from_data_list(cls, data_list, follow_batch=None, exclude_keys=None):
        batch = super().from_data_list(data_list, follow_batch, exclude_keys)
        
        # Manually handle boolean masks
        if hasattr(data_list[0], 'direct_temporal_mask'):
            batch.direct_temporal_mask = torch.cat([data.direct_temporal_mask for data in data_list])
        if hasattr(data_list[0], 'gap_temporal_mask'):
            batch.gap_temporal_mask = torch.cat([data.gap_temporal_mask for data in data_list])
        if hasattr(data_list[0], 'proximity_mask'):
            batch.proximity_mask = torch.cat([data.proximity_mask for data in data_list])
            
        return batch

def create_train_val_split(data_dir: str, val_split: float = 0.2, 
                          random_seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Create train/validation split by chunks to ensure no data leakage.
    
    Args:
        data_dir: Directory with preprocessed data
        val_split: Fraction of data for validation
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_chunk_indices, val_chunk_indices)
    """
    # Load metadata to get number of chunks
    metadata_file = Path(data_dir) / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    num_chunks = metadata['num_chunks']
    
    # Create chunk indices and shuffle
    chunk_indices = list(range(num_chunks))
    random.seed(random_seed)
    random.shuffle(chunk_indices)
    
    # Split chunks
    val_size = int(num_chunks * val_split)
    val_chunks = chunk_indices[:val_size]
    train_chunks = chunk_indices[val_size:]
    
    print(f"📊 Data split created:")
    print(f"   Train chunks: {len(train_chunks)} ({len(train_chunks)/num_chunks*100:.1f}%)")
    print(f"   Val chunks: {len(val_chunks)} ({len(val_chunks)/num_chunks*100:.1f}%)")
    
    return train_chunks, val_chunks

def create_data_loaders(data_dir: str, batch_size: int = 32, val_split: float = 0.2,
                       num_workers: int = 2, pin_memory: bool = True,
                       preload_chunks: bool = False, random_seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders from preprocessed data.
    
    Args:
        data_dir: Directory containing preprocessed data
        batch_size: Batch size for training
        val_split: Fraction of data for validation
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        preload_chunks: Whether to preload all chunks into memory
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create train/val split
    train_chunks, val_chunks = create_train_val_split(data_dir, val_split, random_seed)
    
    # Create datasets
    train_dataset = PreprocessedDataset(
        data_dir, 
        chunk_indices=train_chunks,
        preload_chunks=preload_chunks
    )
    
    val_dataset = PreprocessedDataset(
        data_dir,
        chunk_indices=val_chunks, 
        preload_chunks=preload_chunks
    )
    
    # Create data loaders with custom batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ParticleBatch.from_data_list,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ParticleBatch.from_data_list,
        persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader

class BalancedSampler:
    """
    Sampler that balances rare event classes during training.
    """
    
    def __init__(self, dataset: PreprocessedDataset, target_distribution: Optional[dict] = None):
        """
        Initialize balanced sampler.
        
        Args:
            dataset: Dataset to sample from
            target_distribution: Target distribution for each class (optional)
        """
        self.dataset = dataset
        
        # Analyze class distribution
        print("📊 Analyzing class distribution for balanced sampling...")
        self._analyze_class_distribution()
        
        if target_distribution is None:
            # Default: Equal sampling for all event classes
            self.target_distribution = {
                0: 0.4,   # NORMAL
                1: 0.15,  # MERGE  
                2: 0.15,  # SPLIT
                3: 0.15,  # POST_MERGE
                4: 0.15   # POST_SPLIT
            }
        else:
            self.target_distribution = target_distribution
    
    def _analyze_class_distribution(self):
        """Analyze the distribution of classes in the dataset."""
        self.class_indices = {i: [] for i in range(5)}  # 5 event classes
        
        print("Scanning dataset for class distribution...")
        for idx in tqdm(range(min(len(self.dataset), 10000))):  # Sample first 10k for speed
            graph = self.dataset[idx]
            labels = graph.y
            
            for i, label in enumerate(labels):
                global_idx = idx * len(labels) + i  # Approximate global index
                self.class_indices[label.item()].append(global_idx)
        
        # Print distribution
        total_samples = sum(len(indices) for indices in self.class_indices.values())
        print("Current class distribution:")
        for class_idx, indices in self.class_indices.items():
            percentage = len(indices) / total_samples * 100
            print(f"  Class {class_idx}: {len(indices):,} ({percentage:.2f}%)")
    
    def sample_batch_indices(self, batch_size: int) -> List[int]:
        """Sample balanced batch indices."""
        batch_indices = []
        
        for class_idx, target_prob in self.target_distribution.items():
            num_samples = int(batch_size * target_prob)
            if len(self.class_indices[class_idx]) > 0:
                sampled = random.sample(
                    self.class_indices[class_idx], 
                    min(num_samples, len(self.class_indices[class_idx]))
                )
                batch_indices.extend(sampled)
        
        # Fill remaining slots randomly
        remaining = batch_size - len(batch_indices)
        if remaining > 0:
            all_indices = [idx for indices in self.class_indices.values() for idx in indices]
            batch_indices.extend(random.sample(all_indices, min(remaining, len(all_indices))))
        
        return batch_indices[:batch_size]

def test_data_loading(data_dir: str, batch_size: int = 4):
    """
    Test data loading functionality.
    """
    print(f"🧪 Testing data loading from {data_dir}")
    
    try:
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            val_split=0.2,
            num_workers=0,  # Disable multiprocessing for testing
            preload_chunks=False
        )
        
        print(f"✅ Data loaders created successfully")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
        # Test loading a batch
        print("\n🔄 Testing batch loading...")
        batch = next(iter(train_loader))
        
        print(f"✅ Batch loaded successfully:")
        print(f"   Nodes: {batch.x.shape}")
        print(f"   Edges: {batch.edge_index.shape}")
        print(f"   Edge features: {batch.edge_attr.shape}")
        print(f"   Labels: {batch.y.shape}")
        print(f"   Has masks: {hasattr(batch, 'direct_temporal_mask')}")
        
        if hasattr(batch, 'direct_temporal_mask'):
            print(f"   Direct temporal edges: {batch.direct_temporal_mask.sum()}")
            print(f"   Gap temporal edges: {batch.gap_temporal_mask.sum()}")
            print(f"   Proximity edges: {batch.proximity_mask.sum()}")
        
        # Test normalization stats
        dataset = train_loader.dataset
        stats = dataset.get_normalization_stats()
        if stats:
            print(f"\n✅ Normalization stats available:")
            print(f"   Mass: μ={stats.get('mass_mean', 0):.2f}, σ={stats.get('mass_std', 1):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing data loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the data loading functionality
    import argparse
    
    parser = argparse.ArgumentParser(description="Test preprocessed data loading")
    parser.add_argument("--data_dir", type=str, default="model/processed_data_vf",
                       help="Directory with preprocessed data")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for testing")
    
    args = parser.parse_args()
    
    success = test_data_loading(args.data_dir, args.batch_size)
    
    if success:
        print("\n🎉 Data loading test successful!")
    else:
        print("\n❌ Data loading test failed!")