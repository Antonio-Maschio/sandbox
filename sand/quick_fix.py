#!/usr/bin/env python3
"""
Quick Fix for PyTorch 2.6 Compatibility Issue

This script fixes the weights_only=True issue without reprocessing all data.
Run this if you encounter UnpicklingError during training.

Usage:
    python quick_fix.py --data_dir model/processed_data_vf
"""

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import json

def fix_torch_compatibility(data_dir):
    """
    Fix PyTorch 2.6 compatibility by resaving chunks with proper settings.
    """
    data_path = Path(data_dir)
    
    # Check if metadata exists
    metadata_file = data_path / "metadata.json"
    if not metadata_file.exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        return False
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    num_chunks = metadata['num_chunks']
    print(f"🔧 Fixing {num_chunks} chunks for PyTorch 2.6 compatibility...")
    
    # Process each chunk
    for i in tqdm(range(num_chunks), desc="Fixing chunks"):
        chunk_file = data_path / f"processed_graphs_chunk_{i:04d}.pt"
        
        if not chunk_file.exists():
            print(f"⚠️  Chunk {i} not found: {chunk_file}")
            continue
        
        try:
            # Load with weights_only=False
            chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
            
            # Resave with compatibility settings
            torch.save(chunk_data, chunk_file, _use_new_zipfile_serialization=False)
            
        except Exception as e:
            print(f"❌ Error fixing chunk {i}: {e}")
            return False
    
    print("✅ All chunks fixed for PyTorch 2.6 compatibility!")
    return True

def test_loading(data_dir):
    """
    Test if the fixed chunks can be loaded properly.
    """
    print("🧪 Testing fixed data loading...")
    
    data_path = Path(data_dir)
    
    # Try loading first chunk
    chunk_file = data_path / "processed_graphs_chunk_0000.pt"
    if not chunk_file.exists():
        print("❌ No chunks found to test")
        return False
    
    try:
        # Test with default PyTorch 2.6 settings (should work now)
        chunk_data = torch.load(chunk_file, map_location='cpu')
        print(f"✅ Successfully loaded chunk with {len(chunk_data)} graphs")
        
        # Test graph structure
        if len(chunk_data) > 0:
            graph = chunk_data[0]
            print(f"✅ Graph structure: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")
            
            # Check for masks
            if hasattr(graph, 'direct_temporal_mask'):
                print("✅ Gap-aware masks present")
            else:
                print("⚠️  No gap-aware masks found")
        
        return True
        
    except Exception as e:
        print(f"❌ Loading test failed: {e}")
        return False

def create_safe_data_loader(data_dir, batch_size=4):
    """
    Create a safe data loader that should work with the fixes.
    """
    print("🔄 Creating safe data loader...")
    
    try:
        # Import with updated code
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        
        from data_loader_for_training import create_data_loaders
        
        # Create loaders with safe settings
        train_loader, val_loader = create_data_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            val_split=0.2,
            num_workers=0,  # No multiprocessing to avoid pickle issues
            pin_memory=False,  # Disable for stability
            preload_chunks=False
        )
        
        print(f"✅ Safe data loaders created:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
        # Test loading one batch
        batch = next(iter(train_loader))
        print(f"✅ Batch loading successful: {batch.x.shape[0]} nodes")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix PyTorch 2.6 compatibility")
    parser.add_argument("--data_dir", type=str, default="model/processed_data_vf",
                       help="Directory with processed data")
    parser.add_argument("--test_only", action="store_true",
                       help="Only test loading without fixing")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for testing")
    
    args = parser.parse_args()
    
    print("🔧 PyTorch 2.6 Compatibility Fix")
    print("=" * 40)
    
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    if args.test_only:
        print("🧪 Testing current data loading...")
        success = test_loading(args.data_dir)
        if success:
            success = create_safe_data_loader(args.data_dir, args.batch_size)
    else:
        print("🔧 Fixing data compatibility...")
        success = fix_torch_compatibility(args.data_dir)
        
        if success:
            print("\n🧪 Testing fixed data...")
            success = test_loading(args.data_dir)
            
            if success:
                success = create_safe_data_loader(args.data_dir, args.batch_size)
    
    if success:
        print("\n🎉 Fix completed successfully!")
        print("\nNow you can run training:")
        print("python pipe.py --train")
    else:
        print("\n❌ Fix failed. Manual intervention needed.")
        print("\nAlternative solutions:")
        print("1. Rerun preprocessing: python preprocess_data.py")
        print("2. Check PyTorch version: pip list | grep torch")
        print("3. Use num_workers=0 in data loaders")

if __name__ == "__main__":
    main()