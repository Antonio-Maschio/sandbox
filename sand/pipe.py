#!/usr/bin/env python3
"""
Complete Pipeline Usage Example

This script demonstrates the complete workflow:
1. Preprocess simulation data
2. Load preprocessed data
3. Train gap-aware GNN model
4. Evaluate model performance

Usage:
    python pipeline_example.py --preprocess --train --evaluate
"""

import argparse
import subprocess
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

# Import our modules
from data_loader_for_training import create_data_loaders, test_data_loading

def run_preprocessing(input_dir="data/tracked_simdata_clean", 
                     output_dir="model/processed_data_vf",
                     max_files=1000):
    """
    Run the preprocessing script.
    """
    print("🚀 Starting data preprocessing...")
    
    cmd = [
        sys.executable, "preprocess_data.py",
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--max_files", str(max_files),
        "--radius_buffer", "5.0",
        "--max_gap_frames", "3",
        "--chunk_size", "100"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Preprocessing completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Preprocessing failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ preprocess_data.py not found. Make sure it's in the current directory.")
        return False

def create_simple_model(num_features=1, hidden_channels=256, num_classes=5):
    """
    Create a simple GNN model for demonstration.
    """
    from torch_geometric.nn import GATConv
    import torch.nn as nn
    
    class SimpleGapAwareGNN(nn.Module):
        def __init__(self, num_features, hidden_channels, num_classes):
            super().__init__()
            
            self.conv1 = GATConv(num_features, hidden_channels//4, heads=4, 
                               edge_dim=3, concat=True)
            self.conv2 = GATConv(hidden_channels, hidden_channels//4, heads=4,
                               edge_dim=3, concat=True) 
            self.conv3 = GATConv(hidden_channels, hidden_channels//4, heads=4,
                               edge_dim=3, concat=True)
            
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels//2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_channels//2, num_classes)
            )
            
        def forward(self, x, edge_index, edge_attr, **kwargs):
            # Simple forward pass using enhanced edge attributes
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x = F.dropout(x, p=0.2, training=self.training)
            
            x = F.relu(self.conv2(x, edge_index, edge_attr))
            x = F.dropout(x, p=0.2, training=self.training)
            
            x = F.relu(self.conv3(x, edge_index, edge_attr))
            
            return F.log_softmax(self.classifier(x), dim=-1)
    
    return SimpleGapAwareGNN(num_features, hidden_channels, num_classes)

def train_model(data_dir="model/processed_data_vf", epochs=10, batch_size=32):
    """
    Train the GNN model on preprocessed data.
    """
    print(f"🎯 Starting model training...")
    
    # Check if data exists
    if not Path(data_dir).exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Create data loaders
    try:
        train_loader, val_loader = create_data_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            val_split=0.2,
            num_workers=2 if torch.cuda.is_available() else 0,
            preload_chunks=False  # Set to True if you have enough RAM
        )
        print(f"✅ Data loaders created: {len(train_loader)} train, {len(val_loader)} val batches")
    except Exception as e:
        print(f"❌ Failed to create data loaders: {e}")
        return False
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_simple_model(num_features=1, hidden_channels=256, num_classes=5)
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = F.nll_loss
    
    print(f"🖥️  Training on: {device}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.edge_attr,
                       edge_type=batch.edge_type if hasattr(batch, 'edge_type') else None,
                       batch=batch.batch)
            
            # Compute loss
            loss = criterion(out, batch.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
    
    # Save model
    model_dir = Path("model/trained_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'model_config': {
            'num_features': 1,
            'hidden_channels': 256,
            'num_classes': 5
        }
    }, model_dir / "gap_aware_gnn.pt")
    
    print(f"✅ Model saved to {model_dir / 'gap_aware_gnn.pt'}")
    return True

def evaluate_model(model, val_loader, device):
    """
    Evaluate model on validation set.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            out = model(batch.x, batch.edge_index, batch.edge_attr,
                       edge_type=batch.edge_type if hasattr(batch, 'edge_type') else None,
                       batch=batch.batch)
            
            loss = F.nll_loss(out, batch.y)
            total_loss += loss.item()
            
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    
    model.train()
    return total_loss / len(val_loader), correct / total

def detailed_evaluation(data_dir="model/processed_data_vf"):
    """
    Perform detailed evaluation including per-class metrics.
    """
    print("📊 Performing detailed evaluation...")
    
    # Load data
    _, val_loader = create_data_loaders(data_dir, batch_size=64, val_split=0.2)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_simple_model()
    
    model_path = Path("model/trained_models/gap_aware_gnn.pt")
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model loaded successfully")
    else:
        print("❌ No trained model found. Train first.")
        return
    
    model = model.to(device)
    model.eval()
    
    # Collect predictions
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = batch.to(device)
            
            out = model(batch.x, batch.edge_index, batch.edge_attr,
                       edge_type=batch.edge_type if hasattr(batch, 'edge_type') else None,
                       batch=batch.batch)
            
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    label_names = ['NORMAL', 'MERGE', 'SPLIT', 'POST_MERGE', 'POST_SPLIT']
    
    print("\n📈 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))
    
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Event detection rates
    print("\n🎯 Event Detection Performance:")
    for i, name in enumerate(label_names):
        if i == 0:  # Skip NORMAL class
            continue
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Complete pipeline example")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Detailed evaluation")
    parser.add_argument("--test_loading", action="store_true", help="Test data loading only")
    
    parser.add_argument("--input_dir", type=str, default="data/tracked_simdata_clean",
                       help="Input directory for preprocessing")
    parser.add_argument("--output_dir", type=str, default="model/processed_data_vf",
                       help="Output directory for processed data")
    parser.add_argument("--max_files", type=int, default=1000,
                       help="Maximum files to process")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    
    args = parser.parse_args()
    
    if not any([args.preprocess, args.train, args.evaluate, args.test_loading]):
        print("Please specify at least one action: --preprocess, --train, --evaluate, or --test_loading")
        return
    
    print("🌟 Gap-Aware Particle GNN Pipeline")
    print("=" * 50)
    
    if args.test_loading:
        print("🧪 Testing data loading...")
        success = test_data_loading(args.output_dir, args.batch_size)
        if not success:
            return
    
    if args.preprocess:
        print("\n1️⃣ PREPROCESSING DATA")
        print("-" * 30)
        success = run_preprocessing(args.input_dir, args.output_dir, args.max_files)
        if not success:
            print("❌ Preprocessing failed. Stopping pipeline.")
            return
    
    if args.train:
        print("\n2️⃣ TRAINING MODEL")
        print("-" * 30)
        success = train_model(args.output_dir, args.epochs, args.batch_size)
        if not success:
            print("❌ Training failed. Stopping pipeline.")
            return
    
    if args.evaluate:
        print("\n3️⃣ EVALUATING MODEL")
        print("-" * 30)
        detailed_evaluation(args.output_dir)
    
    print("\n🎉 Pipeline completed successfully!")
    print("\nNext steps:")
    print("- Scale up with your A5000 GPU (increase batch_size to 64-128)")
    print("- Try the advanced gap-aware model architecture")
    print("- Increase training data (3000-5000 simulations)")
    print("- Tune hyperparameters for optimal performance")

if __name__ == "__main__":
    main()