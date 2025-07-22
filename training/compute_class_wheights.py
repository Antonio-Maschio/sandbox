import torch
import numpy as np
from torch.utils.data import DataLoader


def compute_class_weights_original(train_loader: DataLoader) -> torch.Tensor:
    """Original implementation for comparison."""
    y_train = torch.cat([data.y for data in train_loader.dataset])
    class_counts = torch.bincount(y_train)
    print("Class counts:", class_counts)
    
    if len(class_counts) < 5:
        padded_counts = torch.zeros(5, dtype=torch.float)
        padded_counts[:len(class_counts)] = class_counts
        class_counts = padded_counts
    
    total_samples = len(y_train)
    num_classes = 5
    
    class_weights = total_samples / (num_classes * class_counts.float())
    class_weights = torch.clamp(class_weights, min=0.1, max=200.0)
    
    return class_weights


def compute_class_weights_effective(train_loader: DataLoader, beta: float = 0.9999) -> torch.Tensor:
    """
    Compute class weights using effective number of samples.
    
    More stable for extreme imbalance than inverse frequency.
    Formula: (1 - beta) / (1 - beta^n) where n is class count.
    
    Args:
        beta: Controls weighting strength. Higher beta = more aggressive weighting.
              Try 0.99, 0.999, 0.9999 for increasing focus on minorities.
    """
    y_train = torch.cat([data.y for data in train_loader.dataset])
    class_counts = torch.bincount(y_train)
    print("Class counts:", class_counts)
    
    if len(class_counts) < 5:
        padded_counts = torch.zeros(5, dtype=torch.float)
        padded_counts[:len(class_counts)] = class_counts
        class_counts = padded_counts
    
    # Effective number weighting
    effective_num = 1.0 - torch.pow(beta, class_counts.float())
    class_weights = (1.0 - beta) / effective_num
    
    # Normalize to prevent extreme values
    class_weights = class_weights / class_weights.sum() * 5.0
    
    print(f"Effective weights (beta={beta}):", class_weights.numpy())
    return class_weights


def compute_class_weights_sqrt(train_loader: DataLoader) -> torch.Tensor:
    """
    Square root of inverse frequency weighting.
    
    Less aggressive than full inverse frequency, often works better
    for extreme imbalance scenarios.
    """
    y_train = torch.cat([data.y for data in train_loader.dataset])
    class_counts = torch.bincount(y_train)
    print("Class counts:", class_counts)
    
    if len(class_counts) < 5:
        padded_counts = torch.zeros(5, dtype=torch.float)
        padded_counts[:len(class_counts)] = class_counts
        class_counts = padded_counts
    
    total_samples = len(y_train)
    
    # Square root of inverse frequency
    inv_freq = total_samples / class_counts.float()
    class_weights = torch.sqrt(inv_freq)
    
    # Normalize
    class_weights = class_weights / class_weights.sum() * 5.0
    
    print("Sqrt weights:", class_weights.numpy())
    return class_weights


def compute_class_weights_log(train_loader: DataLoader, smoothing: float = 1.0) -> torch.Tensor:
    """
    Log-based weighting with smoothing.
    
    Provides moderate weighting that's less extreme than inverse frequency.
    
    Args:
        smoothing: Smoothing factor to prevent log(0). Higher = less aggressive.
    """
    y_train = torch.cat([data.y for data in train_loader.dataset])
    class_counts = torch.bincount(y_train)
    print("Class counts:", class_counts)
    
    if len(class_counts) < 5:
        padded_counts = torch.zeros(5, dtype=torch.float)
        padded_counts[:len(class_counts)] = class_counts
        class_counts = padded_counts
    
    total_samples = len(y_train)
    
    # Log-based weighting
    class_weights = torch.log(total_samples / (class_counts.float() + smoothing) + 1.0)
    
    # Normalize
    class_weights = class_weights / class_weights.sum() * 5.0
    
    print(f"Log weights (smoothing={smoothing}):", class_weights.numpy())
    return class_weights


def compute_class_weights_custom(train_loader: DataLoader) -> torch.Tensor:
    """
    Custom weighting specifically tuned for your extreme imbalance.
    
    Based on your distribution: [12345985, 44187, 47051, 22211, 99448]
    Gives moderate boost to minorities without being too extreme.
    """
    y_train = torch.cat([data.y for data in train_loader.dataset])
    class_counts = torch.bincount(y_train)
    print("Class counts:", class_counts)
    
    if len(class_counts) < 5:
        padded_counts = torch.zeros(5, dtype=torch.float)
        padded_counts[:len(class_counts)] = class_counts
        class_counts = padded_counts
    
    # Custom weights tuned for your specific imbalance
    # Class 0: Keep low weight for majority
    # Classes 1-4: Moderate boost for minorities
    total_samples = len(y_train)
    
    # Use a power of 0.75 to reduce extremes
    inv_freq = total_samples / class_counts.float()
    class_weights = torch.pow(inv_freq, 0.75)
    
    # Normalize and apply custom scaling
    class_weights = class_weights / class_weights.sum() * 5.0
    
    # Apply specific adjustments for your classes
    class_weights[0] = class_weights[0] * 0.5  # Reduce majority weight further
    class_weights[1:] = class_weights[1:] * 1.2  # Boost minorities slightly
    
    print("Custom weights:", class_weights.numpy())
    return class_weights


def compare_weighting_methods(train_loader: DataLoader):
    """
    Compare all weighting methods side by side.
    Call this once to see which works best for your data.
    """
    print("=" * 60)
    print("COMPARING CLASS WEIGHTING METHODS")
    print("=" * 60)
    
    print("\n1. Original (clamped inverse frequency):")
    w1 = compute_class_weights_original(train_loader)
    
    print("\n2. Effective number (beta=0.9999):")
    w2 = compute_class_weights_effective(train_loader, beta=0.9999)
    
    print("\n3. Square root inverse frequency:")
    w3 = compute_class_weights_sqrt(train_loader)
    
    print("\n4. Log-based weighting:")
    w4 = compute_class_weights_log(train_loader, smoothing=1.0)
    
    print("\n5. Custom tuned weights:")
    w5 = compute_class_weights_custom(train_loader)
    
    print("\n" + "=" * 60)
    print("WEIGHT COMPARISON SUMMARY:")
    print("=" * 60)
    methods = ["Original", "Effective", "Sqrt", "Log", "Custom"]
    weights = [w1, w2, w3, w4, w5]
    
    for i, (method, weight) in enumerate(zip(methods, weights)):
        print(f"{method:12}: {weight.numpy()}")
        
    print("\nRecommendation: Try 'Effective' or 'Custom' first")
    print("If minorities still struggle, try 'Sqrt' or adjust Focal gamma")




import torch
import numpy as np
from torch.utils.data import DataLoader


def compute_class_weights_harsh(train_loader: DataLoader) -> torch.Tensor:
    """
    EXTREMELY harsh weighting - maximum aggression for minority classes.
    
    Uses inverse frequency raised to high power + manual extreme scaling.
    """
    y_train = torch.cat([data.y for data in train_loader.dataset])
    class_counts = torch.bincount(y_train)
    print("Class counts:", class_counts)
    
    if len(class_counts) < 5:
        padded_counts = torch.zeros(5, dtype=torch.float)
        padded_counts[:len(class_counts)] = class_counts
        class_counts = padded_counts
    
    total_samples = len(y_train)
    
    # Extremely harsh: inverse frequency to power of 2
    class_weights = total_samples / class_counts.float()
    # class_weights = torch.pow(inv_freq, 2.0)
    
    # Additional harsh scaling for minority classes
    class_weights[0] = class_weights[0] * 0.8  # Severely reduce majority class
    class_weights[1:] = class_weights[1:] * 1.2 # Extreme boost for minorities
    
    # Extra boost for rarest class (Class 3)
    # class_weights[3] = class_weights[3] * 3.0
    class_weights[2] = class_weights[2] * 2
    class_weights[4] = class_weights[4] * 2


    
    print("EXTREMELY HARSH weights:", class_weights.numpy())
    return class_weights


def compute_class_weights_effective_ultra_harsh(train_loader: DataLoader, beta: float = 0.999999) -> torch.Tensor:
    """
    Ultra harsh effective number weighting with beta very close to 1.0.
    
    Args:
        beta: Use 0.999999 or higher for maximum harshness
    """
    y_train = torch.cat([data.y for data in train_loader.dataset])
    class_counts = torch.bincount(y_train)
    print("Class counts:", class_counts)
    
    if len(class_counts) < 5:
        padded_counts = torch.zeros(5, dtype=torch.float)
        padded_counts[:len(class_counts)] = class_counts
        class_counts = padded_counts
    
    # Ultra-harsh effective number weighting
    effective_num = 1.0 - torch.pow(beta, class_counts.float())
    class_weights = (1.0 - beta) / effective_num
    
    # Additional harsh adjustments
    class_weights[0] = class_weights[0] * 0.05  # Extremely low weight for majority
    class_weights[1:] = class_weights[1:] * 2.0  # Double minority weights
    
    # Normalize but keep extreme ratios
    class_weights = class_weights / class_weights.sum() * 20.0  # Higher total to maintain extremes
    
    print(f"ULTRA HARSH effective weights (beta={beta}):", class_weights.numpy())
    return class_weights


def compute_class_weights_exponential_harsh(train_loader: DataLoader, base: float = 10.0) -> torch.Tensor:
    """
    Exponential harsh weighting - uses exponential function of log ratios.
    
    Args:
        base: Higher values = more extreme weighting (try 10.0, 50.0, 100.0)
    """
    y_train = torch.cat([data.y for data in train_loader.dataset])
    class_counts = torch.bincount(y_train)
    print("Class counts:", class_counts)
    
    if len(class_counts) < 5:
        padded_counts = torch.zeros(5, dtype=torch.float)
        padded_counts[:len(class_counts)] = class_counts
        class_counts = padded_counts
    
    total_samples = len(y_train)
    
    # Exponential harsh weighting
    log_ratios = torch.log(total_samples / class_counts.float())
    class_weights = torch.exp(log_ratios / base)
    
    # Additional manual scaling
    class_weights[0] = class_weights[0] * 0.01  # Crush majority class weight
    class_weights[3] = class_weights[3] * 10.0  # Extreme boost for rarest class
    
    print(f"EXPONENTIAL HARSH weights (base={base}):", class_weights.numpy())
    return class_weights


# def compute_class_weights_custom_nuclear(train_loader: DataLoader) -> torch.Tensor:
#     """
#     'Nuclear option' - manually crafted extreme weights for your specific distribution.
    
#     Designed specifically for your [12M, 44K, 47K, 22K, 99K] distribution.
#     """
#     y_train = torch.cat([data.y for data in train_loader.dataset])
#     class_counts = torch.bincount(y_train)
#     print("Class counts:", class_counts)
    
#     if len(class_counts) < 5:
#         padded_counts = torch.zeros(5, dtype=torch.float)
#         padded_counts[:len(class_counts)] = class_counts
#         class_counts = padded_counts
    
#     # Manual extreme weights based on your distribution
#     # These are hand-tuned for maximum minority class focus
#     class_weights = torch.tensor([
#         0.01,    # Class 0 (majority): extremely low
#         50.0,    # Class 1 (merge): very high
#         48.0,    # Class 2 (split): very high
#         100.0,   #
# Usage examples:
"""
# Quick test - compare all methods:
compare_weighting_methods(train_loader)

# Then replace your compute_class_weights call with best performer:
class_weights = compute_class_weights_effective(train_loader, beta=0.9999)
# or
class_weights = compute_class_weights_custom(train_loader)
"""


