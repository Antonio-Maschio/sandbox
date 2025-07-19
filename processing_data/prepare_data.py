import argparse
from pathlib import Path

from dataset import ParticleDataset
# from processing_data.preprocessor import DataPreprocessor, save_preprocessed_data, load_preprocessed_data
from preprocessor import DataPreprocessor, save_preprocessed_data, load_preprocessed_data


def prepare_particle_data(
    input_pattern,
    output_dir,
    radius_buffer=0.0,
    train_ratio=0.7,
    val_ratio=0.15,
    normalize_method='standard',
    max_workers=4,
    seed=42
):
    print(f"Loading data from: {input_pattern}")
    dataset = ParticleDataset(
        pattern=input_pattern,
        radius_buffer=radius_buffer,
        max_workers=max_workers,
        labeled=True
    )
    print(f"Loaded {len(dataset)} graphs")
    
    preprocessor = DataPreprocessor(dataset)
    
    print(f"Normalizing features ({normalize_method})")
    preprocessor.normalize_features(method=normalize_method)
    preprocessor.normalize_edges(method=normalize_method)
    
    print("Splitting dataset")
    splits = preprocessor.split_dataset(train_ratio, val_ratio, seed)
    
    print(f"Saving to: {output_dir}")
    save_path = save_preprocessed_data(output_dir, dataset, preprocessor, splits)
    
    print("\nDataset statistics:")
    for key, value in dataset.stats.items():
        if key == 'class_distribution':
            print(f"  Class distribution: {value.tolist()}")
        else:
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input CSV pattern')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--radius-buffer', type=float, default=0.0)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--normalize', default='standard', choices=['standard', 'minmax'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    prepare_particle_data(
        input_pattern=args.input,
        output_dir=args.output,
        radius_buffer=args.radius_buffer,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        normalize_method=args.normalize,
        max_workers=args.workers,
        seed=args.seed
    )


if __name__ == '__main__':
    main()