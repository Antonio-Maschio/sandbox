from prepare_data import prepare_particle_data
from preprocessor import load_preprocessed_data
from utils import inspect_dataset, visualize_graphs, check_data_quality

"""most recent method to get the data normalized and stuff"""
def example_preprocessing(input_pattern,output_dir):
    processed_path = prepare_particle_data(
        input_pattern=input_pattern,
        output_dir=output_dir,
        radius_buffer=7,
        train_ratio=0.7,
        val_ratio=0.15,
        normalize_method='standard',
        max_workers=8,
        seed=42
    )
    
    print(f"\nData saved to: {processed_path}")


def example_loading():
    train_loader, val_loader, test_loader, metadata = load_preprocessed_data(
        data_dir="data/dirty_processed",
        batch_size=32,
        num_workers=4
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    for batch in train_loader:
        print(f"\nBatch shape: {batch.x.shape}")
        print(f"Edge shape: {batch.edge_attr.shape}")
        print(f"Labels: {batch.y.unique()}")
        break
    
    return train_loader, val_loader, test_loader


def example_inspection():
    inspect_dataset("data/dirty_processed")
    
    _, _, _, data = load_preprocessed_data("data/dirty_processed", batch_size=1)
    graphs = data['graphs'][:5]
    
    visualize_graphs(graphs, num_samples=3)
    check_data_quality(graphs)


if __name__ == "__main__":
    example_preprocessing(input_pattern="data/tracked_simdata_dirty_80detec/tracked_particles_3d_*.csv",output_dir="data/dirty_processed80detec")

    # example_preprocessing(input_pattern="data/tracked_simdata_dirty_90detec/tracked_particles_3d_*.csv",output_dir="data/dirty_processed90detec")
    # train_loader, val_loader, test_loader = example_loading(,)
    # example_inspection()