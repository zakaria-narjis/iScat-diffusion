import h5py
import numpy as np
import argparse
from tqdm import tqdm

def compute_hdf5_stats(file_path, dataset_name="image_patches", batch_size=100):
    """
    Computes mean and std of an HDF5 dataset efficiently using a running sum.
    """
    with h5py.File(file_path, "r") as f:
        if dataset_name not in f:
            raise KeyError(f"Dataset '{dataset_name}' not found in {file_path}")
        
        ds = f[dataset_name]
        n_samples = ds.shape[0]
        
        # We assume the data is (N, Z, H, W) or (N, H, W)
        # We treat all pixels as one population
        total_sum = 0.0
        total_sum_sq = 0.0
        total_pixel_count = 0

        print(f"Computing stats for {n_samples} samples...")

        for i in tqdm(range(0, n_samples, batch_size)):
            # Load a chunk of data
            end_idx = min(i + batch_size, n_samples)
            data = ds[i:end_idx].astype(np.float64)
            
            # Update running stats
            total_sum += np.sum(data)
            total_sum_sq += np.sum(np.square(data))
            total_pixel_count += data.size

        # Final Calculations
        mean = total_sum / total_pixel_count
        # Variance = E[X^2] - (E[X])^2
        var = (total_sum_sq / total_pixel_count) - (mean ** 2)
        std = np.sqrt(max(var, 0)) # Max to avoid tiny negative numbers due to precision

        return mean, std

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Mean and STD of HDF5 microscopy data.")
    parser.add_argument("path", type=str, help="Path to the HDF5 file")
    parser.add_argument("--ds", type=str, default="image_patches", help="Dataset name inside HDF5")
    parser.add_argument("--batch", type=int, default=128, help="Batch size for processing")
    
    args = parser.parse_args()

    try:
        mean, std = compute_hdf5_stats(args.path, args.ds, args.batch)
        print(f"\nResults for '{args.ds}':")
        print(f"  Mean: {mean:.6f}")
        print(f"  Std:  {std:.6f}")
    except Exception as e:
        print(f"Error: {e}")

# python compute_stats.py /scratch/narjis/iscat_data/brightfield.hdf5