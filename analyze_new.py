import os
import json
import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.random_projection import GaussianRandomProjection


def compare_tensors_random_projection(tensor1, tensor2, n_components=2, n_projections=10):
    """
    Compare tensors using multiple random projections and average the results
    
    Args:
        tensor1, tensor2: Input tensors to compare
        n_components: Number of dimensions to project to
        n_projections: Number of random projections to average over
    
    Returns:
        tuple: (average cosine similarity, average angle in degrees)
    """
    # Flatten tensors
    flat1 = tensor1.reshape(1, -1)
    flat2 = tensor2.reshape(1, -1)
    X = np.vstack((flat1, flat2))
    
    similarities = []
    angles = []
    
    # Perform multiple random projections
    for i in range(n_projections):
        # Create and apply random projection
        transformer = GaussianRandomProjection(n_components=n_components, random_state=i)
        X_projected = transformer.fit_transform(X)
        
        # Get projected vectors
        proj1 = X_projected[0]
        proj2 = X_projected[1]
        
        # Normalize projected vectors
        norm1 = proj1 / np.linalg.norm(proj1)
        norm2 = proj2 / np.linalg.norm(proj2)
        
        # Calculate similarity
        sim = np.dot(norm1, norm2)
        angle = np.arccos(np.clip(sim, -1.0, 1.0)) * 180 / np.pi
        
        similarities.append(sim)
        angles.append(angle)
    
    # Return average similarity and angle
    return np.mean(similarities), np.mean(angles)


def compare_tensors_multiple(tensor1, tensor2, n_projections=10):
    # Flatten and normalize
    flat1 = tensor1.reshape(-1)
    flat2 = tensor2.reshape(-1)
    norm1 = flat1 / np.linalg.norm(flat1)
    norm2 = flat2 / np.linalg.norm(flat2)
    
    # Calculate original angle directly
    cos_sim = np.dot(norm1, norm2)
    angle = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * 180 / np.pi
    
    return cos_sim, angle

def plot_heatmaps(similarities, angles, guidance_values, idx, row_idx, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    
    labels = [f'{g:.1f}' for g in guidance_values]
    
    # Create mask for diagonal and reference row/column
    mask = np.zeros_like(similarities, dtype=bool)
    mask[row_idx, :] = True  # Mask reference row
    mask[:, row_idx] = True  # Mask reference column
    np.fill_diagonal(mask, True)  # Mask diagonal
    
    # Plot similarity heatmap
    sns.heatmap(similarities, 
                ax=ax,
                cmap='RdBu_r', 
                vmin=-1, 
                vmax=1, 
                annot=True,
                fmt='.2f',
                xticklabels=labels,
                yticklabels=labels,
                annot_kws={'size': 6},
                cbar_kws={'label': ''},
                mask=mask | np.isnan(similarities))
    
    ax.set_title(f'Cosine Similarity (ref idx: {row_idx})', fontsize=10)
    ax.set_xlabel('Guidance Value', fontsize=8)
    ax.set_ylabel('Guidance Value', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.suptitle(f'Pairwise Comparisons for idx {idx}', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.close()

def process_single_idx(idx, items, root, save_path, method='cosine'):
    # Sort items by guidance
    items = sorted(items, key=lambda x: x['guidance'])
    
    # Load all samples and guidance values
    Z_tensors = []
    guidance_values = []
    
    print(f"Loading data for idx {idx}...")
    for item in tqdm(items):
        with h5py.File(os.path.join(root, item['h5_url']), 'r') as f:
            Z_tensors.append(f['samples'][:])
            guidance_values.append(item['guidance'])
    
    Z_tensors = np.array(Z_tensors)
    
    # Compute differences
    print("Computing differences...")
    Z_diffs = np.zeros((len(Z_tensors), len(Z_tensors), Z_tensors.shape[1], Z_tensors.shape[2]))
    for i in range(len(Z_tensors)):
        for j in range(i+1, len(Z_tensors)):
            diff = Z_tensors[j] - Z_tensors[i]
            Z_diffs[i, j] = diff
            Z_diffs[j, i] = diff
    
    # Process each reference index
    for row_idx in range(len(Z_diffs)):
        print(f"Processing reference index {row_idx}...")
        Z_diff_0 = Z_diffs[row_idx]
        n = len(Z_diff_0)
        
        # Initialize arrays
        similarities = np.zeros((n, n))
        angles = np.zeros((n, n))
        
        # Fill everything with NaN first
        similarities[:] = np.nan
        angles[:] = np.nan
        
        # Compute similarities for all valid pairs
        for i in range(n):
            for j in range(i+1, n):  # Only compute upper triangle
                if i != row_idx and j != row_idx:  # Skip reference row/column
                    if method == 'cosine':
                        sim, angle = compare_tensors_multiple(Z_diff_0[i], Z_diff_0[j])
                    else:
                        sim, angle = compare_tensors_random_projection(Z_diff_0[i], Z_diff_0[j])
                    similarities[i, j] = sim
                    similarities[j, i] = sim  # Matrix is symmetric
                    angles[i, j] = angle
                    angles[j, i] = angle  # Matrix is symmetric
        
        plot_heatmaps(similarities, angles, guidance_values, idx, row_idx, os.path.join(save_path, f'{idx}_{row_idx}'))
    
    return

def main():
    method = 'cosine'
    root = '/data/DiffEntropy/flux-dedisill/'
    save_path = f'/data/DiffEntropy/flux-dedisill/samples/heatmaps_{method}'
    
    os.makedirs(save_path, exist_ok=True)
    
    # Load metadata
    with open('/data/DiffEntropy/flux-dedisill/samples/inter_seed25/data.json', 'r') as f:
        info = json.load(f)
    
    # Group items by idx
    items_by_idx = defaultdict(list)
    for item in info:
        items_by_idx[item['idx']].append(item)
    
    # Process each idx
    for idx, items in items_by_idx.items():
        process_single_idx(idx, items, root, save_path, method)
        break  # Remove this if you want to process all indices

if __name__ == "__main__":
    main()