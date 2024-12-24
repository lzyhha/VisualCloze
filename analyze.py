import json
import h5py
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import linregress


def compute_linearity_ratios(Z_tensors, time_points):
    """
    Z_tensors: list of tensors [Z₀, Z₁, Z₂, ...]
    time_points: list of corresponding time values [t₀, t₁, t₂, ...]
    """
    # Convert to numpy arrays if they aren't already
    Z_diffs = []
    ratios = []
    
    # Compute consecutive differences
    for i in range(len(Z_tensors)-1):
        Z_diffs.append(Z_tensors[i+1] - Z_tensors[i])
    
    # Compute ratios of consecutive differences
    for i in range(len(Z_diffs)-1):
        # We might have multiple elements in tensors, so take norm
        ratio = np.linalg.norm(Z_diffs[i+1]) / np.linalg.norm(Z_diffs[i])
        dt_ratio = (time_points[i+2] - time_points[i+1]) / (time_points[i+1] - time_points[i])
        # Normalize by time differences
        normalized_ratio = ratio / dt_ratio
        ratios.append(normalized_ratio)
    
    return ratios

def assess_linearity(Z_tensors, time_points):
    ratios = compute_linearity_ratios(Z_tensors, time_points)
    
    # If f is linear, all ratios should be close to 1
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    
    # Compute coefficient of variation (CV) as a measure of consistency
    cv = std_ratio / mean_ratio
    
    return {
        'ratios': ratios,
        'mean_ratio': mean_ratio,
        'std_ratio': std_ratio,
        'cv': cv
    }
    
def multi_scale_linearity(Z_tensors, time_points, window_sizes=[3]):
    results = {}
    
    for w in window_sizes:
        # Slide window over time points
        window_ratios = []
        for i in range(len(time_points) - w + 1):
            window_Z = Z_tensors[i:i+w]
            window_t = time_points[i:i+w]
            ratios = compute_linearity_ratios(window_Z, window_t)
            window_ratios.extend(ratios)
            
        results[f'window_{w}'] = {
            'mean': np.mean(window_ratios),
            'std': np.std(window_ratios),
            'cv': np.std(window_ratios) / np.mean(window_ratios)
        }
    
    return results

def interpret_results(results):
    """
    Returns a linearity score between 0 and 1
    where 1 indicates highly linear and 0 indicates highly nonlinear
    """
    # Combine CV values from different windows
    cvs = [res['cv'] for res in results.values()]
    mean_cv = np.mean(cvs)
    
    # Convert to a score between 0 and 1
    linearity_score = 1 / (1 + mean_cv)
    return linearity_score

def analyze_guidance_segments(Z_tensors, guidance_values, segment_size=4):
    """
    Analyzes linearity for overlapping segments of guidance values
    Returns linearity scores for each segment's midpoint
    """
    segment_scores = []
    segment_centers = []
    
    for i in range(len(guidance_values) - segment_size + 1):
        segment_Z = Z_tensors[i:i+segment_size]
        segment_guidance = guidance_values[i:i+segment_size]
        
        # Calculate linearity for this segment
        linearity_results = multi_scale_linearity(segment_Z, segment_guidance)
        linearity_score = interpret_results(linearity_results)
        
        # Store result with the center guidance value
        segment_scores.append(linearity_score)
        segment_centers.append(np.mean(segment_guidance))
    
    return segment_centers, segment_scores

def cosine_similarity(Z1, Z2):
    return np.dot(Z1, Z2) / (np.linalg.norm(Z1) * np.linalg.norm(Z2))

def analyze_guidance_linearity(data_path, info_path):
    """
    Analyzes the linearity of samples with respect to guidance values,
    grouped by idx and averaged across all idx values.
    
    Args:
        data_path: Path to directory containing h5 files
        info_path: Path to data.json containing sample metadata
    """
    # Load metadata
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # Group items by idx
    items_by_idx = defaultdict(list)
    for item in info:
        items_by_idx[item['idx']].append(item)
    
    # Store results for each idx
    all_results = []
    all_similarity = []
    for idx, items in items_by_idx.items():
        # Sort items by guidance within each idx group
        items = sorted(items, key=lambda x: x['guidance'])
        
        # Load samples and guidance values for this idx
        Z_tensors = []
        guidance_values = []
        
        for item in items:
            with h5py.File(item['h5_url'], 'r') as f:
                sample = f['samples'][:]
                Z_tensors.append(sample)
                guidance_values.append(item['guidance'])
        
        Z_tensors = np.array(Z_tensors)
        
        Z_diffs = []
        for i in range(len(Z_tensors)):
            Z_diff = []
            for j in range(len(Z_tensors)):
                Z_diff.append(Z_tensors[j] - Z_tensors[i])
            Z_diffs.append(Z_diff)
            
        # Compute cosine similarity for each pair of samples
        for i in range(len(Z_diffs) - 1):
            if idx == 0:
                all_similarity.append([0] * len(Z_diffs[i]))
            for j in range(len(Z_diffs[i])):
                if i == j:
                    cosine_sim = 0
                else:
                    cosine_sim = cosine_similarity(Z_diffs[i][i + 1].flatten(), Z_diffs[i][j].flatten())
                if idx == 0:
                    all_similarity[i].append(cosine_sim / len(items_by_idx.items()))
                else:
                    all_similarity[i][j] += cosine_sim / len(items_by_idx.items())
        
            
        # Analyze linearity for this idx
        # linearity_results = multi_scale_linearity(Z_tensors, guidance_values)
        # linearity_score = interpret_results(linearity_results)
        
        # Add segment analysis
        # segment_centers, segment_scores = analyze_guidance_segments(Z_tensors, guidance_values)
        # all_results.append({
        #     'idx': idx,
        #     'linearity_score': linearity_score,
        #     'detailed_results': linearity_results,
        #     'guidance_values': guidance_values,
        #     'segment_centers': segment_centers,
        #     'segment_scores': segment_scores
        # })
    breakpoint()
    # Compute averaged results
    avg_linearity_score = np.mean([r['linearity_score'] for r in all_results])
    std_linearity_score = np.std([r['linearity_score'] for r in all_results])
    
    # Average detailed results across all idx
    avg_detailed_results = defaultdict(lambda: {'mean': [], 'std': [], 'cv': []})
    for result in all_results:
        for window, metrics in result['detailed_results'].items():
            for metric, value in metrics.items():
                avg_detailed_results[window][metric].append(value)
    
    # Convert lists to means
    for window in avg_detailed_results:
        for metric in avg_detailed_results[window]:
            avg_detailed_results[window][metric] = np.mean(avg_detailed_results[window][metric])
    
    return {
        'per_idx_results': all_results,
        'avg_linearity_score': avg_linearity_score,
        'std_linearity_score': std_linearity_score,
        'avg_detailed_results': dict(avg_detailed_results)
    }

def plot_linearity_analysis(results):
    """
    Plots linearity scores across guidance values with mean line
    """
    plt.figure(figsize=(10, 6))
    
    # Plot individual idx results
    for result in results['per_idx_results']:
        plt.plot(result['segment_centers'], result['segment_scores'], 
                'o-', alpha=0.4, label=f"idx_{result['idx']}")
    
    # Calculate and plot mean trend
    all_centers = np.array([])
    all_scores = np.array([])
    for result in results['per_idx_results']:
        all_centers = np.append(all_centers, result['segment_centers'])
        all_scores = np.append(all_scores, result['segment_scores'])
    
    # Sort by centers to ensure line is drawn correctly
    sort_idx = np.argsort(all_centers)
    unique_centers = np.unique(all_centers)
    mean_scores = [np.mean(all_scores[all_centers == center]) for center in unique_centers]
    
    # Plot mean line
    plt.plot(unique_centers, mean_scores, 'r-', 
             linewidth=2, alpha=1.0, label='Mean')
    
    plt.xlabel('Guidance Value')
    plt.ylabel('Linearity Score')
    plt.title('Linearity Score vs Guidance Value')
    # plt.legend()
    plt.grid(True)
    return plt

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="samples/inter_seed25/data")
    parser.add_argument('--info_path', type=str, default="samples/inter_seed25/data.json")
    args = parser.parse_args()
    
    results = analyze_guidance_linearity(args.data_path, args.info_path)
    
    # Print averaged results
    print(f"\nAverage Linearity score: {results['avg_linearity_score']:.3f} ± {results['std_linearity_score']:.3f}")
    print("\nDetailed results by window size (averaged across all idx):")
    for window, metrics in results['avg_detailed_results'].items():
        print(f"  {window}: CV = {metrics['cv']:.3f}")
    
    # Plot the results
    plt = plot_linearity_analysis(results)
    plt.savefig('guidance_linearity_analysis.png')
    plt.close()