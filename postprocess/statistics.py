"""
Compare inference results with ground truth for scratching behavior.

This script:
1. Loads behavior predictions from video_inference directory
2. Loads ground truth from CSV file
3. Calculates scratching statistics for both
4. Compares the results
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


def time_to_seconds(time_str: str) -> float:
    """Convert time string (HH:MM:SS.mmm) to seconds."""
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def calculate_statistics_from_predictions(predictions: np.ndarray, fps: float) -> Dict:
    """
    Calculate scratching statistics from predictions array.
    Same logic as in generate_video.py
    """
    scratch_durations = []
    scratch_count = 0
    total_scratch_frames = 0
    
    # Find scratching segments
    in_scratch = False
    scratch_start = 0
    
    for i, pred in enumerate(predictions):
        if pred == 1:  # scratching
            if not in_scratch:
                in_scratch = True
                scratch_start = i
            total_scratch_frames += 1
        else:  # no behavior
            if in_scratch:
                # End of a scratching segment
                duration = (i - scratch_start) / fps
                scratch_durations.append(duration)
                scratch_count += 1
                in_scratch = False
    
    # Handle case where predictions end during scratching
    if in_scratch:
        duration = (len(predictions) - scratch_start) / fps
        scratch_durations.append(duration)
        scratch_count += 1
    
    # Calculate statistics
    total_scratch_time = total_scratch_frames / fps
    mean_scratch_duration = np.mean(scratch_durations) if scratch_durations else 0.0
    total_duration_sec = len(predictions) / fps
    frequency_per_min = (scratch_count / total_duration_sec * 60.0) if total_duration_sec > 0 else 0.0
    
    return {
        'total_count': scratch_count,
        'total_duration_sec': total_scratch_time,
        'mean_duration_sec': mean_scratch_duration,
        'frequency_per_min': frequency_per_min,
        'durations': scratch_durations,
        'total_duration_sec_video': total_duration_sec
    }


def load_ground_truth(csv_path: Path, start_time: float, end_time: float, fps: float) -> Tuple[np.ndarray, Dict]:
    """
    Load ground truth from CSV and convert to frame-based predictions.
    
    Args:
        csv_path: Path to ground truth CSV file
        start_time: Start time in seconds (from video beginning)
        end_time: End time in seconds (from video beginning)
        fps: Frames per second
    
    Returns:
        predictions: Array of predictions (0 or 1) for each frame
        stats: Statistics dictionary
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Convert time strings to seconds
    df['start_sec'] = df['Start'].apply(time_to_seconds)
    df['end_sec'] = df['End'].apply(time_to_seconds)
    
    # Filter out invalid segments where end_time < start_time (data errors)
    invalid_segments = df[df['end_sec'] < df['start_sec']]
    if len(invalid_segments) > 0:
        print(f"Warning: Found {len(invalid_segments)} invalid segments in {csv_path.name} where end < start:")
        for idx, row in invalid_segments.iterrows():
            print(f"  Row {idx}: {row['Start']} -> {row['End']} ({row['Notes']})")
        print(f"  These segments will be skipped.")
        print()
    
    df = df[df['end_sec'] >= df['start_sec']].copy()
    
    # Calculate number of frames in the time range
    num_frames = int((end_time - start_time) * fps)
    predictions = np.zeros(num_frames, dtype=int)
    
    # Fill in predictions based on ground truth
    for _, row in df.iterrows():
        seg_start = row['start_sec']
        seg_end = row['end_sec']
        label = row['Notes'].strip().lower()
        
        # Check if this segment overlaps with our time range
        if seg_end <= start_time or seg_start >= end_time:
            continue
        
        # Clip segment to our time range
        seg_start = max(seg_start, start_time)
        seg_end = min(seg_end, end_time)
        
        # Convert to frame indices (relative to start_time)
        frame_start = int((seg_start - start_time) * fps)
        frame_end = int((seg_end - start_time) * fps)
        
        # Clip to valid range
        frame_start = max(0, frame_start)
        frame_end = min(num_frames, frame_end)
        
        # Set prediction values
        if label in ('scratching', 'scracthing'):
            predictions[frame_start:frame_end] = 1
    
    # Calculate statistics
    stats = calculate_statistics_from_predictions(predictions, fps)
    
    return predictions, stats


def compare_statistics(pred_stats: Dict, gt_stats: Dict) -> Dict:
    """Compare prediction statistics with ground truth."""
    comparison = {
        'total_count': {
            'prediction': pred_stats['total_count'],
            'ground_truth': gt_stats['total_count'],
            'difference': pred_stats['total_count'] - gt_stats['total_count'],
            'error_rate': abs(pred_stats['total_count'] - gt_stats['total_count']) / gt_stats['total_count'] if gt_stats['total_count'] > 0 else 0
        },
        'total_duration_sec': {
            'prediction': pred_stats['total_duration_sec'],
            'ground_truth': gt_stats['total_duration_sec'],
            'difference': pred_stats['total_duration_sec'] - gt_stats['total_duration_sec'],
            'error_rate': abs(pred_stats['total_duration_sec'] - gt_stats['total_duration_sec']) / gt_stats['total_duration_sec'] if gt_stats['total_duration_sec'] > 0 else 0
        },
        'mean_duration_sec': {
            'prediction': pred_stats['mean_duration_sec'],
            'ground_truth': gt_stats['mean_duration_sec'],
            'difference': pred_stats['mean_duration_sec'] - gt_stats['mean_duration_sec'],
            'error_rate': abs(pred_stats['mean_duration_sec'] - gt_stats['mean_duration_sec']) / gt_stats['mean_duration_sec'] if gt_stats['mean_duration_sec'] > 0 else 0
        },
        'frequency_per_min': {
            'prediction': pred_stats['frequency_per_min'],
            'ground_truth': gt_stats['frequency_per_min'],
            'difference': pred_stats['frequency_per_min'] - gt_stats['frequency_per_min'],
            'error_rate': abs(pred_stats['frequency_per_min'] - gt_stats['frequency_per_min']) / gt_stats['frequency_per_min'] if gt_stats['frequency_per_min'] > 0 else 0
        }
    }
    
    return comparison


def calculate_frame_accuracy(pred_array: np.ndarray, gt_array: np.ndarray) -> Dict:
    """Calculate frame-level accuracy metrics."""
    # Ensure same length
    min_len = min(len(pred_array), len(gt_array))
    pred = pred_array[:min_len]
    gt = gt_array[:min_len]
    
    # Calculate confusion matrix
    tp = np.sum((pred == 1) & (gt == 1))  # True positives
    tn = np.sum((pred == 0) & (gt == 0))  # True negatives
    fp = np.sum((pred == 1) & (gt == 0))  # False positives
    fn = np.sum((pred == 0) & (gt == 1))  # False negatives
    
    # Calculate metrics
    accuracy = (tp + tn) / len(pred) if len(pred) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'confusion_matrix': {
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn)
        },
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def analyze_combined_videos(video_configs: List[Dict], output_path: str = None):
    """
    Analyze multiple videos together as a combined dataset.
    
    Args:
        video_configs: List of dicts with 'inference_dir' and 'ground_truth' keys
        output_path: Optional path to save results JSON
    """
    print("=" * 80)
    print("Combined Multi-Video Statistics Comparison")
    print("=" * 80)
    print(f"Analyzing {len(video_configs)} videos together")
    print()
    
    all_predictions = []
    all_gt_predictions = []
    total_frames = 0
    video_metadata = []
    
    # Load and combine all predictions and ground truth
    for config in video_configs:
        inference_dir = Path(config['inference_dir'])
        gt_path = Path(config['ground_truth'])
        video_name = inference_dir.name
        
        print(f"Loading {video_name}...")
        
        # Load predictions
        pred_path = inference_dir / "predictions.npy"
        predictions = np.load(pred_path)
        
        # Load statistics
        stats_path = inference_dir / "statistics.json"
        with open(stats_path, 'r') as f:
            stats_json = json.load(f)
        
        fps = stats_json['fps']
        frame_offset = stats_json['frame_offset']
        duration_sec = stats_json['duration_sec']
        
        # Load ground truth for the same time range
        start_time = frame_offset / fps
        end_time = start_time + duration_sec
        gt_predictions, _ = load_ground_truth(gt_path, start_time, end_time, fps)
        
        # Append to combined arrays
        all_predictions.append(predictions)
        all_gt_predictions.append(gt_predictions)
        total_frames += len(predictions)
        
        video_metadata.append({
            'video_name': video_name,
            'frames': len(predictions),
            'fps': fps,
            'frame_offset': frame_offset,
            'duration_sec': duration_sec,
            'time_range': {
                'start_sec': start_time,
                'end_sec': end_time
            }
        })
        
        print(f"  Loaded {len(predictions)} frames, FPS: {fps:.2f}, Duration: {duration_sec:.2f}s")
    
    print()
    print(f"Total combined frames: {total_frames}")
    print()
    
    # Concatenate all arrays
    combined_predictions = np.concatenate(all_predictions)
    combined_gt = np.concatenate(all_gt_predictions)
    
    # Use average FPS for statistics calculation (assuming all videos have same FPS)
    avg_fps = np.mean([meta['fps'] for meta in video_metadata])
    
    # Calculate statistics
    print("Calculating combined statistics...")
    pred_stats = calculate_statistics_from_predictions(combined_predictions, avg_fps)
    gt_stats = calculate_statistics_from_predictions(combined_gt, avg_fps)
    
    # Compare statistics
    print("=" * 80)
    print("COMBINED STATISTICS COMPARISON")
    print("=" * 80)
    print()
    
    comparison = compare_statistics(pred_stats, gt_stats)
    
    print(f"{'Metric':<25} {'Prediction':<15} {'Ground Truth':<15} {'Difference':<15} {'Error Rate':<15}")
    print("-" * 85)
    
    for metric, values in comparison.items():
        metric_name = metric.replace('_', ' ').title()
        print(f"{metric_name:<25} {values['prediction']:<15.4f} {values['ground_truth']:<15.4f} "
              f"{values['difference']:<15.4f} {values['error_rate']*100:<14.2f}%")
    
    print()
    print("=" * 80)
    print("COMBINED FRAME-LEVEL ACCURACY")
    print("=" * 80)
    print()
    
    accuracy_metrics = calculate_frame_accuracy(combined_predictions, combined_gt)
    
    cm = accuracy_metrics['confusion_matrix']
    print("Confusion Matrix:")
    print(f"  True Positive (TP):  {cm['true_positive']:>6} (predicted scratching, actual scratching)")
    print(f"  True Negative (TN):  {cm['true_negative']:>6} (predicted no behavior, actual no behavior)")
    print(f"  False Positive (FP): {cm['false_positive']:>6} (predicted scratching, actual no behavior)")
    print(f"  False Negative (FN): {cm['false_negative']:>6} (predicted no behavior, actual scratching)")
    print()
    
    print(f"Accuracy:  {accuracy_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {accuracy_metrics['precision']*100:.2f}%")
    print(f"Recall:    {accuracy_metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {accuracy_metrics['f1_score']*100:.2f}%")
    print()
    
    # Prepare output
    output_data = {
        'metadata': {
            'num_videos': len(video_configs),
            'videos': video_metadata,
            'total_frames': total_frames,
            'average_fps': avg_fps,
            'total_duration_sec': total_frames / avg_fps
        },
        'prediction_statistics': pred_stats,
        'ground_truth_statistics': gt_stats,
        'comparison': comparison,
        'frame_accuracy': accuracy_metrics
    }
    
    # Save or print results
    if output_path:
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {output_file}")
    else:
        print("=" * 80)
        print("DETAILED RESULTS (JSON)")
        print("=" * 80)
        print(json.dumps(output_data, indent=2))
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Compare inference results with ground truth"
    )
    
    parser.add_argument(
        '--inference-dir',
        type=str,
        default=None,
        help='Directory containing predictions.npy and statistics.json'
    )
    
    parser.add_argument(
        '--ground-truth',
        type=str,
        default=None,
        help='Path to ground truth CSV file'
    )
    
    parser.add_argument(
        '--combined',
        action='store_true',
        help='Analyze CQ_2, CQ_3, and CQ_4 together as a combined dataset'
    )
    
    parser.add_argument(
        '--videos',
        type=str,
        nargs='+',
        default=None,
        help='Specify which videos to analyze together (e.g., --videos CQ_3 CQ_4)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for comparison results (default: print to console)'
    )
    
    args = parser.parse_args()
    
    # Handle combined mode or custom video selection
    if args.combined or args.videos:
        # Get absolute paths
        # results_base = Path('/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/video_inference') # original
        results_base = Path('/brtx/605-nvme2/ylu174/research/mouse_bahavior_2/preprocess_dataset') # new
        # gt_base = Path('/data/zhaozhenghao/Projects/Mouse/datasets/UMich_CQ') # original
        gt_base = Path('/brtx/605-nvme2/ylu174/research/mouse_bahavior_2/preprocess_dataset') # new
        
        # Determine which videos to analyze
        if args.videos:
            video_names = args.videos
        else:
            # Default combined: all three videos
            video_names = ['CQ_2', 'CQ_3', 'CQ_4']
        
        video_configs = []
        for video_name in video_names:
            video_configs.append({
                'inference_dir': str(results_base / video_name),
                'ground_truth': str(gt_base / f'{video_name}.csv')
            })
        
        analyze_combined_videos(video_configs, args.output)
        return
    
    # Single video mode
    if not args.inference_dir or not args.ground_truth:
        parser.error("--inference-dir and --ground-truth are required unless using --combined")
    
    inference_dir = Path(args.inference_dir)
    gt_path = Path(args.ground_truth)
    
    print("=" * 80)
    print("Scratching Behavior Statistics Comparison")
    print("=" * 80)
    print(f"Inference directory: {inference_dir}")
    print(f"Ground truth: {gt_path}")
    print()
    
    # Load predictions
    print("Loading predictions...")
    pred_path = inference_dir / "predictions.npy"
    predictions = np.load(pred_path)
    
    # Load statistics
    stats_path = inference_dir / "statistics.json"
    with open(stats_path, 'r') as f:
        stats_json = json.load(f)
    
    fps = stats_json['fps']
    frame_offset = stats_json['frame_offset']
    total_frames = stats_json['total_frames']
    duration_sec = stats_json['duration_sec']
    
    print(f"Predictions: {len(predictions)} frames")
    print(f"FPS: {fps:.2f}")
    print(f"Frame offset: {frame_offset}")
    print(f"Duration: {duration_sec:.2f}s")
    print()
    
    # Calculate prediction statistics
    print("Calculating prediction statistics...")
    pred_stats = calculate_statistics_from_predictions(predictions, fps)
    
    # Load ground truth for the same time range
    print("Loading ground truth...")
    start_time = frame_offset / fps
    end_time = start_time + duration_sec
    
    print(f"Time range: {start_time:.2f}s - {end_time:.2f}s")
    print()
    
    gt_predictions, gt_stats = load_ground_truth(gt_path, start_time, end_time, fps)
    
    # Compare statistics
    print("=" * 80)
    print("STATISTICS COMPARISON")
    print("=" * 80)
    print()
    
    comparison = compare_statistics(pred_stats, gt_stats)
    
    print(f"{'Metric':<25} {'Prediction':<15} {'Ground Truth':<15} {'Difference':<15} {'Error Rate':<15}")
    print("-" * 85)
    
    for metric, values in comparison.items():
        metric_name = metric.replace('_', ' ').title()
        print(f"{metric_name:<25} {values['prediction']:<15.4f} {values['ground_truth']:<15.4f} "
              f"{values['difference']:<15.4f} {values['error_rate']*100:<14.2f}%")
    
    print()
    print("=" * 80)
    print("FRAME-LEVEL ACCURACY")
    print("=" * 80)
    print()
    
    accuracy_metrics = calculate_frame_accuracy(predictions, gt_predictions)
    
    cm = accuracy_metrics['confusion_matrix']
    print("Confusion Matrix:")
    print(f"  True Positive (TP):  {cm['true_positive']:>6} (predicted scratching, actual scratching)")
    print(f"  True Negative (TN):  {cm['true_negative']:>6} (predicted no behavior, actual no behavior)")
    print(f"  False Positive (FP): {cm['false_positive']:>6} (predicted scratching, actual no behavior)")
    print(f"  False Negative (FN): {cm['false_negative']:>6} (predicted no behavior, actual scratching)")
    print()
    
    print(f"Accuracy:  {accuracy_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {accuracy_metrics['precision']*100:.2f}%")
    print(f"Recall:    {accuracy_metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {accuracy_metrics['f1_score']*100:.2f}%")
    print()
    
    # Prepare output
    output_data = {
        'metadata': {
            'inference_dir': str(inference_dir),
            'ground_truth': str(gt_path),
            'fps': fps,
            'frame_offset': frame_offset,
            'total_frames': total_frames,
            'duration_sec': duration_sec,
            'time_range': {
                'start_sec': start_time,
                'end_sec': end_time
            }
        },
        'prediction_statistics': pred_stats,
        'ground_truth_statistics': gt_stats,
        'comparison': comparison,
        'frame_accuracy': accuracy_metrics
    }
    
    # Save or print results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {output_path}")
    else:
        print("=" * 80)
        print("DETAILED RESULTS (JSON)")
        print("=" * 80)
        print(json.dumps(output_data, indent=2))


if __name__ == '__main__':
    main()
