#!/usr/bin/env python3
"""
Advanced Dataset Split Tool (Unified)

This tool creates train/val/test splits from an existing preprocessed dataset
without re-extracting features. Useful when you want to change the split 
strategy after running the preprocessing pipeline.

Supports:
1. Video-based split with train/val (no test)
2. Video-based split with train/test (no val)  
3. Video-based split with train/val/test
4. Temporal split (within each video)

Usage:
    # Video-based splits:
    # CQ_2 + CQ_3 → train, CQ_4 → val (no test)
    python advanced_split.py --train_videos CQ_2 CQ_3 --val_videos CQ_4
    
    # CQ_2 + CQ_3 → train, CQ_4 → test (no val)
    python advanced_split.py --train_videos CQ_2 CQ_3 --test_videos CQ_4
    
    # CQ_2 → train, CQ_3 → val, CQ_4 → test
    python advanced_split.py --train_videos CQ_2 --val_videos CQ_3 --test_videos CQ_4
    
    # Temporal split (split within each video, 80/20):
    python advanced_split.py --split_mode temporal --train_ratio 0.8
    
    # Simple temporal split (simplified syntax):
    python advanced_split.py --split_mode temporal
    
    # Video-based with temporal split for train/val:
    python advanced_split.py --split_mode video --test_videos CQ_4 --train_ratio 0.8

Note: This tool replaces the functionality of the old resplit_dataset.py
"""

import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np


def load_existing_clips(dataset_root):
    """Load information about existing clips"""
    features_dir = Path(dataset_root) / 'features'
    
    if not features_dir.exists():
        raise ValueError(f"Features directory not found: {features_dir}")
    
    # Get all clip files
    clip_files = sorted(features_dir.glob('*.npy'))
    
    # Group by video
    all_clips_info = {}
    for clip_file in clip_files:
        clip_name = clip_file.stem
        # Extract video name (e.g., 'CQ_2' from 'CQ_2_clip_000')
        video_name = '_'.join(clip_name.split('_')[:-2])
        
        if video_name not in all_clips_info:
            all_clips_info[video_name] = []
        all_clips_info[video_name].append(clip_name)
    
    # Sort clips for each video
    for video_name in all_clips_info:
        all_clips_info[video_name] = sorted(all_clips_info[video_name])
    
    return all_clips_info


def create_video_based_split(all_clips_info, train_videos, val_videos=None, test_videos=None):
    """Create video-based split with flexible train/val/test assignment"""
    train_clips = []
    val_clips = []
    test_clips = []
    
    # Validate that all specified videos exist
    all_specified = set()
    if train_videos:
        all_specified.update(train_videos)
    if val_videos:
        all_specified.update(val_videos)
    if test_videos:
        all_specified.update(test_videos)
    
    available_videos = set(all_clips_info.keys())
    missing = all_specified - available_videos
    if missing:
        raise ValueError(f"Specified videos not found in dataset: {missing}\nAvailable: {available_videos}")
    
    # Check for overlaps
    sets_to_check = []
    if train_videos:
        sets_to_check.append(('train', set(train_videos)))
    if val_videos:
        sets_to_check.append(('val', set(val_videos)))
    if test_videos:
        sets_to_check.append(('test', set(test_videos)))
    
    for i, (name1, set1) in enumerate(sets_to_check):
        for name2, set2 in sets_to_check[i+1:]:
            overlap = set1 & set2
            if overlap:
                raise ValueError(f"Videos appear in both {name1} and {name2}: {overlap}")
    
    # Assign clips to splits
    print(f"\n{'='*70}")
    print("Video-based Split Assignment")
    print(f"{'='*70}\n")
    
    if train_videos:
        print("Training videos:")
        for video in sorted(train_videos):
            clips = all_clips_info[video]
            train_clips.extend(clips)
            print(f"  {video}: {len(clips)} clips")
    
    if val_videos:
        print("\nValidation videos:")
        for video in sorted(val_videos):
            clips = all_clips_info[video]
            val_clips.extend(clips)
            print(f"  {video}: {len(clips)} clips")
    
    if test_videos:
        print("\nTest videos:")
        for video in sorted(test_videos):
            clips = all_clips_info[video]
            test_clips.extend(clips)
            print(f"  {video}: {len(clips)} clips")
    
    # Check if all videos are assigned
    assigned_videos = set()
    if train_videos:
        assigned_videos.update(train_videos)
    if val_videos:
        assigned_videos.update(val_videos)
    if test_videos:
        assigned_videos.update(test_videos)
    
    unassigned = available_videos - assigned_videos
    if unassigned:
        print(f"\n⚠️  Warning: Unassigned videos (will not be in any split): {unassigned}")
    
    return train_clips, val_clips, test_clips


def create_temporal_split(all_clips_info, train_ratio):
    """Create temporal split within each video"""
    train_clips = []
    val_clips = []
    
    print(f"\n{'='*70}")
    print(f"Temporal Split (Train ratio: {train_ratio:.0%})")
    print(f"{'='*70}\n")
    
    for video_name, clip_names in sorted(all_clips_info.items()):
        num_clips = len(clip_names)
        split_idx = int(num_clips * train_ratio)
        
        train_clips.extend(clip_names[:split_idx])
        val_clips.extend(clip_names[split_idx:])
        print(f"  {video_name}: {split_idx} clips → TRAIN, {num_clips - split_idx} clips → VAL")
    
    return train_clips, val_clips, []


def save_splits(train_clips, val_clips, test_clips, output_dir):
    """Save split files"""
    splits_dir = Path(output_dir) / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Always save train
    with open(splits_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(train_clips))
    
    # Save val only if non-empty
    if val_clips:
        with open(splits_dir / 'val.txt', 'w') as f:
            f.write('\n'.join(val_clips))
    else:
        # Remove val.txt if it exists but we have no val clips
        val_file = splits_dir / 'val.txt'
        if val_file.exists():
            val_file.unlink()
    
    # Save test only if non-empty
    if test_clips:
        with open(splits_dir / 'test.txt', 'w') as f:
            f.write('\n'.join(test_clips))
    else:
        # Remove test.txt if it exists but we have no test clips
        test_file = splits_dir / 'test.txt'
        if test_file.exists():
            test_file.unlink()
    
    print(f"\n{'='*70}")
    print("Final Split Statistics")
    print(f"{'='*70}")
    print(f"  Training clips:   {len(train_clips)}")
    if val_clips:
        print(f"  Validation clips: {len(val_clips)}")
    if test_clips:
        print(f"  Test clips:       {len(test_clips)}")


def compute_split_stats(train_clips, val_clips, test_clips, dataset_root):
    """Compute and display label distribution for each split"""
    labels_dir = Path(dataset_root) / 'labels'
    meta_dir = Path(dataset_root) / 'meta'
    
    # Load class mapping
    with open(meta_dir / 'class_mapping.json', 'r') as f:
        mapping = json.load(f)
        idx_to_label = {int(k): v for k, v in mapping['idx_to_label'].items()}
    
    def get_label_dist(clips):
        if not clips:
            return None, 0
        
        all_labels = []
        for clip_name in clips:
            labels = np.load(labels_dir / f'{clip_name}.npy')
            all_labels.extend(labels.tolist())
        
        label_counts = Counter(all_labels)
        total = len(all_labels)
        
        dist = {}
        for idx in sorted(label_counts.keys()):
            label_name = idx_to_label[idx]
            count = label_counts[idx]
            pct = (count / total) * 100
            dist[label_name] = {'count': count, 'percentage': pct}
        
        return dist, total
    
    print(f"\n{'='*70}")
    print("Label Distribution per Split")
    print(f"{'='*70}")
    
    # Training set
    if train_clips:
        print("\n📊 Training Set:")
        train_dist, train_total = get_label_dist(train_clips)
        print(f"   Total frames: {train_total:,}")
        for label, info in train_dist.items():
            print(f"   {label}: {info['count']:,} ({info['percentage']:.2f}%)")
    
    # Validation set
    if val_clips:
        print("\n📊 Validation Set:")
        val_dist, val_total = get_label_dist(val_clips)
        print(f"   Total frames: {val_total:,}")
        for label, info in val_dist.items():
            print(f"   {label}: {info['count']:,} ({info['percentage']:.2f}%)")
    
    # Test set
    if test_clips:
        print("\n📊 Test Set:")
        test_dist, test_total = get_label_dist(test_clips)
        print(f"   Total frames: {test_total:,}")
        for label, info in test_dist.items():
            print(f"   {label}: {info['count']:,} ({info['percentage']:.2f}%)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Advanced dataset split tool - Re-split dataset without re-extracting features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Video-based splits:
  python advanced_split.py --train_videos CQ_2 CQ_3 --val_videos CQ_4
  python advanced_split.py --train_videos CQ_2 CQ_3 --test_videos CQ_4
  python advanced_split.py --train_videos CQ_2 --val_videos CQ_3 --test_videos CQ_4
  
  # Temporal split (split each video 80/20):
  python advanced_split.py --split_mode temporal --train_ratio 0.8
  
  # Hybrid: video-based test set with temporal train/val split:
  python advanced_split.py --split_mode video --test_videos CQ_4 --train_ratio 0.8
  
Note: Replaces the old resplit_dataset.py with enhanced functionality.
        """
    )
    
    parser.add_argument('--dataset_root', type=str,
                        default='/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/action_segmentation',
                        help='Root directory of preprocessed dataset')
    
    # Video-based split options
    parser.add_argument('--train_videos', type=str, nargs='+', default=None,
                        help='Videos for training set')
    parser.add_argument('--val_videos', type=str, nargs='+', default=None,
                        help='Videos for validation set')
    parser.add_argument('--test_videos', type=str, nargs='+', default=None,
                        help='Videos for test set')
    
    # Temporal split options
    parser.add_argument('--split_mode', type=str, default=None, choices=['video', 'temporal'],
                        help='Split mode: "video" (split by video) or "temporal" (split within videos). '
                             'Auto-detected if not specified.')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train ratio for temporal split or train/val split within remaining videos (default: 0.8)')
    
    # Additional options
    parser.add_argument('--show_stats', action='store_true', default=True,
                        help='Show label distribution statistics (default: True)')
    parser.add_argument('--no_stats', dest='show_stats', action='store_false',
                        help='Do not show label distribution statistics')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Advanced Dataset Split Tool")
    print("="*70)
    print(f"Dataset: {args.dataset_root}")
    
    # Load existing clips
    print(f"\n🔍 Loading existing clips...")
    all_clips_info = load_existing_clips(args.dataset_root)
    
    print(f"\n📊 Found {len(all_clips_info)} videos:")
    for video_name, clips in sorted(all_clips_info.items()):
        print(f"  {video_name}: {len(clips)} clips")
    
    # Auto-detect split mode if not specified
    if args.split_mode is None:
        if any([args.train_videos, args.val_videos, args.test_videos]):
            args.split_mode = 'video'
            print(f"\n🔍 Auto-detected split mode: VIDEO-BASED")
        else:
            args.split_mode = 'temporal'
            print(f"\n🔍 Auto-detected split mode: TEMPORAL")
    
    # Validate test videos if specified
    if args.test_videos:
        for test_video in args.test_videos:
            if test_video not in all_clips_info:
                print(f"\n❌ Error: Test video '{test_video}' not found in dataset")
                print(f"   Available videos: {list(all_clips_info.keys())}")
                return
    
    # Determine split mode and create splits
    if args.split_mode == 'video' or any([args.train_videos, args.val_videos, args.test_videos]):
        # Video-based split
        if not args.train_videos and not args.test_videos and not args.val_videos:
            print("\n❌ Error: At least one of --train_videos, --val_videos, or --test_videos must be specified for video-based split")
            return
        
        if not args.train_videos:
            print("\n❌ Error: --train_videos must be specified for video-based split")
            return
        
        train_clips, val_clips, test_clips = create_video_based_split(
            all_clips_info,
            train_videos=args.train_videos,
            val_videos=args.val_videos,
            test_videos=args.test_videos
        )
    else:
        # Temporal split
        train_clips, val_clips, test_clips = create_temporal_split(
            all_clips_info,
            train_ratio=args.train_ratio
        )
    
    # Save splits
    save_splits(train_clips, val_clips, test_clips, args.dataset_root)
    
    # Compute statistics if requested
    if args.show_stats:
        compute_split_stats(train_clips, val_clips, test_clips, args.dataset_root)
    
    print("="*70)
    print("✅ Split Complete!")
    print("="*70)
    print(f"\n📁 Split files saved to: {Path(args.dataset_root) / 'splits'}")
    print("  - train.txt")
    if val_clips:
        print("  - val.txt")
    if test_clips:
        print("  - test.txt")
    print()


if __name__ == '__main__':
    main()
