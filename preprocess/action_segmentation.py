"""
Action Segmentation Dataset Preprocessing Pipeline

Steps:
1. Convert time intervals to frame-level labels
2. Extract and save frame-level features
3. Segment videos with sliding window (window_length=512, stride=256)
4. Train/val split with temporal continuity, ensuring similar class distributions
5. File organization structure:
   dataset_root/
     features/ 
       video1_clip_000.npy
       video1_clip_001.npy
       video2_clip_000.npy
       ...
     labels/
       video1_clip_000.npy
       video1_clip_001.npy
       ...
     splits/
       train.txt
       val.txt
     meta/
       class_mapping.json
       dataset_stats.json
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from tqdm import tqdm


def parse_timestamp(ts_str):
    """Parse timestamp string HH:MM:SS.mmm to seconds."""
    parts = ts_str.split(':')
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def create_frame_labels(csv_path, video_path, fps=30):
    """
    Step 1: Convert time intervals to frame-level labels.
    
    Args:
        csv_path: Path to CSV with Start, End, Notes columns
        video_path: Path to video file (to get total frames)
        fps: Frames per second
    
    Returns:
        numpy array of frame labels (one label per frame)
    """
    # Get total frames from video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Initialize all frames as background (we'll use "no behavior" as default)
    frame_labels = np.zeros(total_frames, dtype=np.int32)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Create label mapping
    unique_labels = sorted(df['Notes'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Assign labels to frames
    for _, row in df.iterrows():
        start_sec = parse_timestamp(row['Start'])
        end_sec = parse_timestamp(row['End'])
        
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        # Clip to valid range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames))
        
        label_idx = label_to_idx[row['Notes']]
        frame_labels[start_frame:end_frame] = label_idx
    
    return frame_labels, label_to_idx


def extract_features(video_path, device='cuda', batch_size=32):
    """
    Step 2: Extract frame-level features using ResNet50 with GPU batch processing.
    
    Args:
        video_path: Path to video file
        device: 'cuda' or 'cpu'
        batch_size: Number of frames to process in parallel on GPU
    
    Returns:
        numpy array of shape (num_frames, feature_dim)
    """
    # Load pretrained ResNet50
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Remove final classification layer to get features
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Extract features with batching
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_features = []
    batch_frames = []
    
    with torch.no_grad():
        pbar = tqdm(total=total_frames, desc=f"Extracting features from {Path(video_path).name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB and preprocess
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = transform(frame_rgb)
            batch_frames.append(img_tensor)
            
            # Process batch when full or at end of video
            if len(batch_frames) == batch_size:
                # Stack batch and move to GPU
                batch_tensor = torch.stack(batch_frames).to(device)
                
                # Extract features for entire batch
                batch_features = model(batch_tensor)
                batch_features = batch_features.squeeze(-1).squeeze(-1)  # Remove spatial dims
                
                # Move back to CPU and store
                all_features.append(batch_features.cpu().numpy())
                
                batch_frames = []
                pbar.update(batch_size)
        
        # Process remaining frames
        if len(batch_frames) > 0:
            batch_tensor = torch.stack(batch_frames).to(device)
            batch_features = model(batch_tensor)
            batch_features = batch_features.squeeze(-1).squeeze(-1)
            all_features.append(batch_features.cpu().numpy())
            pbar.update(len(batch_frames))
        
        pbar.close()
    
    cap.release()
    
    # Concatenate all batches
    features = np.concatenate(all_features, axis=0)
    
    return features


def segment_video(features, labels, window_length=512, stride=256):
    """
    Step 3: Segment video into clips with sliding window.
    
    Args:
        features: numpy array of shape (num_frames, feature_dim)
        labels: numpy array of shape (num_frames,)
        window_length: Length of each clip
        stride: Stride for sliding window
    
    Returns:
        List of (feature_clip, label_clip) tuples
    """
    num_frames = len(features)
    clips = []
    
    for start_idx in range(0, num_frames - window_length + 1, stride):
        end_idx = start_idx + window_length
        
        feature_clip = features[start_idx:end_idx]
        label_clip = labels[start_idx:end_idx]
        
        clips.append((feature_clip, label_clip))
    
    # Handle remaining frames if any
    if num_frames % stride != 0:
        start_idx = num_frames - window_length
        if start_idx >= 0 and start_idx not in range(0, num_frames - window_length + 1, stride):
            feature_clip = features[start_idx:num_frames]
            label_clip = labels[start_idx:num_frames]
            clips.append((feature_clip, label_clip))
    
    return clips


def save_clips(clips, video_name, output_dir):
    """
    Save clips to disk.
    
    Args:
        clips: List of (feature_clip, label_clip) tuples
        video_name: Name of the video (e.g., 'CQ_2')
        output_dir: Root output directory
    
    Returns:
        List of clip names
    """
    features_dir = Path(output_dir) / 'features'
    labels_dir = Path(output_dir) / 'labels'
    
    features_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    clip_names = []
    
    for idx, (feature_clip, label_clip) in enumerate(clips):
        clip_name = f"{video_name}_clip_{idx:03d}"
        clip_names.append(clip_name)
        
        # Save features and labels
        np.save(features_dir / f"{clip_name}.npy", feature_clip)
        np.save(labels_dir / f"{clip_name}.npy", label_clip)
    
    return clip_names


def create_splits(all_clips_info, output_dir, train_ratio=0.8, split_mode='temporal', test_videos=None):
    """
    Step 4: Create train/val/test splits.
    
    Args:
        all_clips_info: Dict mapping video_name -> list of clip names
        output_dir: Root output directory
        train_ratio: Ratio of training data (only used in 'temporal' mode)
        split_mode: 'temporal' or 'video'
            - 'temporal': Split each video temporally (first X% train, last Y% val)
            - 'video': Split by video (specified videos for test, others split train/val)
        test_videos: List of video names to use as test set (only for 'video' mode)
    """
    splits_dir = Path(output_dir) / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    train_clips = []
    val_clips = []
    test_clips = []
    
    if split_mode == 'video' and test_videos:
        # Video-based split: use specified videos as test set
        print(f"\n📊 Split Mode: Video-based")
        print(f"   Test videos: {test_videos}")
        
        # Separate test videos from train/val videos
        train_val_videos = {k: v for k, v in all_clips_info.items() if k not in test_videos}
        test_video_dict = {k: v for k, v in all_clips_info.items() if k in test_videos}
        
        # Add all clips from test videos to test set
        for video_name, clip_names in test_video_dict.items():
            test_clips.extend(clip_names)
            print(f"   {video_name}: {len(clip_names)} clips → TEST")
        
        # Split remaining videos temporally for train/val
        for video_name, clip_names in train_val_videos.items():
            num_clips = len(clip_names)
            split_idx = int(num_clips * train_ratio)
            
            train_clips.extend(clip_names[:split_idx])
            val_clips.extend(clip_names[split_idx:])
            print(f"   {video_name}: {split_idx} clips → TRAIN, {num_clips - split_idx} clips → VAL")
    
    else:
        # Temporal split: split each video temporally
        print(f"\n📊 Split Mode: Temporal (within each video)")
        print(f"   Train ratio: {train_ratio:.0%}")
        
        for video_name, clip_names in all_clips_info.items():
            num_clips = len(clip_names)
            split_idx = int(num_clips * train_ratio)
            
            train_clips.extend(clip_names[:split_idx])
            val_clips.extend(clip_names[split_idx:])
            print(f"   {video_name}: {split_idx} clips → TRAIN, {num_clips - split_idx} clips → VAL")
    
    # Save splits
    with open(splits_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(train_clips))
    
    with open(splits_dir / 'val.txt', 'w') as f:
        f.write('\n'.join(val_clips))
    
    if test_clips:
        with open(splits_dir / 'test.txt', 'w') as f:
            f.write('\n'.join(test_clips))
    
    print(f"\n📊 Final Split Statistics:")
    print(f"   Training clips: {len(train_clips)}")
    print(f"   Validation clips: {len(val_clips)}")
    if test_clips:
        print(f"   Test clips: {len(test_clips)}")
    
    return train_clips, val_clips, test_clips


def compute_stats(all_clips_info, output_dir, label_to_idx):
    """
    Compute and save dataset statistics.
    """
    meta_dir = Path(output_dir) / 'meta'
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Save class mapping
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    with open(meta_dir / 'class_mapping.json', 'w') as f:
        json.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label,
            'num_classes': len(label_to_idx)
        }, f, indent=2)
    
    # Compute label distribution
    labels_dir = Path(output_dir) / 'labels'
    all_labels = []
    
    for video_name, clip_names in all_clips_info.items():
        for clip_name in clip_names:
            label_clip = np.load(labels_dir / f"{clip_name}.npy")
            all_labels.extend(label_clip.tolist())
    
    label_counts = Counter(all_labels)
    label_distribution = {idx_to_label[idx]: count for idx, count in sorted(label_counts.items())}
    
    # Save statistics
    stats = {
        'total_clips': sum(len(clips) for clips in all_clips_info.values()),
        'total_frames': len(all_labels),
        'num_videos': len(all_clips_info),
        'label_distribution': label_distribution,
        'videos': {video: len(clips) for video, clips in all_clips_info.items()}
    }
    
    with open(meta_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total videos: {stats['num_videos']}")
    print(f"   Total clips: {stats['total_clips']}")
    print(f"   Total frames: {stats['total_frames']}")
    print(f"   Label distribution:")
    for label, count in label_distribution.items():
        percentage = (count / len(all_labels)) * 100
        print(f"      {label}: {count} ({percentage:.2f}%)")


def process_dataset(video_paths, csv_paths, output_dir, window_length=512, stride=256, 
                    device='cuda', batch_size=32, split_mode='temporal', test_videos=None, train_ratio=0.8):
    """
    Main pipeline to process all videos.
    
    Args:
        batch_size: Number of frames to process in parallel on GPU
        split_mode: 'temporal' or 'video' - how to split the dataset
        test_videos: List of video names for test set (only for 'video' mode)
        train_ratio: Ratio for train/val split (default 0.8)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_clips_info = {}
    global_label_to_idx = None
    
    for video_path, csv_path in zip(video_paths, csv_paths):
        video_name = Path(video_path).stem
        print(f"\n{'='*60}")
        print(f"Processing {video_name}")
        print(f"{'='*60}")
        
        # Step 1: Create frame labels
        print("Step 1: Creating frame-level labels...")
        frame_labels, label_to_idx = create_frame_labels(csv_path, video_path)
        
        # Ensure consistent label mapping across all videos
        if global_label_to_idx is None:
            global_label_to_idx = label_to_idx
        else:
            # Verify all videos have the same labels
            if set(label_to_idx.keys()) != set(global_label_to_idx.keys()):
                print(f"Warning: Inconsistent labels across videos. Merging...")
                # Merge label sets
                all_labels = sorted(set(list(label_to_idx.keys()) + list(global_label_to_idx.keys())))
                global_label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
                # Remap current labels
                old_to_new = {old_idx: global_label_to_idx[label] for label, old_idx in label_to_idx.items()}
                frame_labels = np.array([old_to_new[old_idx] for old_idx in frame_labels])
        
        print(f"   Total frames: {len(frame_labels)}")
        print(f"   Unique labels: {list(label_to_idx.keys())}")
        
        # Step 2: Extract features (GPU batch processing)
        print(f"Step 2: Extracting features (batch_size={batch_size}, device={device})...")
        features = extract_features(video_path, device=device, batch_size=batch_size)
        print(f"   Feature shape: {features.shape}")
        
        # Step 3: Segment video
        print("Step 3: Segmenting video...")
        clips = segment_video(features, frame_labels, window_length=window_length, stride=stride)
        print(f"   Number of clips: {len(clips)}")
        
        # Save clips
        print("Step 4: Saving clips...")
        clip_names = save_clips(clips, video_name, output_dir)
        all_clips_info[video_name] = clip_names
    
    # Step 4: Create splits
    print(f"\n{'='*60}")
    print("Creating train/val/test splits...")
    print(f"{'='*60}")
    create_splits(all_clips_info, output_dir, train_ratio=train_ratio, 
                  split_mode=split_mode, test_videos=test_videos)
    
    # Step 5: Compute statistics
    print(f"\n{'='*60}")
    print("Computing dataset statistics...")
    print(f"{'='*60}")
    compute_stats(all_clips_info, output_dir, global_label_to_idx)
    
    print(f"\n✅ Dataset preprocessing complete!")
    print(f"   Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess videos for action segmentation task')
    parser.add_argument('--video_dir', type=str, default='/data/zhaozhenghao/Projects/Mouse/datasets/UMich_CQ',
                        help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, default='/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/action_segmentation',
                        help='Output directory for preprocessed dataset')
    parser.add_argument('--videos', type=str, nargs='+', default=['CQ_2.mp4', 'CQ_3.mp4', 'CQ_4.mp4'],
                        help='List of video filenames')
    parser.add_argument('--csvs', type=str, nargs='+', default=['CQ_2.csv', 'CQ_3.csv', 'CQ_4.csv'],
                        help='List of CSV filenames (must match videos order)')
    parser.add_argument('--window_length', type=int, default=512,
                        help='Window length for segmentation')
    parser.add_argument('--stride', type=int, default=256,
                        help='Stride for sliding window')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for GPU feature extraction (increase for faster processing)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device for feature extraction (cuda or cpu)')
    parser.add_argument('--split_mode', type=str, default='temporal', choices=['temporal', 'video'],
                        help='Split mode: temporal (split each video) or video (split by video)')
    parser.add_argument('--test_videos', type=str, nargs='+', default=None,
                        help='Video names to use as test set (only for video mode), e.g., CQ_4')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train ratio for temporal split (default: 0.8)')
    
    args = parser.parse_args()
    
    # Construct full paths
    video_paths = [os.path.join(args.video_dir, v) for v in args.videos]
    csv_paths = [os.path.join(args.video_dir, c) for c in args.csvs]
    
    # Verify files exist
    for vp, cp in zip(video_paths, csv_paths):
        if not os.path.exists(vp):
            raise FileNotFoundError(f"Video not found: {vp}")
        if not os.path.exists(cp):
            raise FileNotFoundError(f"CSV not found: {cp}")
    
    print("="*60)
    print("Action Segmentation Dataset Preprocessing")
    print("="*60)
    print(f"Videos: {args.videos}")
    print(f"Labels: {args.csvs}")
    print(f"Window length: {args.window_length}")
    print(f"Stride: {args.stride}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Split mode: {args.split_mode}")
    if args.split_mode == 'video' and args.test_videos:
        print(f"Test videos: {args.test_videos}")
    else:
        print(f"Train ratio: {args.train_ratio}")
    print(f"Output: {args.output_dir}")
    
    process_dataset(
        video_paths=video_paths,
        csv_paths=csv_paths,
        output_dir=args.output_dir,
        window_length=args.window_length,
        stride=args.stride,
        device=args.device,
        batch_size=args.batch_size,
        split_mode=args.split_mode,
        test_videos=args.test_videos,
        train_ratio=args.train_ratio
    )


if __name__ == '__main__':
    main()