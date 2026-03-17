"""
Data Loader for Action Segmentation Dataset
Use this after preprocessing to load data for training
"""

import numpy as np
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import os


class ActionSegmentationDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed action segmentation clips
    Supports optional keypoint features for multimodal learning
    """
    
    def __init__(self, dataset_root, split='train', transform=None, 
                 use_keypoints=False, keypoint_root=None):
        """
        Args:
            dataset_root: Path to preprocessed dataset directory
            split: 'train' or 'val'
            transform: Optional transform to apply to features
            use_keypoints: Whether to load and concatenate keypoint features
            keypoint_root: Path to directory containing keypoint features (if different from dataset_root)
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.use_keypoints = use_keypoints
        
        # Set keypoint root
        if keypoint_root is None:
            # Default: look for keypoint_features directory in parent of dataset_root
            self.keypoint_root = self.dataset_root.parent / 'keypoint_features'
        else:
            self.keypoint_root = Path(keypoint_root)
        
        # Load clip names
        split_file = self.dataset_root / 'splits' / f'{split}.txt'
        with open(split_file, 'r') as f:
            self.clip_names = f.read().splitlines()
        
        # Load class mapping
        with open(self.dataset_root / 'meta' / 'class_mapping.json', 'r') as f:
            mapping = json.load(f)
            self.num_classes = mapping['num_classes']
            self.idx_to_label = mapping['idx_to_label']
            # Convert string keys to int
            self.idx_to_label = {int(k): v for k, v in self.idx_to_label.items()}
        
        # Check keypoint availability if requested
        if self.use_keypoints:
            self._check_keypoint_availability()
        
        print(f"Loaded {split} dataset: {len(self.clip_names)} clips")
        print(f"Number of classes: {self.num_classes}")
        if self.use_keypoints:
            print(f"Using keypoint features from: {self.keypoint_root}")
            print(f"  - Videos with keypoints: {len(self.video_has_keypoints)} / {len(self.video_keypoint_cache)}")
    
    def _check_keypoint_availability(self):
        """Check which videos have keypoint features available"""
        self.video_keypoint_cache = {}
        self.video_has_keypoints = {}
        
        # Get unique video names from clip names
        video_names = set()
        for clip_name in self.clip_names:
            # Extract video name from clip name (e.g., "CQ_2_clip_000" -> "CQ_2")
            video_name = '_'.join(clip_name.split('_')[:-2])
            video_names.add(video_name)
        
        # Check each video for keypoint features
        for video_name in video_names:
            keypoint_path = self.keypoint_root / f"{video_name}_keypoints.npy"
            if keypoint_path.exists():
                self.video_has_keypoints[video_name] = True
                # Lazy load - cache will be populated on first access
                self.video_keypoint_cache[video_name] = None
            else:
                self.video_has_keypoints[video_name] = False
                print(f"  Warning: No keypoint features found for {video_name}")
    
    def _load_video_keypoints(self, video_name):
        """Load full video keypoint features (lazy loading)"""
        if video_name not in self.video_keypoint_cache:
            return None
        
        if self.video_keypoint_cache[video_name] is None:
            keypoint_path = self.keypoint_root / f"{video_name}_keypoints.npy"
            self.video_keypoint_cache[video_name] = np.load(keypoint_path)
        
        return self.video_keypoint_cache[video_name]
    
    def _extract_clip_keypoints(self, clip_name):
        """Extract keypoint features for a specific clip"""
        # Parse clip name to get video name and clip index
        parts = clip_name.split('_')
        video_name = '_'.join(parts[:-2])  # e.g., "CQ_2"
        clip_idx = int(parts[-1])  # e.g., 0 from "clip_000"
        
        # Check if keypoints available
        if not self.video_has_keypoints.get(video_name, False):
            return None
        
        # Load video keypoints
        video_keypoints = self._load_video_keypoints(video_name)
        if video_keypoints is None:
            return None
        
        # Load the corresponding ResNet features to get clip frame range
        features = np.load(self.dataset_root / 'features' / f'{clip_name}.npy')
        clip_length = features.shape[0]
        
        # Calculate clip frame range (assuming clips are created sequentially)
        # This assumes each clip has a meta file or we can infer from clip index
        # For now, we'll try to match by loading metadata if available
        meta_path = self.dataset_root / 'meta' / f'{clip_name}_meta.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                clip_start = meta.get('start_frame', clip_idx * clip_length)
                clip_end = meta.get('end_frame', clip_start + clip_length)
        else:
            # Fallback: assume sequential non-overlapping clips
            clip_start = clip_idx * clip_length
            clip_end = clip_start + clip_length
        
        # Extract keypoint features for this clip
        clip_keypoints = video_keypoints[clip_start:clip_end]
        
        # Ensure clip_keypoints matches features length
        if clip_keypoints.shape[0] != clip_length:
            # Pad or truncate if necessary
            if clip_keypoints.shape[0] < clip_length:
                # Pad with zeros
                pad_length = clip_length - clip_keypoints.shape[0]
                clip_keypoints = np.pad(clip_keypoints, ((0, pad_length), (0, 0)), mode='constant')
            else:
                # Truncate
                clip_keypoints = clip_keypoints[:clip_length]
        
        return clip_keypoints
    
    def __len__(self):
        return len(self.clip_names)
    
    def __getitem__(self, idx):
        """
        Returns:
            features: numpy array of shape (T, D) where T=window_length, D=feature_dim
                     If use_keypoints=True, D = resnet_dim (2048) + keypoint_dim (117) = 2165
            labels: numpy array of shape (T,)
        """
        clip_name = self.clip_names[idx]
        
        # Load ResNet features and labels
        features = np.load(self.dataset_root / 'features' / f'{clip_name}.npy')
        labels = np.load(self.dataset_root / 'labels' / f'{clip_name}.npy')
        
        # Load and concatenate keypoint features if requested
        if self.use_keypoints:
            clip_keypoints = self._extract_clip_keypoints(clip_name)
            if clip_keypoints is not None:
                # Concatenate ResNet features (2048) with keypoint features (117)
                features = np.concatenate([features, clip_keypoints], axis=1)
            else:
                # No keypoints available, pad with zeros
                keypoint_dim = 117
                zero_keypoints = np.zeros((features.shape[0], keypoint_dim), dtype=features.dtype)
                features = np.concatenate([features, zero_keypoints], axis=1)
        
        if self.transform:
            features = self.transform(features)
        
        return {
            'features': features,
            'labels': labels,
            'clip_name': clip_name
        }
    
    def get_label_name(self, label_idx):
        """Convert label index to label name"""
        return self.idx_to_label[label_idx]
    
    def get_feature_dim(self):
        """Return the feature dimension"""
        base_dim = 2048  # ResNet feature dim
        if self.use_keypoints:
            return base_dim + 117  # 39 keypoints × 3 values
        return base_dim


def get_dataloader(dataset_root, split='train', batch_size=8, shuffle=None, num_workers=4, 
                   use_keypoints=False, keypoint_root=None):
    """
    Create DataLoader for action segmentation dataset
    
    Args:
        dataset_root: Path to preprocessed dataset
        split: 'train' or 'val'
        batch_size: Batch size
        shuffle: Whether to shuffle (default: True for train, False for val)
        num_workers: Number of worker processes
        use_keypoints: Whether to load and concatenate keypoint features
        keypoint_root: Path to keypoint features directory
    
    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = ActionSegmentationDataset(
        dataset_root, 
        split=split, 
        use_keypoints=use_keypoints,
        keypoint_root=keypoint_root
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# Example usage
if __name__ == '__main__':
    # Path to preprocessed dataset
    dataset_root = '/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/action_segmentation'
    
    print("="*60)
    print("Action Segmentation Dataset Loader - Example Usage")
    print("="*60)
    
    # Create dataset
    train_dataset = ActionSegmentationDataset(dataset_root, split='train')
    val_dataset = ActionSegmentationDataset(dataset_root, split='val')
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} clips")
    print(f"  Validation: {len(val_dataset)} clips")
    
    # Create dataloaders
    train_loader = get_dataloader(dataset_root, split='train', batch_size=4)
    val_loader = get_dataloader(dataset_root, split='val', batch_size=4)
    
    print(f"\nDataLoader info:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Load one batch
    print(f"\nLoading sample batch...")
    batch = next(iter(train_loader))
    
    features = batch['features']
    labels = batch['labels']
    clip_names = batch['clip_name']
    
    print(f"\nBatch shape:")
    print(f"  Features: {features.shape}  # (batch_size, window_length, feature_dim)")
    print(f"  Labels: {labels.shape}      # (batch_size, window_length)")
    
    print(f"\nSample clips in batch:")
    for i, name in enumerate(clip_names):
        print(f"  {i+1}. {name}")
    
    print(f"\nLabel distribution in first clip:")
    first_clip_labels = labels[0].numpy()
    unique, counts = np.unique(first_clip_labels, return_counts=True)
    for label_idx, count in zip(unique, counts):
        label_name = train_dataset.get_label_name(label_idx)
        percentage = (count / len(first_clip_labels)) * 100
        print(f"  {label_name}: {count} frames ({percentage:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ Dataset loader working correctly!")
    print("="*60)
