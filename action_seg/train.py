"""
Training Script for Temporal Action Segmentation

This script trains a Multi-Stage Temporal Convolutional Network (MS-TCN)
for frame-level action segmentation on mouse scratching behavior videos.

Based on:
- MS-TCN: "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation"
  (CVPR 2019) - https://arxiv.org/abs/1903.01945
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from tqdm import tqdm

# Add parent directory to path to import dataloader
import importlib.util
# Get absolute path to dataloader
current_dir = Path(__file__).resolve().parent  # action_seg/
src_dir = current_dir.parent  # src/
dataloader_path = src_dir / 'preprocess' / 'dataloader_example.py'
if not dataloader_path.exists():
    raise FileNotFoundError(f"Cannot find dataloader at {dataloader_path}")
spec = importlib.util.spec_from_file_location("dataloader_example", str(dataloader_path))
dataloader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataloader_module)
ActionSegmentationDataset = dataloader_module.ActionSegmentationDataset


# Import model from separate module
from .model import MS_TCN


# ============================================================================
# DATASET
# ============================================================================

class ActionSegmentationTrainDataset(ActionSegmentationDataset):
    """Extended dataset with data augmentation for training"""
    
    def __getitem__(self, idx):
        # Call parent class to get data (which handles keypoints internally)
        data_dict = super().__getitem__(idx)
        
        # Convert to tensors
        features = torch.from_numpy(data_dict['features']).float()  # (T, D)
        labels = torch.from_numpy(data_dict['labels']).long()       # (T,)
        
        # Transpose features to (D, T) for Conv1d
        features = features.transpose(0, 1)  # (D, T)
        
        return {
            'features': features,
            'labels': labels,
            'clip_name': data_dict['clip_name']
        }


def compute_sample_weights(dataset_root, split='train', power=2.0):
    """
    Compute sample weights for oversampling based on minority class proportion
    
    Args:
        dataset_root: Path to dataset
        split: 'train' or 'val'
        power: Exponent to amplify weight differences (higher = more aggressive)
    
    Returns:
        List of weights for each sample (clip)
    """
    dataset_root = Path(dataset_root)
    
    # Load clip names
    split_file = dataset_root / 'splits' / f'{split}.txt'
    with open(split_file, 'r') as f:
        clip_names = f.read().splitlines()
    
    # Compute minority class proportion for each clip
    sample_weights = []
    minority_ratios = []
    
    for clip_name in clip_names:
        labels = np.load(dataset_root / 'labels' / f'{clip_name}.npy')
        
        # Count each class
        unique, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique, counts))
        
        # Get minority class proportion (higher weight for clips with more minority class)
        # Assume class 1 is minority (scratching)
        minority_count = label_counts.get(1, 0)
        total_count = len(labels)
        minority_ratio = minority_count / total_count
        minority_ratios.append(minority_ratio)
    
    # Apply power transformation to amplify differences
    # Clips with more scratching get exponentially higher weights
    for ratio in minority_ratios:
        if ratio > 0:
            # Use power transformation: weight = (ratio)^power
            # Then scale to make weights more aggressive
            weight = (ratio ** power) * 100  # Scale up significantly
        else:
            # Still give small weight to clips with no scratching
            weight = 0.1
        sample_weights.append(weight)
    
    return sample_weights



# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class ActionSegmentationLoss(nn.Module):
    """Combined loss for action segmentation with class weighting and focal loss"""
    
    def __init__(self, num_classes, class_weights=None, tmse_weight=0.15, use_focal=True, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.tmse_weight = tmse_weight
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: List of predictions from each stage (batch, num_classes, T)
            targets: Ground truth labels (batch, T)
        """
        loss = 0
        
        for p in predictions:
            # Cross-entropy loss (potentially with focal loss)
            if self.use_focal:
                ce = self.focal_loss(p, targets)
            else:
                ce = self.ce_loss(p, targets).mean()
            
            # Temporal MSE loss (smoothness)
            tmse = self.temporal_mse(p, targets)
            # tmse = torch.tensor(0.0, device=p.device)  # Disable TMSE
            
            loss += ce + self.tmse_weight * tmse
        
        return loss
    
    def focal_loss(self, pred, target):
        """Focal loss to handle class imbalance"""
        # Get cross entropy loss per sample
        ce_loss = self.ce_loss(pred, target)
        
        # Get probabilities
        pt = torch.exp(-ce_loss)
        
        # Focal loss: (1 - pt)^gamma * ce_loss
        focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
        
        return focal_loss.mean()
    
    def temporal_mse(self, pred, target):
        """Temporal smoothness loss"""
        # Convert predictions to probabilities
        pred = F.softmax(pred, dim=1)
        
        # Create one-hot encoding of targets
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        target_one_hot = target_one_hot.transpose(1, 2)  # (batch, num_classes, T)
        
        # MSE between predictions and targets
        loss = self.mse_loss(pred, target_one_hot)
        
        # Temporal smoothness: MSE between consecutive frames
        pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
        target_diff = target_one_hot[:, :, 1:] - target_one_hot[:, :, :-1]
        smooth_loss = self.mse_loss(pred_diff, target_diff)
        
        return loss.mean() + smooth_loss.mean()


# ============================================================================
# METRICS
# ============================================================================

def frame_accuracy(predictions, targets):
    """Compute frame-level accuracy"""
    pred_labels = torch.argmax(predictions, dim=1)  # (batch, T)
    correct = (pred_labels == targets).float()
    return correct.mean().item()


def edit_distance(predicted, target):
    """Compute Levenshtein edit distance between two sequences"""
    n, m = len(predicted), len(target)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if predicted[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[n][m]


def segment_edit_score(predictions, targets):
    """Compute segment-level edit score"""
    # Convert frame-level predictions to segments
    pred_segments = []
    target_segments = []
    
    for pred, target in zip(predictions, targets):
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        # Get segments (consecutive same labels)
        pred_seg = [pred[0]]
        for p in pred[1:]:
            if p != pred_seg[-1]:
                pred_seg.append(p)
        
        target_seg = [target[0]]
        for t in target[1:]:
            if t != target_seg[-1]:
                target_seg.append(t)
        
        # Compute edit distance
        ed = edit_distance(pred_seg, target_seg)
        normalized_ed = ed / max(len(pred_seg), len(target_seg))
        
        pred_segments.append(1 - normalized_ed)  # Higher is better
    
    return np.mean(pred_segments)


def compute_f1_scores(predictions, targets, num_classes):
    """Compute F1 score for each class"""
    pred_labels = torch.argmax(predictions, dim=1)
    
    f1_scores = []
    for c in range(num_classes):
        tp = ((pred_labels == c) & (targets == c)).sum().item()
        fp = ((pred_labels == c) & (targets != c)).sum().item()
        fn = ((pred_labels != c) & (targets == c)).sum().item()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        f1_scores.append(f1)
    
    return f1_scores


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    # Track label distribution in predictions
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        features = batch['features'].to(device)  # (batch, D, T)
        labels = batch['labels'].to(device)      # (batch, T)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(features)
        
        # Compute loss
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            acc = frame_accuracy(predictions[-1], labels)
            pred_labels = torch.argmax(predictions[-1], dim=1)
            all_predictions.append(pred_labels.cpu())
            all_labels.append(labels.cpu())
        
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    # Print class distribution
    all_predictions = torch.cat(all_predictions).numpy()  # Shape: (total_batches, T)
    all_labels = torch.cat(all_labels).numpy()  # Shape: (total_batches, T)
    
    # Flatten to get all frames (IMPORTANT: len() on 2D array only gives first dimension!)
    all_predictions_flat = all_predictions.flatten()
    all_labels_flat = all_labels.flatten()
    
    pred_unique, pred_counts = np.unique(all_predictions_flat, return_counts=True)
    label_unique, label_counts = np.unique(all_labels_flat, return_counts=True)
    
    # Debug: Check total counts
    total_pred_frames = len(all_predictions_flat)
    total_label_frames = len(all_labels_flat)
    
    # print(f"\n  [DEBUG] Total frames: Predictions={total_pred_frames}, Labels={total_label_frames}")
    # print(f"  [DEBUG] Array shapes: Predictions={all_predictions.shape}, Labels={all_labels.shape}")
    # print(f"  [DEBUG] Prediction unique values: {pred_unique}")
    # print(f"  [DEBUG] Label unique values: {label_unique}")
    
    print(f"\n  Label distribution (Ground Truth):")
    label_sum = 0
    for cls, count in zip(label_unique, label_counts):
        pct = 100 * count / total_label_frames
        label_sum += count
        print(f"    Class {cls}: {count:6d} ({pct:5.2f}%)")
    # print(f"  [DEBUG] Label sum: {label_sum}, should equal {total_label_frames}")
    
    print(f"\n  Label distribution (Predictions):")
    pred_sum = 0
    for cls, count in zip(pred_unique, pred_counts):
        pct = 100 * count / total_pred_frames
        pred_sum += count
        print(f"    Class {cls}: {count:6d} ({pct:5.2f}%)")
    # print(f"  [DEBUG] Prediction sum: {pred_sum}, should equal {total_pred_frames}")
    
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes):
    """Evaluate on validation set"""
    model.eval()
    
    total_loss = 0
    total_acc = 0
    total_edit = 0
    all_f1_scores = defaultdict(list)
    num_batches = 0
    
    # Track label distribution in predictions
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Validation')
    for batch in pbar:
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        predictions = model(features)
        
        # Compute loss
        loss = criterion(predictions, labels)
        
        # Metrics
        acc = frame_accuracy(predictions[-1], labels)
        edit = segment_edit_score(torch.argmax(predictions[-1], dim=1), labels)
        f1_scores = compute_f1_scores(predictions[-1], labels, num_classes)
        
        # Track predictions and labels
        pred_labels = torch.argmax(predictions[-1], dim=1)
        all_predictions.append(pred_labels.cpu())
        all_labels.append(labels.cpu())
        
        total_loss += loss.item()
        total_acc += acc
        total_edit += edit
        for i, f1 in enumerate(f1_scores):
            all_f1_scores[i].append(f1)
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.4f}',
            'edit': f'{edit:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    avg_edit = total_edit / num_batches
    avg_f1_scores = {i: np.mean(scores) for i, scores in all_f1_scores.items()}
    
    # Print validation class distribution
    all_predictions = torch.cat(all_predictions).numpy()  # Shape: (total_batches, T)
    all_labels = torch.cat(all_labels).numpy()  # Shape: (total_batches, T)
    
    # Flatten to get all frames
    all_predictions_flat = all_predictions.flatten()
    all_labels_flat = all_labels.flatten()
    
    pred_unique, pred_counts = np.unique(all_predictions_flat, return_counts=True)
    label_unique, label_counts = np.unique(all_labels_flat, return_counts=True)
    
    total_pred_frames = len(all_predictions_flat)
    total_label_frames = len(all_labels_flat)
    
    print(f"\n  [VALIDATION] Total frames: {total_label_frames}")
    print(f"  [VALIDATION] Label distribution (Ground Truth):")
    for cls, count in zip(label_unique, label_counts):
        pct = 100 * count / total_label_frames
        print(f"    Class {cls}: {count:6d} ({pct:5.2f}%)")
    
    print(f"  [VALIDATION] Label distribution (Predictions):")
    for cls, count in zip(pred_unique, pred_counts):
        pct = 100 * count / total_pred_frames
        print(f"    Class {cls}: {count:6d} ({pct:5.2f}%)")
    
    return avg_loss, avg_acc, avg_edit, avg_f1_scores


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MS-TCN for action segmentation')
    
    # Data parameters
    parser.add_argument('--dataset_root', type=str,
                        default='/data/zhaozhenghao/Projects/Mouse/datasets/UMich_CQ/all_data_80train_20_val',
                        help='Path to preprocessed dataset')
    parser.add_argument('--output_dir', type=str,
                        default='/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/action_seg_training/all_video_80triain_20val',
                        help='Output directory for checkpoints and logs')
    
    # Model parameters
    parser.add_argument('--num_stages', type=int, default=4,
                        help='Number of MS-TCN stages')
    parser.add_argument('--num_layers', type=int, default=10,
                        help='Number of layers per stage')
    parser.add_argument('--num_f_maps', type=int, default=64,
                        help='Number of feature maps')
    parser.add_argument('--feature_dim', type=int, default=2048,
                        help='Input feature dimension')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--tmse_weight', type=float, default=0.15,
                        help='Weight for temporal MSE loss')
    parser.add_argument('--class_weight', type=float, default=None,
                        help='Weight for minority class (auto-computed if None)')
    parser.add_argument('--use_oversampling', action='store_true', default=True,
                        help='Use oversampling to balance classes')
    parser.add_argument('--no_oversampling', dest='use_oversampling', action='store_false',
                        help='Disable oversampling')
    parser.add_argument('--oversample_power', type=float, default=2.0,
                        help='Power for oversampling weight (higher = more aggressive)')
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                        help='Use focal loss for class imbalance')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Multimodal parameters
    parser.add_argument('--use_keypoints', action='store_true',
                        help='Use keypoint features alongside ResNet features')
    parser.add_argument('--keypoint_root', type=str, default=None,
                        help='Path to keypoint features directory (default: dataset_root/../keypoint_features)')
    
    args = parser.parse_args()
    
    # Update feature_dim if using keypoints
    if args.use_keypoints:
        args.feature_dim = 2048 + 117  # ResNet (2048) + Keypoints (39*3=117)
        print(f"Using multimodal features: ResNet(2048) + Keypoints(117) = {args.feature_dim}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 80)
    print("Training MS-TCN for Action Segmentation")
    print("=" * 80)
    print(f"Dataset: {args.dataset_root}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Oversampling: {args.use_oversampling}")
    print(f"Focal loss: {args.use_focal_loss}")
    
    # Load dataset metadata
    with open(Path(args.dataset_root) / 'meta' / 'class_mapping.json', 'r') as f:
        mapping = json.load(f)
        num_classes = mapping['num_classes']
        idx_to_label = {int(k): v for k, v in mapping['idx_to_label'].items()}
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(idx_to_label.values())}")
    
    # Compute class weights
    if args.class_weight is None:
        with open(Path(args.dataset_root) / 'meta' / 'dataset_stats.json', 'r') as f:
            stats = json.load(f)
            label_dist = stats['label_distribution']
            total_frames = sum(label_dist.values())
            
            # Inverse frequency weighting
            class_weights = []
            for i in range(num_classes):
                label_name = idx_to_label[i]
                count = label_dist[label_name]
                weight = total_frames / (num_classes * count)
                class_weights.append(weight)
            
            class_weights = torch.FloatTensor(class_weights)
            print(f"Auto-computed class weights: {class_weights.tolist()}")
    else:
        class_weights = torch.FloatTensor([1.0, args.class_weight])
        print(f"Using class weights: {class_weights.tolist()}")
    
    # Create datasets and dataloaders
    print("\nLoading datasets...")
    train_dataset = ActionSegmentationTrainDataset(
        args.dataset_root, 
        split='train',
        use_keypoints=args.use_keypoints,
        keypoint_root=args.keypoint_root
    )
    val_dataset = ActionSegmentationTrainDataset(
        args.dataset_root, 
        split='val',
        use_keypoints=args.use_keypoints,
        keypoint_root=args.keypoint_root
    )
    
    # Compute sample weights for oversampling
    train_sampler = None
    if args.use_oversampling:
        print("\nComputing sample weights for oversampling...")
        sample_weights = compute_sample_weights(
            args.dataset_root, 
            split='train',
            power=args.oversample_power
        )
        
        # Print weight statistics
        print(f"Sample weight statistics:")
        print(f"  Min: {min(sample_weights):.4f}")
        print(f"  Max: {max(sample_weights):.4f}")
        print(f"  Mean: {np.mean(sample_weights):.4f}")
        print(f"  Std: {np.std(sample_weights):.4f}")
        print(f"  Power: {args.oversample_power}")
        
        # Create weighted sampler
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights) * 2,  # Sample 2x to ensure minority class coverage
            replacement=True
        )
        print("✅ Oversampling enabled for training set (2x samples)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # For validation, also use oversampling to evaluate on balanced data
    val_sampler = None
    if args.use_oversampling:
        val_sample_weights = compute_sample_weights(
            args.dataset_root, 
            split='val',
            power=args.oversample_power
        )
        val_sampler = WeightedRandomSampler(
            weights=val_sample_weights,
            num_samples=len(val_sample_weights) * 2,  # Sample 2x for validation too
            replacement=True
        )
        print("✅ Oversampling enabled for validation set (2x samples)")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} clips, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} clips, {len(val_loader)} batches")
    
    # Create model
    print("\nInitializing model...")
    model = MS_TCN(
        num_stages=args.num_stages,
        num_layers=args.num_layers,
        num_f_maps=args.num_f_maps,
        dim=args.feature_dim,
        num_classes=num_classes
    ).to(args.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = ActionSegmentationLoss(
        num_classes=num_classes,
        class_weights=class_weights.to(args.device),
        tmse_weight=args.tmse_weight,
        use_focal=args.use_focal_loss,
        focal_gamma=args.focal_gamma
    )
    
    print(f"\nLoss configuration:")
    print(f"  Class weights: {class_weights.tolist()}")
    print(f"  Focal loss: {args.use_focal_loss}")
    if args.use_focal_loss:
        print(f"  Focal gamma: {args.focal_gamma}")
    print(f"  TMSE weight: {args.tmse_weight}")
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Create training log file
    log_file = output_dir / 'training_log.csv'
    with open(log_file, 'w') as f:
        header = 'epoch,train_loss,train_acc,val_loss,val_acc,val_edit,lr'
        for i in range(num_classes):
            header += f',f1_{idx_to_label[i].replace(" ", "_")}'
        f.write(header + '\n')
    print(f"Logging to: {log_file}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0)
        print(f"Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch + 1
        )
        
        # Evaluate
        val_loss, val_acc, val_edit, val_f1_scores = evaluate(
            model, val_loader, criterion, args.device, num_classes
        )
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Log metrics to CSV
        with open(log_file, 'a') as f:
            log_line = f"{epoch + 1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{val_edit:.6f},{optimizer.param_groups[0]['lr']:.8f}"
            for i in range(num_classes):
                log_line += f",{val_f1_scores[i]:.6f}"
            f.write(log_line + '\n')
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Edit: {val_edit:.4f}")
        print(f"  F1 scores:")
        for i, f1 in val_f1_scores.items():
            print(f"    {idx_to_label[i]}: {f1:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_edit': val_edit,
            'val_f1_scores': val_f1_scores,
            'best_val_acc': best_val_acc,
            'config': vars(args)
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint['best_val_acc'] = best_val_acc
            torch.save(checkpoint, checkpoint_dir / 'best.pth')
            print(f"  ✅ New best model! Val Acc: {val_acc:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch + 1}.pth')
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
