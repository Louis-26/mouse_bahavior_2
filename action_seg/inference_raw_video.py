"""
Inference on Raw Video with Trained MS-TCN Model

This script performs end-to-end inference on a raw video file:
1. Extract frames from video
2. Extract features using ResNet50
3. Run MS-TCN inference
4. Generate visualization and save results

Usage:
    python inference_raw_video.py --video_path path/to/video.mp4 --checkpoint path/to/best.pth
"""

import argparse
import json
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

import sys
# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import MS-TCN model from action_seg/train.py
# Import here to avoid circular dependency
def load_model_class():
    """Dynamically load MS_TCN class to avoid import conflicts"""
    import importlib.util
    train_path = Path(__file__).parent / 'train.py'
    spec = importlib.util.spec_from_file_location("train_module", train_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    return train_module.MS_TCN

MS_TCN = load_model_class()


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """Extract features from video frames using pretrained ResNet50"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Check CUDA availability
        if device == 'cuda':
            if torch.cuda.is_available():
                print(f"✓ CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                print("⚠️  CUDA requested but not available! Falling back to CPU")
                self.device = 'cpu'
        
        # Load pretrained ResNet50 (MUST use same weights as training!)
        # Training uses ResNet50_Weights.IMAGENET1K_V2
        print(f"Loading ResNet50 on {self.device.upper()}...")
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove final classification layer
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ ResNet50 loaded on {self.device.upper()}")
        
        # Image preprocessing (MUST match training preprocessing!)
        # Training uses: Resize(256) -> CenterCrop(224)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @torch.no_grad()
    def extract_from_frames(self, frames, batch_size=32):
        """
        Extract features from a list of frames
        
        Args:
            frames: List of frames (H, W, 3) in BGR format
            batch_size: Batch size for processing
        
        Returns:
            features: Array of shape (num_frames, 2048)
        """
        num_frames = len(frames)
        features = []
        
        print(f"Extracting features from {num_frames} frames using {self.device.upper()}...")
        print(f"Batch size: {batch_size}")
        
        # Warm-up: process first batch to trigger any lazy initialization
        if num_frames > 0:
            if self.device == 'cuda':
                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
        
        for i in tqdm(range(0, num_frames, batch_size)):
            batch_frames = frames[i:i+batch_size]
            
            # Preprocess frames
            batch = []
            for frame in batch_frames:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Transform
                frame_tensor = self.transform(frame_rgb)
                batch.append(frame_tensor)
            
            batch = torch.stack(batch).to(self.device)
            
            # Extract features
            batch_features = self.model(batch)  # (batch, 2048, 1, 1)
            batch_features = batch_features.squeeze(-1).squeeze(-1)  # (batch, 2048)
            
            features.append(batch_features.cpu().numpy())
            
            # For first batch, show device info
            if i == 0 and self.device == 'cuda':
                print(f"\n✓ First batch processed on GPU")
                print(f"  Input tensor device: {batch.device}")
                print(f"  GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"  GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        features = np.concatenate(features, axis=0)  # (num_frames, 2048)
        
        print(f"\nExtracted features shape: {features.shape}")
        
        # Debug: Print feature statistics
        print("\n🔍 Debug: Feature statistics:")
        print(f"  Mean: {features.mean():.4f}")
        print(f"  Std: {features.std():.4f}")
        print(f"  Min: {features.min():.4f}")
        print(f"  Max: {features.max():.4f}")
        print(f"  Contains NaN: {np.isnan(features).any()}")
        print(f"  Contains Inf: {np.isinf(features).any()}")
        print()
        
        return features


# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def extract_frames_from_video(video_path):
    """
    Extract all frames from a video file
    
    Args:
        video_path: Path to video file
    
    Returns:
        frames: List of frames (H, W, 3) in BGR format
        fps: Frame rate of the video
        total_frames: Total number of frames
    """
    print(f"Reading video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    
    frames = []
    
    print("Extracting frames...")
    pbar = tqdm(total=total_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"Extracted {len(frames)} frames")
    
    return frames, fps, total_frames


# ============================================================================
# INFERENCE
# ============================================================================

@torch.no_grad()
def segment_and_predict(model, features, window_length=512, stride=256, device='cuda'):
    """
    Segment features into clips and run inference
    
    Args:
        model: Trained MS-TCN model
        features: Feature array (num_frames, 2048)
        window_length: Length of each clip
        stride: Stride between clips
        device: torch device
    
    Returns:
        frame_predictions: Predicted label for each frame (num_frames,)
        frame_probabilities: Prediction probabilities (num_frames, num_classes)
    """
    model.eval()
    
    num_frames = len(features)
    num_classes = model.stage1.conv_out.out_channels
    
    # Initialize arrays to accumulate predictions
    prediction_sum = np.zeros((num_frames, num_classes), dtype=np.float32)
    prediction_count = np.zeros(num_frames, dtype=np.int32)
    
    # Generate clips with sliding window
    clips = []
    clip_positions = []
    
    for start in range(0, num_frames, stride):
        end = min(start + window_length, num_frames)
        
        if end - start < window_length:
            # Pad the last clip if needed
            clip_features = np.zeros((window_length, features.shape[1]), dtype=np.float32)
            clip_features[:end-start] = features[start:end]
            clip_length = end - start
        else:
            clip_features = features[start:end]
            clip_length = window_length
        
        clips.append(clip_features)
        clip_positions.append((start, end, clip_length))
    
    print(f"Created {len(clips)} clips (window={window_length}, stride={stride})")
    
    # Run inference on each clip
    print("Running inference on clips...")
    
    # Debug: Track raw predictions before averaging
    debug_raw_predictions = []
    debug_raw_probs = []
    
    for clip_features, (start, end, clip_length) in tqdm(zip(clips, clip_positions), total=len(clips)):
        # Prepare input
        clip_tensor = torch.from_numpy(clip_features).float()  # (T, D)
        clip_tensor = clip_tensor.transpose(0, 1).unsqueeze(0)  # (1, D, T)
        clip_tensor = clip_tensor.to(device)
        
        # Forward pass
        outputs = model(clip_tensor)
        
        # Get final stage predictions
        final_output = outputs[-1]  # (1, num_classes, T)
        
        # Convert to probabilities
        probs = F.softmax(final_output, dim=1)  # (1, num_classes, T)
        probs = probs.squeeze(0).transpose(0, 1).cpu().numpy()  # (T, num_classes)
        
        # Debug: Store raw predictions for first few clips
        if len(debug_raw_predictions) < 5:
            raw_pred = np.argmax(probs[:clip_length], axis=1)
            debug_raw_predictions.append(raw_pred)
            debug_raw_probs.append(probs[:clip_length])
        
        # Accumulate predictions (only for valid frames)
        prediction_sum[start:end] += probs[:clip_length]
        prediction_count[start:end] += 1
    
    # Debug: Print statistics from raw predictions
    if debug_raw_predictions:
        print("\n🔍 Debug: Raw predictions from first 5 clips (BEFORE averaging):")
        for i, (raw_pred, raw_prob) in enumerate(zip(debug_raw_predictions, debug_raw_probs)):
            unique, counts = np.unique(raw_pred, return_counts=True)
            print(f"  Clip {i}:")
            for cls, count in zip(unique, counts):
                pct = 100 * count / len(raw_pred)
                print(f"    Class {cls}: {count}/{len(raw_pred)} ({pct:.1f}%)")
            # Print probability statistics
            mean_probs = raw_prob.mean(axis=0)
            print(f"    Mean probabilities: {mean_probs}")
            max_prob_cls1 = raw_prob[:, 1].max()
            print(f"    Max prob for class 1: {max_prob_cls1:.4f}")
        print()
    
    # Average predictions across overlapping clips
    frame_probabilities = prediction_sum / np.maximum(prediction_count[:, None], 1)
    frame_predictions = np.argmax(frame_probabilities, axis=1)
    
    # Debug: Print final prediction statistics
    print("\n🔍 Debug: Final predictions (AFTER averaging):")
    unique, counts = np.unique(frame_predictions, return_counts=True)
    for cls, count in zip(unique, counts):
        pct = 100 * count / num_frames
        print(f"  Class {cls}: {count}/{num_frames} ({pct:.1f}%)")
    
    # Print probability statistics
    mean_probs = frame_probabilities.mean(axis=0)
    print(f"  Mean probabilities across all frames: {mean_probs}")
    print(f"  Max probability for class 1: {frame_probabilities[:, 1].max():.4f}")
    print(f"  Min probability for class 0: {frame_probabilities[:, 0].min():.4f}")
    
    # Check if any frame has class 1 probability > 0.5
    high_prob_frames = (frame_probabilities[:, 1] > 0.5).sum()
    print(f"  Frames with P(class 1) > 0.5: {high_prob_frames}")
    print()
    
    return frame_predictions, frame_probabilities


# ============================================================================
# KEYPOINT LOADING
# ============================================================================

def load_ground_truth(gt_path, num_frames, fps, label_to_idx, frame_offset=0):
    """
    Load ground truth labels from CSV file
    
    Args:
        gt_path: Path to ground truth CSV file (with Start, End, Notes columns)
        num_frames: Number of frames in the video/predictions
        fps: Frame rate of the video
        label_to_idx: Mapping from label name to class index
        frame_offset: Frame offset from original video (for time range filtering)
    
    Returns:
        ground_truth: Binary array (num_frames,) with frame-level labels
    """
    print(f"📋 Loading ground truth from: {gt_path}")
    
    # Read CSV
    df = pd.read_csv(gt_path)
    
    # Initialize ground truth array
    ground_truth = np.zeros(num_frames, dtype=np.int32)
    
    # Parse each segment
    for _, row in df.iterrows():
        start_str = row['Start']
        end_str = row['End']
        label_str = row['Notes']
        
        # Parse time strings (format: HH:MM:SS.mmm)
        def parse_time(time_str):
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        
        start_time = parse_time(start_str)
        end_time = parse_time(end_str)
        
        # Convert to frames (absolute frame numbers in original video)
        start_frame_abs = int(start_time * fps)
        end_frame_abs = int(end_time * fps)
        
        # Account for frame offset
        start_frame = start_frame_abs - frame_offset
        end_frame = end_frame_abs - frame_offset
        
        # Clip to valid range
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        
        # Skip if segment is outside the range
        if start_frame >= num_frames or end_frame <= 0:
            continue
        
        # Get class index
        if label_str in label_to_idx:
            class_idx = label_to_idx[label_str]
            ground_truth[start_frame:end_frame] = class_idx
    
    # Print statistics
    unique, counts = np.unique(ground_truth, return_counts=True)
    print(f"   Ground truth statistics:")
    for cls, count in zip(unique, counts):
        pct = 100 * count / num_frames
        print(f"     Class {cls}: {count}/{num_frames} ({pct:.1f}%)")
    
    return ground_truth


# ============================================================================
# KEYPOINT LOADING AND PROCESSING
# ============================================================================

def load_keypoints(video_path, keypoint_dir):
    """
    Load keypoints from JSON file
    
    Args:
        video_path: Path to video file (to extract video name)
        keypoint_dir: Directory containing keypoint JSON files
    
    Returns:
        keypoints_data: List of keypoint data per frame, or None if not found
    """
    video_name = Path(video_path).stem
    keypoint_dir = Path(keypoint_dir)
    
    # Look for matching keypoint file
    # Pattern: {video_name}_*_before_adapt.json or {video_name}_*.json
    json_files = list(keypoint_dir.glob(f"{video_name}_*_before_adapt.json"))
    if not json_files:
        json_files = list(keypoint_dir.glob(f"{video_name}_*.json"))
    
    if not json_files:
        print(f"⚠️  No keypoint file found for {video_name}")
        return None
    
    json_path = json_files[0]
    print(f"📍 Loading keypoints from: {json_path.name}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"   Loaded {len(data)} frames of keypoint data")
        return data
    except Exception as e:
        print(f"⚠️  Error loading keypoints: {e}")
        return None


def extract_keypoint_features(keypoints_data, num_frames, num_keypoints=39, start_frame=0):
    """
    Extract keypoint features for multimodal input
    
    Args:
        keypoints_data: List of keypoint data per frame (from JSON)
        num_frames: Number of frames to extract (after time range filtering)
        num_keypoints: Number of keypoints per animal (default: 39)
        start_frame: Starting frame index (for time range filtering)
    
    Returns:
        keypoint_features: Array of shape (num_frames, num_keypoints * 3)
                          Each keypoint has (x, y, confidence)
    """
    print(f"\n📍 Extracting keypoint features for multimodal input...")
    print(f"   Target frames: {num_frames} (starting from frame {start_frame})")
    
    keypoint_dim = num_keypoints * 3  # x, y, confidence for each keypoint
    keypoint_features = np.zeros((num_frames, keypoint_dim), dtype=np.float32)
    
    if keypoints_data is None:
        print(f"   ⚠️  No keypoint data available, using zero features")
        return keypoint_features
    
    # Process each frame
    frames_with_keypoints = 0
    for i in range(num_frames):
        frame_idx = start_frame + i
        
        # Check if frame is within keypoint data range
        if frame_idx >= len(keypoints_data):
            continue
        
        frame_data = keypoints_data[frame_idx]
        
        # Check if frame has bodyparts
        if frame_data is None or 'bodyparts' not in frame_data:
            continue
        
        bodyparts_list = frame_data['bodyparts']
        
        # Take first animal if multiple detected
        if len(bodyparts_list) > 0:
            bodyparts = bodyparts_list[0]  # Shape: (num_keypoints, 3)
            bodyparts = np.array(bodyparts)
            
            # Flatten to 1D: [x1, y1, conf1, x2, y2, conf2, ...]
            if bodyparts.shape[0] == num_keypoints:
                keypoint_features[i] = bodyparts.flatten()
                frames_with_keypoints += 1
            elif bodyparts.shape[0] < num_keypoints:
                # Pad if fewer keypoints
                flat_features = bodyparts.flatten()
                keypoint_features[i, :len(flat_features)] = flat_features
                frames_with_keypoints += 1
            else:
                # Truncate if more keypoints
                keypoint_features[i] = bodyparts[:num_keypoints].flatten()
                frames_with_keypoints += 1
    
    print(f"   ✓ Extracted keypoint features: {keypoint_features.shape}")
    print(f"   Frames with valid keypoints: {frames_with_keypoints} / {num_frames} ({100*frames_with_keypoints/num_frames:.1f}%)")
    
    # Debug: Print feature statistics
    print(f"\n🔍 Debug: Keypoint feature statistics:")
    print(f"  Mean: {keypoint_features.mean():.4f}")
    print(f"  Std: {keypoint_features.std():.4f}")
    print(f"  Min: {keypoint_features.min():.4f}")
    print(f"  Max: {keypoint_features.max():.4f}")
    print(f"  Non-zero elements: {(keypoint_features != 0).sum()} / {keypoint_features.size} ({100*(keypoint_features != 0).sum()/keypoint_features.size:.1f}%)")
    print()
    
    return keypoint_features


def draw_keypoints(frame, keypoints_frame_data, confidence_threshold=0.3):
    """
    Draw keypoints and skeleton on frame
    
    Args:
        frame: Frame to draw on
        keypoints_frame_data: Keypoint data for this frame (dict with 'bodyparts', 'bboxes', etc.)
        confidence_threshold: Minimum confidence to draw keypoint
    
    Returns:
        frame with keypoints drawn
    """
    if keypoints_frame_data is None or 'bodyparts' not in keypoints_frame_data:
        return frame
    
    bodyparts_list = keypoints_frame_data['bodyparts']
    
    # Process each detected animal (usually just one)
    for animal_idx, bodyparts in enumerate(bodyparts_list):
        # bodyparts is a list of [x, y, confidence] for each keypoint
        keypoints = np.array(bodyparts)  # Shape: (num_keypoints, 3)
        
        # Define skeleton connections (approximate quadruped skeleton)
        # This is a simplified skeleton - adjust based on your specific bodypart model
        skeleton = [
            # Head/neck
            (0, 1), (1, 2),
            # Spine
            (5, 6), (6, 7), (7, 8),
            # Front left leg
            (10, 11), (11, 12),
            # Front right leg
            (15, 16), (16, 17),
            # Back left leg
            (20, 21), (21, 22),
            # Back right leg
            (30, 31), (31, 32),
        ]
        
        # Draw skeleton connections first (so they appear under keypoints)
        for pt1_idx, pt2_idx in skeleton:
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                x1, y1, conf1 = keypoints[pt1_idx]
                x2, y2, conf2 = keypoints[pt2_idx]
                
                # Only draw if both points have sufficient confidence
                if conf1 > confidence_threshold and conf2 > confidence_threshold:
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    # cv2.line(frame, pt1, pt2, (0, 255, 255), 2)  # Yellow lines
        
        # Draw keypoints
        for kp_idx, (x, y, conf) in enumerate(keypoints):
            if conf > confidence_threshold:
                center = (int(x), int(y))
                
                # Color based on confidence: red (low) -> green (high)
                color_intensity = int(255 * conf)
                color = (0, color_intensity, 255 - color_intensity)  # BGR
                
                # Draw circle for keypoint
                cv2.circle(frame, center, 4, color, -1)
                cv2.circle(frame, center, 4, (255, 255, 255), 1)  # White outline
        
        # Draw bounding box if available
        if 'bboxes' in keypoints_frame_data and len(keypoints_frame_data['bboxes']) > animal_idx:
            bbox = keypoints_frame_data['bboxes'][animal_idx]
            x, y, w, h = bbox
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)  # Blue bbox
            
            # Draw bbox score
            if 'bbox_scores' in keypoints_frame_data and len(keypoints_frame_data['bbox_scores']) > animal_idx:
                score = keypoints_frame_data['bbox_scores'][animal_idx]
                label = f"Det: {score:.2f}"
                cv2.putText(frame, label, (int(x), int(y) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return frame


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_prediction_video(frames, predictions, idx_to_label, output_path, fps=30, 
                           keypoints_data=None, frame_offset=0):
    """
    Create a video with prediction overlays and keypoints
    
    Args:
        frames: List of video frames
        predictions: Predicted labels for each frame
        idx_to_label: Mapping from index to label name
        output_path: Path to save output video
        fps: Frame rate
        keypoints_data: Optional keypoint data for each frame
        frame_offset: Frame offset if using time range filtering
    """
    print(f"Creating prediction video: {output_path}")
    
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Color map for classes
    colors_rgb = plt.cm.Set3(range(len(idx_to_label)))
    colors_bgr = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in colors_rgb]
    
    for frame_idx, (frame, pred) in enumerate(tqdm(zip(frames, predictions), total=len(frames), desc="Writing video")):
        frame_copy = frame.copy()
        
        # Draw keypoints if available
        if keypoints_data is not None:
            # Account for frame offset when indexing keypoints
            keypoint_idx = frame_idx + frame_offset
            if keypoint_idx < len(keypoints_data):
                frame_copy = draw_keypoints(frame_copy, keypoints_data[keypoint_idx])
        
        # Add colored banner at top
        label = idx_to_label[pred]
        color = colors_bgr[pred]
        
        # Draw filled rectangle
        cv2.rectangle(frame_copy, (0, 0), (width, 50), color, -1)
        
        # Add text
        text = f"{label}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Center text
        text_x = (width - text_width) // 2
        text_y = (50 + text_height) // 2
        
        # Draw text with white outline
        cv2.putText(frame_copy, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness + 2)
        cv2.putText(frame_copy, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        out.write(frame_copy)
    
    out.release()
    print(f"Saved prediction video to {output_path}")


def visualize_timeline(predictions, idx_to_label, fps, output_path):
    """
    Create a timeline visualization of predictions
    
    Args:
        predictions: Predicted labels for each frame
        idx_to_label: Mapping from index to label name
        fps: Frame rate
        output_path: Path to save figure
    """
    print(f"Creating timeline visualization: {output_path}")
    
    num_frames = len(predictions)
    duration_sec = num_frames / fps
    
    fig, ax = plt.subplots(figsize=(20, 4))
    
    # Color map for classes
    colors = plt.cm.Set3(range(len(idx_to_label)))
    color_map = {i: colors[i] for i in range(len(idx_to_label))}
    
    # Plot timeline
    for i in range(num_frames):
        time_start = i / fps
        time_end = (i + 1) / fps
        ax.axvspan(time_start, time_end, facecolor=color_map[predictions[i]], alpha=0.8)
    
    ax.set_xlim(0, duration_sec)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Action', fontsize=14, fontweight='bold')
    ax.set_title('Action Segmentation Timeline', fontsize=16, fontweight='bold')
    ax.set_yticks([])
    
    # Add legend
    patches = [mpatches.Patch(color=color_map[i], label=idx_to_label[i]) 
               for i in range(len(idx_to_label))]
    ax.legend(handles=patches, loc='upper right', ncol=len(idx_to_label))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved timeline to {output_path}")

def plot_behavior_timeline(predictions, probabilities, ground_truth, fps=30, output_path='behavior_timeline.png'):
    """Plot behavior predictions vs ground truth over time."""
    time_seconds = np.arange(len(predictions)) / fps
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Ground truth
    axes[0].fill_between(time_seconds, 0, ground_truth, 
                        alpha=0.7, color='green', label='Ground Truth Scratching')
    axes[0].set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
    axes[0].set_title('Scratching Behavior Detection Timeline', fontsize=14, fontweight='bold')
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Predictions
    axes[1].fill_between(time_seconds, 0, predictions, 
                        alpha=0.7, color='red', label='Predicted Scratching')
    axes[1].set_ylabel('Predictions', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Probabilities
    # axes[2].plot(time_seconds, probabilities, color='blue', alpha=0.8, linewidth=1)
    # axes[2].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Threshold')
    # axes[2].fill_between(time_seconds, 0, probabilities, alpha=0.3, color='blue')
    # axes[2].set_ylabel('Scratching Probability', fontsize=12, fontweight='bold')
    # axes[2].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    # axes[2].set_ylim(0, 1)
    # axes[2].legend()
    # axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Behavior timeline saved as '{output_path}'")
    plt.close()
    
def compute_statistics(predictions, idx_to_label, fps, frame_offset=0, ground_truth=None):
    """
    Compute statistics about predictions
    
    Args:
        predictions: Predicted labels for each frame
        idx_to_label: Mapping from index to label name
        fps: Frame rate
        frame_offset: Frame offset from original video (for time range filtering)
        ground_truth: Optional ground truth labels for comparison
    
    Returns:
        stats: Dictionary of statistics
    """
    num_frames = len(predictions)
    duration_sec = num_frames / fps
    
    stats = {
        'total_frames': num_frames,
        'duration_sec': duration_sec,
        'fps': fps,
        'frame_offset': frame_offset,
        'time_offset': frame_offset / fps,
        'class_distribution': {},
        'segments': []
    }
    
    # Class distribution
    for i in range(len(idx_to_label)):
        count = (predictions == i).sum()
        percentage = count / num_frames * 100
        duration = count / fps
        
        stats['class_distribution'][idx_to_label[i]] = {
            'frames': int(count),
            'percentage': float(percentage),
            'duration_sec': float(duration)
        }
    
    # Segment analysis
    if len(predictions) > 0:
        current_label = predictions[0]
        segment_start = 0
        
        for i in range(1, len(predictions)):
            if predictions[i] != current_label:
                # End of segment
                segment_end = i
                segment_duration = (segment_end - segment_start) / fps
                
                stats['segments'].append({
                    'label': idx_to_label[current_label],
                    'start_frame': int(segment_start),
                    'end_frame': int(segment_end),
                    'start_time': float(segment_start / fps + frame_offset / fps),
                    'end_time': float(segment_end / fps + frame_offset / fps),
                    'duration': float(segment_duration),
                    'start_frame_absolute': int(segment_start + frame_offset),
                    'end_frame_absolute': int(segment_end + frame_offset)
                })
                
                current_label = predictions[i]
                segment_start = i
        
        # Last segment
        segment_end = len(predictions)
        segment_duration = (segment_end - segment_start) / fps
        stats['segments'].append({
            'label': idx_to_label[current_label],
            'start_frame': int(segment_start),
            'end_frame': int(segment_end),
            'start_time': float(segment_start / fps + frame_offset / fps),
            'end_time': float(segment_end / fps + frame_offset / fps),
            'duration': float(segment_duration),
            'start_frame_absolute': int(segment_start + frame_offset),
            'end_frame_absolute': int(segment_end + frame_offset)
        })
    
    stats['num_segments'] = len(stats['segments'])
    
    # Compute scratching frequency if ground truth is provided
    if ground_truth is not None:
        # Compute scratching bouts (continuous scratching episodes)
        def count_bouts(labels):
            """Count number of scratching bouts and their durations"""
            bouts = []
            in_bout = False
            bout_start = None
            
            for i, label in enumerate(labels):
                if label == 1:  # Scratching
                    if not in_bout:
                        in_bout = True
                        bout_start = i
                else:  # Not scratching
                    if in_bout:
                        in_bout = False
                        bout_duration = (i - bout_start) / fps
                        bouts.append({
                            'start_frame': bout_start,
                            'end_frame': i,
                            'duration_sec': bout_duration
                        })
            
            # Handle case where last bout extends to end
            if in_bout:
                bout_duration = (len(labels) - bout_start) / fps
                bouts.append({
                    'start_frame': bout_start,
                    'end_frame': len(labels),
                    'duration_sec': bout_duration
                })
            
            return bouts
        
        # Count bouts for ground truth and predictions
        gt_bouts = count_bouts(ground_truth)
        pred_bouts = count_bouts(predictions)
        
        # Compute statistics
        gt_scratching_frames = (ground_truth == 1).sum()
        pred_scratching_frames = (predictions == 1).sum()
        
        gt_scratching_duration = gt_scratching_frames / fps
        pred_scratching_duration = pred_scratching_frames / fps
        
        # Compute frequency (bouts per minute)
        duration_minutes = duration_sec / 60
        gt_frequency = len(gt_bouts) / duration_minutes if duration_minutes > 0 else 0
        pred_frequency = len(pred_bouts) / duration_minutes if duration_minutes > 0 else 0
        
        # Compute average bout duration
        gt_avg_bout_duration = np.mean([b['duration_sec'] for b in gt_bouts]) if gt_bouts else 0
        pred_avg_bout_duration = np.mean([b['duration_sec'] for b in pred_bouts]) if pred_bouts else 0
        
        # Add to stats
        stats['scratching_frequency'] = {
            'ground_truth': {
                'num_bouts': len(gt_bouts),
                'frequency_per_minute': float(gt_frequency),
                'total_scratching_frames': int(gt_scratching_frames),
                'total_scratching_duration_sec': float(gt_scratching_duration),
                'scratching_percentage': float(100 * gt_scratching_frames / num_frames),
                'average_bout_duration_sec': float(gt_avg_bout_duration),
                'bouts': gt_bouts
            },
            'predictions': {
                'num_bouts': len(pred_bouts),
                'frequency_per_minute': float(pred_frequency),
                'total_scratching_frames': int(pred_scratching_frames),
                'total_scratching_duration_sec': float(pred_scratching_duration),
                'scratching_percentage': float(100 * pred_scratching_frames / num_frames),
                'average_bout_duration_sec': float(pred_avg_bout_duration),
                'bouts': pred_bouts
            },
            'comparison': {
                'bout_count_difference': len(pred_bouts) - len(gt_bouts),
                'frequency_difference_per_minute': float(pred_frequency - gt_frequency),
                'duration_difference_sec': float(pred_scratching_duration - gt_scratching_duration),
                'percentage_difference': float(100 * (pred_scratching_frames - gt_scratching_frames) / num_frames)
            }
        }
    
    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run inference on raw video with trained MS-TCN')
    
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as video with _results suffix)')
    parser.add_argument('--keypoint_dir', type=str, default=None,
                        help='Directory containing keypoint JSON files (e.g., results/UMich_CQ/keypoint)')
    parser.add_argument('--ground_truth', type=str, default=None,
                        help='Path to ground truth CSV file (e.g., datasets/UMich_CQ/CQ_3.csv)')
    
    # Multimodal parameters
    parser.add_argument('--use_keypoints', action='store_true',
                        help='Use keypoint features alongside ResNet features (multimodal)')
    
    # Inference parameters
    parser.add_argument('--window_length', type=int, default=512,
                        help='Length of each clip for inference')
    parser.add_argument('--stride', type=int, default=256,
                        help='Stride between clips')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    
    # Time range parameters
    parser.add_argument('--start_time', type=float, default=None,
                        help='Start time in seconds (default: from beginning)')
    parser.add_argument('--end_time', type=float, default=None,
                        help='End time in seconds (default: until end)')
    
    # Output options
    parser.add_argument('--save_video', action='store_true',
                        help='Create video with prediction overlay')
    parser.add_argument('--save_features', action='store_true',
                        help='Save extracted features')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup output directory
    video_path = Path(args.video_path)
    if args.output_dir is None:
        output_dir = video_path.parent / f'{video_path.stem}_results'
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MS-TCN Inference on Raw Video")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Multimodal: {args.use_keypoints}")
    if args.keypoint_dir:
        print(f"Keypoint dir: {args.keypoint_dir}")
    if args.ground_truth:
        print(f"Ground truth: {args.ground_truth}")
    
    # Load keypoints if directory is provided or if using multimodal
    keypoints_data = None
    if args.keypoint_dir or args.use_keypoints:
        if args.keypoint_dir is None:
            print("\n⚠️  Warning: --use_keypoints enabled but --keypoint_dir not provided")
            print("    Multimodal mode will use zero-padded keypoint features")
        else:
            print("\n" + "=" * 80)
            print("Loading Keypoints")
            print("=" * 80)
            keypoints_data = load_keypoints(video_path, args.keypoint_dir)
    
    # Load checkpoint
    print("\n" + "=" * 80)
    print("Loading Model")
    print("=" * 80)
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint['config']
    
    # Get class mapping from checkpoint or use default
    if 'class_mapping' in checkpoint:
        mapping = checkpoint['class_mapping']
    else:
        # Try to load from default location
        default_mapping_path = Path(config['dataset_root']) / 'meta' / 'class_mapping.json'
        if default_mapping_path.exists():
            with open(default_mapping_path, 'r') as f:
                mapping = json.load(f)
        else:
            # Default for 2-class problem
            mapping = {
                'num_classes': 2,
                'idx_to_label': {0: 'no behavior', 1: 'scracthing'}
            }
    
    num_classes = mapping['num_classes']
    idx_to_label = {int(k): v for k, v in mapping['idx_to_label'].items()}
    
    # Create reverse mapping for ground truth loading
    label_to_idx = {v: int(k) for k, v in mapping['idx_to_label'].items()}
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(idx_to_label.values())}")
    
    # Determine feature dimension
    # Check if checkpoint was trained with keypoints
    checkpoint_feature_dim = config.get('feature_dim', 2048)
    checkpoint_uses_keypoints = checkpoint_feature_dim > 2048
    
    if checkpoint_uses_keypoints:
        print(f"\n✓ Checkpoint was trained with multimodal features (dim={checkpoint_feature_dim})")
        if not args.use_keypoints:
            print(f"⚠️  WARNING: Model expects keypoint features but --use_keypoints not enabled!")
            print(f"   Enabling multimodal mode automatically...")
            args.use_keypoints = True
    
    if args.use_keypoints:
        feature_dim = 2048 + 117  # ResNet (2048) + Keypoints (39*3=117)
        print(f"Using multimodal features: ResNet(2048) + Keypoints(117) = {feature_dim}")
        
        if checkpoint_feature_dim != feature_dim:
            print(f"⚠️  WARNING: Feature dimension mismatch!")
            print(f"   Expected: {checkpoint_feature_dim}, Got: {feature_dim}")
            print(f"   This may cause inference errors!")
    else:
        feature_dim = 2048
        print(f"Using ResNet features only (dim={feature_dim})")
    
    # Create model
    model = MS_TCN(
        num_stages=config['num_stages'],
        num_layers=config['num_layers'],
        num_f_maps=config['num_f_maps'],
        dim=feature_dim,
        num_classes=num_classes
    ).to(args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    if 'val_acc' in checkpoint:
        print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
    
    # Extract frames from video
    print("\n" + "=" * 80)
    print("Step 1: Extract Frames")
    print("=" * 80)
    
    frames, fps, total_frames = extract_frames_from_video(video_path)
    
    # Apply time range filter if specified
    if args.start_time is not None or args.end_time is not None:
        start_frame = int(args.start_time * fps) if args.start_time is not None else 0
        end_frame = int(args.end_time * fps) if args.end_time is not None else len(frames)
        
        # Validate range
        start_frame = max(0, start_frame)
        end_frame = min(len(frames), end_frame)
        
        if start_frame >= end_frame:
            raise ValueError(f"Invalid time range: start_time={args.start_time}s (frame {start_frame}) >= end_time={args.end_time}s (frame {end_frame})")
        
        print(f"\n⏱️  Applying time range filter:")
        print(f"  Original video: {len(frames)} frames ({len(frames)/fps:.2f}s)")
        print(f"  Time range: {start_frame/fps:.2f}s - {end_frame/fps:.2f}s")
        print(f"  Frame range: {start_frame} - {end_frame}")
        print(f"  Selected: {end_frame - start_frame} frames ({(end_frame - start_frame)/fps:.2f}s)")
        
        frames = frames[start_frame:end_frame]
        
        # Store offset for later use in statistics
        frame_offset = start_frame
    else:
        frame_offset = 0
    
    # Extract features
    print("\n" + "=" * 80)
    print("Step 2: Extract Features")
    print("=" * 80)
    
    feature_extractor = FeatureExtractor(device=args.device)
    resnet_features = feature_extractor.extract_from_frames(frames, batch_size=args.batch_size)
    
    print(f"Extracted ResNet features shape: {resnet_features.shape}")
    
    # Combine with keypoint features if using multimodal
    if args.use_keypoints:
        print("\n" + "-" * 80)
        print("Combining with Keypoint Features (Multimodal)")
        print("-" * 80)
        
        keypoint_features = extract_keypoint_features(
            keypoints_data, 
            num_frames=len(frames),
            num_keypoints=39,
            start_frame=frame_offset
        )
        
        # Concatenate ResNet and keypoint features
        features = np.concatenate([resnet_features, keypoint_features], axis=1)
        print(f"Combined features shape: {features.shape}")
        print(f"  ResNet: {resnet_features.shape[1]} dims")
        print(f"  Keypoints: {keypoint_features.shape[1]} dims")
        print(f"  Total: {features.shape[1]} dims")
    else:
        features = resnet_features
        print(f"Using ResNet features only: {features.shape}")
    
    if args.save_features:
        if args.use_keypoints:
            # Save separate and combined features
            resnet_path = output_dir / 'resnet_features.npy'
            keypoint_path = output_dir / 'keypoint_features.npy'
            combined_path = output_dir / 'features.npy'
            np.save(resnet_path, resnet_features)
            np.save(keypoint_path, keypoint_features)
            np.save(combined_path, features)
            print(f"Saved ResNet features to {resnet_path}")
            print(f"Saved keypoint features to {keypoint_path}")
            print(f"Saved combined features to {combined_path}")
        else:
            feature_path = output_dir / 'features.npy'
            np.save(feature_path, features)
            print(f"Saved features to {feature_path}")
    
    # Run inference
    print("\n" + "=" * 80)
    print("Step 3: Run Inference")
    print("=" * 80)
    
    predictions, probabilities = segment_and_predict(
        model, features,
        window_length=args.window_length,
        stride=args.stride,
        device=args.device
    )
    
    print(f"Generated predictions for {len(predictions)} frames")
    
    # Save predictions
    pred_path = output_dir / 'predictions.npy'
    prob_path = output_dir / 'probabilities.npy'
    np.save(pred_path, predictions)
    np.save(prob_path, probabilities)
    print(f"Saved predictions to {pred_path}")
    print(f"Saved probabilities to {prob_path}")
    
    # Load ground truth if provided (before computing statistics)
    ground_truth = None
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        if gt_path.exists():
            print(f"\n📋 Loading ground truth labels...")
            ground_truth = load_ground_truth(
                gt_path, 
                num_frames=len(predictions), 
                fps=fps, 
                label_to_idx=label_to_idx,
                frame_offset=frame_offset
            )
        else:
            print(f"⚠️  Ground truth file not found: {gt_path}")
    
    # Compute statistics
    print("\n" + "=" * 80)
    print("Step 4: Compute Statistics")
    print("=" * 80)
    
    stats = compute_statistics(predictions, idx_to_label, fps, frame_offset, ground_truth)
    
    print(f"\nVideo Duration: {stats['duration_sec']:.2f} seconds ({stats['total_frames']} frames)")
    if frame_offset > 0:
        print(f"Time offset: {stats['time_offset']:.2f} seconds (absolute time in original video)")
    print(f"FPS: {stats['fps']:.2f}")
    print(f"\nClass Distribution:")
    for label, info in stats['class_distribution'].items():
        print(f"  {label}:")
        print(f"    Frames: {info['frames']} ({info['percentage']:.2f}%)")
        print(f"    Duration: {info['duration_sec']:.2f} seconds")
    
    print(f"\nNumber of segments: {stats['num_segments']}")
    print(f"First 5 segments:")
    for seg in stats['segments'][:5]:
        print(f"  {seg['label']}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s ({seg['duration']:.2f}s)")
    
    # Print scratching frequency if available
    if 'scratching_frequency' in stats:
        print("\n" + "-" * 80)
        print("Scratching Frequency Analysis")
        print("-" * 80)
        
        sf = stats['scratching_frequency']
        
        print("\n📊 Ground Truth:")
        print(f"  Number of scratching bouts: {sf['ground_truth']['num_bouts']}")
        print(f"  Frequency: {sf['ground_truth']['frequency_per_minute']:.2f} bouts/minute")
        print(f"  Total scratching duration: {sf['ground_truth']['total_scratching_duration_sec']:.2f} seconds ({sf['ground_truth']['scratching_percentage']:.2f}%)")
        print(f"  Average bout duration: {sf['ground_truth']['average_bout_duration_sec']:.2f} seconds")
        
        print("\n📊 Predictions:")
        print(f"  Number of scratching bouts: {sf['predictions']['num_bouts']}")
        print(f"  Frequency: {sf['predictions']['frequency_per_minute']:.2f} bouts/minute")
        print(f"  Total scratching duration: {sf['predictions']['total_scratching_duration_sec']:.2f} seconds ({sf['predictions']['scratching_percentage']:.2f}%)")
        print(f"  Average bout duration: {sf['predictions']['average_bout_duration_sec']:.2f} seconds")
        
        print("\n📊 Comparison:")
        print(f"  Bout count difference: {sf['comparison']['bout_count_difference']:+d} bouts")
        print(f"  Frequency difference: {sf['comparison']['frequency_difference_per_minute']:+.2f} bouts/minute")
        print(f"  Duration difference: {sf['comparison']['duration_difference_sec']:+.2f} seconds ({sf['comparison']['percentage_difference']:+.2f}%)")
        print("-" * 80)
    
    # Save statistics
    stats_path = output_dir / 'statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to {stats_path}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("Step 5: Create Visualizations")
    print("=" * 80)
    
    # Timeline visualization
    timeline_path = output_dir / 'timeline.png'
    
    if ground_truth is not None:
        # Use plot_behavior_timeline if ground truth is available
        print(f"Using plot_behavior_timeline (ground truth provided)")
        # For binary classification, get probability of class 1 (scratching)
        if probabilities.shape[1] >= 2:
            scratching_probs = probabilities[:, 1]
        else:
            scratching_probs = probabilities[:, 0]
        
        plot_behavior_timeline(
            predictions=predictions,
            probabilities=scratching_probs,
            ground_truth=ground_truth,
            fps=fps,
            output_path=timeline_path
        )
    else:
        # Use visualize_timeline if no ground truth
        print(f"Using visualize_timeline (no ground truth)")
        visualize_timeline(predictions, idx_to_label, fps, timeline_path)
    
    # Create prediction video
    if args.save_video:
        video_output_path = output_dir / f'{video_path.stem}_predictions.mp4'
        create_prediction_video(frames, predictions, idx_to_label, video_output_path, fps,
                               keypoints_data=keypoints_data, frame_offset=frame_offset)
    
    print("\n" + "=" * 80)
    print("Inference Complete!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"  - predictions.npy: Frame-level predictions")
    print(f"  - probabilities.npy: Prediction probabilities")
    print(f"  - statistics.json: Detailed statistics")
    print(f"  - timeline.png: Timeline visualization")
    if args.save_features:
        if args.use_keypoints:
            print(f"  - resnet_features.npy: ResNet50 features")
            print(f"  - keypoint_features.npy: Keypoint features")
            print(f"  - features.npy: Combined multimodal features")
        else:
            print(f"  - features.npy: Extracted features")
    if args.save_video:
        print(f"  - {video_path.stem}_predictions.mp4: Video with predictions")
    if args.use_keypoints:
        print(f"\n✓ Multimodal inference used: ResNet + Keypoints")
    print("=" * 80)


if __name__ == '__main__':
    main()
