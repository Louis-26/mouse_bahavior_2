"""
Extract keypoint features from JSON files and save as numpy arrays.
This script processes keypoint detection results to create feature vectors 
that can be used alongside ResNet features for multimodal learning.

Keypoint format:
- 39 keypoints per frame
- Each keypoint has [x, y, confidence]
- Output: (T, 117) numpy array where T is number of frames
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_keypoint_json(json_path: str) -> List[Dict]:
    """Load keypoint detection results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_keypoint_features(keypoint_data: List[Dict]) -> np.ndarray:
    """
    Extract keypoint features from loaded JSON data.
    
    Args:
        keypoint_data: List of dicts, each containing 'bodyparts', 'bboxes', 'bbox_scores'
    
    Returns:
        numpy array of shape (num_frames, 117) containing flattened keypoint coordinates and confidences
    """
    num_frames = len(keypoint_data)
    features = np.zeros((num_frames, 117), dtype=np.float32)
    
    for i, frame_data in enumerate(keypoint_data):
        if 'bodyparts' in frame_data and len(frame_data['bodyparts']) > 0:
            # Take the first detection (highest confidence bbox)
            bodyparts = frame_data['bodyparts'][0]  # shape: (39, 3)
            # Flatten to (117,)
            features[i] = np.array(bodyparts, dtype=np.float32).flatten()
        else:
            # No detection in this frame, keep zeros
            features[i] = 0.0
    
    return features


def normalize_keypoint_features(features: np.ndarray, 
                                image_width: int = 640, 
                                image_height: int = 480) -> np.ndarray:
    """
    Normalize keypoint features.
    
    Args:
        features: (T, 117) array where every 3 values are [x, y, conf]
        image_width: width of the video frame
        image_height: height of the video frame
    
    Returns:
        Normalized features (T, 117)
    """
    normalized = features.copy()
    
    # Normalize x coordinates (every 3rd value starting from index 0)
    normalized[:, 0::3] /= image_width
    
    # Normalize y coordinates (every 3rd value starting from index 1)
    normalized[:, 1::3] /= image_height
    
    # Confidence scores (index 2::3) are already in [0, 1], no normalization needed
    
    return normalized


def match_keypoints_to_clip(keypoint_features: np.ndarray,
                           clip_start: int,
                           clip_end: int) -> np.ndarray:
    """
    Extract keypoint features for a specific video clip.
    
    Args:
        keypoint_features: Full video keypoint features (num_frames, 117)
        clip_start: Start frame index
        clip_end: End frame index (exclusive)
    
    Returns:
        Keypoint features for the clip (clip_length, 117)
    """
    return keypoint_features[clip_start:clip_end]


def process_video_keypoints(json_path: str,
                           output_dir: str,
                           video_name: str,
                           image_width: int = 640,
                           image_height: int = 480):
    """
    Process keypoint JSON file for a full video and save features.
    
    Args:
        json_path: Path to keypoint JSON file
        output_dir: Directory to save processed features
        video_name: Name of the video (e.g., 'CQ_2')
        image_width: Video frame width
        image_height: Video frame height
    """
    print(f"Processing keypoints for {video_name}...")
    
    # Load keypoint data
    keypoint_data = load_keypoint_json(json_path)
    print(f"  Loaded {len(keypoint_data)} frames")
    
    # Extract features
    features = extract_keypoint_features(keypoint_data)
    print(f"  Extracted features shape: {features.shape}")
    
    # Normalize features
    features_normalized = normalize_keypoint_features(features, image_width, image_height)
    
    # Save processed features
    output_path = Path(output_dir) / f"{video_name}_keypoints.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features_normalized)
    print(f"  Saved to {output_path}")
    
    return features_normalized


def extract_clip_keypoints(full_keypoint_path: str,
                          clip_start: int,
                          clip_end: int,
                          output_path: str,
                          image_width: int = 640,
                          image_height: int = 480):
    """
    Extract and save keypoint features for a specific clip.
    
    Args:
        full_keypoint_path: Path to full video keypoint numpy file
        clip_start: Start frame index
        clip_end: End frame index (exclusive)
        output_path: Path to save clip keypoint features
        image_width: Video frame width  
        image_height: Video frame height
    """
    # Load full video keypoints
    full_keypoints = np.load(full_keypoint_path)
    
    # Extract clip
    clip_keypoints = match_keypoints_to_clip(full_keypoints, clip_start, clip_end)
    
    # Save clip keypoints
    np.save(output_path, clip_keypoints)
    
    return clip_keypoints


def main():
    parser = argparse.ArgumentParser(description='Extract keypoint features from JSON files')
    parser.add_argument('--keypoint_dir', type=str, required=True,
                       help='Directory containing keypoint JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save processed keypoint features')
    parser.add_argument('--video_name', type=str, required=True,
                       help='Video name (e.g., CQ_2)')
    parser.add_argument('--image_width', type=int, default=640,
                       help='Video frame width')
    parser.add_argument('--image_height', type=int, default=480,
                       help='Video frame height')
    
    args = parser.parse_args()
    
    # Find the keypoint JSON file
    keypoint_dir = Path(args.keypoint_dir)
    json_files = list(keypoint_dir.glob(f"{args.video_name}_*.json"))
    
    if not json_files:
        print(f"Error: No keypoint JSON files found for {args.video_name}")
        return
    
    # Use the first matching file
    json_path = str(json_files[0])
    print(f"Found keypoint file: {json_path}")
    
    # Process
    process_video_keypoints(
        json_path=json_path,
        output_dir=args.output_dir,
        video_name=args.video_name,
        image_width=args.image_width,
        image_height=args.image_height
    )


if __name__ == '__main__':
    main()
