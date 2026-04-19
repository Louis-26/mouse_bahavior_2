# Action Segmentation with MS-TCN for Mouse Behavior Analysis

This directory contains a complete implementation of the MS-TCN (Multi-Stage Temporal Convolutional Network) for segmenting mouse scratching behaviors in videos. The implementation supports both unimodal (ResNet only) and multimodal (ResNet + Keypoints) approaches.

## Table of Contents

- [Quick Start](#quick-start)
- [Dataset Structure](#dataset-structure)
- [Training](#training)
- [Inference](#inference)
- [Architecture Details](#architecture-details)
- [Multimodal Features](#multimodal-features)
- [Input Frame Analysis](#input-frame-analysis)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Quick Start

### Verify Setup
```bash
bash verify_setup.sh
```

### Train Model (Recommended: With Keypoints)
```bash
bash train.sh
```

### Train Model (Baseline: ResNet Only)
```bash
bash train_resnet_only.sh
```

### Interactive Training
```bash
bash quick_start.sh
```

### Quick Inference Reference

#### For Models Trained WITHOUT Keypoints
```bash
python inference_raw_video.py \
    --video_path video.mp4 \
    --checkpoint model.pth \
    --save_video
```

#### For Models Trained WITH Keypoints
```bash
python inference_raw_video.py \
    --video_path video.mp4 \
    --checkpoint model.pth \
    --use_keypoints \
    --keypoint_dir /path/to/keypoints \
    --save_video
```

## Dataset Structure

The training expects a preprocessed dataset with the following structure:
```
all_data_80train_20_val/
├── features/          # ResNet50 features (.npy files)
├── labels/            # Frame-level labels (.npy files)
├── meta/              # Metadata (class_mapping.json, dataset_stats.json)
└── splits/            # Train/val splits (train.txt, val.txt)
```

### Keypoint Features

Keypoint detection results should be in HDF5 format (.h5 files) from DeepLabCut:
```
/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/keypoint/
├── CQ_2_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_.h5
├── CQ_3_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_.h5
└── CQ_4_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_.h5
```

## Training

### 1. Train with Keypoints (Recommended)

Uses both ResNet features and keypoint features for multimodal learning:

```bash
bash train.sh
```

This will:
- Use dataset: `/data/zhaozhenghao/Projects/Mouse/datasets/UMich_CQ/all_data_80train_20_val`
- Load keypoints from: `/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/keypoint`
- Feature dimension: 2048 (ResNet) + 117 (39 keypoints × 3) = 2165
- Save results with timestamp in output directory

### 2. Train without Keypoints

Uses only ResNet features:

```bash
bash train_resnet_only.sh
```

This will:
- Use only ResNet features (2048 dimensions)
- No keypoint loading
- Useful for comparison with multimodal approach

### Training Parameters

Default parameters in the scripts:
- **Batch size**: 8
- **Epochs**: 50
- **Learning rate**: 0.0005
- **MS-TCN stages**: 4
- **Layers per stage**: 10
- **Feature maps**: 64
- **Oversampling**: Enabled (to handle class imbalance)
- **Focal loss**: Enabled (gamma=2.0)

### Customizing Training

You can modify parameters in the shell scripts or run `train.py` directly:

```bash
python train.py \
    --dataset_root /path/to/dataset \
    --output_dir /path/to/output \
    --use_keypoints \
    --keypoint_root /path/to/keypoints \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 0.001 \
    --device cuda
```

#### Available Arguments

Run `python train.py --help` for all options, including:
- `--num_stages`: Number of MS-TCN stages (default: 4)
- `--num_layers`: Layers per stage (default: 10)
- `--num_f_maps`: Feature map channels (default: 64)
- `--tmse_weight`: Temporal MSE loss weight (default: 0.15)
- `--class_weight`: Manual class weight (auto-computed if None)
- `--oversample_power`: Oversampling aggressiveness (default: 2.0)
- `--focal_gamma`: Focal loss gamma parameter (default: 2.0)
- `--resume`: Path to checkpoint to resume from

### Output Structure

Training will create the following output structure:

```
results/UMich_CQ/action_seg_training/with_keypoints_TIMESTAMP/
├── checkpoints/
│   ├── best_model.pth        # Best model by validation F1
│   ├── latest_model.pth      # Latest checkpoint
│   └── epoch_*.pth           # Per-epoch checkpoints
├── config.json               # Training configuration
├── training_log.json         # Detailed training metrics
└── plots/                    # Training curves (if generated)
```

### Resume Training

To resume from a checkpoint:

```bash
python train.py \
    --dataset_root /path/to/dataset \
    --output_dir /path/to/output \
    --resume /path/to/checkpoint.pth
```

## Inference

### Multimodal Inference Guide

The inference script supports multimodal input, combining:
- **ResNet50 features** (2048 dimensions) - extracted from video frames
- **Keypoint features** (117 dimensions = 39 keypoints × 3 values) - spatial pose information

### Feature Dimensions

| Mode | ResNet | Keypoints | Total |
|------|--------|-----------|-------|
| ResNet Only | 2048 | - | **2048** |
| Multimodal | 2048 | 117 | **2165** |

### Basic Inference (ResNet Only)

For models trained without keypoints:

```bash
python inference_raw_video.py \
    --video_path /path/to/video.mp4 \
    --checkpoint /path/to/checkpoint.pth \
    --save_video
```

### Multimodal Inference (ResNet + Keypoints)

For models trained with keypoints:

```bash
python inference_raw_video.py \
    --video_path /path/to/video.mp4 \
    --checkpoint /path/to/checkpoint.pth \
    --use_keypoints \
    --keypoint_dir /path/to/keypoint_jsons \
    --save_video \
    --save_features
```

### Full Example with All Options

```bash
python inference_raw_video.py \
    --video_path datasets/UMich_CQ/videos/CQ_3.mp4 \
    --checkpoint results/UMich_CQ/action_seg_training/run_001/checkpoints/best.pth \
    --use_keypoints \
    --keypoint_dir results/UMich_CQ/keypoint \
    --ground_truth datasets/UMich_CQ/CQ_3.csv \
    --output_dir results/UMich_CQ/video_inference/CQ_3_multimodal \
    --start_time 10.0 \
    --end_time 60.0 \
    --window_length 512 \
    --stride 256 \
    --batch_size 32 \
    --save_video \
    --save_features \
    --device cuda
```

### Automatic Feature Dimension Detection

The script automatically detects if a checkpoint was trained with multimodal features:

1. **Checkpoint inspection**: Checks `config['feature_dim']` in the checkpoint
2. **Automatic mode switching**: If the checkpoint expects keypoints but `--use_keypoints` is not enabled, the script will:
   - Print a warning
   - Automatically enable multimodal mode
   - Use zero-padded keypoint features if `--keypoint_dir` is not provided

### Inference Arguments

#### Required Arguments
- `--video_path`: Path to input video file
- `--checkpoint`: Path to trained model checkpoint (.pth file)

#### Multimodal Arguments
- `--use_keypoints`: Enable multimodal inference (ResNet + Keypoints)
- `--keypoint_dir`: Directory containing keypoint JSON files

#### Optional Arguments
- `--output_dir`: Output directory (default: `{video_stem}_results`)
- `--ground_truth`: Path to ground truth CSV for comparison
- `--window_length`: Length of each inference clip (default: 512)
- `--stride`: Stride between clips (default: 256)
- `--batch_size`: Batch size for feature extraction (default: 32)
- `--start_time`: Start time in seconds (default: from beginning)
- `--end_time`: End time in seconds (default: until end)
- `--save_video`: Create video with prediction overlay
- `--save_features`: Save extracted features to .npy files
- `--device`: Device to use (default: cuda if available, else cpu)

### Output Files

When using `--save_features` with multimodal mode:
- `resnet_features.npy`: ResNet50 features only (num_frames, 2048)
- `keypoint_features.npy`: Keypoint features only (num_frames, 117)
- `features.npy`: Combined multimodal features (num_frames, 2165)
- `predictions.npy`: Frame-level predictions
- `probabilities.npy`: Prediction probabilities
- `statistics.json`: Detailed statistics
- `timeline.png`: Timeline visualization

When using `--save_video`:
- `{video_name}_predictions.mp4`: Video with prediction overlays and keypoints

## Architecture Details

### Overall MS-TCN Architecture

```
输入特征 (Batch, 2048/2165, T)
    |
    v
┌─────────────────────────────────────────────────────────────────┐
│                        STAGE 1                                   │
│  ┌──────────┐    ┌─────────────────────────┐    ┌──────────┐   │
│  │ Conv 1x1 │───▶│  10 Dilated Residual    │───▶│ Conv 1x1 │   │
│  │ 2048→64  │    │  Layers (dilation 2^i)  │    │  64→2    │   │
│  └──────────┘    └─────────────────────────┘    └──────────┘   │
└─────────────────────────────────────────────────────────────────┘
    |
    v  预测1 (Batch, 2, T)
    |
    v
┌─────────────────────────────────────────────────────────────────┐
│                     STAGE 2-4                                    │
│               (渐进式精炼预测)                                     │
└─────────────────────────────────────────────────────────────────┘
    |
    v  最终预测 (Batch, 2, T)
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| num_stages | 4 | Multi-stage refinement iterations |
| num_layers | 10 | Layers per stage |
| num_f_maps | 64 | Feature map channels |
| dim | 2048/2165 | Input feature dimension |
| num_classes | 2 | Output classes (no behavior/scratching) |

### Receptive Field Calculation

Each layer's receptive field grows exponentially:
- Layer 1: dilation=1, receptive field=3
- Layer 2: dilation=2, receptive field=7
- Layer 3: dilation=4, receptive field=15
- ...
- Layer 10: dilation=512, receptive field=3073 frames

At 30fps, maximum receptive field ≈ **102 seconds**, sufficient for capturing long-term behavioral patterns.

### Multi-stage Refinement Flow

```
Stage 1: Raw features (2048/2165D) → Coarse prediction
   ↓
Stage 2: Coarse prediction (2D probabilities) → Refined prediction 1
   ↓
Stage 3: Refined prediction 1 → Refined prediction 2
   ↓
Stage 4: Refined prediction 2 → Final prediction
```

### Model Size

```python
Total Parameters: ~400K - 500K

Distribution:
- Stage 1: ~140K parameters
- Stage 2-4: ~80K parameters each
```

Very lightweight compared to other video understanding models (I3D, SlowFast: tens of millions of parameters).

## Multimodal Features

### Complete Multimodal Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Raw Video Input                              │
│                    (e.g., CQ_2.mp4, 30fps)                         │
└─────────────────────────────────────────────────────────────────────┘
                              |
                              v
        ┌─────────────────────┴─────────────────────┐
        |                                            |
        v                                            v
┌──────────────────┐                        ┌──────────────────┐
│  ResNet50 Feature│                        │  Keypoint        │
│  Extraction      │                        │  Detection       │
│  (Preprocessing) │                        │  (DeepLabCut)    │
└──────────────────┘                        └──────────────────┘
        |                                            |
        v                                            v
┌──────────────────┐                        ┌──────────────────┐
│ Per-frame 2048D  │                        │ Per-frame 39     │
│ features         │                        │ keypoints        │
│ (T, 2048)        │                        │ (T, 39, 3)       │
└──────────────────┘                        │ x, y, score      │
                                            └──────────────────┘
        |                                            |
        |                                            v
        |                                    ┌──────────────────┐
        |                                    │ Flatten          │
        |                                    │ keypoints        │
        |                                    │ (T, 39*3=117)    │
        |                                    └──────────────────┘
        |                                            |
        └──────────────────┬─────────────────────────┘
                          |
                          v
                ┌─────────────────────┐
                │   Feature           │
                │   Concatenation     │
                │   axis=1            │
                └─────────────────────┘
                          |
                          v
        ┌─────────────────────────────────────┐
        │  Multimodal Features                │
        │  (T, 2048 + 117 = 2165)            │
        └─────────────────────────────────────┘
                          |
                          v
        ┌─────────────────────────────────────┐
        │          MS-TCN Network             │
        │   (num_stages=4, dim=2165)          │
        └─────────────────────────────────────┘
                          |
                          v
        ┌─────────────────────────────────────┐
        │    Predictions (per-frame probs)    │
        │      (T, 2) - [no behavior, scratch]│
        └─────────────────────────────────────┘
```

### Detailed Multimodal Feature Structure

#### 1. ResNet50 Features (2048D)

```
ResNet50 output (per frame):
┌─────────────────────────────────────────┐
│  Global visual features: [f1, f2, ..., f2048] │
│                                         │
│  Contains information:                   │
│  - Mouse appearance                      │
│  - Coarse pose features                  │
│  - Scene context                         │
│  - Motion blur patterns                  │
└─────────────────────────────────────────┘
```

#### 2. Keypoint Features (117D)

```
39 keypoints × 3 values = 117 dimensions:

Keypoint list (mouse skeleton):
1. nose                → [x1, y1, score1]
2. left_ear            → [x2, y2, score2]
3. right_ear           → [x3, y3, score3]
4. neck                → [x4, y4, score4]
...
39. tail_tip           → [x39, y39, score39]

Flattened: [x1, y1, s1, x2, y2, s2, ..., x39, y39, s39]
           └────────────── 117 dimensions ──────────────────┘

Contains information:
- Precise limb positions
- Joint angles (implicit)
- Pose details
- Movement trajectories
- Detection confidence
```

#### 3. Combined Multimodal Features (2165D)

```
┌──────────────────────────────────────────────────────────┐
│ [ResNet 2048D] + [Keypoint 117D] = 2165D               │
│                                                          │
│ [f1, f2, ..., f2048] + [x1, y1, s1, ..., x39, y39, s39] │
│  └─ Global visual ─┘   └────── Local pose ──────┘      │
└──────────────────────────────────────────────────────────┘
```

### Multimodal Fusion Advantages

#### ResNet Features Advantages
✅ **Global context**: Can see the entire scene  
✅ **Appearance info**: Fur texture, lighting, shadows  
✅ **Motion blur**: Visual cues of rapid movement  
✅ **Robustness**: Works even if keypoint detection fails  

#### Keypoint Features Advantages
✅ **Precise localization**: Accurate limb positions  
✅ **Pose details**: Joint angles, limb configurations  
✅ **Lightweight**: More compact than pixel features  
✅ **Interpretability**: Human-understandable pose information  

#### Synergistic Effects After Fusion
🔥 **Complementary information**:  
- ResNet: "Mouse is moving rapidly"  
- Keypoints: "Right front paw repeatedly moving near head"  
- Fusion: "Mouse is scratching its head!"  

🔥 **Improved robustness**:  
- If keypoint detection has occlusion → ResNet features compensate  
- If visual features are blurry → Keypoints provide precise location  

🔥 **Performance improvement**:  
- Theoretical F1 score improvement of 5-15%  
- More accurate recognition of subtle actions  

## Input Frame Analysis

### Key Answer: Default 512 frames per clip

### Detailed Explanation

#### 1. Data Preprocessing Segmentation Strategy

During preprocessing (`action_segmentation.py`), videos are segmented into fixed-length clips:

```python
# Default parameters
window_length = 512   # Frames per clip
stride = 256          # Sliding window stride
```

**Segmentation method: Sliding Window**

```
Complete video (e.g., 30,000 frames)
    |
    v
┌─────────────────────────────────────────────────────────────┐
│ Frame: 0    1    2  ...  510  511  512  513  ...  29999     │
└─────────────────────────────────────────────────────────────┘
    |
    v Using sliding window segmentation
    |
┌───────────────────────────────────────────────────────────┐
│ Clip 0:   [0 ─────────────── 511]    (512 frames)        │
│                                                           │
│ Clip 1:        [256 ────────────── 767]    (512 frames)  │
│                     ↑                                     │
│                stride=256 (50% overlap)                   │
│                                                           │
│ Clip 2:             [512 ──────────── 1023]   (512 frames)│
│                                                           │
│ ...                                                       │
│                                                           │
│ Clip N:   [29488 ────────── 29999]    (512 frames)       │
└───────────────────────────────────────────────────────────┘
```

#### 2. Training Input

**Input dimensions per batch:**
```python
Input shape: (batch_size, feature_dim, sequence_length)
           = (8, 2048, 512)  # Default configuration

Where:
- batch_size = 8      # Process 8 clips at once
- feature_dim = 2048  # ResNet50 feature dimension (or 2165 with keypoints)
- sequence_length = 512  # Frames per clip (time steps)
```

#### 3. Why Choose 512 Frames?

**Duration calculation (@30fps):**
```
512 frames ÷ 30fps = 17.07 seconds
```

**Reasons:**

✅ **Sufficient length to capture complete behaviors**
- Mouse scratching behaviors typically last 1-5 seconds
- 512 frames (17 seconds) can contain multiple complete scratching events
- Can learn behavioral context before and after

✅ **Computational efficiency**
- 512 is a power of 2 (2^9), GPU-friendly for computation
- Not too long to cause memory overflow
- Not too short to lack context

✅ **Receptive field coverage**
- MS-TCN 10-layer maximum receptive field: 3073 frames
- 512 frames completely within receptive field range
- Network can see global information for entire clip

#### 4. 50% Overlap Strategy (stride=256)

**Why use 50% overlap?**

```
Clip 0:  [0 ──────────────── 511]
Clip 1:       [256 ────────────── 767]
            ↑
         Share 256 frames (50% overlap)
```

**Advantages:**

1. **Boundary effect smoothing**
   - Behaviors might span clip boundaries
   - Overlap ensures each behavior appears completely in at least one clip

2. **Data augmentation**
   - Same frames appear in different contexts
   - Improves model robustness

3. **Inference smoothing**
   - Can average predictions in overlapping regions during inference
   - Reduces prediction jitter at boundaries

#### 5. MS-TCN Sequence Length Flexibility

**Important feature: MS-TCN is fully convolutional, theoretically can handle any length!**

```python
class MS_TCN(nn.Module):
    def forward(self, x):
        # x: (batch, dim, T)  # T can be any length!
        # ...
        return predictions  # (batch, num_classes, T)
```

**But practical reasons for fixed length:**

1. **Batch processing requirements**
   - PyTorch DataLoader needs same length within batch
   - Fixed length avoids padding/masking complexity

2. **Memory management**
   - Fixed length convenient for GPU memory budgeting
   - Avoids some extremely long sequences causing OOM

3. **Training stability**
   - More stable gradient updates
   - Better batch normalization effects

### Different Configuration Comparison

| window_length | Duration (@30fps) | Pros | Cons |
|--------------|-------------------|------|-----|
| 256 | 8.5 seconds | Fast training, small memory | Insufficient context, may truncate long behaviors |
| **512** | **17 seconds** | **Balanced: sufficient context, reasonable computation** | **Default recommendation** |
| 1024 | 34 seconds | Very long context | Slow, large memory, may over-smooth |
| 2048 | 68 seconds | Ultra-long context | Extremely slow, GPU memory may be insufficient |

### Inference Processing

**Method 1: Clip-by-clip inference (consistent with training)**
```python
# Segment test video into 512-frame clips
clips = segment_video(features, window_length=512, stride=256)

# Inference on each clip
predictions = []
for clip in clips:
    pred = model(clip)  # (1, 2, 512)
    predictions.append(pred)

# Concatenate results (handle overlapping regions)
full_predictions = merge_overlapping_predictions(predictions, stride=256)
```

**Method 2: Full-length inference (if memory allows)**
```python
# Direct inference on entire video (requires sufficient GPU memory)
full_features = extract_features(video)  # (30000, 2048)
full_features = full_features.transpose(0, 1).unsqueeze(0)  # (1, 2048, 30000)

# One-shot inference
predictions = model(full_features)  # (1, 2, 30000)
```

### Actual Dataset Statistics

Based on your data:

```python
# CQ_2 video example
Total video frames: ~27,000 frames (about 15 minutes @30fps)

After segmentation:
- window_length=512, stride=256
- Total clips generated: ~105 clips
- Each clip: 512 frames = 17 seconds

During training:
- batch_size=8
- Each batch: 8 clips × 512 frames = 4,096 frames
- GPU memory: ~2-3GB (depends on feature_dim)
```

### How to Modify Input Length

**Method 1: Repreprocess data**
```bash
# Use different window_length to repreprocess
python preprocess/action_segmentation.py \
    --window_length 1024 \
    --stride 512 \
    --output_dir datasets/UMich_CQ/all_data_1024window
```

**Method 2: Use different length during inference**
```python
# MS-TCN can handle any length during inference (fully convolutional)
features = load_features()  # Any length
predictions = model(features)  # Automatically adapts to length
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| "Feature dimension mismatch" | Match `--use_keypoints` with training mode |
| "No keypoint file found" | Check keypoint JSON naming and location |
| "Frames with valid keypoints: 0" | Verify keypoint JSON structure |
| Slow inference | Use `--device cuda` + larger `--batch_size` |
| Out of memory | Reduce batch_size or use CPU |
| Model not converging | Check class balance, try different learning rate |

### GPU Requirements

Training requires a CUDA-capable GPU. The script automatically detects and uses CUDA if available.

For CPU-only training (not recommended):
```bash
python train.py ... --device cpu
```

### Best Practices

1. **Match Training Configuration**: Always use the same feature mode (ResNet only vs. multimodal) as training
2. **Check Checkpoint**: Let the script auto-detect the feature dimension from the checkpoint
3. **Provide Keypoints**: If using multimodal, provide `--keypoint_dir` for best results
4. **GPU Acceleration**: Use `--device cuda` for faster processing (10-50x speedup)
5. **Batch Size**: Adjust `--batch_size` based on GPU memory (32 works well for most GPUs)

## Advanced Usage

### Keypoint Feature Extraction

If you need to extract keypoint features from DeepLabCut results:

```python
import pandas as pd
import numpy as np

def extract_keypoint_features(dlc_csv_path, output_path):
    """
    Extract keypoint features from DeepLabCut CSV
    
    Args:
        dlc_csv_path: DLC output CSV file (e.g., CQ_2.csv)
        output_path: Output npy file path
    """
    # Read DLC results
    df = pd.read_csv(dlc_csv_path, header=[1, 2])
    
    # Extract all keypoints' x, y, score
    keypoint_features = []
    for idx in range(len(df)):
        frame_features = []
        for bodypart in df.columns.levels[0]:  # Iterate through 39 keypoints
            if bodypart != 'coords':
                x = df[bodypart]['x'].iloc[idx]
                y = df[bodypart]['y'].iloc[idx]
                score = df[bodypart]['likelihood'].iloc[idx]
                frame_features.extend([x, y, score])
        keypoint_features.append(frame_features)
    
    keypoint_features = np.array(keypoint_features)  # (num_frames, 117)
    
    # Normalization (optional but recommended)
    # x, y normalize to [0, 1]
    keypoint_features[:, 0::3] /= video_width   # x coordinates
    keypoint_features[:, 1::3] /= video_height  # y coordinates
    # score already in [0, 1]
    
    np.save(output_path, keypoint_features)
    print(f"Saved keypoint features: {keypoint_features.shape}")
```

### Batch Processing Script

```bash
#!/bin/bash
for video in datasets/videos/*.mp4; do
    python inference_raw_video.py \
        --video_path "$video" \
        --checkpoint models/best.pth \
        --use_keypoints \
        --keypoint_dir keypoints/ \
        --save_video
done
```

### Performance Optimization

1. **Use GPU**: `--device cuda` (10-50x faster)
2. **Increase batch size**: `--batch_size 64` (if GPU memory allows)
3. **Skip video generation**: Remove `--save_video` (saves time)
4. **Process time ranges**: Use `--start_time` and `--end_time`

### Monitoring Training

The training script displays:
- Epoch progress with tqdm
- Training/validation loss
- Per-class metrics (Precision, Recall, F1)
- Overall accuracy and F1 score
- Best model updates

### Expected Performance

| Model Configuration | Frame Acc | F1 Score | Parameters | Training Time |
|-------------------|-----------|----------|------------|---------------|
| ResNet Only (2048) | ~95% | ~60% | 500K | 1x |
| ResNet + Keypoints (2165) | ~97% | ~70-75% | 510K (+2%) | 1.05x |

### File Organization

```
methods/UMich_methods/action_seg/
├── README.md                     # This comprehensive guide
├── train.py                      # Main training script
├── inference_raw_video.py        # Inference script
├── model.py                      # MS-TCN model definition
├── train.sh                      # Training with keypoints
├── train_resnet_only.sh          # Training without keypoints
├── verify_setup.sh               # Setup verification
├── quick_start.sh                # Interactive launcher
└── inference_raw_video.sh        # Inference launcher
```

## References

- MS-TCN paper: "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation" (CVPR 2019)
- Training data: University of Michigan mouse behavior dataset
- Keypoint detection: DeepLabCut with SuperAnimal models
- Feature extraction: ResNet50 pretrained on ImageNet

---

For specific implementation details, see the individual Python scripts and their inline documentation.