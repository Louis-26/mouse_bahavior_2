# Preprocessing Tools

A complete toolkit for behavior analysis data preparation and processing.

---

## 📋 Table of Contents

- [Preprocessing Tools](#preprocessing-tools)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Tools Overview](#-tools-overview)
  - [📖 Detailed Usage Instructions](#-detailed-usage-instructions)
    - [1. `label_process.py` - Label Processing](#1-label_processpy---label-processing)
      - [Features](#features)
      - [Usage](#usage)
      - [Parameters](#parameters)
      - [Output Format](#output-format)
    - [2. `split_cell.py` - Video Grid Splitting](#2-split_cellpy---video-grid-splitting)
      - [Features](#features-1)
      - [Usage](#usage-1)
      - [Parameters](#parameters-1)
      - [Workflow](#workflow)
      - [Output Files (2x2 mode)](#output-files-2x2-mode)
      - [Output Files (3x2 mode)](#output-files-3x2-mode)
      - [Interactive Mode Options](#interactive-mode-options)
    - [3. `action_segmentation.py` - Complete Preprocessing Pipeline](#3-action_segmentationpy---complete-preprocessing-pipeline)
      - [Processing Steps](#processing-steps)
      - [Usage](#usage-2)
      - [Main Parameters](#main-parameters)
      - [Output Structure](#output-structure)
    - [4. `advanced_split.py` - Dataset Splitting](#4-advanced_splitpy---dataset-splitting)
      - [Supported Split Modes](#supported-split-modes)
      - [Usage](#usage-3)
      - [Main Parameters](#main-parameters-1)
      - [Features](#features-2)
    - [5. `extract_keypoint_features.py` - Keypoint Feature Extraction](#5-extract_keypoint_featurespy---keypoint-feature-extraction)
      - [Features](#features-3)
      - [Usage](#usage-4)
      - [Main Parameters](#main-parameters-2)
      - [Output](#output)
    - [6. `dataloader_example.py` - Data Loader](#6-dataloader_examplepy---data-loader)
      - [Features](#features-4)
      - [Usage](#usage-5)
  - [🔄 Complete Workflow](#-complete-workflow)
    - [Standard Processing Pipeline](#standard-processing-pipeline)
    - [Directory Structure After Processing](#directory-structure-after-processing)
  - [❓ Frequently Asked Questions](#-frequently-asked-questions)
    - [Q: What video formats are supported?](#q-what-video-formats-are-supported)
    - [Q: How to handle missing keypoint data?](#q-how-to-handle-missing-keypoint-data)
    - [Q: Can I change the sliding window parameters?](#q-can-i-change-the-sliding-window-parameters)
    - [Q: How to add new behavior classes?](#q-how-to-add-new-behavior-classes)
    - [Q: What if GPU memory is insufficient?](#q-what-if-gpu-memory-is-insufficient)

---

## 🎯 Tools Overview

| Tool | Function | Input | Output |
|------|----------|-------|--------|
| `label_process.py` | Excel labels to CSV | `.xlsx` | `.csv` |
| `split_cell.py` | Video grid splitting | Video files | Split videos |
| `action_segmentation.py` | Complete preprocessing pipeline | Video + CSV | Features + splits |
| `advanced_split.py` | Re-split datasets | Preprocessed data | New splits |
| `extract_keypoint_features.py` | Extract keypoint features | JSON | `.npy` |
| `dataloader_example.py` | PyTorch data loading | Preprocessed data | DataLoader |

---

## 📖 Detailed Usage Instructions

### 1. `label_process.py` - Label Processing

Convert Excel behavior annotation files to standard CSV format (HH:MM:SS.mmm timestamps).

#### Features
- 🔄 Automatic time format detection (CQ2, CQ3, CQ4)
- 📝 Single file and batch processing support
- ⏱️ Millisecond precision timestamps
- 🎯 Smart column name detection

#### Usage

```bash
# View help
python label_process.py --help

# Single file processing (auto-detect format)
python label_process.py --input data.xlsx --output data.csv

# Specify format
python label_process.py --input data.xlsx --output data.csv --format cq2

# Batch process entire directory
python label_process.py --input_dir input/ --output_dir output/

# Batch process with specific format
python label_process.py --input_dir input/ --output_dir output/ --format cq3
```

#### Parameters
- `--input`: Input Excel file path
- `--output`: Output CSV file path
- `--input_dir`: Input directory (batch mode)
- `--output_dir`: Output directory (batch mode)
- `--format`: Format type (`auto`, `cq2`, `cq3`, `cq4`)
- `--pattern`: File matching pattern (default: `*.xlsx`)

#### Output Format
```csv
Start,End,Notes
00:01:23.456,00:01:25.789,scratching
00:02:15.123,00:02:18.456,no behavior
```

### 2. `split_cell.py` - Video Grid Splitting

Split multi-cell recorded videos (2x2 or 3x2 grid) into individual video files. Uses ffmpeg for fast processing with automatic grid line detection and interactive adjustment.

#### Features
- 🔄 Automatic grid line position detection (projection method)
- 🎯 Support for 2x2 (4 videos) and 3x2 (6 videos) grids
- 🖼️ Interactive splitting position adjustment and preview
- ⚡ Fast processing using ffmpeg (recommended)
- 🎬 Optional video stabilization (vidstab)
- 🚀 Automatic mode skipping confirmation

#### Usage

```bash
# View help
python split_cell.py --help

# Basic usage - 2x2 grid (4 videos)
python split_cell.py --video input.mp4

# 3x2 grid (6 videos)
python split_cell.py --video input.mp4 --cells 6

# Manual splitting positions (2x2)
python split_cell.py --video input.mp4 --manual_x 640 --manual_y 480

# Manual splitting positions (3x2)
python split_cell.py --video input.mp4 --cells 6 --manual_x "426,852" --manual_y 480

# Enable stabilization
python split_cell.py --video input.mp4 --stabilize

# Automatic mode (skip confirmation)
python split_cell.py --video input.mp4 --yes

# Complete example
python split_cell.py \
    --video CQ_2.mp4 \
    --out_dir cells_out \
    --cells 4 \
    --stabilize \
    --yes
```

#### Parameters
- `--video`: Input video file path (required)
- `--out_dir`: Output directory (default: `cells_out`)
- `--cells`: Number of cells, 4 (2x2) or 6 (3x2) (default: 4)
- `--manual_x`: Manual vertical split line x coordinates
  - 4 cells: single number (e.g., `640`)
  - 6 cells: comma-separated two values (e.g., `"426,852"`)
- `--manual_y`: Manual horizontal split line y coordinate
- `--stabilize`: Enable video stabilization (ffmpeg vidstab filter)
- `--yes` / `-y`: Skip confirmation, process directly

#### Workflow
1. **Auto-detection**: Use projection method to detect optimal split positions
2. **Preview generation**: Create annotated preview image `split_preview.png`
3. **Interactive adjustment**: User confirms or manually adjusts split positions
4. **Fast splitting**: Use ffmpeg to process all regions in parallel

#### Output Files (2x2 mode)
```
cells_out/
├── split_preview.png       # Split preview image
├── top_left.mp4           # Top-left video
├── top_right.mp4          # Top-right video
├── bottom_left.mp4        # Bottom-left video
└── bottom_right.mp4       # Bottom-right video
```

#### Output Files (3x2 mode)
```
cells_out/
├── split_preview.png       # Split preview image
├── top_left.mp4           # Top-left video
├── top_center.mp4         # Top-center video
├── top_right.mp4          # Top-right video
├── bottom_left.mp4        # Bottom-left video
├── bottom_center.mp4      # Bottom-center video
└── bottom_right.mp4       # Bottom-right video
```

#### Interactive Mode Options
- `[Enter]` - Confirm current position, start processing
- `[a]` - Manually adjust split position
- `[r]` - Re-run auto-detection
- `[q]` - Cancel and exit

### 3. `action_segmentation.py` - Complete Preprocessing Pipeline

Execute complete action segmentation data preprocessing, from raw videos to training-ready datasets.

#### Processing Steps
1. Convert time intervals to frame-level labels
2. Extract ResNet50 features (GPU batch processing)
3. Segment videos using sliding window
4. Create train/val/test splits
5. Calculate dataset statistics

#### Usage

```bash
# View help
python action_segmentation.py --help

# Basic usage
python action_segmentation.py \
    --video_dir /path/to/videos \
    --videos CQ_2.mp4 CQ_3.mp4 CQ_4.mp4 \
    --csvs CQ_2.csv CQ_3.csv CQ_4.csv \
    --output_dir /path/to/output

# Use CQ_4 as test set
python action_segmentation.py \
    --video_dir /path/to/videos \
    --videos CQ_2.mp4 CQ_3.mp4 CQ_4.mp4 \
    --csvs CQ_2.csv CQ_3.csv CQ_4.csv \
    --output_dir /path/to/output \
    --split_mode video \
    --test_videos CQ_4

# Temporal split mode (80/20 split per video)
python action_segmentation.py \
    --video_dir /path/to/videos \
    --videos CQ_2.mp4 CQ_3.mp4 CQ_4.mp4 \
    --csvs CQ_2.csv CQ_3.csv CQ_4.csv \
    --output_dir /path/to/output \
    --split_mode temporal \
    --train_ratio 0.8
```

#### Main Parameters
- `--video_dir`: Video file directory
- `--videos`: Video file name list
- `--csvs`: CSV label file name list (corresponding to videos)
- `--output_dir`: Output directory
- `--window_length`: Window length (default: 512 frames)
- `--stride`: Sliding stride (default: 256 frames)
- `--batch_size`: GPU batch processing size (default: 32)
- `--device`: Device (`cuda` or `cpu`)
- `--split_mode`: Split mode (`temporal` or `video`)
- `--test_videos`: Test set videos (video mode)
- `--train_ratio`: Training set ratio (default: 0.8)

#### Output Structure
```
output_dir/
├── features/          # ResNet50 features (.npy)
├── labels/            # Frame-level labels (.npy)
├── splits/            # Data splits
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── meta/              # Metadata
    ├── class_mapping.json
    └── dataset_stats.json
```

### 4. `advanced_split.py` - Dataset Splitting

Re-split preprocessed datasets without re-extracting features. Supports multiple splitting strategies.

#### Supported Split Modes
- ✅ Video splitting (train/val, train/test, train/val/test)
- ✅ Temporal splitting (within each video)
- ✅ Mixed mode (video split + temporal split)
- ✅ Automatic mode detection

#### Usage

```bash
# View help
python advanced_split.py --help

# Video split: CQ_2+CQ_3 train, CQ_4 validation
python advanced_split.py --train_videos CQ_2 CQ_3 --val_videos CQ_4

# Video split: CQ_2+CQ_3 train, CQ_4 test (no validation set)
python advanced_split.py --train_videos CQ_2 CQ_3 --test_videos CQ_4

# Complete three-set split
python advanced_split.py \
    --train_videos CQ_2 \
    --val_videos CQ_3 \
    --test_videos CQ_4

# Temporal split: 80/20 split per video
python advanced_split.py --split_mode temporal --train_ratio 0.8

# Temporal split: use default ratio
python advanced_split.py --split_mode temporal

# Mixed mode: CQ_4 test, other videos 80/20 train/val split
python advanced_split.py --test_videos CQ_4 --split_mode temporal

# Fast mode (no statistics display)
python advanced_split.py \
    --train_videos CQ_2 CQ_3 \
    --val_videos CQ_4 \
    --no_stats

# Specify dataset directory
python advanced_split.py \
    --dataset_root /path/to/dataset \
    --train_videos CQ_2 CQ_3 \
    --val_videos CQ_4
```

#### Main Parameters
- `--dataset_root`: Preprocessed dataset root directory
- `--train_videos`: Training set video list
- `--val_videos`: Validation set video list
- `--test_videos`: Test set video list
- `--split_mode`: Split mode (`video` or `temporal`, auto-detectable)
- `--train_ratio`: Training set ratio (default: 0.8)
- `--show_stats`: Show statistics (default)
- `--no_stats`: Don't show statistics (fast mode)

#### Features
- 🔄 No feature re-extraction, fast re-splitting
- 🎯 Automatic split mode detection
- 📊 Display label distribution statistics for each split
- ⚡ Fast mode support

### 5. `extract_keypoint_features.py` - Keypoint Feature Extraction

Extract and process keypoint features from keypoint detection JSON files.

#### Features
- 39 keypoints × 3 values (x, y, confidence) = 117-dimensional features
- Normalize coordinates to [0, 1]
- Output numpy array format

#### Usage

```bash
# View help
python extract_keypoint_features.py --help

# Basic usage
python extract_keypoint_features.py \
    --keypoint_dir /path/to/keypoint_jsons \
    --output_dir /path/to/output \
    --video_name CQ_2 \
    --image_width 640 \
    --image_height 480

# Process multiple videos
for video in CQ_2 CQ_3 CQ_4; do
    python extract_keypoint_features.py \
        --keypoint_dir /path/to/keypoints \
        --output_dir /path/to/output \
        --video_name $video
done
```

#### Main Parameters
- `--keypoint_dir`: Keypoint JSON file directory
- `--output_dir`: Output feature directory
- `--video_name`: Video name (e.g., `CQ_2`)
- `--image_width`: Video width (default: 640)
- `--image_height`: Video height (default: 480)

#### Output
Generates `{video_name}_keypoints.npy` file with shape `(num_frames, 117)`

### 6. `dataloader_example.py` - Data Loader

PyTorch data loader for model training. Supports multimodal features (ResNet + Keypoints).

#### Features
- Automatic feature loading and preprocessing
- Multimodal support (ResNet + keypoints)
- Flexible dataset configuration
- Memory-efficient loading

#### Usage

```python
from dataloader_example import ActionSegmentationDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = ActionSegmentationDataset(
    dataset_root='/path/to/preprocessed/data',
    split='train',
    use_keypoints=True,
    keypoint_root='/path/to/keypoint/features'
)

# Create data loader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

# Iterate through data
for batch in dataloader:
    features = batch['features']  # (batch, channels, time)
    labels = batch['labels']      # (batch, time)
    clip_names = batch['clip_name']
```

---

## 🔄 Complete Workflow

### Standard Processing Pipeline

1. **Label Processing** (if needed)
   ```bash
   python label_process.py --input raw_labels.xlsx --output labels.csv
   ```

2. **Video Splitting** (if multi-cell)
   ```bash
   python split_cell.py --video multi_cell.mp4 --cells 4
   ```

3. **Complete Preprocessing**
   ```bash
   python action_segmentation.py \
       --video_dir videos/ \
       --videos CQ_2.mp4 CQ_3.mp4 CQ_4.mp4 \
       --csvs CQ_2.csv CQ_3.csv CQ_4.csv \
       --output_dir preprocessed_data/
   ```

4. **Keypoint Feature Extraction** (optional)
   ```bash
   python extract_keypoint_features.py \
       --keypoint_dir keypoint_jsons/ \
       --output_dir keypoint_features/ \
       --video_name CQ_2
   ```

5. **Re-split if Needed**
   ```bash
   python advanced_split.py \
       --dataset_root preprocessed_data/ \
       --train_videos CQ_2 CQ_3 \
       --test_videos CQ_4
   ```

### Directory Structure After Processing

```
project_root/
├── raw_data/
│   ├── videos/
│   │   ├── CQ_2.mp4
│   │   ├── CQ_3.mp4
│   │   └── CQ_4.mp4
│   └── labels/
│       ├── CQ_2.csv
│       ├── CQ_3.csv
│       └── CQ_4.csv
├── preprocessed_data/
│   ├── features/
│   │   ├── CQ_2_clip_000.npy
│   │   ├── CQ_2_clip_001.npy
│   │   └── ...
│   ├── labels/
│   │   ├── CQ_2_clip_000.npy
│   │   ├── CQ_2_clip_001.npy
│   │   └── ...
│   ├── splits/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   └── test.txt
│   └── meta/
│       ├── class_mapping.json
│       └── dataset_stats.json
└── keypoint_features/
    ├── CQ_2_keypoints.npy
    ├── CQ_3_keypoints.npy
    └── CQ_4_keypoints.npy
```

---

## ❓ Frequently Asked Questions

### Q: What video formats are supported?
A: All formats supported by OpenCV and ffmpeg, including MP4, AVI, MOV, etc.

### Q: How to handle missing keypoint data?
A: The system fills missing keypoints with zeros. Set appropriate confidence thresholds during extraction.

### Q: Can I change the sliding window parameters?
A: Yes, modify `--window_length` and `--stride` in `action_segmentation.py`. Default is 512/256 frames.

### Q: How to add new behavior classes?
A: Modify the CSV label files and the preprocessing pipeline will automatically detect new classes.

### Q: What if GPU memory is insufficient?
A: Reduce `--batch_size` in preprocessing or use `--device cpu` for CPU processing.

---

For detailed technical information and advanced usage, refer to the individual script documentation and the main project README.