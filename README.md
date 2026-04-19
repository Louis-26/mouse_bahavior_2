# UMich Methods - Mouse Behavior Analysis Toolkit

A comprehensive toolkit for mouse behavior analysis using deep learning, specifically designed for temporal action segmentation of scratching behaviors. This repository contains the complete pipeline from data preprocessing to model training and inference.

## 🎯 Overview

This toolkit implements a Multi-Stage Temporal Convolutional Network (MS-TCN) for frame-level action segmentation on mouse behavior videos. It supports both unimodal (ResNet features) and multimodal (ResNet + keypoints) approaches for improved accuracy.

## 📁 Directory Structure

```
methods/UMich_methods/
├── README.md                    # This file
├── CLEANUP_PLAN.md             # Code cleanup documentation
├── action_seg/                 # Action segmentation models and training
│   ├── README.md               # Detailed action segmentation guide
│   ├── model.py                # MS-TCN model definitions
│   ├── train.py                # Training script
│   ├── inference_raw_video.py  # Inference on raw videos
│   └── *.sh                    # Training and inference scripts
├── preprocess/                 # Data preprocessing tools
│   ├── README.md               # Preprocessing tools documentation
│   ├── action_segmentation.py  # Complete preprocessing pipeline
│   ├── extract_keypoint_features.py  # Keypoint feature extraction
│   ├── dataloader_example.py   # PyTorch data loader
│   └── ...                     # Additional preprocessing utilities
└── postprocess/                # Post-processing and analysis
    ├── README.md               # Post-processing documentation (to be created)
    ├── statistics.py           # Statistical analysis
    ├── generate_video.py       # Video generation with overlays
    └── ...                     # Additional analysis tools
```

## 🚀 Quick Start

### 1. Action Segmentation Training

For detailed training instructions, see [`action_seg/README.md`](action_seg/README.md).

**Train with keypoints (recommended):**
```bash
cd action_seg
bash train.sh
```

**Train with ResNet only:**
```bash
cd action_seg
bash train_resnet_only.sh
```

### 2. Preprocessing Pipeline

For detailed preprocessing instructions, see [`preprocess/README.md`](preprocess/README.md).

**Complete preprocessing pipeline:**
```bash
cd preprocess
python action_segmentation.py \
    --video_dir /path/to/videos \
    --videos CQ_2.mp4 CQ_3.mp4 CQ_4.mp4 \
    --csvs CQ_2.csv CQ_3.csv CQ_4.csv \
    --output_dir /path/to/output
```

### 3. Inference on New Videos

```bash
cd action_seg
python inference_raw_video.py \
    --video_path /path/to/video.mp4 \
    --checkpoint /path/to/model.pth \
    --use_keypoints \
    --save_video
```

### 4. Post-processing and Analysis

```bash
cd postprocess
python statistics.py \
    --inference_dir /path/to/inference_results \
    --ground_truth /path/to/ground_truth.csv
```

## 🛠️ Core Components

### Action Segmentation (`action_seg/`)

**Purpose:** Train and deploy MS-TCN models for temporal action segmentation

**Key Features:**
- Multi-Stage Temporal Convolutional Network (MS-TCN)
- Multimodal learning (ResNet + keypoints)
- Class imbalance handling with focal loss and oversampling
- Comprehensive evaluation metrics

**Main Files:**
- `model.py` - MS-TCN model definitions
- `train.py` - Training script with advanced features
- `inference_raw_video.py` - End-to-end inference pipeline

### Preprocessing (`preprocess/`)

**Purpose:** Convert raw videos and annotations into training-ready datasets

**Key Features:**
- Frame-level label generation from time intervals
- ResNet50 feature extraction
- Keypoint feature processing
- Sliding window segmentation
- Train/validation splitting

**Main Files:**
- `action_segmentation.py` - Complete preprocessing pipeline
- `extract_keypoint_features.py` - Keypoint feature extraction
- `dataloader_example.py` - PyTorch data loading

### Post-processing (`postprocess/`)

**Purpose:** Analyze model outputs and generate visualizations

**Key Features:**
- Statistical analysis and comparison with ground truth
- Video generation with prediction overlays
- Performance metrics calculation
- Visualization tools

**Main Files:**
- `statistics.py` - Statistical analysis of predictions
- `generate_video.py` - Video overlay generation

<!-- ## 📊 Model Performance

| Configuration | Frame Accuracy | F1 Score | Parameters |
|---------------|----------------|----------|------------|
| ResNet Only (2048D) | ~95% | ~60% | 500K |
| ResNet + Keypoints (2165D) | ~97% | ~70-75% | 510K | -->

## 🔧 Requirements

- Python 3.8+
- PyTorch 1.9+
- OpenCV
- NumPy, Pandas
- tqdm
- pathlib

## 📖 Documentation

- **[Action Segmentation Guide](action_seg/README.md)** - Comprehensive training and inference documentation
- **[Preprocessing Guide](preprocess/README.md)** - Data preparation and preprocessing tools
- **[Post-processing Guide](postprocess/README.md)** - Analysis and visualization tools

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and support, please open an issue in the repository.

---

**Note:** This toolkit is specifically designed for mouse scratching behavior analysis using the University of Michigan dataset. For adaptation to other behaviors or datasets, modifications to the preprocessing and model configuration may be required.