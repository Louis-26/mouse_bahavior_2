# Post-processing Tools

Comprehensive analysis and visualization tools for mouse behavior inference results.

---

## 📋 Table of Contents

- [Tools Overview](#-tools-overview)
- [Detailed Usage Instructions](#-detailed-usage-instructions)
  - [1. statistics.py - Statistical Analysis](#1-statisticspy---statistical-analysis)
  - [2. generate_video.py - Video Generation](#2-generate_videopy---video-generation)
  - [3. to_csv.py - CSV Export](#3-to_csvpy---csv-export)
  - [4. idea_from_UMich.py - UMich Analysis](#4-idea_from_umichpy---umich-analysis)
- [Batch Processing Scripts](#-batch-processing-scripts)
- [Analysis Workflows](#-analysis-workflows)
- [Output Examples](#-output-examples)

---

## 🎯 Tools Overview

| Tool | Function | Input | Output |
|------|----------|-------|--------|
| `statistics.py` | Statistical analysis and comparison | Inference results + Ground truth | Statistics reports, metrics |
| `generate_video.py` | Create annotated videos | Predictions + Original video | Annotated MP4 with overlays |
| `to_csv.py` | Export predictions to CSV | Prediction arrays | CSV timeline files |
| `idea_from_UMich.py` | UMich-specific analysis | Custom analysis data | Analysis reports |

---

## 📖 Detailed Usage Instructions

### 1. `statistics.py` - Statistical Analysis

Compare inference results with ground truth annotations and generate comprehensive statistics.

#### Features
- 🔍 Frame-level accuracy metrics (precision, recall, F1)
- 📊 Behavior statistics (count, duration, frequency)
- 📈 Confusion matrix analysis
- 🎯 Combined multi-video analysis
- 📋 Detailed error analysis

#### Usage

**Single Video Analysis:**
```bash
# Basic comparison
python statistics.py \
    --inference-dir /path/to/inference/results \
    --ground-truth /path/to/ground_truth.csv

# Save results to file
python statistics.py \
    --inference-dir /path/to/inference/results \
    --ground-truth /path/to/ground_truth.csv \
    --output results_analysis.json
```

**Multi-Video Combined Analysis:**
```bash
# Analyze all CQ videos together
python statistics.py --combined

# Analyze specific videos together
python statistics.py \
    --videos CQ_3 CQ_4 \
    --output combined_analysis.json

# Custom paths for combined analysis
python statistics.py --combined \
    --videos CQ_2 CQ_3 CQ_4 \
    --output multi_video_results.json
```

#### Parameters
- `--inference-dir`: Directory containing `predictions.npy` and `statistics.json`
- `--ground-truth`: Path to ground truth CSV file with time annotations
- `--combined`: Analyze CQ_2, CQ_3, and CQ_4 together as combined dataset
- `--videos`: Specify which videos to analyze together (e.g., `--videos CQ_3 CQ_4`)
- `--output`: Save detailed results to JSON file (default: print to console)

#### Output Metrics

**Frame-Level Accuracy:**
- **Accuracy**: Overall frame-level classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

**Behavior Statistics:**
- **Total Count**: Number of scratching episodes
- **Total Duration**: Total time spent scratching (seconds)
- **Mean Duration**: Average scratching episode length
- **Frequency**: Scratching episodes per minute

**Confusion Matrix:**
- True Positive (TP): Correctly identified scratching frames
- True Negative (TN): Correctly identified non-behavior frames
- False Positive (FP): Incorrectly identified scratching frames
- False Negative (FN): Missed scratching frames

#### Example Output
```
STATISTICS COMPARISON
=====================

Metric                    Prediction      Ground Truth    Difference      Error Rate
---------------------------------------------------------------------------------
Total Count               12.0000         10.0000         2.0000          20.00%
Total Duration Sec         8.5000          7.2000          1.3000          18.06%
Mean Duration Sec          0.7083          0.7200         -0.0117          1.63%
Frequency Per Min          4.8000          4.0000          0.8000          20.00%

FRAME-LEVEL ACCURACY
====================

Confusion Matrix:
  True Positive (TP):    245 (predicted scratching, actual scratching)
  True Negative (TN):   1678 (predicted no behavior, actual no behavior)
  False Positive (FP):    67 (predicted scratching, actual no behavior)
  False Negative (FN):    34 (predicted no behavior, actual scratching)

Accuracy:  94.89%
Precision: 78.53%
Recall:    87.81%
F1 Score:  83.95%
```

### 2. `generate_video.py` - Video Generation

Create annotated videos with prediction overlays, behavior highlighting, and optional keypoint visualization.

#### Features
- 🎥 Overlay predictions on original video
- 🎨 Color-coded behavior visualization
- 📊 Real-time statistics display
- 🔍 Keypoint overlay support
- ⏱️ Timeline and progress indicators

#### Usage

```bash
# Basic video generation
python generate_video.py \
    --video_path /path/to/original_video.mp4 \
    --predictions_path /path/to/predictions.npy \
    --output_path /path/to/annotated_video.mp4

# With ground truth comparison
python generate_video.py \
    --video_path /path/to/original_video.mp4 \
    --predictions_path /path/to/predictions.npy \
    --ground_truth_path /path/to/ground_truth.csv \
    --output_path /path/to/comparison_video.mp4

# With keypoint overlay
python generate_video.py \
    --video_path /path/to/original_video.mp4 \
    --predictions_path /path/to/predictions.npy \
    --keypoints_path /path/to/keypoints.json \
    --output_path /path/to/keypoint_video.mp4

# Time range selection
python generate_video.py \
    --video_path /path/to/original_video.mp4 \
    --predictions_path /path/to/predictions.npy \
    --output_path /path/to/segment_video.mp4 \
    --start_time 30.0 \
    --end_time 120.0

# Custom visualization settings
python generate_video.py \
    --video_path /path/to/original_video.mp4 \
    --predictions_path /path/to/predictions.npy \
    --output_path /path/to/custom_video.mp4 \
    --fps 30 \
    --quality high \
    --show_stats \
    --show_timeline
```

#### Parameters
- `--video_path`: Path to original input video
- `--predictions_path`: Path to predictions numpy array
- `--output_path`: Output annotated video path
- `--ground_truth_path`: Optional ground truth CSV for comparison
- `--keypoints_path`: Optional keypoints JSON for overlay
- `--start_time`: Start time in seconds (default: from beginning)
- `--end_time`: End time in seconds (default: until end)
- `--fps`: Output video frame rate (default: same as input)
- `--quality`: Video quality (`low`, `medium`, `high`)
- `--show_stats`: Display real-time statistics overlay
- `--show_timeline`: Show progress timeline at bottom

#### Visual Elements

**Color Coding:**
- 🟢 **Green**: No behavior / background
- 🔴 **Red**: Scratching behavior (predictions)
- 🟡 **Yellow**: Ground truth scratching (if provided)
- 🟠 **Orange**: Prediction/ground truth overlap

**Overlays:**
- Behavior prediction bars
- Confidence scores
- Frame-by-frame statistics
- Keypoint skeleton (if enabled)
- Timeline progress indicator

### 3. `to_csv.py` - CSV Export

Convert prediction arrays to CSV format for further analysis or visualization in external tools.

#### Features
- 📄 Export frame-by-frame predictions
- ⏰ Time-based formatting (HH:MM:SS.mmm)
- 📊 Behavior episode segmentation
- 🔢 Confidence score inclusion

#### Usage

```bash
# Basic CSV export
python to_csv.py \
    --predictions_path /path/to/predictions.npy \
    --output_path /path/to/predictions.csv \
    --fps 30

# With confidence scores
python to_csv.py \
    --predictions_path /path/to/predictions.npy \
    --probabilities_path /path/to/probabilities.npy \
    --output_path /path/to/detailed_predictions.csv \
    --fps 30

# Episode-based export (segments)
python to_csv.py \
    --predictions_path /path/to/predictions.npy \
    --output_path /path/to/episodes.csv \
    --fps 30 \
    --segment_mode

# Custom time offset
python to_csv.py \
    --predictions_path /path/to/predictions.npy \
    --output_path /path/to/predictions.csv \
    --fps 30 \
    --start_offset 60.0
```

#### Parameters
- `--predictions_path`: Path to predictions numpy array
- `--probabilities_path`: Optional probabilities for confidence scores
- `--output_path`: Output CSV file path
- `--fps`: Video frame rate for time calculation
- `--segment_mode`: Export behavior episodes instead of frame-by-frame
- `--start_offset`: Time offset in seconds for the first frame

#### Output Formats

**Frame-by-Frame Mode:**
```csv
Frame,Time,Prediction,Confidence
0,00:00:00.000,0,0.95
1,00:00:00.033,0,0.92
2,00:00:00.066,1,0.78
3,00:00:00.100,1,0.85
```

**Episode Mode:**
```csv
Start,End,Duration,Notes
00:01:23.456,00:01:25.789,2.333,scratching
00:02:15.123,00:02:18.456,3.333,scratching
```

### 4. `idea_from_UMich.py` - UMich Analysis

Specialized analysis tools following University of Michigan research methodologies.

#### Features
- 📈 Advanced behavioral metrics
- 🔬 Research-specific analysis protocols
- 📊 Statistical comparisons with published baselines
- 📋 Formatted reports for academic use

#### Usage

```bash
# Standard UMich analysis
python idea_from_UMich.py \
    --data_dir /path/to/analysis/data \
    --output_dir /path/to/umich/results

# Comparison with baseline
python idea_from_UMich.py \
    --data_dir /path/to/analysis/data \
    --baseline_dir /path/to/baseline/data \
    --output_dir /path/to/comparison/results

# Generate research report
python idea_from_UMich.py \
    --data_dir /path/to/analysis/data \
    --output_dir /path/to/results \
    --generate_report \
    --report_format academic
```

---

## 🔧 Batch Processing Scripts

### Batch Analysis Script (`run_statistics.sh`)

```bash
#!/bin/bash
# Analyze multiple videos in batch

# Run individual video analyses
./run_statistics.sh

# This will analyze:
# - CQ_2 individually
# - CQ_3 individually  
# - CQ_4 individually
# - All three combined
```

### Batch Visualization Script (`run_visualization.sh`)

```bash
#!/bin/bash
# Generate videos for all results

# Run batch video generation
./run_visualization.sh

# Creates annotated videos for all inference results
```

### Combined Statistics Script (`run_combined_statistics.sh`)

```bash
#!/bin/bash
# Run combined analysis for specific video sets

# Analyze CQ_3 and CQ_4 together
./run_cq3_cq4_statistics.sh
```

---

## 🔄 Analysis Workflows

### Standard Post-Processing Pipeline

1. **Statistical Analysis**
   ```bash
   # Individual video analysis
   python statistics.py \
       --inference-dir results/CQ_2 \
       --ground-truth data/CQ_2.csv \
       --output analysis/CQ_2_stats.json
   ```

2. **Combined Analysis**
   ```bash
   # Multi-video combined analysis
   python statistics.py --combined \
       --output analysis/combined_stats.json
   ```

3. **Video Generation**
   ```bash
   # Create annotated videos
   python generate_video.py \
       --video_path data/CQ_2.mp4 \
       --predictions_path results/CQ_2/predictions.npy \
       --ground_truth_path data/CQ_2.csv \
       --output_path videos/CQ_2_annotated.mp4
   ```

4. **CSV Export**
   ```bash
   # Export for external analysis
   python to_csv.py \
       --predictions_path results/CQ_2/predictions.npy \
       --output_path exports/CQ_2_timeline.csv \
       --fps 30
   ```

### Comparative Analysis Workflow

```bash
# 1. Analyze baseline method
python statistics.py \
    --inference-dir results/baseline/CQ_2 \
    --ground-truth data/CQ_2.csv \
    --output analysis/baseline_CQ_2.json

# 2. Analyze improved method
python statistics.py \
    --inference-dir results/improved/CQ_2 \
    --ground-truth data/CQ_2.csv \
    --output analysis/improved_CQ_2.json

# 3. Generate comparison videos
python generate_video.py \
    --video_path data/CQ_2.mp4 \
    --predictions_path results/improved/CQ_2/predictions.npy \
    --ground_truth_path data/CQ_2.csv \
    --output_path videos/CQ_2_comparison.mp4
```

---

## 📊 Output Examples

### Statistical Analysis Output

```json
{
  "metadata": {
    "inference_dir": "results/CQ_2",
    "ground_truth": "data/CQ_2.csv",
    "fps": 29.97,
    "total_frames": 15420,
    "duration_sec": 514.35
  },
  "prediction_statistics": {
    "total_count": 12,
    "total_duration_sec": 8.5,
    "mean_duration_sec": 0.708,
    "frequency_per_min": 1.4
  },
  "ground_truth_statistics": {
    "total_count": 10,
    "total_duration_sec": 7.2,
    "mean_duration_sec": 0.72,
    "frequency_per_min": 1.17
  },
  "frame_accuracy": {
    "accuracy": 0.9489,
    "precision": 0.7853,
    "recall": 0.8781,
    "f1_score": 0.8395,
    "confusion_matrix": {
      "true_positive": 245,
      "true_negative": 1678,
      "false_positive": 67,
      "false_negative": 34
    }
  }
}
```

### Video Generation Features

- **Prediction Overlay**: Color-coded behavior predictions
- **Ground Truth Comparison**: Side-by-side or overlay comparison
- **Statistics Panel**: Real-time metrics display
- **Timeline Indicator**: Progress and behavior timeline
- **Keypoint Skeleton**: Optional pose visualization

### CSV Export Formats

**Timeline Export:**
```csv
Frame,Time,Prediction,Confidence
0,00:00:00.000,0,0.95
1,00:00:00.033,0,0.92
2,00:00:00.066,1,0.78
```

**Episode Export:**
```csv
Start,End,Duration,Notes
00:01:23.456,00:01:25.789,2.333,scratching
00:02:15.123,00:02:18.456,3.333,scratching
```

---

## 🎯 Integration with Other Components

### With Action Segmentation
```bash
# Direct pipeline from inference to analysis
cd ../action_seg
python inference_raw_video.py --video_path data/CQ_2.mp4 --checkpoint model.pth --output_dir ../postprocess/results/CQ_2

cd ../postprocess
python statistics.py --inference-dir results/CQ_2 --ground-truth ../data/CQ_2.csv
```

### With Preprocessing
```bash
# Use preprocessed labels for validation
python statistics.py \
    --inference-dir results/CQ_2 \
    --ground-truth ../preprocess/preprocessed_data/original_labels/CQ_2.csv
```

---

## ❓ Frequently Asked Questions

### Q: How to handle videos with different frame rates?
A: Specify the correct `--fps` parameter in analysis tools. The system will automatically adjust time calculations.

### Q: Can I analyze custom time segments?
A: Yes, use `--start_time` and `--end_time` parameters in video generation and analysis tools.

### Q: How to export results for publication?
A: Use the JSON output format with `--output` parameter, then process with your preferred analysis tools (R, MATLAB, etc.).

### Q: What video formats are supported for generation?
A: MP4, AVI, MOV, and other formats supported by OpenCV. MP4 with H.264 encoding is recommended.

### Q: How to batch process multiple experiments?
A: Use the provided shell scripts (`run_*.sh`) or create custom batch scripts following the examples.

---

For detailed technical information and integration examples, refer to the main project documentation and the action segmentation guide.