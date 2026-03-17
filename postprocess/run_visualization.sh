#!/bin/bash

# Example script to run visualization for CQ_3 video
# Adjust paths as needed for different videos

# Set paths
VIDEO_PATH="/data/zhaozhenghao/Projects/Mouse/datasets/UMich_CQ/CQ_3.mp4"
INFERENCE_DIR="/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/video_inference/CQ_3"
KEYPOINT_FILE="/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/keypoint/CQ_3_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2__before_adapt.json"
OUTPUT_VIDEO="/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/video_inference/CQ_3/CQ_3_visualized.mp4"

# Run visualization
python3 generate_video.py \
    --video "$VIDEO_PATH" \
    --inference-dir "$INFERENCE_DIR" \
    --keypoint-file "$KEYPOINT_FILE" \
    --output "$OUTPUT_VIDEO"

echo "Visualization complete! Output saved to: $OUTPUT_VIDEO"
