#!/bin/bash

# Training script for MS-TCN action segmentation
# Uses only ResNet features (no keypoints)

# Set dataset paths
DATASET_ROOT="/data/zhaozhenghao/Projects/Mouse/datasets/UMich_CQ/all_data_80train_20_val"
OUTPUT_DIR="/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/action_seg_training/resnet_only_$(date +%Y%m%d_%H%M%S)"

# Training parameters
BATCH_SIZE=8
NUM_EPOCHS=50
LEARNING_RATE=0.0005

# Model parameters
NUM_STAGES=4
NUM_LAYERS=10
NUM_F_MAPS=64
FEATURE_DIM=2048  # ResNet50 features only

# Run training without keypoint features
python train.py \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LEARNING_RATE} \
    --num_stages ${NUM_STAGES} \
    --num_layers ${NUM_LAYERS} \
    --num_f_maps ${NUM_F_MAPS} \
    --feature_dim ${FEATURE_DIM} \
    --use_oversampling \
    --use_focal_loss \
    --device cuda

echo "Training completed. Results saved to: ${OUTPUT_DIR}"
