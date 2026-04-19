#!/bin/bash

# Training script for MS-TCN action segmentation
# Uses preprocessed dataset with keypoint features

# Set dataset paths
DATASET_ROOT="/data/zhaozhenghao/Projects/Mouse/datasets/UMich_CQ/all_data_train"
KEYPOINT_ROOT="/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/keypoint"
OUTPUT_DIR="/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/action_seg_training/all_train"

# Training parameters
BATCH_SIZE=8
NUM_EPOCHS=50
LEARNING_RATE=0.0005

# Model parameters
NUM_STAGES=4
NUM_LAYERS=10
NUM_F_MAPS=64

# Run training with keypoint features
CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --use_keypoints \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LEARNING_RATE} \
    --num_stages ${NUM_STAGES} \
    --num_layers ${NUM_LAYERS} \
    --num_f_maps ${NUM_F_MAPS} \
    --use_oversampling \
    --use_focal_loss \
    --device cuda \
    # --keypoint_root "${KEYPOINT_ROOT}" \

echo "Training completed. Results saved to: ${OUTPUT_DIR}" 