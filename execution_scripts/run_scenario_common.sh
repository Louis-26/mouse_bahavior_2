#!/bin/bash

set -euo pipefail

: "${SCENARIO_SUFFIX:?SCENARIO_SUFFIX is required}"
: "${DATASET_DIR:?DATASET_DIR is required}"
: "${TRAIN_SCRIPT:?TRAIN_SCRIPT is required}"

DATA_PREP_SCRIPT="${DATA_PREP_SCRIPT:-}"
ENV_NAME="${ENV_NAME:-mouse_behavior}"
ROOT_DIR="$(git rev-parse --show-toplevel)"

if [[ -f "${ROOT_DIR}/requirements.txt" ]]; then
    REQUIREMENTS_FILE="${ROOT_DIR}/requirements.txt"
else
    REQUIREMENTS_FILE="${ROOT_DIR}/requriements.txt"
fi

echo "step 1: environment configuration"
source /data/svillar3/ylu174/Anaconda3/etc/profile.d/conda.sh

if ! conda info --envs | grep -q "/${ENV_NAME}$"; then
    conda create -n "${ENV_NAME}" python=3.9 -y
    conda activate "${ENV_NAME}"
    pip install -r "${REQUIREMENTS_FILE}"
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
    conda activate "${ENV_NAME}"
fi

echo "step 2: scenario data preparation"
cd "${ROOT_DIR}"
if [ -d "${ROOT_DIR}/${DATASET_DIR}" ] && [ -n "$(ls -A "${ROOT_DIR}/${DATASET_DIR}")" ]; then
    echo "Dataset folder already exists and is not empty, skipping data preparation"
else
    mkdir -p "${ROOT_DIR}/${DATASET_DIR}"
    if [[ -n "${DATA_PREP_SCRIPT}" ]]; then
        python "${DATA_PREP_SCRIPT}"
    fi
fi

echo "step 3: preprocessing"
cd "${ROOT_DIR}/preprocess"
mkdir -p "${ROOT_DIR}/video_segmentation_output_${SCENARIO_SUFFIX}"
python action_segmentation.py \
    --video_dir "../${DATASET_DIR}" \
    --videos CQ_2.mp4 CQ_3.mp4 CQ_4.mp4 \
    --csvs CQ_2.csv CQ_3.csv CQ_4.csv \
    --output_dir "../video_segmentation_output_${SCENARIO_SUFFIX}" \
    --split_mode video \
    --test_videos CQ_4

python advanced_split.py \
    --dataset_root "../video_segmentation_output_${SCENARIO_SUFFIX}" \
    --train_videos CQ_2 CQ_3 \
    --val_videos CQ_4 \
    --split_mode video

echo "step 4: training"
cd "${ROOT_DIR}/action_seg"
mkdir -p "${ROOT_DIR}/train_result/resnet_only_${SCENARIO_SUFFIX}"

BATCH_SIZE=8
NUM_EPOCHS=50
LEARNING_RATE=0.0005
NUM_STAGES=4
NUM_LAYERS=10
NUM_F_MAPS=64
FEATURE_DIM=2048

python "${TRAIN_SCRIPT}" \
    --dataset_root "../video_segmentation_output_${SCENARIO_SUFFIX}" \
    --output_dir "../train_result/resnet_only_${SCENARIO_SUFFIX}" \
    --batch_size "${BATCH_SIZE}" \
    --num_epochs "${NUM_EPOCHS}" \
    --lr "${LEARNING_RATE}" \
    --num_stages "${NUM_STAGES}" \
    --num_layers "${NUM_LAYERS}" \
    --num_f_maps "${NUM_F_MAPS}" \
    --feature_dim "${FEATURE_DIM}" \
    --use_oversampling \
    --use_focal_loss \
    --device cuda

echo "step 5: inference"
python inference_raw_video.py \
    --video_path "../${DATASET_DIR}/CQ_4.mp4" \
    --checkpoint "../train_result/resnet_only_${SCENARIO_SUFFIX}/checkpoints/best.pth" \
    --output_dir "../${DATASET_DIR}/CQ_4_results_${SCENARIO_SUFFIX}" \
    --save_video

echo "step 6: postprocessing"
cd "${ROOT_DIR}/postprocess"
mkdir -p "${ROOT_DIR}/statistics_results"
python statistics.py \
    --inference-dir "../${DATASET_DIR}/CQ_4_results_${SCENARIO_SUFFIX}" \
    --ground-truth "../${DATASET_DIR}/CQ_4.csv" \
    --output "../statistics_results/multi_video_results_${SCENARIO_SUFFIX}.json"

mkdir -p "${ROOT_DIR}/export_csv"
python to_csv.py \
    --json_path "../${DATASET_DIR}/CQ_4_results_${SCENARIO_SUFFIX}/statistics.json" \
    --output "../export_csv/CQ_4_export_${SCENARIO_SUFFIX}.csv"

grep -qxF "/statistics_results/" "${ROOT_DIR}/.gitignore" || echo -e "\n/statistics_results/" >> "${ROOT_DIR}/.gitignore"
grep -qxF "/train_result/" "${ROOT_DIR}/.gitignore" || echo -e "\n/train_result/" >> "${ROOT_DIR}/.gitignore"
grep -qxF "/export_csv/" "${ROOT_DIR}/.gitignore" || echo -e "\n/export_csv/" >> "${ROOT_DIR}/.gitignore"

echo "scenario '${SCENARIO_SUFFIX}' completed"
