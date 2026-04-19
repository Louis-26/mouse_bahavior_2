#!/bin/bash

# Compare CQ_3 inference results with ground truth

python3 statistics.py \
    --inference-dir "/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/video_inference/CQ_3" \
    --ground-truth "/data/zhaozhenghao/Projects/Mouse/datasets/UMich_CQ/CQ_3.csv" \
    --output "/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/video_inference/CQ_3/comparison_results.json"

echo ""
echo "Comparison complete! Results saved to:"
echo "/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/video_inference/CQ_3/comparison_results.json"
