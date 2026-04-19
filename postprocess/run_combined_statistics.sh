#!/bin/bash

# Run combined statistics analysis for CQ_2, CQ_3, and CQ_4

python3 /data/zhaozhenghao/Projects/Mouse/methods/UMich_methods/postprocess/statistics.py \
    --combined \
    --output /data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/combined_comparison_results.json
