#!/bin/bash

# Run combined statistics analysis for CQ_3 and CQ_4 only (excluding CQ_2)

python3 /data/zhaozhenghao/Projects/Mouse/methods/UMich_methods/postprocess/statistics.py \
    --videos CQ_3 CQ_4 \
    --output /data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/cq3_cq4_comparison_results.json
