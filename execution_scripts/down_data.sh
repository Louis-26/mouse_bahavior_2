#!/bin/bash

set -euo pipefail

SCENARIO_SUFFIX="down"
DATASET_DIR="down_sample_dataset"
DATA_PREP_SCRIPT="data_augmentation/down_sample.py"
TRAIN_SCRIPT="train.py"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_scenario_common.sh"
