#!/bin/bash

set -euo pipefail

SCENARIO_SUFFIX="focal"
DATASET_DIR="preprocess_dataset"
DATA_PREP_SCRIPT=""
TRAIN_SCRIPT="train_focal.py"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_scenario_common.sh"
