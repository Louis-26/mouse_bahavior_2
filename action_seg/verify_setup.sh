#!/bin/bash

# Verification script to check dataset and keypoint setup before training

echo "==================================================================="
echo "Training Setup Verification"
echo "==================================================================="

DATASET_ROOT="/data/zhaozhenghao/Projects/Mouse/datasets/UMich_CQ/all_data_80train_20_val"
KEYPOINT_ROOT="/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/keypoint"

echo ""
echo "1. Checking dataset directory..."
if [ -d "${DATASET_ROOT}" ]; then
    echo "   ✓ Dataset directory exists: ${DATASET_ROOT}"
else
    echo "   ✗ Dataset directory NOT found: ${DATASET_ROOT}"
    exit 1
fi

echo ""
echo "2. Checking dataset structure..."
for subdir in features labels meta splits; do
    if [ -d "${DATASET_ROOT}/${subdir}" ]; then
        count=$(ls -1 "${DATASET_ROOT}/${subdir}" 2>/dev/null | wc -l)
        echo "   ✓ ${subdir}/ exists with ${count} files"
    else
        echo "   ✗ ${subdir}/ directory missing"
    fi
done

echo ""
echo "3. Checking split files..."
for split in train val; do
    split_file="${DATASET_ROOT}/splits/${split}.txt"
    if [ -f "${split_file}" ]; then
        count=$(wc -l < "${split_file}")
        echo "   ✓ ${split}.txt exists with ${count} samples"
    else
        echo "   ✗ ${split}.txt not found"
    fi
done

echo ""
echo "4. Checking keypoint directory..."
if [ -d "${KEYPOINT_ROOT}" ]; then
    echo "   ✓ Keypoint directory exists: ${KEYPOINT_ROOT}"
    h5_count=$(ls -1 "${KEYPOINT_ROOT}"/*.h5 2>/dev/null | wc -l)
    echo "   ✓ Found ${h5_count} .h5 keypoint files"
    
    if [ ${h5_count} -gt 0 ]; then
        echo ""
        echo "   Keypoint files:"
        ls -1 "${KEYPOINT_ROOT}"/*.h5 | head -5 | while read file; do
            echo "     - $(basename $file)"
        done
        if [ ${h5_count} -gt 5 ]; then
            echo "     ... and $((h5_count - 5)) more"
        fi
    fi
else
    echo "   ✗ Keypoint directory NOT found: ${KEYPOINT_ROOT}"
    echo "   (Training without keypoints will still work)"
fi

echo ""
echo "5. Checking Python dependencies..."
python -c "import torch; print('   ✓ PyTorch version:', torch.__version__)" 2>/dev/null || echo "   ✗ PyTorch not found"
python -c "import numpy; print('   ✓ NumPy version:', numpy.__version__)" 2>/dev/null || echo "   ✗ NumPy not found"
python -c "import h5py; print('   ✓ h5py version:', h5py.__version__)" 2>/dev/null || echo "   ✗ h5py not found (required for keypoints)"

echo ""
echo "6. Checking CUDA availability..."
python -c "import torch; print('   ✓ CUDA available:', torch.cuda.is_available()); print('   ✓ CUDA devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo "   ✗ Cannot check CUDA"

echo ""
echo "==================================================================="
echo "Verification complete!"
echo "==================================================================="
echo ""
echo "To start training:"
echo "  With keypoints:    bash train.sh"
echo "  Without keypoints: bash train_resnet_only.sh"
echo ""
