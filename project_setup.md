## environment configuration
```bash
ENV_NAME="mouse_behavior"
if ! conda info --envs | grep -q "/$ENV_NAME"; then
    conda create -n $ENV_NAME python=3.9 -y
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    
    pip install -r requriements.txt
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi
conda activate $ENV_NAME
```


## step 0: download data
Download data and store it in `dataset` folder
```markdown
├── dataset
    ├── CQ_2.mp4
    ├── CQ_3.mp4
    ├── CQ_4.mp4
    ├── itch_video_analysis_CQ_2.xlsx
    ├── itch_video_analysis_CQ_3.xlsx
    ├── itch_video_analysis_CQ_4.xlsx
```
```bash
cd $(git rev-parse --show-toplevel)
if ! command -v gdown &> /dev/null; then
    pip install gdown
fi

if [ ! -d "orig_dataset" ]; then
    mkdir -p orig_dataset
    # Download the data from Google Drive and move it to the dataset folder
    # You can use gdown or any other method to download the files
    gdown --folder https://drive.google.com/drive/folders/1rm2sGUw9QVswf0HRhoUldcDObH3Tb06L?usp=drive_link 
else
    echo "Dataset already exists. Skipping download."
fi
cd orig_dataset
for f in itch\ video\ analysisi_*.xlsx; do
    mv "$f" "${f#itch video analysisi_}"
done
```

## step 1: preprocess xlsx

### step 1.1: transform xlsx to csv
```bash
cd $(git rev-parse --show-toplevel)
mkdir -p preprocess_dataset
cd preprocess
python label_process.py --input_dir ../dataset/ --output_dir ../preprocess_dataset/
grep -qxF "/preprocessed_data/" ../.gitignore || echo -e "\n/preprocessed_data/" >> ../.gitignore
cp ../dataset/*.mp4 ../preprocess_dataset/
# rm -rf $(git rev-parse --show-toplevel)/dataset
```

### step 1.2: video grid splitting
❌split_cell.py is missing

❌just needed for multiscreen task

### step 1.3: action segmentation
```bash
cd $(git rev-parse --show-toplevel)
mkdir -p video_segmentation_output
cd preprocess
python action_segmentation.py \
    --video_dir ../preprocess_dataset \
    --videos CQ_2.mp4 CQ_3.mp4 CQ_4.mp4 \
    --csvs itch_video_analysis_CQ_2.csv itch_video_analysis_CQ_3.csv itch_video_analysis_CQ_4.csv \
    --output_dir ../video_segmentation_output \
    --split_mode video \
    --test_videos CQ_4
grep -qxF "/video_segmentation_output/" ../.gitignore || echo -e "\n/video_segmentation_output/" >> ../.gitignore
```

modification:
action_segmentation.py > parse_timestamp > function beginning
```python
if pd.isna(ts_str):
    return 0.0
```

feature extraction elasted time: 05:22+06:34+07:43=19 min 39 s, (may vary with different devices)

### step 1.4: advanced split
```bash
python advanced_split.py \
    --dataset_root ../video_segmentation_output \
    --train_videos CQ_2 CQ_3 \
    --val_videos CQ_4 \
    --split_mode video
```

### step 1.5: feature extraction
❌keypoint detection JSON files are missing
```bash
# View help
python extract_keypoint_features.py --help

# Basic usage
python extract_keypoint_features.py \
    --keypoint_dir /path/to/keypoint_jsons \
    --output_dir /path/to/output \
    --video_name CQ_2 \
    --image_width 640 \
    --image_height 480

# Process multiple videos
for video in CQ_2 CQ_3 CQ_4; do
    python extract_keypoint_features.py \
        --keypoint_dir /path/to/keypoints \
        --output_dir /path/to/output \
        --video_name $video
done
```

## step 2: action segmentation
```bash
cd $(git rev-parse --show-toplevel)/action_seg
```

### step 2.0: verify setup
```bash
bash verify_setup.sh

```

### step 2.1: train the model
1. with keypoint features, not available as keypoint is missing❌
```bash
bash train_resnet_only.sh
```

2. train without keypoint features
```bash
mkdir -p ../train_result/resnet_only
# model parameters
# Training parameters
BATCH_SIZE=8
NUM_EPOCHS=50
LEARNING_RATE=0.0005

# Model parameters
NUM_STAGES=4
NUM_LAYERS=10
NUM_F_MAPS=64
FEATURE_DIM=2048  # ResNet50 features only

python train.py \
    --dataset_root "../video_segmentation_output" \
    --output_dir "../train_result/resnet_only" \
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
grep -qxF "/train_result/" ../.gitignore || echo -e "\n/train_result/" >> ../.gitignore
```
modify:
train.py > line 41
```python
# from .model import MS_TCN # original
from model import MS_TCN # now
```
experiment start time: 2026-03-24 1:48 pm

experiment end time: 2026-03-24 1:54 pm

elasted time: 6 min (may vary with different devices)

### step 2.2: model inference
```bash
# option 1, with keypoint features, not available as keypoint is missing❌
python inference_raw_video.py \
    --video_path video.mp4 \
    --checkpoint model.pth \
    --use_keypoints \
    --keypoint_dir /path/to/keypoints \
    --save_video

# option 2, without keypoint features
python inference_raw_video.py \
    --video_path ../preprocess_dataset/CQ_4.mp4 \
    --checkpoint ../train_result/resnet_only/checkpoints/best.pth \
    --output_dir ../preprocess_dataset/CQ_4_results \
    --save_video
```
option 2, test on CQ_4.mp4:
inference start time: 2026-03-24 1:57 pm

inference end time: 2026-03-24 2:10 pm

elasted time: 13 min (may vary with different devices)


## step 3: postprocessing
```bash
cd $(git rev-parse --show-toplevel)/postprocess
```

### step 3.1: statistics analysis
❌need `/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/` information
```bash
mkdir -p ../statistics_results
cp ../preprocess_dataset/itch_video_analysis_CQ_2.csv ../preprocess_dataset/CQ_2.csv
cp ../preprocess_dataset/itch_video_analysis_CQ_3.csv ../preprocess_dataset/CQ_3.csv
cp ../preprocess_dataset/itch_video_analysis_CQ_4.csv ../preprocess_dataset/CQ_4.csv
python statistics.py \
    --ground-truth ../preprocess_dataset \
    --videos CQ_4 \
    --output ../statistics_results/multi_video_results.json
grep -qxF "/statistics_results/" ../.gitignore || echo -e "\n/statistics_results/" >> ../.gitignore

```
modify:
statistics.py > line 410-411
```python
# results_base = Path('/data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/video_inference') # original
current_script_dir = Path(__file__).resolve().parent
results_base = current_script_dir.parent / "preprocess_dataset" # new
# gt_base = Path('/data/zhaozhenghao/Projects/Mouse/datasets/UMich_CQ') # original
gt_base = current_script_dir.parent / "preprocess_dataset" # new
```
statistics.py > line 420-421
```python
# 'inference_dir': str(results_base / video_name), # old
'inference_dir': str(results_base / f"{video_name}_results"), # new
```

### step 3.2: video generation
❌not available, as keypoint file is required
```bash
mkdir -p ../video_generation_output
python generate_video.py \
    --video ../preprocess_dataset/CQ_4.mp4 \
    --inference-dir ../preprocess_dataset/CQ_4/predictions.npy \
    --output_path ../video_generation_output/CQ_4_custom_video.mp4 \
    --fps 30 \
    --quality high \
    --show_stats \
    --show_timeline
grep -qxF "/video_generation_output/" ../.gitignore || echo -e "\n/video_generation_output/" >> ../.gitignore
```
<span style="color: red;">There is some mistake in doc at `postprocess/README.md#2-generate_videopy---video-generation`</span>


### step 3.3: export to csv
```bash
mkdir -p ../export_csv
python to_csv.py \
    --json_path ../preprocess_dataset/CQ_4_results/statistics.json \
    --output ../export_csv/CQ_4_export.csv
grep -qxF "/export_csv/" ../.gitignore || echo -e "\n/export_csv/" >> ../.gitignore
```
<span style="color: red;">There is some mistake in doc at `postprocess/README.md#3-to_csvpy---csv-export`</span>

### step 3.4: idea from UMich
❌not yet implemented