# train 
ACTION_SEG_RETRAIN=false
ADVANCED_SPLIT_RETRAIN=false
MODEL_RETRAIN=false

# dataset
ORIG_DATASET="orig_dataset"
DATASET="dataset"
VIDEO_SEG_OUTPUT="video_segmentation_output"
STATS_OUTPUT="multi_video_results"
INFERENCE_DIR="preprocess_dataset_overall/dataset"
# CSV_EXPORT="CQ_4_export.csv"
# EVAL_DATASET="CQ_4"
CSV_EXPORT="CQ_3_export.csv"
EVAL_DATASET="CQ_3"


# step 1. environment configuration and data preparation
echo "🚀step 1, start for environment configuration and data preparation"
source /data/svillar3/ylu174/Anaconda3/etc/profile.d/conda.sh # hardcode it into the actual path
conda update -n base -c defaults conda -y
## step 1.1: environment 
ENV_NAME="mouse_behavior"
echo "🚀step 1.1, start for environment configuration"

cd $(git rev-parse --show-toplevel)
if ! conda info --envs | grep -q "/$ENV_NAME"; then
    conda create -n $ENV_NAME python=3.9 -y
    conda activate $ENV_NAME
    pip install -r requirements.txt
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi
conda activate $ENV_NAME
echo "✅step 1.1, environment setup is done"

## step 1.2: dataset
if ! command -v gdown &> /dev/null; then
    pip install gdown
fi

if [ ! -d "dataset" ]; then
    mkdir -p dataset
    # Download the data from Google Drive and move it to the dataset folder
    # You can use gdown or any other method to download the files
    gdown --folder https://drive.google.com/drive/folders/1rm2sGUw9QVswf0HRhoUldcDObH3Tb06L?usp=drive_link 
    for f in itch\ video\ analysisi_*.xlsx; do
        mv "$f" "${f#itch video analysisi_}"
    done    
else
    echo "Dataset already exists. Skipping download."
fi
echo "✅step 1.2, dataset preparation is done"

echo "✅step 1, environment configuration and data preparation is done"
# step 2: preprocess the data
echo "🚀step 2, start for data preprocessing"
## step 2.1: transform xlsx to csv
echo "🚀step 2.1, start for xlsx to csv transformation"
cd $(git rev-parse --show-toplevel)
mkdir -p preprocess_dataset_overall/$DATASET
cd preprocess

if [ -d "../preprocess_dataset_overall/$DATASET" ]; then
    echo "preprocess_dataset_overall already exists, skipping creation."
else
    python label_process.py --input_dir ../$ORIG_DATASET/ --output_dir ../preprocess_dataset_overall/$DATASET
fi


grep -qxF "/preprocess_dataset_overall/" ../.gitignore || echo -e "\n/preprocess_dataset_overall/" >> ../.gitignore
if [ -d "../preprocess_dataset_overall/$DATASET" ] && ls ../preprocess_dataset_overall/$DATASET/*.mp4 >/dev/null 2>&1; then
    echo ".mp4 files exist, skipping copy."
else
    
    cp ../$ORIG_DATASET/*.mp4 ../preprocess_dataset_overall/$DATASET/
fi
echo "✅step 2.1, xlsx to csv transformation is done, preprocessed data is ready"
## step 2.2: video grid splitting

# right now this step is skipped as split_cell.py is missing, this is needed only for multiscreen task

## step 2.3: action segmentation
echo "🚀step 2.3, start for action segmentation"
cd $(git rev-parse --show-toplevel)
mkdir -p video_segmentation_output_overall/$VIDEO_SEG_OUTPUT
cd preprocess

if [ ! -z "$(ls -A "../video_segmentation_output_overall/$VIDEO_SEG_OUTPUT" 2>/dev/null)" ] && [ "$ACTION_SEG_RETRAIN" = false ]; then
    echo "Segmented videos already exist and explicitly asked not to retrain, skipping action segmentation."

else
    python action_segmentation.py \
        --video_dir ../preprocess_dataset_overall/$DATASET \
        --videos CQ_2.mp4 CQ_3.mp4 CQ_4.mp4 \
        --csvs CQ_2.csv CQ_3.csv CQ_4.csv \
        --output_dir ../video_segmentation_output_overall/$VIDEO_SEG_OUTPUT \
        --split_mode video \
        --test_videos CQ_4
fi

grep -qxF "/video_segmentation_output_overall/" ../.gitignore || echo -e "\n/video_segmentation_output_overall/" >> ../.gitignore
echo "✅step 2.3, action segmentation is done"

## step 2.4: advanced split
echo "🚀step 2.4, start for advanced split"

if [ ! -z "$(ls -A "../video_segmentation_output_overall/$VIDEO_SEG_OUTPUT/split" 2>/dev/null)" ] && [ "$ADVANCED_SPLIT_RETRAIN" = false ]; then
    echo "Split videos already exist and explicitly asked not to retrain, skipping advanced split."

else
    python advanced_split.py \
        --dataset_root ../video_segmentation_output_overall/$VIDEO_SEG_OUTPUT \
        --train_videos CQ_2 CQ_3 \
        --val_videos CQ_4 \
        --split_mode video
fi

echo "✅step 2.4, advanced split is done"
## step 2.5: feature extraction
# skip as keypoint is missing
echo "✅step 2, data preprocessing is done"

# step 3: action segmentation
echo "🚀step 3, start for action segmentation"
cd $(git rev-parse --show-toplevel)/action_seg

## step 3.1: train the model
echo "🚀step 3.1, start for model training"
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


if [ ! -z "$(ls -A "../train_result/resnet_only" 2>/dev/null)" ] && [ "$MODEL_RETRAIN" = false ]; then
    echo "Model checkpoints already exist and explicitly asked not to retrain, skipping model training."

else
    python train.py \
        --dataset_root "../video_segmentation_output_overall/$VIDEO_SEG_OUTPUT" \
        --output_dir "../train_result/resnet_only" \
        --batch_size ${BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --lr ${LEARNING_RATE} \
        --num_stages ${NUM_STAGES} \
        --num_layers ${NUM_LAYERS} \
        --num_f_maps ${NUM_F_MAPS} \
        --feature_dim ${FEATURE_DIM} \
        --device cuda
fi



grep -qxF "/train_result/" ../.gitignore || echo -e "\n/train_result/" >> ../.gitignore
echo "✅step 3.1, model training is done"

## step 3.2: model inference
echo "🚀step 3.2, start for model inference"
# option 2, without keypoint features
python inference_raw_video.py \
    --video_path "../preprocess_dataset_overall/$DATASET/${EVAL_DATASET}.mp4" \
    --checkpoint "../train_result/resnet_only/checkpoints/best.pth" \
    --save_video
echo "✅step 3.2, model inference is done"

echo "✅step 3, action segmentation is done"

## step 4: postprocessing
echo "🚀step 4, start for postprocessing"
cd $(git rev-parse --show-toplevel)/postprocess

## step 4.1: statistical analysis
echo "🚀step 4.1, start for statistical analysis"
mkdir -p ../statistics_results
python statistics.py \
    --ground-truth ../preprocess_dataset_overall/$DATASET \
    --inference-dir $INFERENCE_DIR \
    --videos $EVAL_DATASET \
    --output "../statistics_results/${STATS_OUTPUT}.json"
grep -qxF "/statistics_results/" ../.gitignore || echo -e "\n/statistics_results/" >> ../.gitignore
echo "✅step 4.1, statistical analysis is done"

## step 4.2: video generation
# skipped as key point is missing

### step 4.3 export to csv
echo "🚀step 4.3, start for csv export"
mkdir -p ../export_csv
python to_csv.py \
    --json_path "../preprocess_dataset_overall/$DATASET/${EVAL_DATASET}_results/statistics.json" \
    --output ../export_csv/$CSV_EXPORT
grep -qxF "/export_csv/" ../.gitignore || echo -e "\n/export_csv/" >> ../.gitignore
echo "✅step 4.3, csv export is done"
echo "✅step 4, postprocessing is done"