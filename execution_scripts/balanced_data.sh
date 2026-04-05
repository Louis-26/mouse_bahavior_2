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
else
    echo "Conda environment '$ENV_NAME' already exists. Skipping environment setup."
    conda activate $ENV_NAME
fi
echo "✅step 1.1, environment setup is done"

## step 1.2: dataset
# if ! command -v gdown &> /dev/null; then
#     pip install gdown
# fi

# if [ ! -d "dataset" ]; then
#     mkdir dataset
#     # Download the data from Google Drive and move it to the dataset folder
#     # You can use gdown or any other method to download the files
#     gdown --folder https://drive.google.com/drive/folders/1rm2sGUw9QVswf0HRhoUldcDObH3Tb06L?usp=drive_link 
#     for f in itch\ video\ analysisi_*.xlsx; do
#         mv "$f" "${f#itch video analysisi_}"
#     done    
# else
#     echo "Dataset already exists. Skipping download."
# fi
# echo "✅step 1.2, dataset preparation is done"

echo "✅step 1, environment configuration and data preparation is done"
# step 2: preprocess the data
echo "🚀step 2, start for data preprocessing"
## step 2.1: transform xlsx to csv
# echo "🚀step 2.1, start for xlsx to csv transformation"
# cd $(git rev-parse --show-toplevel)
# # mkdir -p preprocess_dataset
# cd preprocess
# python label_process.py --input_dir ../dataset/ --output_dir ../preprocess_dataset/
# grep -qxF "/preprocessed_data/" ../.gitignore || echo -e "\n/preprocessed_data/" >> ../.gitignore
# if [ -d "../preprocess_dataset" ] && ls ../preprocess_dataset/*.mp4 >/dev/null 2>&1; then
#     echo ".mp4 files exist, skipping copy."
# else
#     cp ../dataset/*.mp4 ../preprocess_dataset/
# fi
# echo "✅step 2.1, xlsx to csv transformation is done"
## step 2.2: video grid splitting

# right now this step is skipped as split_cell.py is missing, this is needed only for multiscreen task

## step 2.3: action segmentation
echo "🚀step 2.3, start for action segmentation"
cd $(git rev-parse --show-toplevel)
mkdir -p video_segmentation_output_balanced
cd preprocess
python action_segmentation.py \
    --video_dir ../balanced_dataset \
    --videos CQ_2.mp4 CQ_3.mp4 CQ_4.mp4 \
    --csvs CQ_2.csv CQ_3.csv CQ_4.csv \
    --output_dir ../video_segmentation_output_balanced \
    --split_mode video \
    --test_videos CQ_4
grep -qxF "/video_segmentation_output_balanced/" ../.gitignore || echo -e "\n/video_segmentation_output_balanced/" >> ../.gitignore
echo "✅step 2.3, action segmentation is done"

## step 2.4: advanced split
echo "🚀step 2.4, start for advanced split"
python advanced_split.py \
    --dataset_root ../video_segmentation_output_balanced \
    --train_videos CQ_2 CQ_3 \
    --val_videos CQ_4 \
    --split_mode video
echo "✅step 2.4, advanced split is done"
## step 2.5: feature extraction
# skip as keypoint is missing
echo "✅step 2, data preprocessing is done"

# step 3: action segmentation
echo "🚀step 3, start for action segmentation"
cd $(git rev-parse --show-toplevel)/action_seg

## step 3.1: train the model
echo "🚀step 3.1, start for model training"
mkdir -p ../train_result/resnet_only_balanced
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
    --dataset_root "../video_segmentation_output_balanced" \
    --output_dir "../train_result/resnet_only_balanced" \
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
echo "✅step 3.1, model training is done"

## step 3.2: model inference
echo "🚀step 3.2, start for model inference"
# option 2, without keypoint features
python inference_raw_video.py \
    --video_path ../balanced_dataset/CQ_4.mp4 \
    --checkpoint ../train_result/resnet_only_balanced/checkpoints/best.pth \
    --output_dir ../balanced_dataset/CQ_4_results \
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
    --ground-truth ../balanced_dataset \
    --videos CQ_4 \
    --output ../statistics_results/multi_video_results_balanced.json
grep -qxF "/statistics_results/" ../.gitignore || echo -e "\n/statistics_results/" >> ../.gitignore
echo "✅step 4.1, statistical analysis is done"

## step 4.2: video generation
# skipped as key point is missing

### step 4.3 export to csv
echo "🚀step 4.3, start for csv export"
mkdir -p ../export_csv
python to_csv.py \
    --json_path ../balanced_dataset/CQ_4_results/statistics.json \
    --output ../export_csv/CQ_4_export_balanced.csv
grep -qxF "/export_csv/" ../.gitignore || echo -e "\n/export_csv/" >> ../.gitignore
echo "✅step 4.3, csv export is done"
echo "✅step 4, postprocessing is done"