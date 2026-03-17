# UMich CQ_4
CUDA_VISIBLE_DEVICES=0 \
python inference_raw_video.py \
 --video_path /data/zhaozhenghao/Projects/Mouse/datasets/UMich_CQ/CQ_4.mp4 \
 --checkpoint /data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/action_seg_training/all_video_80triain_20val/checkpoints/best.pth \
 --output_dir /data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/video_inference/CQ_4_850-1150 \
 --keypoint_dir /data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/keypoint \
 --ground_truth /data/zhaozhenghao/Projects/Mouse/datasets/UMich_CQ/CQ_4.csv \
 --start_time 850 \
 --end_time 1150 \
 --save_video 

# Run statistics.py for the output results
python ../postprocess/statistics.py \
 --inference-dir /data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/video_inference/CQ_4_850-1150 \
 --ground-truth /data/zhaozhenghao/Projects/Mouse/datasets/UMich_CQ/CQ_4.csv \
 --output /data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/video_inference/CQ_4_850-1150/comparison_results.json