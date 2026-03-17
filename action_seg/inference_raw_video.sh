# UMich CQ_4
CUDA_VISIBLE_DEVICES=1 \
python inference_raw_video.py \
 --video_path /data/zhaozhenghao/Projects/Mouse/datasets/UMich_CQ/CQ_4.mp4 \
 --checkpoint /data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/action_seg_training/all_video_80triain_20val/checkpoints/best.pth \
 --output_dir /data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/video_inference/CQ_4_1150-1330 \
 --keypoint_dir /data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/keypoint \
 --ground_truth /data/zhaozhenghao/Projects/Mouse/datasets/UMich_CQ/CQ_4.csv \
 --start_time 1150 \
 --end_time 1330 \
 --save_video 

# UMich Mechanical Itch Video
# CUDA_VISIBLE_DEVICES=1 \
# python inference_raw_video.py \
#  --video_path /data/zhaozhenghao/Projects/Mouse/datasets/UMich_Mechanical_Itch_Video/Mechanical_Itch_video_1.mp4 \
#  --checkpoint /data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/action_seg_training/all_train_with_keypoints/checkpoints/epoch_10.pth \
#  --output_dir /data/zhaozhenghao/Projects/Mouse/results/UMich_Mechanical_Itch_Video/video_inference/Mechanical_Itch_video_1 \
#  --keypoint_dir /data/zhaozhenghao/Projects/Mouse/results/UMich_Mechanical_Itch_Video/keypoint \
#  --use_keypoints \
#  --start_time 0 \
#  --save_video