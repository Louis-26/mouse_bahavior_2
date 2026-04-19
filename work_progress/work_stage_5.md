## next step
try the following strategies again
- up sample
- down sample
- focal loss
- prediction based on sample(10 frames per sample), only compute one sigmoid output


## finished task
1. do file organization, create folder to combine results from different strategies, and move files to the corresponding folders
- `dataset` -----> `orig_dataset`
- `dataset`, `down_sample_dataset`, `up_sample_dataset` -----> `dataset_overall`
- `preprocess_dataset` -----> `preprocess_dataset_overall`
- `video_segmentation_output`, `video_segmentation_output_focal`, `video_segmentation_output_down`, `video_segmentation_output_up` -----> `video_segmentation_output_overall`
- `video_generation_output` -----> `video_generation_output_overall`