We use upsampling to increase the time of `scratching` both CQ_2.csv and CQ_3.csv, so that the percentage of `scratching` approaches 45-50%.

## step 1: prepare
use [up-sampling script](../data_augmentation/up_sample.py)
then create upsampled dataset
```bash
cd $(git rev-parse --show-toplevel)/data_augmentation
python up_sample.py 
grep -qxF "/up_sample_dataset/" ../.gitignore || echo -e "\n/up_sample_dataset/" >> ../.gitignore
```

## step 2: execute the code
```bash
cd $(git rev-parse --show-toplevel)/SLURM_execution/SLURM_script
sbatch train_focal.sh
cd $(git rev-parse --show-toplevel)
grep -qxF "/video_segmentation_output_up/" .gitignore || echo -e "\n/video_segmentation_output_up/" >> .gitignore
```

## step 3: check the results

```bash
cd $(git rev-parse --show-toplevel)/statistics_results
python -c '
import json

with open("multi_video_results.json") as f:
    data = json.load(f)

print("precision:", data["frame_accuracy"]["precision"])
print("f1_score:", data["frame_accuracy"]["f1_score"])
'
```