We use downsampling to decrease the time of `no behavior` for both CQ_2.csv and CQ_3.csv, so that the percentage of `scratching` approaches 45-50%.

## step 1: prepare
use [down-sampling script](../data_augmentation/down_sample.py)
then create downsampled dataset
```bash
cd $(git rev-parse --show-toplevel)/data_augmentation
python down_sample.py 
grep -qxF "/down_sample_dataset/" ../.gitignore || echo -e "\n/down_sample_dataset/" >> ../.gitignore
```

## step 2: execute the code
```bash
cd $(git rev-parse --show-toplevel)/SLURM_execution/SLURM_script
sbatch down_sample.sh
cd $(git rev-parse --show-toplevel)
grep -qxF "/video_segmentation_output_down/" .gitignore || echo -e "\n/video_segmentation_output_down/" >> .gitignore
```

## step 3: check the results

```bash
cd $(git rev-parse --show-toplevel)/statistics_results
python -c '
import json

with open("multi_video_results.json") as f:
    data_baseline = json.load(f)

with open("multi_video_results_focal.json") as f:
    data_new = json.load(f)

print("precision from baseline to focal:", data_baseline["frame_accuracy"]["precision"], "->", data_new["frame_accuracy"]["precision"])
print("f1_score from baseline to focal:", data_baseline["frame_accuracy"]["f1_score"], "->", data_new["frame_accuracy"]["f1_score"])
'
```

result
```text
precision from baseline to focal: 0.21800760215394363 -> 0.2175413822860281
f1_score from baseline to focal: 0.30158295448321193 -> 0.3018067043970396
```