We use focal loss to prioritize the weight of under represented `scratch` class


## step 1: execute the code
```bash
cd $(git rev-parse --show-toplevel)/SLURM_execution/SLURM_script
sbatch train_focal.sh
cd $(git rev-parse --show-toplevel)
grep -qxF "/video_segmentation_output_focal/" .gitignore || echo -e "\n/video_segmentation_output_focal/" >> .gitignore
```

## step 2: check the results

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


conclusion:
slightly better in f1 score, slightly worse in precision
the two results are very close