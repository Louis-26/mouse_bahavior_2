```bash
python utils/compute_proportion.py --dataset_folder preprocess_dataset_overall/dataset/
python utils/compute_proportion.py --dataset_folder preprocess_dataset_overall/down_sample_dataset/
python utils/compute_proportion.py --dataset_folder preprocess_dataset_overall/up_sample_dataset/

```

## original 
CQ_2.csv:
  no behavior: 2965.960s (87.48%)
  scracthing: 424.306s (12.52%)
  total length: 3390.266s

CQ_3.csv:
  no behavior: 1785.471s (98.56%)
  scracthing: 26.163s (1.44%)
  total length: 1811.634s

CQ_4.csv:
  no behavior: 1766.857s (90.14%)
  scracthing: 193.372s (9.86%)
  total length: 1960.229s

overall time proportion:
  no behavior: 6518.288s (91.01%)
  scracthing: 643.841s (8.99%)
  total length: 7162.129s

## upsampling
CQ_2.csv:
  no behavior: 3016.048s (46.98%)
  scracthing: 3403.592s (53.02%)
  total length: 6419.640s

CQ_3.csv:
  no behavior: 1795.526s (50.00%)
  scracthing: 1795.860s (50.00%)
  total length: 3591.386s

CQ_4.csv:
  no behavior: 1766.857s (90.14%)
  scracthing: 193.372s (9.86%)
  total length: 1960.229s

overall time proportion:
  no behavior: 6578.431s (54.95%)
  scracthing: 5392.824s (45.05%)
  total length: 11971.255s



## downsampling
CQ_2.csv:
  no behavior: 198.070s (50.00%)
  scracthing: 198.070s (50.00%)
  total length: 396.140s

CQ_3.csv:
  no behavior: 194.159s (50.00%)
  scracthing: 194.159s (50.00%)
  total length: 388.318s

CQ_4.csv:
  no behavior: 1766.857s (90.14%)
  scracthing: 193.372s (9.86%)
  total length: 1960.229s

overall time proportion:
  no behavior: 2159.086s (78.66%)
  scracthing: 585.601s (21.34%)
  total length: 2744.687s