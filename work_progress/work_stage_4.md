## task
1. investigate robust methods/off-the-shelf methods for class imbalance issue

2. get loss curve

3. create PPT slide for work summarization

4. conquer imbalance issue techniques
- up sampling
- down sampling
- focal loss 


## progress
1. use the following strategies to handle data imbalance issue and recompute the loss
- up sample `scratch` time in `CQ_2.csv` and `CQ_3.csv`, and keep `CQ_4.csv` as what it was, summarized at [here](../data_augmentation/up_sampling.md)
- down sample `no behavior` time in `CQ_2.csv` and `CQ_3.csv`, and keep `CQ_4.csv` as what it was, summarized at [here](../data_augmentation/down_sampling.md)
- consider focal loss, but keep everything else the same, summarized at [here](../data_augmentation/focal_loss.md)

2. create a PPT slide to summarize the current work, [here](../presentation/result_summary.pptx)

## Next step
As the folder number has increased, I will make it more structured
├── dataset
|   ├── preprocess_dataset
|   ├── up_sample_dataset
|   ├── down_sample_dataset