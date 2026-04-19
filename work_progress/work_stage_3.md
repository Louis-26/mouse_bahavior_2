## Task finished
1. finish a bash script to quickly set up the environment
2. analyze classification metric results for different classes, `no behavior` and `scratching`, 

3. use data augmentation(downsampling, upsampling) to balance the dataset, and retrain the model, then compare the results with the original one.

4. figure out the number of clips for each class in the training set


If time permits & Next steps:
5. give a more comprehensive data analysis, and corresponding feature size per frame

6. give visualization of the classification results, compare with the ground truth

7. get the plot of training loss versus iteration

8. get the illustration document of model for classification per clip

## Progress
1. ✅[here](../execution_scripts/setup_env.sh) 

2. ✅[here](../inference_result_analysis/baseline_results.md)

3. ✅get a new training dataset, from script [here](../data_augmentation/data_balance.py), ⏳training and inference in progress from [here](../execution_scripts/balanced_data.sh)

4. ✅1 prediction per frame