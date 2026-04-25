## task
Training performance for each algorithm​

Finetune focal loss parameter​

Down sampling recheck, (to two times of scratching)​

Colab setup+GPU memory check​



## finished work
✅get result for CQ_2.csv and CQ_3.csv, not sure why, but always scracthing as the prediction, meaning that:
no behavior is never detected, and we have 
902.7s for no behavior, 55.7s for scracthing, and 958.4s for total length

confusion matrix:
|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | 55.7  | 0 |
| Actual Negative | 902.7 | 0 |

- accuracy = 0.058
- precision = 1
- recall = 0.062
- F1 score = 0.117

conclusion: It is quite weird, meaning that the model even can't predict the training set. We need a fundamental change towards the model.
-------------------------------------------------------
✅finetune focal loss parameter, 8.0

- accuracy = 0.7830323436384042
- precision = 0.2184489022271363
- recall = 0.49138390477882393
- F1 score = 0.3024438248318845

almost no difference with previous one
-----------------------------------------------------------
down sample redo, with no behavior of two times of scratching,
process dataset 
```bash
cd $(git rev-parse --show-toplevel)/data_augmentation
conda activate mouse_behavior
python down_sample_2.py --input_dir ../preprocess_dataset_overall/dataset --output_dir ../preprocess_dataset_overall/tmp
```

then get results
- accuracy = 0.4770431588613407,
- precision = 0.13180725716630518,
- recall = 0.7988985610232724,
- F1 score = 0.2262812287719828

still quite bad
--------------------------------------------------------------
✅GPU memory checked
  GPU Memory allocated: 0.13 GB
  GPU Memory cached: 0.67 GB
next step: enlarge batch size
------------------------------------------------------------
❌for colab, unable to create conda environment, thus unable to run the code

## next step
need a way to finetune focal loss parameter, cross validation?