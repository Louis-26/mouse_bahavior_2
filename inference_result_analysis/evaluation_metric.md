# Background
positive class: `scratching`

negative class: `no behavior`

# evaluation metric

## confusion matrix
|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | True Positive (TP)  | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN)  |



## Accuracy
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

## Precision
$$Precision = \frac{TP}{TP + FP}$$

## Recall
$$Recall = \frac{TP}{TP + FN}$$

## F1 Score
$$F1 Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$


# core evaluation metric
As the original dataset is imbalanced, the core evaluation metric would be `F1 score` and `precision`.