## background information
- target video: `CQ_4.mp4`
- video length: 32 min 40 s
- no behavior percentage: 78.52%
- scratching percentage: 21.48%
- number of frame: 58806
- number of clips: 229
- frame per clip: 257
- prediction time: 58806(meaning one prediction per frame)

## baseline confusion matrix results 
"confusion_matrix": {
      "true_positive": 2753,
      "true_negative": 43302,
      "false_positive": 9875,
      "false_negative": 2876
    },

|               | Predicted `scratching` | Predicted `no behavior` |
|---------------|------------------------|-------------------------|
| Actual `scratching` |  2753  | 2876 |
| Actual `no behavior` | 9875 | 43302 |


## accuracy in no behavior
Among all actual `no behavior` time, the rate of correctly predicted `no behavior` time  can be computed as follows
$\text{Accuracy} = \frac{\text{TN}}{\text{TN + FP}}$
in baseline model, it is `43302/(43302 + 9875) = 0.814`

## accuracy in scratching
Among all actual `scratching` time, the rate of correctly predicted `scratching` time can be computed as follows
$\text{Accuracy} = \frac{\text{TP}}{\text{TP + FN}}$
in baseline model, it is `2753/(2753 + 2876) = 0.489`


# F1 score 
The F1 score is the harmonic mean of precision and recall, which can be computed as follows:  
$$F1 Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}=0.3016$$