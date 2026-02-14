# Summary of Ensemble

[<< Go back](../README.md)


## Ensemble structure
| Model                   |   Weight |
|:------------------------|---------:|
| 4_Default_Xgboost       |        5 |
| 5_Default_NeuralNetwork |        1 |

## Metric details
|           |    score |    threshold |
|:----------|---------:|-------------:|
| logloss   | 0.177868 | nan          |
| auc       | 0.981486 | nan          |
| f1        | 0.933333 |   0.355478   |
| accuracy  | 0.932203 |   0.355478   |
| precision | 0.949367 |   0.713068   |
| recall    | 1        |   0.00255891 |
| mcc       | 0.865321 |   0.355478   |


## Metric details with threshold from accuracy metric
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.177868 |  nan        |
| auc       | 0.981486 |  nan        |
| f1        | 0.933333 |    0.355478 |
| accuracy  | 0.932203 |    0.355478 |
| precision | 0.913043 |    0.355478 |
| recall    | 0.954545 |    0.355478 |
| mcc       | 0.865321 |    0.355478 |


## Confusion matrix (at threshold=0.355478)
|                   |   Predicted as Female |   Predicted as Male |
|:------------------|----------------------:|--------------------:|
| Labeled as Female |                    81 |                   8 |
| Labeled as Male   |                     4 |                  84 |

## Learning curves
![Learning curves](learning_curves.png)
## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)


## Normalized Confusion Matrix

![Normalized Confusion Matrix](confusion_matrix_normalized.png)


## ROC Curve

![ROC Curve](roc_curve.png)


## Kolmogorov-Smirnov Statistic

![Kolmogorov-Smirnov Statistic](ks_statistic.png)


## Precision-Recall Curve

![Precision-Recall Curve](precision_recall_curve.png)


## Calibration Curve

![Calibration Curve](calibration_curve_curve.png)


## Cumulative Gains Curve

![Cumulative Gains Curve](cumulative_gains_curve.png)


## Lift Curve

![Lift Curve](lift_curve.png)



[<< Go back](../README.md)
