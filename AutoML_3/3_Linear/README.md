# Summary of 3_Linear

[<< Go back](../README.md)


## Logistic Regression (Linear)
- **n_jobs**: -1
- **num_class**: 7
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.75
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
logloss

## Training time

3.1 seconds

### Metric details
|           |       18 |        19 |        20 |        21 |        22 |       23 |   24 |   accuracy |   macro avg |   weighted avg |   logloss |
|:----------|---------:|----------:|----------:|----------:|----------:|---------:|-----:|-----------:|------------:|---------------:|----------:|
| precision | 1        |  0.464286 |  0.436364 |  0.435897 |  0.510638 | 0.6      |    0 |   0.477528 |    0.492455 |       0.467529 |   1.00858 |
| recall    | 0.8      |  0.317073 |  0.585366 |  0.435897 |  0.648649 | 0.333333 |    0 |   0.477528 |    0.44576  |       0.477528 |   1.00858 |
| f1-score  | 0.888889 |  0.376812 |  0.5      |  0.435897 |  0.571429 | 0.428571 |    0 |   0.477528 |    0.457371 |       0.462886 |   1.00858 |
| support   | 5        | 41        | 41        | 39        | 37        | 9        |    6 |   0.477528 |  178        |     178        |   1.00858 |


## Confusion matrix
|               |   Predicted as 18 |   Predicted as 19 |   Predicted as 20 |   Predicted as 21 |   Predicted as 22 |   Predicted as 23 |   Predicted as 24 |
|:--------------|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|
| Labeled as 18 |                 4 |                 0 |                 0 |                 1 |                 0 |                 0 |                 0 |
| Labeled as 19 |                 0 |                13 |                25 |                 3 |                 0 |                 0 |                 0 |
| Labeled as 20 |                 0 |                13 |                24 |                 4 |                 0 |                 0 |                 0 |
| Labeled as 21 |                 0 |                 2 |                 6 |                17 |                13 |                 1 |                 0 |
| Labeled as 22 |                 0 |                 0 |                 0 |                13 |                24 |                 0 |                 0 |
| Labeled as 23 |                 0 |                 0 |                 0 |                 0 |                 6 |                 3 |                 0 |
| Labeled as 24 |                 0 |                 0 |                 0 |                 1 |                 4 |                 1 |                 0 |

## Learning curves
![Learning curves](learning_curves.png)

## Coefficients

### Coefficients learner #1
|                              |         18 |         19 |          20 |         21 |         22 |         23 |           24 |
|:-----------------------------|-----------:|-----------:|------------:|-----------:|-----------:|-----------:|-------------:|
| intercept                    | -2.53565   |  1.50108   |  0.53205    |  2.38185   |  0.267594  | -0.901646  | -1.24527     |
| Student_ID                   | -2.03283   | -0.743373  | -0.802259   |  0.113828  |  0.29533   |  1.44359   |  1.72572     |
| Gender                       |  0.233255  | -0.440004  | -0.773766   |  0.568815  |  0.605607  | -0.289169  |  0.095262    |
| Academic_Level               |  0.160301  |  2.32306   |  3.17668    |  0.0502488 | -2.18647   | -1.82926   | -1.69456     |
| Country                      | -0.523066  | -0.0254075 | -0.00998646 | -0.0264324 | -0.077061  |  0.407737  |  0.254217    |
| Avg_Daily_Usage_Hours        | -0.0128633 |  0.935849  |  0.390234   |  0.220672  | -0.0835368 | -0.941029  | -0.509326    |
| Most_Used_Platform           |  0.222073  | -0.127191  | -0.0307777  | -0.113387  |  0.129243  | -0.0794925 | -0.000467679 |
| Affects_Academic_Performance | -0.441218  |  0.606891  |  0.939032   |  0.0727404 | -0.79689   | -0.090968  | -0.289587    |
| Sleep_Hours_Per_Night        | -0.765488  |  1.08129   |  0.895395   |  0.511329  |  0.102034  | -1.03309   | -0.79147     |
| Mental_Health_Score          | -0.888252  |  0.40238   |  0.666022   |  0.706241  |  0.136291  | -0.459629  | -0.563052    |
| Relationship_Status          |  0.390946  | -0.239669  | -0.458379   | -0.0443104 |  0.176126  | -0.0733932 |  0.24868     |
| Conflicts_Over_Social_Media  |  0.225325  |  0.535359  |  0.611332   | -0.25734   |  0.212126  | -0.617481  | -0.709321    |
| Addicted_Score               |  0.491081  | -0.359605  | -0.309031   |  1.05295   |  0.270663  | -0.67377   | -0.472284    |


## Permutation-based Importance
![Permutation-based Importance](permutation_importance.png)
## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)


## Normalized Confusion Matrix

![Normalized Confusion Matrix](confusion_matrix_normalized.png)


## ROC Curve

![ROC Curve](roc_curve.png)


## Precision Recall Curve

![Precision Recall Curve](precision_recall_curve.png)



[<< Go back](../README.md)
