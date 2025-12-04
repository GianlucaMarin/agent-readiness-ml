---
COMPREHENSIVE TEST SET EVALUATION REPORT
Agent Readiness ML - Random Forest Model
Generated: 2025-12-04 19:39:15
---

## 1. MODEL INFORMATION

Model Type: RandomForestRegressor
Model Path: models/random_forest_initial.joblib
Number of Features: 41
Training Hyperparameters:
  - n_estimators: 200
  - max_depth: 15
  - min_samples_leaf: 3

## 2. DATA SPLIT SUMMARY

Training Set:    124 samples (69.7%)
Validation Set:   18 samples (10.1%)
Test Set:         36 samples (20.2%) [HELD-OUT]
Total:           178 samples

## 3. PERFORMANCE METRICS (ALL SPLITS)


     Split  N_Samples      MAE     RMSE       R²  MAPE (%)  Median_AE  Max_Error  Mean_Actual  Mean_Predicted  Std_Actual  Std_Predicted
     Train        124 0.597477 1.458503 0.996201  4.464805   0.202526  12.346021    72.508065       72.568167   23.661762      23.268621
Validation         18 1.086759 2.210133 0.994355  8.953203   0.202526   7.265104    67.555556       68.354187   29.416968      27.712570
      Test         36 0.635896 1.543371 0.995877  4.048789   0.202526   8.265104    73.277778       73.434469   24.035403      23.348247


## 4. TEST SET PERFORMANCE BY SCORE RANGE


Split    Score_Range  N_Samples  Percentage      MAE     RMSE       R²  Median_AE  Mean_Actual  Mean_Predicted
 Test     Low (0-30)          2    5.555556 5.095298 6.000811 0.601000   5.095298    16.500000       19.669806
 Test Medium (30-70)         10   27.777778 0.581174 0.817596 0.986792   0.211425    48.300000       48.593788
 Test  High (70-100)         24   66.666667 0.287080 0.541897 0.996337   0.202526    88.416667       88.265142


## 5. CONFUSION MATRIX (TEST SET - CATEGORICAL)

Categories: Low (0-30), Medium (30-70), High (70-100)

Absolute Counts:
              Pred: Low  Pred: Medium  Pred: High
True: Low             2             0           0
True: Medium          0            10           0
True: High            0             0          24

Normalized (Row %):
              Pred: Low  Pred: Medium  Pred: High
True: Low         100.0           0.0         0.0
True: Medium        0.0         100.0         0.0
True: High          0.0           0.0       100.0

Category-wise Accuracy:
  Low     :  100.0%
  Medium  :  100.0%
  High    :  100.0%


## 6. RESIDUAL ANALYSIS (TEST SET)

Mean Residual:          -0.1567
Std Residual:            1.5354
Min Residual:           -8.2651
Max Residual:            1.9255
Median Residual:         0.0854

Statistical Tests:
  Shapiro-Wilk (Normality):       W=0.5191, p=0.0000
    → Residuals NOT normal (α=0.05)
  Levene (Heteroscedasticity):    W=2.7859, p=0.0762
    → Homoscedastic (α=0.05)
  Autocorrelation (Lag-1):        r=-0.1057


## 7. TOP 10 WORST PREDICTIONS (TEST SET)


 Rank  Actual_Score  Predicted_Score  Abs_Error Actual_Category Predicted_Category
    1             7        15.265104   8.265104             Low                Low
    2            26        24.074507   1.925493             Low                Low
    3            99        97.141885   1.858115            High               High
    4            50        51.539729   1.539729          Medium             Medium
    5            78        79.489099   1.489099            High               High
    6            43        44.399522   1.399522          Medium             Medium
    7            53        54.194559   1.194559          Medium             Medium
    8            36        35.116658   0.883342          Medium             Medium
    9            99        98.227744   0.772256            High               High
   10            80        79.392376   0.607624            High               High


## 8. KEY INSIGHTS & RECOMMENDATIONS

✓ EXCELLENT: Model shows strong predictive performance (R² > 0.8)
✓ EXCELLENT: Very low prediction error (MAE < 10)
✓ GOOD: Model generalizes well (minimal overfitting)
✓ EXCELLENT: High categorical accuracy (100.0%)

Recommendations:
  1. Review top 10 worst predictions for data quality issues
  2. Consider feature engineering for underperforming score ranges
  3. Monitor for heteroscedasticity in production predictions
  4. Implement prediction confidence intervals in deployment


## 9. GENERATED ARTIFACTS

CSV Files:
  • metrics_comparison_all_splits.csv
  • performance_by_score_range.csv
  • confusion_matrix.csv
  • residual_statistics.csv
  • worst_10_predictions.csv

Visualizations:
  • predicted_vs_actual_with_ci.png
  • performance_by_score_range.png
  • confusion_matrix_categories.png
  • residual_analysis_comprehensive.png
  • residual_distribution.png
  • worst_10_predictions_analysis.png

---
END OF REPORT
---