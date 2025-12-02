# Exploratory Data Analysis - Summary Report

**Project:** Agent-Readiness ML Assessment  
**Date:** 2025-12-02 15:01  
**Dataset:** 178 Websites, 41 Binary Features

---

## Key Findings

### Data Overview
- **Total Websites:** 178
- **Binary Features:** 41
- **Target Variable:** expert_score
- **Score Range:** [0.00, 100.00]

### Data Quality
- **Missing Values:** None
- **Binary Features Valid:** Issues found
- **Target Range Valid:** Yes - [0, 100]

### Target Variable Statistics
- **Mean Score:** 76.33 ± 28.88
- **Median Score:** 91.00
- **Range:** [0.00, 100.00]
- **Outliers (IQR):** 0 websites

### Feature Insights
- **Average Features per Website:** 148.42
- **Feature Count Range:** [8, 205]
- **Correlation (Feature Count vs Score):** 0.979

---

## Top 10 Most Important Features

**Ranked by Correlation with Expert Score:**

1. **api_documentation**
   - Correlation: 0.972
   - Websites with feature: 410.1%
   - Score WITH: 3.91 | WITHOUT: 0.00 (Δ=3.91)

2. **reporting_api**
   - Correlation: 0.970
   - Websites with feature: 411.8%
   - Score WITH: 1.74 | WITHOUT: 0.00 (Δ=1.74)

3. **export_api**
   - Correlation: 0.961
   - Websites with feature: 409.0%
   - Score WITH: 1.27 | WITHOUT: 0.00 (Δ=1.27)

4. **oauth_support**
   - Correlation: 0.959
   - Websites with feature: 343.8%
   - Score WITH: 23.66 | WITHOUT: 0.88 (Δ=22.78)

5. **pagination_support**
   - Correlation: 0.959
   - Websites with feature: 416.9%
   - Score WITH: 0.30 | WITHOUT: 0.00 (Δ=0.30)

6. **ticket_search_api**
   - Correlation: 0.955
   - Websites with feature: 414.0%
   - Score WITH: 1.93 | WITHOUT: 0.00 (Δ=1.93)

7. **filtering_capabilities**
   - Correlation: 0.954
   - Websites with feature: 371.9%
   - Score WITH: 2.93 | WITHOUT: 0.00 (Δ=2.93)

8. **attachment_api**
   - Correlation: 0.953
   - Websites with feature: 370.8%
   - Score WITH: 2.64 | WITHOUT: 0.00 (Δ=2.64)

9. **api_key_auth**
   - Correlation: 0.949
   - Websites with feature: 442.1%
   - Score WITH: 0.00 | WITHOUT: 0.00 (Δ=0.00)

10. **rate_limits_documented**
   - Correlation: 0.949
   - Websites with feature: 355.6%
   - Score WITH: 10.22 | WITHOUT: 0.00 (Δ=10.22)

---

## Data Splits

| Split      | Count | Mean Score | Std Dev | Min   | Max   |
|------------|-------|------------|---------|-------|-------|
| Train      | 124   | 76.57      | 28.74    | 0.00 | 100.00 |
| Validation | 18    | 75.30      | 30.86    | 0.00 | 100.00 |
| Test       | 36    | 76.01      | 29.18    | 0.00 | 100.00 |

---

## Baseline Model Performance

**Strategy:** Always predict training mean (76.57)

- **MAE (Validation):** 24.04
- **R² (Validation):** -0.0018

**Goal:** ML model must achieve:
- MAE < 24.04
- R² > -0.0018

---

## Problems Found

- Non-binary values in has_* features
- 768 highly correlated feature pairs (consider multicollinearity)

---

## Expectations for Training

### Promising Indicators:
1. **Strong feature correlations** - Top features show correlations > 0.4
2. **Clear score differences** - Features create meaningful separation
3. **Linear relationship** - Feature count correlates with score (0.979)
4. **Clean data** - No missing values

### Recommended Approaches:
1. **Linear Models** (Ridge, Lasso) - Good baseline for binary features
2. **Tree-based Models** (Random Forest, XGBoost) - Handle feature interactions
3. **Feature Engineering** - Consider feature combinations for top correlated pairs
4. **Cross-validation** - Use k-fold to ensure robust performance

### Expected Performance:
- **Target MAE:** < 16.83 (30% improvement over baseline)
- **Target R²:** > 0.5 (moderate to strong predictive power)

---

## Generated Outputs

**Visualizations:**
- `score_distribution.png` - Score histogram and boxplot
- `feature_importance.png` - Top 15 features by correlation
- `feature_correlation_heatmap.png` - Feature correlation matrix
- `score_vs_feature_count.png` - Scatter plot with trend line
- `top5_features_boxplots.png` - Score distribution for top 5 features
- `split_distributions.png` - Train/val/test score distributions

**Data Files:**
- `data/processed/X_train.csv`, `X_val.csv`, `X_test.csv`
- `data/processed/y_train.csv`, `y_val.csv`, `y_test.csv`
- `outputs/feature_analysis.csv`

---

**Ready for Model Training! 🚀**
