"""
Comprehensive Test Set Evaluation Script for Agent Readiness ML
Author: AI Engineering Team
Date: 2025-12-04

This script performs enterprise-grade model evaluation on the held-out test set.
It includes:
- All regression metrics (MAE, RMSE, R², MAPE, Median AE)
- Performance analysis by score ranges
- Confusion matrix for categorical scoring
- Comprehensive residual analysis
- Statistical hypothesis testing
- Professional visualizations with confidence intervals
- Detailed error analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    confusion_matrix
)
from scipy import stats
from scipy.stats import shapiro, levene
import joblib
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Professional visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = Path('outputs/test_evaluation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = 'models/random_forest_initial.joblib'
DATA_DIR = Path('data/processed')

# Score range boundaries
SCORE_RANGES = {
    'Low (0-30)': (0, 30),
    'Medium (30-70)': (30, 70),
    'High (70-100)': (70, 100)
}

print("=" * 80)
print("COMPREHENSIVE TEST SET EVALUATION")
print("Enterprise-Grade Model Performance Analysis")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD MODEL AND DATA
# ============================================================================
print("\n[STEP 1/10] Loading model and data...")

# Load model
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
feature_names = model_data['feature_names']

# Load all splits
X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
X_val = pd.read_csv(DATA_DIR / 'X_val.csv')
X_test = pd.read_csv(DATA_DIR / 'X_test.csv')

y_train = pd.read_csv(DATA_DIR / 'y_train.csv').values.ravel()
y_val = pd.read_csv(DATA_DIR / 'y_val.csv').values.ravel()
y_test = pd.read_csv(DATA_DIR / 'y_test.csv').values.ravel()

print(f"✓ Model loaded: {MODEL_PATH}")
print(f"  Features: {len(feature_names)}")
print(f"  Model type: {type(model).__name__}")
print(f"\n✓ Data loaded:")
print(f"  Train: {len(y_train)} samples")
print(f"  Val:   {len(y_val)} samples")
print(f"  Test:  {len(y_test)} samples (HELD-OUT)")

# ============================================================================
# STEP 2: GENERATE PREDICTIONS
# ============================================================================
print("\n[STEP 2/10] Generating predictions on all splits...")

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

print(f"✓ Predictions generated for all splits")

# ============================================================================
# STEP 3: CALCULATE ALL REGRESSION METRICS
# ============================================================================
print("\n[STEP 3/10] Calculating comprehensive regression metrics...")

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error, handling zero values."""
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_all_metrics(y_true, y_pred, split_name):
    """Calculate all regression metrics for a given split."""
    metrics = {
        'Split': split_name,
        'N_Samples': len(y_true),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred),
        'MAPE (%)': calculate_mape(y_true, y_pred),
        'Median_AE': median_absolute_error(y_true, y_pred),
        'Max_Error': np.max(np.abs(y_true - y_pred)),
        'Mean_Actual': np.mean(y_true),
        'Mean_Predicted': np.mean(y_pred),
        'Std_Actual': np.std(y_true),
        'Std_Predicted': np.std(y_pred)
    }
    return metrics

# Calculate for all splits
metrics_train = calculate_all_metrics(y_train, y_train_pred, 'Train')
metrics_val = calculate_all_metrics(y_val, y_val_pred, 'Validation')
metrics_test = calculate_all_metrics(y_test, y_test_pred, 'Test')

# Create comparison dataframe
metrics_comparison = pd.DataFrame([metrics_train, metrics_val, metrics_test])

print("✓ Metrics calculated for all splits:")
print("\n" + "=" * 80)
print(metrics_comparison.to_string(index=False))
print("=" * 80)

# Save metrics
metrics_comparison.to_csv(OUTPUT_DIR / 'metrics_comparison_all_splits.csv', index=False)
print(f"\n✓ Saved: {OUTPUT_DIR / 'metrics_comparison_all_splits.csv'}")

# ============================================================================
# STEP 4: PERFORMANCE BY SCORE RANGE
# ============================================================================
print("\n[STEP 4/10] Analyzing performance by score ranges...")

def analyze_score_ranges(y_true, y_pred, ranges, split_name):
    """Analyze performance for different score ranges."""
    results = []

    for range_name, (min_score, max_score) in ranges.items():
        mask = (y_true >= min_score) & (y_true < max_score)
        n_samples = mask.sum()

        if n_samples > 0:
            y_true_range = y_true[mask]
            y_pred_range = y_pred[mask]

            results.append({
                'Split': split_name,
                'Score_Range': range_name,
                'N_Samples': n_samples,
                'Percentage': (n_samples / len(y_true)) * 100,
                'MAE': mean_absolute_error(y_true_range, y_pred_range),
                'RMSE': np.sqrt(mean_squared_error(y_true_range, y_pred_range)),
                'R²': r2_score(y_true_range, y_pred_range),
                'Median_AE': median_absolute_error(y_true_range, y_pred_range),
                'Mean_Actual': np.mean(y_true_range),
                'Mean_Predicted': np.mean(y_pred_range)
            })

    return pd.DataFrame(results)

# Analyze all splits
range_analysis_train = analyze_score_ranges(y_train, y_train_pred, SCORE_RANGES, 'Train')
range_analysis_val = analyze_score_ranges(y_val, y_val_pred, SCORE_RANGES, 'Validation')
range_analysis_test = analyze_score_ranges(y_test, y_test_pred, SCORE_RANGES, 'Test')

range_analysis_all = pd.concat([range_analysis_train, range_analysis_val, range_analysis_test],
                                 ignore_index=True)

print("✓ Performance by score range:")
print("\n" + range_analysis_all.to_string(index=False))

# Save range analysis
range_analysis_all.to_csv(OUTPUT_DIR / 'performance_by_score_range.csv', index=False)
print(f"\n✓ Saved: {OUTPUT_DIR / 'performance_by_score_range.csv'}")

# Visualize performance by range
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Test set only for main visualization
test_ranges = range_analysis_test.copy()

# Plot 1: Sample distribution
ax = axes[0, 0]
ax.bar(test_ranges['Score_Range'], test_ranges['N_Samples'],
       color='steelblue', edgecolor='black', alpha=0.7)
ax.set_title('Test Set: Sample Distribution by Score Range', fontweight='bold')
ax.set_ylabel('Number of Samples')
ax.set_xlabel('Score Range')
ax.grid(axis='y', alpha=0.3)

# Plot 2: MAE by range
ax = axes[0, 1]
ax.bar(test_ranges['Score_Range'], test_ranges['MAE'],
       color='coral', edgecolor='black', alpha=0.7)
ax.set_title('Test Set: MAE by Score Range', fontweight='bold')
ax.set_ylabel('Mean Absolute Error')
ax.set_xlabel('Score Range')
ax.grid(axis='y', alpha=0.3)

# Plot 3: R² by range
ax = axes[1, 0]
ax.bar(test_ranges['Score_Range'], test_ranges['R²'],
       color='mediumseagreen', edgecolor='black', alpha=0.7)
ax.set_title('Test Set: R² by Score Range', fontweight='bold')
ax.set_ylabel('R² Score')
ax.set_xlabel('Score Range')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax.grid(axis='y', alpha=0.3)

# Plot 4: Mean predictions vs actual
ax = axes[1, 1]
x_pos = np.arange(len(test_ranges))
width = 0.35
ax.bar(x_pos - width/2, test_ranges['Mean_Actual'], width,
       label='Mean Actual', color='steelblue', edgecolor='black', alpha=0.7)
ax.bar(x_pos + width/2, test_ranges['Mean_Predicted'], width,
       label='Mean Predicted', color='orange', edgecolor='black', alpha=0.7)
ax.set_title('Test Set: Mean Scores by Range', fontweight='bold')
ax.set_ylabel('Mean Score')
ax.set_xlabel('Score Range')
ax.set_xticks(x_pos)
ax.set_xticklabels(test_ranges['Score_Range'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'performance_by_score_range.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR / 'performance_by_score_range.png'}")

# ============================================================================
# STEP 5: CONFUSION MATRIX FOR SCORE CATEGORIES
# ============================================================================
print("\n[STEP 5/10] Creating confusion matrix for score categories...")

def categorize_scores(scores):
    """Categorize scores into Low/Medium/High."""
    categories = np.empty(len(scores), dtype=object)
    categories[(scores >= 0) & (scores < 30)] = 'Low'
    categories[(scores >= 30) & (scores < 70)] = 'Medium'
    categories[(scores >= 70) & (scores <= 100)] = 'High'
    return categories

# Categorize test set
y_test_cat = categorize_scores(y_test)
y_test_pred_cat = categorize_scores(y_test_pred)

# Create confusion matrix
cm = confusion_matrix(y_test_cat, y_test_pred_cat, labels=['Low', 'Medium', 'High'])
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Calculate category-wise accuracy
category_accuracy = np.diag(cm) / cm.sum(axis=1)

print("✓ Confusion Matrix (Test Set):")
print("\nAbsolute counts:")
cm_df = pd.DataFrame(cm,
                     index=['True: Low', 'True: Medium', 'True: High'],
                     columns=['Pred: Low', 'Pred: Medium', 'Pred: High'])
print(cm_df)

print("\nNormalized (row percentages):")
cm_norm_df = pd.DataFrame(cm_normalized * 100,
                          index=['True: Low', 'True: Medium', 'True: High'],
                          columns=['Pred: Low', 'Pred: Medium', 'Pred: High'])
print(cm_norm_df.round(1))

print("\nCategory-wise Accuracy:")
for i, cat in enumerate(['Low', 'Medium', 'High']):
    print(f"  {cat}: {category_accuracy[i]*100:.1f}%")

# Visualize confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Absolute counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix: Absolute Counts', fontweight='bold')
axes[0].set_ylabel('True Category')
axes[0].set_xlabel('Predicted Category')

# Normalized percentages
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            ax=axes[1], cbar_kws={'label': 'Proportion'})
axes[1].set_title('Confusion Matrix: Normalized (Row %)', fontweight='bold')
axes[1].set_ylabel('True Category')
axes[1].set_xlabel('Predicted Category')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix_categories.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR / 'confusion_matrix_categories.png'}")

# Save confusion matrix data
cm_results = {
    'confusion_matrix': cm,
    'confusion_matrix_normalized': cm_normalized,
    'category_accuracy': dict(zip(['Low', 'Medium', 'High'], category_accuracy))
}
pd.DataFrame(cm).to_csv(OUTPUT_DIR / 'confusion_matrix.csv')
print(f"✓ Saved: {OUTPUT_DIR / 'confusion_matrix.csv'}")

# ============================================================================
# STEP 6: COMPREHENSIVE RESIDUAL ANALYSIS
# ============================================================================
print("\n[STEP 6/10] Performing comprehensive residual analysis...")

# Calculate residuals for test set
residuals_test = y_test - y_test_pred

# Statistical tests
print("\n✓ Residual Statistics (Test Set):")
print(f"  Mean residual: {np.mean(residuals_test):.4f}")
print(f"  Std residual: {np.std(residuals_test):.4f}")
print(f"  Min residual: {np.min(residuals_test):.4f}")
print(f"  Max residual: {np.max(residuals_test):.4f}")
print(f"  Median residual: {np.median(residuals_test):.4f}")

# Shapiro-Wilk test for normality
shapiro_stat, shapiro_p = shapiro(residuals_test)
print(f"\n✓ Shapiro-Wilk Test (Normality):")
print(f"  Statistic: {shapiro_stat:.4f}")
print(f"  p-value: {shapiro_p:.4f}")
print(f"  Result: {'Residuals appear normal' if shapiro_p > 0.05 else 'Residuals NOT normal'} (α=0.05)")

# Test for heteroscedasticity (Levene's test across predicted score ranges)
pred_terciles = pd.qcut(y_test_pred, q=3, labels=['Low', 'Med', 'High'])
groups = [residuals_test[pred_terciles == label] for label in ['Low', 'Med', 'High']]
levene_stat, levene_p = levene(*groups)
print(f"\n✓ Levene's Test (Heteroscedasticity):")
print(f"  Statistic: {levene_stat:.4f}")
print(f"  p-value: {levene_p:.4f}")
print(f"  Result: {'Homoscedastic' if levene_p > 0.05 else 'Heteroscedastic'} (α=0.05)")

# Calculate autocorrelation (if residuals are ordered by index)
autocorr = np.corrcoef(residuals_test[:-1], residuals_test[1:])[0, 1]
print(f"\n✓ Autocorrelation of Residuals:")
print(f"  Lag-1 correlation: {autocorr:.4f}")

# Save residual statistics
residual_stats = {
    'mean': np.mean(residuals_test),
    'std': np.std(residuals_test),
    'min': np.min(residuals_test),
    'max': np.max(residuals_test),
    'median': np.median(residuals_test),
    'shapiro_statistic': shapiro_stat,
    'shapiro_pvalue': shapiro_p,
    'levene_statistic': levene_stat,
    'levene_pvalue': levene_p,
    'autocorr_lag1': autocorr
}
pd.DataFrame([residual_stats]).to_csv(OUTPUT_DIR / 'residual_statistics.csv', index=False)
print(f"\n✓ Saved: {OUTPUT_DIR / 'residual_statistics.csv'}")

# ============================================================================
# STEP 7: VISUALIZATIONS - PREDICTED VS ACTUAL
# ============================================================================
print("\n[STEP 7/10] Creating Predicted vs Actual visualizations...")

# Calculate prediction intervals (using standard deviation as proxy)
# In a real scenario, you'd use quantile regression or bootstrapping
prediction_std = np.std(residuals_test)
lower_bound = y_test_pred - 1.96 * prediction_std
upper_bound = y_test_pred + 1.96 * prediction_std

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Left: Test set with confidence intervals
ax = axes[0]
ax.scatter(y_test, y_test_pred, alpha=0.6, s=60, edgecolors='black',
           linewidth=0.5, c=residuals_test, cmap='RdYlGn_r')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', lw=2, label='Perfect Prediction', zorder=5)

# Add confidence band (approximation)
sorted_idx = np.argsort(y_test)
ax.fill_between(y_test[sorted_idx],
                lower_bound[sorted_idx],
                upper_bound[sorted_idx],
                alpha=0.2, color='blue', label='95% Prediction Interval')

ax.set_xlabel('Actual Score', fontweight='bold')
ax.set_ylabel('Predicted Score', fontweight='bold')
ax.set_title(f'Test Set: Predicted vs Actual\n(N={len(y_test)}, MAE={metrics_test["MAE"]:.2f}, R²={metrics_test["R²"]:.3f})',
             fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Residual', rotation=270, labelpad=20)

# Right: All splits comparison
ax = axes[1]
ax.scatter(y_train, y_train_pred, alpha=0.4, s=40, label='Train', edgecolors='none')
ax.scatter(y_val, y_val_pred, alpha=0.5, s=50, label='Validation', edgecolors='none')
ax.scatter(y_test, y_test_pred, alpha=0.6, s=60, label='Test', edgecolors='black', linewidth=0.5)
ax.plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction', zorder=5)

ax.set_xlabel('Actual Score', fontweight='bold')
ax.set_ylabel('Predicted Score', fontweight='bold')
ax.set_title('All Splits: Predicted vs Actual', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'predicted_vs_actual_with_ci.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR / 'predicted_vs_actual_with_ci.png'}")

# ============================================================================
# STEP 8: RESIDUAL VISUALIZATIONS
# ============================================================================
print("\n[STEP 8/10] Creating comprehensive residual visualizations...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Residuals vs Predicted
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(y_test_pred, residuals_test, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax1.axhline(y=np.mean(residuals_test), color='blue', linestyle=':', linewidth=2,
            label=f'Mean: {np.mean(residuals_test):.2f}')
ax1.axhline(y=2*prediction_std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax1.axhline(y=-2*prediction_std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
            label=f'±2σ: ±{2*prediction_std:.2f}')
ax1.set_xlabel('Predicted Score', fontweight='bold')
ax1.set_ylabel('Residual (Actual - Predicted)', fontweight='bold')
ax1.set_title('Residuals vs Predicted Values', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Q-Q Plot
ax2 = fig.add_subplot(gs[0, 2])
stats.probplot(residuals_test, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot (Normality)', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Get top feature importances for residual analysis
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

top_features = feature_importance.head(6)['feature'].tolist()

# Plots 3-8: Residuals vs Top Features
for idx, feature in enumerate(top_features):
    ax = fig.add_subplot(gs[1 + idx//3, idx%3])
    feature_values = X_test[feature].values
    ax.scatter(feature_values, residuals_test, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel(feature.replace('has_', ''), fontweight='bold')
    ax.set_ylabel('Residual', fontweight='bold')
    ax.set_title(f'Residuals vs {feature.replace("has_", "")}', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.savefig(OUTPUT_DIR / 'residual_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR / 'residual_analysis_comprehensive.png'}")

# Additional residual distribution plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
ax = axes[0]
ax.hist(residuals_test, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax.axvline(x=np.mean(residuals_test), color='blue', linestyle=':', linewidth=2,
           label=f'Mean: {np.mean(residuals_test):.2f}')
ax.set_xlabel('Residual', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Distribution of Residuals', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Box plot by predicted score range
ax = axes[1]
pred_ranges = pd.cut(y_test_pred, bins=[0, 30, 70, 100], labels=['Low', 'Medium', 'High'])
data_for_box = [residuals_test[pred_ranges == label] for label in ['Low', 'Medium', 'High']]
bp = ax.boxplot(data_for_box, labels=['Low\n(0-30)', 'Medium\n(30-70)', 'High\n(70-100)'],
                patch_artist=True, showmeans=True)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
    patch.set_facecolor(color)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Score Range', fontweight='bold')
ax.set_ylabel('Residual', fontweight='bold')
ax.set_title('Residuals by Predicted Score Range', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'residual_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR / 'residual_distribution.png'}")

# ============================================================================
# STEP 9: WORST PREDICTIONS ANALYSIS
# ============================================================================
print("\n[STEP 9/10] Analyzing worst predictions (Top 10)...")

# Calculate absolute errors
abs_errors = np.abs(residuals_test)
worst_10_idx = np.argsort(abs_errors)[-10:][::-1]

# Create detailed analysis
worst_predictions = pd.DataFrame({
    'Rank': range(1, 11),
    'Sample_Index': worst_10_idx,
    'Actual_Score': y_test[worst_10_idx],
    'Predicted_Score': y_test_pred[worst_10_idx],
    'Error': residuals_test[worst_10_idx],
    'Abs_Error': abs_errors[worst_10_idx],
    'Percent_Error': (abs_errors[worst_10_idx] / y_test[worst_10_idx]) * 100,
    'Actual_Category': categorize_scores(y_test[worst_10_idx]),
    'Predicted_Category': categorize_scores(y_test_pred[worst_10_idx])
})

# Add feature values for worst predictions
for feature in top_features[:5]:
    worst_predictions[feature] = X_test[feature].values[worst_10_idx]

print("\n✓ Top 10 Worst Predictions (Test Set):")
print("\n" + "=" * 120)
print(worst_predictions.to_string(index=False))
print("=" * 120)

# Save worst predictions
worst_predictions.to_csv(OUTPUT_DIR / 'worst_10_predictions.csv', index=False)
print(f"\n✓ Saved: {OUTPUT_DIR / 'worst_10_predictions.csv'}")

# Visualize worst predictions
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Top plot: Actual vs Predicted for worst cases
ax = axes[0]
x_pos = np.arange(10)
width = 0.35
ax.bar(x_pos - width/2, worst_predictions['Actual_Score'], width,
       label='Actual', color='steelblue', edgecolor='black', alpha=0.7)
ax.bar(x_pos + width/2, worst_predictions['Predicted_Score'], width,
       label='Predicted', color='coral', edgecolor='black', alpha=0.7)
ax.set_xlabel('Rank (1 = Worst)', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Top 10 Worst Predictions: Actual vs Predicted', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(worst_predictions['Rank'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Bottom plot: Absolute error
ax = axes[1]
colors = ['red' if cat != pred_cat else 'orange'
          for cat, pred_cat in zip(worst_predictions['Actual_Category'],
                                   worst_predictions['Predicted_Category'])]
ax.bar(x_pos, worst_predictions['Abs_Error'], color=colors, edgecolor='black', alpha=0.7)
ax.set_xlabel('Rank (1 = Worst)', fontweight='bold')
ax.set_ylabel('Absolute Error', fontweight='bold')
ax.set_title('Top 10 Worst Predictions: Absolute Error (Red = Category Mismatch)',
             fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(worst_predictions['Rank'])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'worst_10_predictions_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR / 'worst_10_predictions_analysis.png'}")

# ============================================================================
# STEP 10: GENERATE COMPREHENSIVE REPORT
# ============================================================================
print("\n[STEP 10/10] Generating comprehensive evaluation report...")

report_lines = [
    "=" * 80,
    "COMPREHENSIVE TEST SET EVALUATION REPORT",
    "Agent Readiness ML - Random Forest Model",
    f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 80,
    "",
    "## 1. MODEL INFORMATION",
    "-" * 80,
    f"Model Type: {type(model).__name__}",
    f"Model Path: {MODEL_PATH}",
    f"Number of Features: {len(feature_names)}",
    f"Training Hyperparameters:",
    f"  - n_estimators: {model.n_estimators}",
    f"  - max_depth: {model.max_depth}",
    f"  - min_samples_leaf: {model.min_samples_leaf}",
    "",
    "## 2. DATA SPLIT SUMMARY",
    "-" * 80,
    f"Training Set:   {len(y_train):4d} samples ({len(y_train)/(len(y_train)+len(y_val)+len(y_test))*100:.1f}%)",
    f"Validation Set: {len(y_val):4d} samples ({len(y_val)/(len(y_train)+len(y_val)+len(y_test))*100:.1f}%)",
    f"Test Set:       {len(y_test):4d} samples ({len(y_test)/(len(y_train)+len(y_val)+len(y_test))*100:.1f}%) [HELD-OUT]",
    f"Total:          {len(y_train)+len(y_val)+len(y_test):4d} samples",
    "",
    "## 3. PERFORMANCE METRICS (ALL SPLITS)",
    "-" * 80,
]

# Add metrics table
report_lines.append("\n" + metrics_comparison.to_string(index=False))
report_lines.extend([
    "",
    "",
    "## 4. TEST SET PERFORMANCE BY SCORE RANGE",
    "-" * 80,
    "\n" + range_analysis_test.to_string(index=False),
    "",
    "",
    "## 5. CONFUSION MATRIX (TEST SET - CATEGORICAL)",
    "-" * 80,
    "Categories: Low (0-30), Medium (30-70), High (70-100)",
    "",
    "Absolute Counts:",
    cm_df.to_string(),
    "",
    "Normalized (Row %):",
    cm_norm_df.round(1).to_string(),
    "",
    "Category-wise Accuracy:"
])

for i, cat in enumerate(['Low', 'Medium', 'High']):
    report_lines.append(f"  {cat:8s}: {category_accuracy[i]*100:6.1f}%")

report_lines.extend([
    "",
    "",
    "## 6. RESIDUAL ANALYSIS (TEST SET)",
    "-" * 80,
    f"Mean Residual:         {np.mean(residuals_test):8.4f}",
    f"Std Residual:          {np.std(residuals_test):8.4f}",
    f"Min Residual:          {np.min(residuals_test):8.4f}",
    f"Max Residual:          {np.max(residuals_test):8.4f}",
    f"Median Residual:       {np.median(residuals_test):8.4f}",
    "",
    "Statistical Tests:",
    f"  Shapiro-Wilk (Normality):       W={shapiro_stat:.4f}, p={shapiro_p:.4f}",
    f"    → {'Residuals appear normal' if shapiro_p > 0.05 else 'Residuals NOT normal'} (α=0.05)",
    f"  Levene (Heteroscedasticity):    W={levene_stat:.4f}, p={levene_p:.4f}",
    f"    → {'Homoscedastic' if levene_p > 0.05 else 'Heteroscedastic'} (α=0.05)",
    f"  Autocorrelation (Lag-1):        r={autocorr:.4f}",
    "",
    "",
    "## 7. TOP 10 WORST PREDICTIONS (TEST SET)",
    "-" * 80,
    "\n" + worst_predictions[['Rank', 'Actual_Score', 'Predicted_Score',
                             'Abs_Error', 'Actual_Category', 'Predicted_Category']].to_string(index=False),
    "",
    "",
    "## 8. KEY INSIGHTS & RECOMMENDATIONS",
    "-" * 80,
])

# Generate insights
mae_test = metrics_test['MAE']
r2_test = metrics_test['R²']
mae_improvement = ((baseline_mae := np.mean(np.abs(y_test - np.mean(y_train)))) - mae_test) / baseline_mae * 100

insights = []
if r2_test > 0.8:
    insights.append("✓ EXCELLENT: Model shows strong predictive performance (R² > 0.8)")
elif r2_test > 0.6:
    insights.append("✓ GOOD: Model shows good predictive performance (R² > 0.6)")
else:
    insights.append("⚠ MODERATE: Model shows moderate performance (R² < 0.6)")

if mae_test < 10:
    insights.append("✓ EXCELLENT: Very low prediction error (MAE < 10)")
elif mae_test < 15:
    insights.append("✓ GOOD: Low prediction error (MAE < 15)")
else:
    insights.append("⚠ ATTENTION: Consider model improvement (MAE > 15)")

# Check for overfitting
if metrics_train['R²'] - metrics_test['R²'] > 0.15:
    insights.append("⚠ WARNING: Possible overfitting detected (Train R² >> Test R²)")
else:
    insights.append("✓ GOOD: Model generalizes well (minimal overfitting)")

# Check category accuracy
overall_cat_acc = np.mean(category_accuracy) * 100
if overall_cat_acc > 80:
    insights.append(f"✓ EXCELLENT: High categorical accuracy ({overall_cat_acc:.1f}%)")
else:
    insights.append(f"⚠ ATTENTION: Categorical accuracy could improve ({overall_cat_acc:.1f}%)")

# Add insights to report
for insight in insights:
    report_lines.append(insight)

report_lines.extend([
    "",
    "Recommendations:",
    "  1. Review top 10 worst predictions for data quality issues",
    "  2. Consider feature engineering for underperforming score ranges",
    "  3. Monitor for heteroscedasticity in production predictions",
    "  4. Implement prediction confidence intervals in deployment",
    "",
    "",
    "## 9. GENERATED ARTIFACTS",
    "-" * 80,
    "CSV Files:",
    "  • metrics_comparison_all_splits.csv",
    "  • performance_by_score_range.csv",
    "  • confusion_matrix.csv",
    "  • residual_statistics.csv",
    "  • worst_10_predictions.csv",
    "",
    "Visualizations:",
    "  • predicted_vs_actual_with_ci.png",
    "  • performance_by_score_range.png",
    "  • confusion_matrix_categories.png",
    "  • residual_analysis_comprehensive.png",
    "  • residual_distribution.png",
    "  • worst_10_predictions_analysis.png",
    "",
    "=" * 80,
    "END OF REPORT",
    "=" * 80
])

# Save report
report_text = "\n".join(report_lines)
with open(OUTPUT_DIR / 'TEST_EVALUATION_REPORT.txt', 'w') as f:
    f.write(report_text)

print(f"\n✓ Saved: {OUTPUT_DIR / 'TEST_EVALUATION_REPORT.txt'}")

# Also save as markdown
report_md = report_text.replace("=" * 80, "---").replace("-" * 80, "")
with open(OUTPUT_DIR / 'TEST_EVALUATION_REPORT.md', 'w') as f:
    f.write(report_md)

print(f"✓ Saved: {OUTPUT_DIR / 'TEST_EVALUATION_REPORT.md'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ COMPREHENSIVE TEST EVALUATION COMPLETE!")
print("=" * 80)
print(f"\n📊 Test Set Performance:")
print(f"  MAE:  {mae_test:.2f}")
print(f"  RMSE: {metrics_test['RMSE']:.2f}")
print(f"  R²:   {r2_test:.3f}")
print(f"  MAPE: {metrics_test['MAPE (%)']:.1f}%")
print(f"\n🎯 Overall Categorical Accuracy: {overall_cat_acc:.1f}%")
print(f"\n📁 All outputs saved to: {OUTPUT_DIR}/")
print(f"\n📄 Full report available: TEST_EVALUATION_REPORT.txt")
print("\n" + "=" * 80)
