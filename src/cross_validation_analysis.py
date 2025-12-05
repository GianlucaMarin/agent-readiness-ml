"""
STEP 1.2: Cross-Validation Deep Dive
=====================================

Comprehensive cross-validation analysis to verify model performance stability
across different data splits.

Author: Sandra Marin
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from scipy import stats
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
OUTPUT_DIR = Path('outputs/02_cross_validation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STEP 1.2: CROSS-VALIDATION DEEP DIVE")
print("=" * 80)
print()

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("Loading data...")
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
X_val = pd.read_csv('data/processed/X_val.csv')
y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# Combine train + val for cross-validation
X_cv = pd.concat([X_train, X_val], ignore_index=True)
y_cv = np.concatenate([y_train, y_val])

print(f"CV dataset: {len(X_cv)} samples (Train: {len(X_train)}, Val: {len(X_val)})")
print()

# Features
feature_cols = [col for col in X_cv.columns if col.startswith('has_')]

print(f"Features: {len(feature_cols)}")
print(f"CV samples: {len(X_cv)}")
print(f"Test samples: {len(X_test)}")
print()

# ============================================================================
# 2. CREATE STRATIFIED BINS FOR CV
# ============================================================================

def create_stratified_bins(scores, n_bins=5):
    """Create stratified bins for cross-validation"""
    bins = pd.qcut(scores, q=n_bins, labels=False, duplicates='drop')
    return bins

# Create bins for stratification
cv_bins = create_stratified_bins(y_cv, n_bins=5)
print("Score distribution in bins:")
print(pd.Series(cv_bins).value_counts().sort_index())
print()

# ============================================================================
# 3. DEFINE MODEL (Same hyperparameters as original)
# ============================================================================

def get_model():
    """Return model with same hyperparameters as test evaluation"""
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

# ============================================================================
# 4. CUSTOM SCORING FUNCTIONS
# ============================================================================

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def custom_scorer(estimator, X, y):
    """Custom scorer for cross-validation"""
    y_pred = estimator.predict(X)
    return {
        'mae': mean_absolute_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'r2': r2_score(y, y_pred),
        'mape': calculate_mape(y, y_pred),
        'median_ae': median_absolute_error(y, y_pred)
    }

# ============================================================================
# 5. PERFORM 5-FOLD CROSS-VALIDATION
# ============================================================================

print("=" * 80)
print("PERFORMING 5-FOLD CROSS-VALIDATION")
print("=" * 80)
print()

skf_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_results_5 = []
fold_info_5 = []

for fold_idx, (train_idx, val_idx) in enumerate(skf_5.split(X_cv, cv_bins), 1):
    print(f"Fold {fold_idx}/5...")

    X_train_fold = X_cv.iloc[train_idx]
    y_train_fold = y_cv[train_idx]
    X_val_fold = X_cv.iloc[val_idx]
    y_val_fold = y_cv[val_idx]

    # Train model
    model = get_model()
    model.fit(X_train_fold, y_train_fold)

    # Predict
    y_train_pred = model.predict(X_train_fold)
    y_val_pred = model.predict(X_val_fold)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train_fold, y_train_pred)
    val_mae = mean_absolute_error(y_val_fold, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
    val_r2 = r2_score(y_val_fold, y_val_pred)
    val_mape = calculate_mape(y_val_fold, y_val_pred)
    val_median_ae = median_absolute_error(y_val_fold, y_val_pred)

    # Store results
    fold_results_5.append({
        'Fold': fold_idx,
        'Train_MAE': train_mae,
        'Val_MAE': val_mae,
        'Val_RMSE': val_rmse,
        'Val_R2': val_r2,
        'Val_MAPE': val_mape,
        'Val_Median_AE': val_median_ae,
        'Train_Size': len(train_idx),
        'Val_Size': len(val_idx)
    })

    # Analyze fold characteristics
    val_scores = y_val_fold  # Already a numpy array
    low_pct = np.sum(val_scores < 30) / len(val_scores) * 100
    med_pct = np.sum((val_scores >= 30) & (val_scores < 70)) / len(val_scores) * 100
    high_pct = np.sum(val_scores >= 70) / len(val_scores) * 100

    fold_info_5.append({
        'Fold': fold_idx,
        'Low_%': low_pct,
        'Med_%': med_pct,
        'High_%': high_pct,
        'Mean_Score': np.mean(val_scores),
        'Std_Score': np.std(val_scores),
        'Val_MAE': val_mae
    })

    print(f"  Train MAE: {train_mae:.3f} | Val MAE: {val_mae:.3f} | Val R²: {val_r2:.4f}")
    print(f"  Fold composition - Low: {low_pct:.1f}%, Med: {med_pct:.1f}%, High: {high_pct:.1f}%")
    print()

df_5fold = pd.DataFrame(fold_results_5)
df_5fold_info = pd.DataFrame(fold_info_5)

# ============================================================================
# 6. PERFORM 10-FOLD CROSS-VALIDATION
# ============================================================================

print("=" * 80)
print("PERFORMING 10-FOLD CROSS-VALIDATION")
print("=" * 80)
print()

skf_10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_results_10 = []
fold_info_10 = []

for fold_idx, (train_idx, val_idx) in enumerate(skf_10.split(X_cv, cv_bins), 1):
    print(f"Fold {fold_idx}/10...")

    X_train_fold = X_cv.iloc[train_idx]
    y_train_fold = y_cv[train_idx]
    X_val_fold = X_cv.iloc[val_idx]
    y_val_fold = y_cv[val_idx]

    # Train model
    model = get_model()
    model.fit(X_train_fold, y_train_fold)

    # Predict
    y_train_pred = model.predict(X_train_fold)
    y_val_pred = model.predict(X_val_fold)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train_fold, y_train_pred)
    val_mae = mean_absolute_error(y_val_fold, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
    val_r2 = r2_score(y_val_fold, y_val_pred)
    val_mape = calculate_mape(y_val_fold, y_val_pred)
    val_median_ae = median_absolute_error(y_val_fold, y_val_pred)

    # Store results
    fold_results_10.append({
        'Fold': fold_idx,
        'Train_MAE': train_mae,
        'Val_MAE': val_mae,
        'Val_RMSE': val_rmse,
        'Val_R2': val_r2,
        'Val_MAPE': val_mape,
        'Val_Median_AE': val_median_ae,
        'Train_Size': len(train_idx),
        'Val_Size': len(val_idx)
    })

    # Analyze fold characteristics
    val_scores = y_val_fold  # Already a numpy array
    low_pct = np.sum(val_scores < 30) / len(val_scores) * 100
    med_pct = np.sum((val_scores >= 30) & (val_scores < 70)) / len(val_scores) * 100
    high_pct = np.sum(val_scores >= 70) / len(val_scores) * 100

    fold_info_10.append({
        'Fold': fold_idx,
        'Low_%': low_pct,
        'Med_%': med_pct,
        'High_%': high_pct,
        'Mean_Score': np.mean(val_scores),
        'Std_Score': np.std(val_scores),
        'Val_MAE': val_mae
    })

    print(f"  Train MAE: {train_mae:.3f} | Val MAE: {val_mae:.3f} | Val R²: {val_r2:.4f}")

df_10fold = pd.DataFrame(fold_results_10)
df_10fold_info = pd.DataFrame(fold_info_10)

print()

# ============================================================================
# 7. CALCULATE AGGREGATE STATISTICS
# ============================================================================

print("=" * 80)
print("AGGREGATE STATISTICS")
print("=" * 80)
print()

def calculate_aggregate_stats(df, name):
    """Calculate aggregate statistics for cross-validation results"""
    stats_dict = {}

    for metric in ['Val_MAE', 'Val_RMSE', 'Val_R2', 'Val_MAPE']:
        values = df[metric].values
        stats_dict[metric] = {
            'Mean': np.mean(values),
            'Std': np.std(values, ddof=1),
            'Min': np.min(values),
            'Max': np.max(values),
            'CV%': (np.std(values, ddof=1) / np.mean(values)) * 100 if np.mean(values) != 0 else 0,
            'CI_95_Lower': np.mean(values) - 1.96 * np.std(values, ddof=1) / np.sqrt(len(values)),
            'CI_95_Upper': np.mean(values) + 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
        }

    return pd.DataFrame(stats_dict).T

stats_5fold = calculate_aggregate_stats(df_5fold, "5-Fold")
stats_10fold = calculate_aggregate_stats(df_10fold, "10-Fold")

print("5-FOLD CV STATISTICS:")
print(stats_5fold.round(4))
print()

print("10-FOLD CV STATISTICS:")
print(stats_10fold.round(4))
print()

# ============================================================================
# 8. COMPARISON WITH SINGLE VAL/TEST
# ============================================================================

print("=" * 80)
print("COMPARISON WITH SINGLE VAL/TEST SPLITS")
print("=" * 80)
print()

# Load original test results for comparison
val_mae_original = 1.09
test_mae_original = 0.64

comparison_df = pd.DataFrame({
    'Approach': ['Single Val (18 samples)', 'Single Test (36 samples)',
                 '5-Fold CV', '10-Fold CV'],
    'MAE': [val_mae_original, test_mae_original,
            stats_5fold.loc['Val_MAE', 'Mean'], stats_10fold.loc['Val_MAE', 'Mean']],
    'Std': ['-', '-',
            stats_5fold.loc['Val_MAE', 'Std'], stats_10fold.loc['Val_MAE', 'Std']],
    'CI_95_Lower': ['-', '-',
                    stats_5fold.loc['Val_MAE', 'CI_95_Lower'],
                    stats_10fold.loc['Val_MAE', 'CI_95_Lower']],
    'CI_95_Upper': ['-', '-',
                    stats_5fold.loc['Val_MAE', 'CI_95_Upper'],
                    stats_10fold.loc['Val_MAE', 'CI_95_Upper']],
    'Notes': ['Small sample', 'Lucky split?', 'More robust', 'Most robust']
})

print(comparison_df.to_string(index=False))
print()

# ============================================================================
# 9. STATISTICAL SIGNIFICANCE TESTS
# ============================================================================

print("=" * 80)
print("STATISTICAL SIGNIFICANCE TESTS")
print("=" * 80)
print()

# Test 1: Is Test MAE significantly different from 5-Fold CV mean?
mae_5fold_values = df_5fold['Val_MAE'].values
t_stat_test_vs_5fold, p_val_test_vs_5fold = stats.ttest_1samp(mae_5fold_values, test_mae_original)

print("Test 1: Single Test (0.64) vs 5-Fold CV Mean")
print(f"  t-statistic: {t_stat_test_vs_5fold:.4f}")
print(f"  p-value: {p_val_test_vs_5fold:.4f}")
print(f"  Result: {'Significantly different' if p_val_test_vs_5fold < 0.05 else 'NOT significantly different'} (α=0.05)")
print()

# Test 2: Is Val MAE significantly different from 5-Fold CV mean?
t_stat_val_vs_5fold, p_val_val_vs_5fold = stats.ttest_1samp(mae_5fold_values, val_mae_original)

print("Test 2: Single Val (1.09) vs 5-Fold CV Mean")
print(f"  t-statistic: {t_stat_val_vs_5fold:.4f}")
print(f"  p-value: {p_val_val_vs_5fold:.4f}")
print(f"  Result: {'Significantly different' if p_val_val_vs_5fold < 0.05 else 'NOT significantly different'} (α=0.05)")
print()

# Test 3: Outlier detection in folds
print("Test 3: Outlier Detection (Z-score analysis)")
print("\n5-Fold CV:")
z_scores_5 = stats.zscore(df_5fold['Val_MAE'])
for i, (fold, mae, z) in enumerate(zip(df_5fold['Fold'], df_5fold['Val_MAE'], z_scores_5), 1):
    outlier_flag = "⚠️ OUTLIER" if abs(z) > 2 else "✓"
    print(f"  Fold {fold}: MAE={mae:.3f}, Z-score={z:.3f} {outlier_flag}")

print("\n10-Fold CV:")
z_scores_10 = stats.zscore(df_10fold['Val_MAE'])
for i, (fold, mae, z) in enumerate(zip(df_10fold['Fold'], df_10fold['Val_MAE'], z_scores_10), 1):
    outlier_flag = "⚠️ OUTLIER" if abs(z) > 2 else "✓"
    print(f"  Fold {fold}: MAE={mae:.3f}, Z-score={z:.3f} {outlier_flag}")

print()

# ============================================================================
# 10. LEARNING CURVES
# ============================================================================

print("=" * 80)
print("GENERATING LEARNING CURVES")
print("=" * 80)
print()

train_sizes = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
train_sizes_abs = (train_sizes * len(X_cv)).astype(int)

model = get_model()
train_sizes_actual, train_scores, val_scores = learning_curve(
    model, X_cv, y_cv,
    train_sizes=train_sizes,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state=42
)

# Convert to positive MAE
train_scores_mae = -train_scores
val_scores_mae = -val_scores

# Calculate means and stds
train_mean = np.mean(train_scores_mae, axis=1)
train_std = np.std(train_scores_mae, axis=1)
val_mean = np.mean(val_scores_mae, axis=1)
val_std = np.std(val_scores_mae, axis=1)

print("Learning Curve Results:")
for size, tr_m, tr_s, val_m, val_s in zip(train_sizes_actual, train_mean, train_std, val_mean, val_std):
    print(f"  {size:3d} samples: Train MAE={tr_m:.3f}±{tr_s:.3f}, Val MAE={val_m:.3f}±{val_s:.3f}")
print()

# ============================================================================
# 11. STABILITY SCORE CALCULATION
# ============================================================================

print("=" * 80)
print("STABILITY ASSESSMENT")
print("=" * 80)
print()

cv_5_pct = stats_5fold.loc['Val_MAE', 'CV%']
cv_10_pct = stats_10fold.loc['Val_MAE', 'CV%']

def calculate_stability_score(cv_percentage):
    """Calculate stability score (0-10) based on CV%"""
    if cv_percentage < 5:
        return 10
    elif cv_percentage < 10:
        return 9
    elif cv_percentage < 15:
        return 7
    elif cv_percentage < 20:
        return 5
    elif cv_percentage < 30:
        return 3
    else:
        return 1

stability_5 = calculate_stability_score(cv_5_pct)
stability_10 = calculate_stability_score(cv_10_pct)

print(f"5-Fold CV Coefficient of Variation: {cv_5_pct:.2f}%")
print(f"5-Fold Stability Score: {stability_5}/10")
print()
print(f"10-Fold CV Coefficient of Variation: {cv_10_pct:.2f}%")
print(f"10-Fold Stability Score: {stability_10}/10")
print()

# Interpretation
if cv_10_pct < 5:
    interpretation = "EXCELLENT - Performance is highly stable across different data splits"
elif cv_10_pct < 15:
    interpretation = "GOOD - Performance is reasonably stable with minor variations"
elif cv_10_pct < 30:
    interpretation = "MODERATE - Performance shows moderate variability across splits"
else:
    interpretation = "POOR - Performance is highly unstable, model is not robust"

print(f"Interpretation: {interpretation}")
print()

# ============================================================================
# 12. SAVE RESULTS TO CSV
# ============================================================================

print("Saving results to CSV...")
df_5fold.to_csv(OUTPUT_DIR / '5fold_results.csv', index=False)
df_10fold.to_csv(OUTPUT_DIR / '10fold_results.csv', index=False)
stats_5fold.to_csv(OUTPUT_DIR / '5fold_statistics.csv')
stats_10fold.to_csv(OUTPUT_DIR / '10fold_statistics.csv')
comparison_df.to_csv(OUTPUT_DIR / 'comparison_single_vs_cv.csv', index=False)
df_5fold_info.to_csv(OUTPUT_DIR / '5fold_characteristics.csv', index=False)
df_10fold_info.to_csv(OUTPUT_DIR / '10fold_characteristics.csv', index=False)
print("✓ CSV files saved")
print()

# ============================================================================
# 13. VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)
print()

# Visualization 1: Box Plot - Fold Performance Distribution
print("Creating Visualization 1: Box Plot...")
fig, ax = plt.subplots(figsize=(12, 6))

data_to_plot = [
    df_5fold['Val_MAE'].values,
    df_10fold['Val_MAE'].values
]

bp = ax.boxplot(data_to_plot, labels=['5-Fold CV', '10-Fold CV'],
                patch_artist=True, widths=0.6)

# Color boxes
colors = ['#3498db', '#2ecc71']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add horizontal lines for single val/test
ax.axhline(y=val_mae_original, color='orange', linestyle='--', linewidth=2,
           label=f'Single Val: {val_mae_original:.2f}', alpha=0.8)
ax.axhline(y=test_mae_original, color='red', linestyle='--', linewidth=2,
           label=f'Single Test: {test_mae_original:.2f}', alpha=0.8)

# Add mean markers
ax.plot([1, 2], [stats_5fold.loc['Val_MAE', 'Mean'], stats_10fold.loc['Val_MAE', 'Mean']],
        'D', color='darkred', markersize=10, label='Mean', zorder=10)

ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax.set_title('Cross-Validation Performance Distribution\n(Comparison with Single Val/Test Splits)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cv_boxplot_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: cv_boxplot_comparison.png")

# Visualization 2: Fold-by-Fold Bar Chart
print("Creating Visualization 2: Fold-by-Fold Bar Chart...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 5-Fold
x_5 = df_5fold_info['Fold'].values
y_5 = df_5fold_info['Val_MAE'].values
colors_5 = plt.cm.RdYlGn_r(y_5 / y_5.max())

bars1 = ax1.bar(x_5, y_5, color=colors_5, edgecolor='black', linewidth=1.5)
ax1.axhline(y=stats_5fold.loc['Val_MAE', 'Mean'], color='blue', linestyle='--',
            linewidth=2, label=f'Mean: {stats_5fold.loc["Val_MAE", "Mean"]:.3f}')
ax1.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax1.set_ylabel('Validation MAE', fontsize=12, fontweight='bold')
ax1.set_title('5-Fold CV: MAE per Fold', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xticks(x_5)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 10-Fold
x_10 = df_10fold_info['Fold'].values
y_10 = df_10fold_info['Val_MAE'].values
colors_10 = plt.cm.RdYlGn_r(y_10 / y_10.max())

bars2 = ax2.bar(x_10, y_10, color=colors_10, edgecolor='black', linewidth=1.5)
ax2.axhline(y=stats_10fold.loc['Val_MAE', 'Mean'], color='blue', linestyle='--',
            linewidth=2, label=f'Mean: {stats_10fold.loc["Val_MAE", "Mean"]:.3f}')
ax2.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax2.set_ylabel('Validation MAE', fontsize=12, fontweight='bold')
ax2.set_title('10-Fold CV: MAE per Fold', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(x_10)

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fold_by_fold_mae.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: fold_by_fold_mae.png")

# Visualization 3: Learning Curve
print("Creating Visualization 3: Learning Curve...")
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(train_sizes_actual, train_mean, 'o-', color='blue', linewidth=2,
        markersize=8, label='Training MAE')
ax.fill_between(train_sizes_actual, train_mean - train_std, train_mean + train_std,
                alpha=0.2, color='blue')

ax.plot(train_sizes_actual, val_mean, 'o-', color='red', linewidth=2,
        markersize=8, label='Validation MAE')
ax.fill_between(train_sizes_actual, val_mean - val_std, val_mean + val_std,
                alpha=0.2, color='red')

ax.set_xlabel('Training Set Size (samples)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
ax.set_title('Learning Curve: Training Size vs Performance\n(Shaded area = ±1 Std)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

# Add convergence annotation
gap = val_mean[-1] - train_mean[-1]
ax.annotate(f'Train-Val Gap: {gap:.3f}',
            xy=(train_sizes_actual[-1], (train_mean[-1] + val_mean[-1])/2),
            xytext=(train_sizes_actual[-1] - 20, (train_mean[-1] + val_mean[-1])/2 + 0.3),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: learning_curve.png")

# Visualization 4: Fold Characteristics Heatmap
print("Creating Visualization 4: Fold Characteristics Heatmap...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 5-Fold Heatmap
data_5 = df_5fold_info[['Low_%', 'Med_%', 'High_%', 'Mean_Score', 'Val_MAE']].values
sns.heatmap(data_5.T, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Value'},
            xticklabels=[f'Fold {i}' for i in range(1, 6)],
            yticklabels=['Low %', 'Med %', 'High %', 'Mean Score', 'Val MAE'],
            ax=ax1, linewidths=1, linecolor='black')
ax1.set_title('5-Fold CV: Fold Characteristics', fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel('Fold', fontsize=11, fontweight='bold')

# 10-Fold Heatmap
data_10 = df_10fold_info[['Low_%', 'Med_%', 'High_%', 'Mean_Score', 'Val_MAE']].values
sns.heatmap(data_10.T, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Value'},
            xticklabels=[f'F{i}' for i in range(1, 11)],
            yticklabels=['Low %', 'Med %', 'High %', 'Mean Score', 'Val MAE'],
            ax=ax2, linewidths=1, linecolor='black')
ax2.set_title('10-Fold CV: Fold Characteristics', fontsize=13, fontweight='bold', pad=15)
ax2.set_xlabel('Fold', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fold_characteristics_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: fold_characteristics_heatmap.png")

# Visualization 5: CV Confidence Intervals Comparison
print("Creating Visualization 5: CV Confidence Intervals...")
fig, ax = plt.subplots(figsize=(12, 7))

approaches = ['Single\nVal', 'Single\nTest', '5-Fold\nCV', '10-Fold\nCV']
mae_values = [
    val_mae_original,
    test_mae_original,
    stats_5fold.loc['Val_MAE', 'Mean'],
    stats_10fold.loc['Val_MAE', 'Mean']
]

# Error bars (only for CV methods)
errors_lower = [
    0,  # No error for single val
    0,  # No error for single test
    stats_5fold.loc['Val_MAE', 'Mean'] - stats_5fold.loc['Val_MAE', 'CI_95_Lower'],
    stats_10fold.loc['Val_MAE', 'Mean'] - stats_10fold.loc['Val_MAE', 'CI_95_Lower']
]

errors_upper = [
    0,
    0,
    stats_5fold.loc['Val_MAE', 'CI_95_Upper'] - stats_5fold.loc['Val_MAE', 'Mean'],
    stats_10fold.loc['Val_MAE', 'CI_95_Upper'] - stats_10fold.loc['Val_MAE', 'Mean']
]

x_pos = np.arange(len(approaches))
colors_bar = ['orange', 'red', '#3498db', '#2ecc71']

bars = ax.bar(x_pos, mae_values, yerr=[errors_lower, errors_upper],
              color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2,
              capsize=10, error_kw={'linewidth': 2, 'ecolor': 'black'})

# Add value labels
for i, (bar, val) in enumerate(zip(bars, mae_values)):
    height = bar.get_height()
    if i < 2:
        label_text = f'{val:.2f}'
    else:
        ci_lower = mae_values[i] - errors_lower[i]
        ci_upper = mae_values[i] + errors_upper[i]
        label_text = f'{val:.2f}\n[{ci_lower:.2f}, {ci_upper:.2f}]'

    ax.text(bar.get_x() + bar.get_width()/2., height + (errors_upper[i] if i >= 2 else 0),
            label_text, ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison: Single Splits vs Cross-Validation\n(Error bars = 95% Confidence Intervals)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(approaches, fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(mae_values) * 1.4)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cv_confidence_intervals.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: cv_confidence_intervals.png")

print()
print("✓ All visualizations generated successfully!")
print()

# ============================================================================
# 14. GENERATE COMPREHENSIVE REPORT
# ============================================================================

print("=" * 80)
print("GENERATING COMPREHENSIVE REPORT")
print("=" * 80)
print()

report_lines = []
report_lines.append("=" * 80)
report_lines.append("STEP 1.2: CROSS-VALIDATION DEEP DIVE - COMPREHENSIVE REPORT")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append(f"Date: December 2025")
report_lines.append(f"Author: Sandra Marin")
report_lines.append("")

# Executive Summary
report_lines.append("=" * 80)
report_lines.append("EXECUTIVE SUMMARY")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append(f"Dataset: {len(X_cv)} samples (Train + Val combined for CV)")
report_lines.append(f"Features: {len(feature_cols)} has_* features")
report_lines.append(f"Model: Random Forest Regressor (n_estimators=200, max_depth=15)")
report_lines.append("")

# Key Findings
report_lines.append("KEY FINDINGS:")
report_lines.append("")
report_lines.append(f"1. 5-Fold CV Mean MAE: {stats_5fold.loc['Val_MAE', 'Mean']:.3f} ± {stats_5fold.loc['Val_MAE', 'Std']:.3f}")
report_lines.append(f"   95% CI: [{stats_5fold.loc['Val_MAE', 'CI_95_Lower']:.3f}, {stats_5fold.loc['Val_MAE', 'CI_95_Upper']:.3f}]")
report_lines.append(f"   Coefficient of Variation: {cv_5_pct:.2f}%")
report_lines.append(f"   Stability Score: {stability_5}/10")
report_lines.append("")

report_lines.append(f"2. 10-Fold CV Mean MAE: {stats_10fold.loc['Val_MAE', 'Mean']:.3f} ± {stats_10fold.loc['Val_MAE', 'Std']:.3f}")
report_lines.append(f"   95% CI: [{stats_10fold.loc['Val_MAE', 'CI_95_Lower']:.3f}, {stats_10fold.loc['Val_MAE', 'CI_95_Upper']:.3f}]")
report_lines.append(f"   Coefficient of Variation: {cv_10_pct:.2f}%")
report_lines.append(f"   Stability Score: {stability_10}/10")
report_lines.append("")

report_lines.append(f"3. Comparison with Single Splits:")
report_lines.append(f"   - Single Val MAE: {val_mae_original:.2f}")
report_lines.append(f"   - Single Test MAE: {test_mae_original:.2f}")
report_lines.append(f"   - Test MAE is {'within' if test_mae_original >= stats_10fold.loc['Val_MAE', 'CI_95_Lower'] and test_mae_original <= stats_10fold.loc['Val_MAE', 'CI_95_Upper'] else 'OUTSIDE'} 10-Fold CV 95% CI")
report_lines.append("")

report_lines.append(f"4. Statistical Significance:")
report_lines.append(f"   - Test (0.64) vs 5-Fold CV: p={p_val_test_vs_5fold:.4f} ({'Significant' if p_val_test_vs_5fold < 0.05 else 'Not significant'})")
report_lines.append(f"   - Val (1.09) vs 5-Fold CV: p={p_val_val_vs_5fold:.4f} ({'Significant' if p_val_val_vs_5fold < 0.05 else 'Not significant'})")
report_lines.append("")

report_lines.append(f"5. Overall Stability: {interpretation}")
report_lines.append("")

# Detailed Results
report_lines.append("=" * 80)
report_lines.append("5-FOLD CV DETAILED RESULTS")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append(df_5fold.to_string(index=False))
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("10-FOLD CV DETAILED RESULTS")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append(df_10fold.to_string(index=False))
report_lines.append("")

# Fold Characteristics
report_lines.append("=" * 80)
report_lines.append("FOLD CHARACTERISTICS ANALYSIS")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("5-Fold Characteristics:")
report_lines.append(df_5fold_info.to_string(index=False))
report_lines.append("")
report_lines.append("10-Fold Characteristics:")
report_lines.append(df_10fold_info.to_string(index=False))
report_lines.append("")

# Learning Curve Analysis
report_lines.append("=" * 80)
report_lines.append("LEARNING CURVE ANALYSIS")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Training Size | Train MAE (±Std) | Val MAE (±Std) | Gap")
report_lines.append("-" * 70)
for size, tr_m, tr_s, val_m, val_s in zip(train_sizes_actual, train_mean, train_std, val_mean, val_std):
    gap = val_m - tr_m
    report_lines.append(f"{size:13d} | {tr_m:7.3f} ± {tr_s:.3f}  | {val_m:7.3f} ± {val_s:.3f} | {gap:.3f}")
report_lines.append("")

convergence_gap = val_mean[-1] - train_mean[-1]
if convergence_gap < 0.2:
    lc_interpretation = "✓ EXCELLENT - Model converges with minimal train-val gap"
elif convergence_gap < 0.5:
    lc_interpretation = "✓ GOOD - Model shows good convergence"
else:
    lc_interpretation = "⚠ WARNING - Significant train-val gap suggests overfitting"

report_lines.append(f"Convergence Assessment: {lc_interpretation}")
report_lines.append(f"Final Train-Val Gap: {convergence_gap:.3f}")
report_lines.append("")

# Critical Questions Answered
report_lines.append("=" * 80)
report_lines.append("CRITICAL QUESTIONS ANSWERED")
report_lines.append("=" * 80)
report_lines.append("")

# Q1: Is Test MAE representative or lucky?
test_in_ci = (test_mae_original >= stats_10fold.loc['Val_MAE', 'CI_95_Lower'] and
              test_mae_original <= stats_10fold.loc['Val_MAE', 'CI_95_Upper'])
if test_in_ci:
    q1_answer = f"✓ REPRESENTATIVE - Test MAE ({test_mae_original:.2f}) is within 10-Fold CV 95% CI"
else:
    q1_answer = f"⚠ LUCKY SPLIT - Test MAE ({test_mae_original:.2f}) is outside 10-Fold CV 95% CI"

report_lines.append(f"Q1: Is Test MAE (0.64) representative or lucky?")
report_lines.append(f"    {q1_answer}")
report_lines.append("")

# Q2: What's the true expected MAE?
expected_mae = stats_10fold.loc['Val_MAE', 'Mean']
expected_ci_lower = stats_10fold.loc['Val_MAE', 'CI_95_Lower']
expected_ci_upper = stats_10fold.loc['Val_MAE', 'CI_95_Upper']

report_lines.append(f"Q2: What's the true expected MAE (with confidence interval)?")
report_lines.append(f"    Expected MAE: {expected_mae:.3f} [95% CI: {expected_ci_lower:.3f} - {expected_ci_upper:.3f}]")
report_lines.append("")

# Q3: Is performance stable?
if cv_10_pct < 10:
    q3_answer = f"✓ YES - CV% = {cv_10_pct:.2f}% (< 10% threshold)"
elif cv_10_pct < 20:
    q3_answer = f"⚠ MODERATE - CV% = {cv_10_pct:.2f}% (10-20% range)"
else:
    q3_answer = f"✗ NO - CV% = {cv_10_pct:.2f}% (> 20% threshold)"

report_lines.append(f"Q3: Is performance stable across different data splits?")
report_lines.append(f"    {q3_answer}")
report_lines.append("")

# Q4: Should we proceed or tune?
if stability_10 >= 7 and cv_10_pct < 15:
    q4_answer = "✓ PROCEED - Model is stable and performs well. No tuning needed."
elif stability_10 >= 5:
    q4_answer = "⚠ CONSIDER TUNING - Moderate stability. Hyperparameter optimization might help."
else:
    q4_answer = "✗ TUNE REQUIRED - Poor stability. Hyperparameter tuning strongly recommended."

report_lines.append(f"Q4: Should we proceed with this model or tune further?")
report_lines.append(f"    {q4_answer}")
report_lines.append("")

# Q5: Is hyperparameter choice robust?
if cv_10_pct < 10 and convergence_gap < 0.3:
    q5_answer = "✓ YES - Low variance across folds and good learning curve convergence"
elif cv_10_pct < 20:
    q5_answer = "⚠ ACCEPTABLE - Reasonable robustness, but monitoring recommended"
else:
    q5_answer = "✗ NO - High variance suggests hyperparameters may not be optimal"

report_lines.append(f"Q5: Is our current hyperparameter choice robust?")
report_lines.append(f"    {q5_answer}")
report_lines.append("")

# Final Verdict
report_lines.append("=" * 80)
report_lines.append("FINAL VERDICT")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append(f"STABILITY SCORE: {stability_10}/10")
report_lines.append("")
report_lines.append(f"EXPECTED PRODUCTION PERFORMANCE:")
report_lines.append(f"Based on 10-fold cross-validation, we expect MAE of {expected_mae:.2f} ± {stats_10fold.loc['Val_MAE', 'Std']:.2f}")
report_lines.append(f"(95% CI: [{expected_ci_lower:.2f}, {expected_ci_upper:.2f}]) in production,")
report_lines.append(f"assuming similar data distribution.")
report_lines.append("")

report_lines.append(f"ROBUSTNESS ASSESSMENT:")
report_lines.append(f"✓ Robust to different data splits: {'YES' if cv_10_pct < 15 else 'NO'}")
report_lines.append(f"✓ Robust to sample compositions: {'YES' if len([z for z in z_scores_10 if abs(z) > 2]) == 0 else 'NO'}")
report_lines.append(f"✓ Robust to training size variations: {'YES' if convergence_gap < 0.3 else 'NO'}")
report_lines.append("")

report_lines.append(f"RECOMMENDATIONS:")
if stability_10 >= 8 and cv_10_pct < 10:
    report_lines.append("✓ Model is highly stable and ready for production")
    report_lines.append("✓ No hyperparameter tuning required")
    report_lines.append("✓ Proceed to next validation steps")
elif stability_10 >= 6:
    report_lines.append("⚠ Model shows good stability but could benefit from:")
    report_lines.append("  - Light hyperparameter tuning (grid search)")
    report_lines.append("  - Monitoring performance on additional validation sets")
else:
    report_lines.append("✗ Model shows instability. Recommended actions:")
    report_lines.append("  - Comprehensive hyperparameter tuning")
    report_lines.append("  - Feature engineering review")
    report_lines.append("  - Consider ensemble methods or alternative algorithms")

report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("END OF REPORT")
report_lines.append("=" * 80)

# Save report
report_text = "\n".join(report_lines)
with open(OUTPUT_DIR / 'CROSS_VALIDATION_REPORT.txt', 'w') as f:
    f.write(report_text)

print("✓ Comprehensive report saved: CROSS_VALIDATION_REPORT.txt")
print()

# ============================================================================
# 15. FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("CROSS-VALIDATION ANALYSIS COMPLETE")
print("=" * 80)
print()
print("SUMMARY:")
print(f"  10-Fold CV Mean MAE: {expected_mae:.3f} ± {stats_10fold.loc['Val_MAE', 'Std']:.3f}")
print(f"  95% Confidence Interval: [{expected_ci_lower:.3f}, {expected_ci_upper:.3f}]")
print(f"  Stability Score: {stability_10}/10")
print(f"  Coefficient of Variation: {cv_10_pct:.2f}%")
print()
print(f"VERDICT: {interpretation}")
print()
print("FILES GENERATED:")
print("  📊 5 Visualizations (PNG)")
print("  📄 7 CSV files with detailed results")
print("  📝 1 Comprehensive report (TXT)")
print()
print(f"All outputs saved to: {OUTPUT_DIR}")
print()
print("=" * 80)
