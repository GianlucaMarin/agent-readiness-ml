"""
Hyperparameter Tuning with Nested Cross-Validation
===================================================

Performs systematic hyperparameter optimization using Nested CV:
- Outer Loop: 5-Fold CV for unbiased performance estimation
- Inner Loop: 3-Fold Grid Search for hyperparameter selection

Focus: Improve stability (reduce CV variance) and overall MAE

Author: Sandra Marin
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

print("=" * 80)
print("HYPERPARAMETER TUNING - NESTED CROSS-VALIDATION")
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

# Combine train + val for CV
X_cv = pd.concat([X_train, X_val], ignore_index=True)
y_cv = np.concatenate([y_train, y_val])

print(f"✓ CV dataset: {len(X_cv)} samples")
print(f"✓ Test dataset: {len(X_test)} samples")
print()

# ============================================================================
# 2. DEFINE HYPERPARAMETER GRID
# ============================================================================

print("Defining hyperparameter search space...")

param_grid = {
    'max_depth': [12, 15, 18, 20],
    'min_samples_leaf': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.5]
}

# Fixed parameters
fixed_params = {
    'n_estimators': 200,
    'random_state': 42,
    'n_jobs': -1
}

total_combinations = (len(param_grid['max_depth']) *
                     len(param_grid['min_samples_leaf']) *
                     len(param_grid['min_samples_split']) *
                     len(param_grid['max_features']))

print(f"✓ Total hyperparameter combinations: {total_combinations}")
print(f"✓ Fixed: n_estimators={fixed_params['n_estimators']}")
print()

print("Parameter Grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")
print()

# ============================================================================
# 3. CREATE STRATIFIED BINS FOR CV
# ============================================================================

print("Creating stratified bins for cross-validation...")

# Create bins for stratification
bins = [0, 30, 70, 100]
cv_bins = pd.cut(y_cv, bins=bins, labels=['Low', 'Medium', 'High'])

print("CV Data Distribution:")
print(f"  Low (0-30):     {np.sum(cv_bins == 'Low')} samples ({np.sum(cv_bins == 'Low')/len(cv_bins)*100:.1f}%)")
print(f"  Medium (30-70): {np.sum(cv_bins == 'Medium')} samples ({np.sum(cv_bins == 'Medium')/len(cv_bins)*100:.1f}%)")
print(f"  High (70-100):  {np.sum(cv_bins == 'High')} samples ({np.sum(cv_bins == 'High')/len(cv_bins)*100:.1f}%)")
print()

# ============================================================================
# 4. NESTED CROSS-VALIDATION
# ============================================================================

print("=" * 80)
print("NESTED CROSS-VALIDATION: 5-Fold Outer × 3-Fold Inner Grid Search")
print("=" * 80)
print()

# Outer CV: 5-Fold for performance estimation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV: 3-Fold for hyperparameter selection
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Results storage
nested_scores = []
best_params_per_fold = []
fold_predictions = []

print("Starting Nested CV...")
print()

for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_cv, cv_bins), 1):
    print(f"Outer Fold {fold_idx}/5")
    print("-" * 80)

    # Split data
    X_train_fold = X_cv.iloc[train_idx]
    y_train_fold = y_cv[train_idx]
    X_val_fold = X_cv.iloc[val_idx]
    y_val_fold = y_cv[val_idx]

    # Stratification bins for inner CV
    train_bins = cv_bins[train_idx]

    # Inner Grid Search
    print(f"  Running Grid Search ({total_combinations} combinations)...")

    base_model = RandomForestRegressor(**fixed_params)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=inner_cv.split(X_train_fold, train_bins),
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train_fold, y_train_fold)

    # Best model from grid search
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"  ✓ Best params: {best_params}")
    print(f"  ✓ Best CV MAE: {-grid_search.best_score_:.3f}")

    # Evaluate on outer validation fold
    y_val_pred = best_model.predict(X_val_fold)
    val_mae = mean_absolute_error(y_val_fold, y_val_pred)
    val_r2 = r2_score(y_val_fold, y_val_pred)

    print(f"  ✓ Outer Fold MAE: {val_mae:.3f}")
    print(f"  ✓ Outer Fold R²: {val_r2:.4f}")
    print()

    # Store results
    nested_scores.append({
        'Fold': fold_idx,
        'MAE': val_mae,
        'R2': val_r2,
        'Best_CV_MAE': -grid_search.best_score_
    })

    best_params_per_fold.append({
        'Fold': fold_idx,
        **best_params
    })

    fold_predictions.append({
        'Fold': fold_idx,
        'y_true': y_val_fold,
        'y_pred': y_val_pred
    })

# ============================================================================
# 5. AGGREGATE NESTED CV RESULTS
# ============================================================================

print("=" * 80)
print("NESTED CV RESULTS")
print("=" * 80)
print()

nested_df = pd.DataFrame(nested_scores)
params_df = pd.DataFrame(best_params_per_fold)

mean_mae = nested_df['MAE'].mean()
std_mae = nested_df['MAE'].std(ddof=1)
mean_r2 = nested_df['R2'].mean()

print(f"Outer Fold MAE: {mean_mae:.3f} ± {std_mae:.3f}")
print(f"Outer Fold R²:  {mean_r2:.4f}")
print()

print("MAE per Fold:")
for _, row in nested_df.iterrows():
    print(f"  Fold {int(row['Fold'])}: {row['MAE']:.3f}")
print()

# ============================================================================
# 6. DETERMINE BEST HYPERPARAMETERS
# ============================================================================

print("=" * 80)
print("BEST HYPERPARAMETERS ANALYSIS")
print("=" * 80)
print()

print("Best parameters selected per fold:")
print(params_df.to_string(index=False))
print()

# Most common parameters (mode)
best_params_final = {}
for param in param_grid.keys():
    mode_value = params_df[param].mode()[0]
    best_params_final[param] = mode_value
    print(f"Most common {param}: {mode_value} (appeared in {(params_df[param] == mode_value).sum()}/5 folds)")

print()
print(f"FINAL BEST HYPERPARAMETERS: {best_params_final}")
print()

# ============================================================================
# 7. TRAIN FINAL MODEL WITH BEST HYPERPARAMETERS
# ============================================================================

print("=" * 80)
print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
print("=" * 80)
print()

final_model = RandomForestRegressor(
    **fixed_params,
    **best_params_final
)

# Train on full CV dataset (train + val)
print("Training on full CV dataset (142 samples)...")
final_model.fit(X_cv, y_cv)
print("✓ Training complete")
print()

# Evaluate on held-out test set
print("Evaluating on test set (36 samples)...")
y_test_pred = final_model.predict(X_test)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"✓ Test MAE:  {test_mae:.3f}")
print(f"✓ Test R²:   {test_r2:.4f}")
print(f"✓ Test RMSE: {test_rmse:.3f}")
print()

# ============================================================================
# 8. COMPARE WITH BASELINE MODEL
# ============================================================================

print("=" * 80)
print("COMPARISON WITH BASELINE MODEL")
print("=" * 80)
print()

# Load baseline model
baseline_data = joblib.load('models/random_forest_initial.joblib')
baseline_model = baseline_data['model']
y_test_baseline = baseline_model.predict(X_test)

baseline_mae = mean_absolute_error(y_test, y_test_baseline)
baseline_r2 = r2_score(y_test, y_test_baseline)

print("Baseline Hyperparameters:")
print(f"  max_depth: 15")
print(f"  min_samples_leaf: 3")
print(f"  min_samples_split: 2")
print(f"  max_features: None")
print()

print("Tuned Hyperparameters:")
for param, value in best_params_final.items():
    print(f"  {param}: {value}")
print()

print("Test Set Performance:")
print(f"  Baseline MAE:  {baseline_mae:.3f}")
print(f"  Tuned MAE:     {test_mae:.3f}")
print(f"  Improvement:   {(baseline_mae - test_mae) / baseline_mae * 100:+.1f}%")
print()

print(f"  Baseline R²:   {baseline_r2:.4f}")
print(f"  Tuned R²:      {test_r2:.4f}")
print()

# CV Stability Comparison
print("CV Stability:")
print(f"  Nested CV MAE: {mean_mae:.3f} ± {std_mae:.3f}")
print(f"  CV%: {(std_mae / mean_mae * 100):.1f}%")
print()

# ============================================================================
# 9. SAVE TUNED MODEL
# ============================================================================

print("=" * 80)
print("SAVING TUNED MODEL")
print("=" * 80)
print()

# Metadata
tuned_metadata = {
    'model': final_model,
    'hyperparameters': {**fixed_params, **best_params_final},
    'test_mae': test_mae,
    'test_r2': test_r2,
    'nested_cv_mae': mean_mae,
    'nested_cv_std': std_mae,
    'improvement_vs_baseline': (baseline_mae - test_mae) / baseline_mae * 100,
    'training_date': 'December 2025'
}

# Decide if tuned model is better
if test_mae < baseline_mae:
    output_path = 'models/random_forest_tuned.joblib'
    joblib.dump(tuned_metadata, output_path)
    print(f"✓ Saved tuned model: {output_path}")
    print("✓ Tuned model is BETTER than baseline")
    recommendation = "USE TUNED MODEL"
else:
    output_path = 'models/experiments/random_forest_tuned.joblib'
    joblib.dump(tuned_metadata, output_path)
    print(f"✓ Saved tuned model: {output_path}")
    print("✗ Tuned model is NOT better than baseline")
    recommendation = "KEEP BASELINE MODEL"

print()
print(f"RECOMMENDATION: {recommendation}")
print()

# ============================================================================
# 10. CREATE VISUALIZATIONS
# ============================================================================

print("Creating visualizations...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Nested CV Fold Performance
ax1 = fig.add_subplot(gs[0, 0])
folds = nested_df['Fold'].astype(int)
maes = nested_df['MAE']
colors = ['#2ecc71' if mae < mean_mae else '#e74c3c' for mae in maes]
bars = ax1.bar(folds, maes, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax1.axhline(mean_mae, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_mae:.3f}')
ax1.set_xlabel('Outer Fold', fontsize=11, fontweight='bold')
ax1.set_ylabel('MAE', fontsize=11, fontweight='bold')
ax1.set_title('Nested CV: Fold Performance', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

# 2. Baseline vs Tuned Comparison
ax2 = fig.add_subplot(gs[0, 1])
models = ['Baseline', 'Tuned']
test_maes = [baseline_mae, test_mae]
colors = ['#3498db', '#2ecc71' if test_mae < baseline_mae else '#e74c3c']
bars = ax2.bar(models, test_maes, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
ax2.set_ylabel('Test MAE', fontsize=11, fontweight='bold')
ax2.set_title('Test Set: Baseline vs Tuned', fontsize=12, fontweight='bold')
ax2.set_ylim(0, max(test_maes) * 1.2)
for bar, mae in zip(bars, test_maes):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{mae:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Improvement Percentage
ax3 = fig.add_subplot(gs[0, 2])
improvement = (baseline_mae - test_mae) / baseline_mae * 100
color = '#2ecc71' if improvement > 0 else '#e74c3c'
bar = ax3.barh(['Improvement'], [improvement], color=color, edgecolor='black', linewidth=2, alpha=0.8)
ax3.axvline(0, color='black', linestyle='-', linewidth=2)
ax3.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
ax3.set_title('Test MAE Improvement', fontsize=12, fontweight='bold')
ax3.text(improvement, 0, f'{improvement:+.1f}%', ha='left' if improvement > 0 else 'right',
         va='center', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# 4. Hyperparameter Frequency - max_depth
ax4 = fig.add_subplot(gs[1, 0])
max_depth_counts = params_df['max_depth'].value_counts().sort_index()
ax4.bar(max_depth_counts.index.astype(str), max_depth_counts.values,
        color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.8)
ax4.set_xlabel('max_depth', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Hyperparameter Frequency: max_depth', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 5)
ax4.grid(True, alpha=0.3, axis='y')

# 5. Hyperparameter Frequency - min_samples_leaf
ax5 = fig.add_subplot(gs[1, 1])
msl_counts = params_df['min_samples_leaf'].value_counts().sort_index()
ax5.bar(msl_counts.index.astype(str), msl_counts.values,
        color='#9b59b6', edgecolor='black', linewidth=1.5, alpha=0.8)
ax5.set_xlabel('min_samples_leaf', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title('Hyperparameter Frequency: min_samples_leaf', fontsize=12, fontweight='bold')
ax5.set_ylim(0, 5)
ax5.grid(True, alpha=0.3, axis='y')

# 6. Hyperparameter Frequency - min_samples_split
ax6 = fig.add_subplot(gs[1, 2])
mss_counts = params_df['min_samples_split'].value_counts().sort_index()
ax6.bar(mss_counts.index.astype(str), mss_counts.values,
        color='#e67e22', edgecolor='black', linewidth=1.5, alpha=0.8)
ax6.set_xlabel('min_samples_split', fontsize=11, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax6.set_title('Hyperparameter Frequency: min_samples_split', fontsize=12, fontweight='bold')
ax6.set_ylim(0, 5)
ax6.grid(True, alpha=0.3, axis='y')

# 7. Test Set: True vs Predicted (Baseline)
ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(y_test, y_test_baseline, alpha=0.6, color='#3498db', edgecolors='black', linewidths=0.5)
ax7.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect')
ax7.set_xlabel('True Score', fontsize=11, fontweight='bold')
ax7.set_ylabel('Predicted Score', fontsize=11, fontweight='bold')
ax7.set_title(f'Baseline: Test Set (MAE={baseline_mae:.3f})', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# 8. Test Set: True vs Predicted (Tuned)
ax8 = fig.add_subplot(gs[2, 1])
ax8.scatter(y_test, y_test_pred, alpha=0.6, color='#2ecc71', edgecolors='black', linewidths=0.5)
ax8.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect')
ax8.set_xlabel('True Score', fontsize=11, fontweight='bold')
ax8.set_ylabel('Predicted Score', fontsize=11, fontweight='bold')
ax8.set_title(f'Tuned: Test Set (MAE={test_mae:.3f})', fontsize=12, fontweight='bold')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)

# 9. Summary Text
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')

summary_text = f'''
HYPERPARAMETER TUNING SUMMARY

Nested CV Results:
  Outer Folds: 5
  Inner Folds: 3
  Total Combinations: {total_combinations}

Best Hyperparameters:
  max_depth: {best_params_final['max_depth']}
  min_samples_leaf: {best_params_final['min_samples_leaf']}
  min_samples_split: {best_params_final['min_samples_split']}
  max_features: {best_params_final['max_features']}

Performance:
  Baseline Test MAE: {baseline_mae:.3f}
  Tuned Test MAE: {test_mae:.3f}
  Improvement: {improvement:+.1f}%

CV Stability:
  Nested CV MAE: {mean_mae:.3f} ± {std_mae:.3f}
  CV%: {(std_mae / mean_mae * 100):.1f}%

Recommendation:
  {recommendation}
'''

ax9.text(0.5, 0.5, summary_text, ha='center', va='center',
         fontsize=9, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Hyperparameter Tuning: Nested Cross-Validation Results',
             fontsize=14, fontweight='bold', y=0.995)

plot_path = 'outputs/hyperparameter_tuning_results.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved visualization: {plot_path}")
print()

# ============================================================================
# 11. SAVE DETAILED REPORT
# ============================================================================

print("Saving detailed report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("HYPERPARAMETER TUNING - NESTED CROSS-VALIDATION REPORT")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Author: Sandra Marin")
report_lines.append("Date: December 2025")
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("METHODOLOGY")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Nested Cross-Validation:")
report_lines.append("  - Outer Loop: 5-Fold Stratified CV for performance estimation")
report_lines.append("  - Inner Loop: 3-Fold Stratified Grid Search for hyperparameter selection")
report_lines.append("  - Stratification: Low (0-30), Medium (30-70), High (70-100)")
report_lines.append("")
report_lines.append(f"Total Hyperparameter Combinations Tested: {total_combinations}")
report_lines.append("")

report_lines.append("Parameter Grid:")
for param, values in param_grid.items():
    report_lines.append(f"  {param}: {values}")
report_lines.append("")

report_lines.append("Fixed Parameters:")
for param, value in fixed_params.items():
    report_lines.append(f"  {param}: {value}")
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("NESTED CV RESULTS")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append(f"Outer Fold MAE: {mean_mae:.3f} ± {std_mae:.3f}")
report_lines.append(f"CV%: {(std_mae / mean_mae * 100):.1f}%")
report_lines.append("")

report_lines.append("Fold-by-Fold Results:")
for _, row in nested_df.iterrows():
    report_lines.append(f"  Fold {int(row['Fold'])}: MAE={row['MAE']:.3f}, R²={row['R2']:.4f}, Best_CV_MAE={row['Best_CV_MAE']:.3f}")
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("BEST HYPERPARAMETERS")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Parameters selected per fold:")
report_lines.append(params_df.to_string(index=False))
report_lines.append("")

report_lines.append("Final Best Hyperparameters (mode across folds):")
for param, value in best_params_final.items():
    freq = (params_df[param] == value).sum()
    report_lines.append(f"  {param}: {value} (selected in {freq}/5 folds)")
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("TEST SET PERFORMANCE")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Baseline Model:")
report_lines.append(f"  Test MAE: {baseline_mae:.3f}")
report_lines.append(f"  Test R²:  {baseline_r2:.4f}")
report_lines.append("")
report_lines.append("Tuned Model:")
report_lines.append(f"  Test MAE: {test_mae:.3f}")
report_lines.append(f"  Test R²:  {test_r2:.4f}")
report_lines.append("")
report_lines.append(f"Improvement: {improvement:+.1f}%")
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("FINAL RECOMMENDATION")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append(f"RECOMMENDATION: {recommendation}")
report_lines.append("")

if test_mae < baseline_mae:
    report_lines.append("RATIONALE:")
    report_lines.append(f"  - Tuned model achieves {improvement:.1f}% lower Test MAE")
    report_lines.append(f"  - Model saved to: models/random_forest_tuned.joblib")
    report_lines.append("  - Use this model for production")
else:
    report_lines.append("RATIONALE:")
    report_lines.append(f"  - Tuned model performs {-improvement:.1f}% WORSE on test set")
    report_lines.append("  - No significant improvement from hyperparameter tuning")
    report_lines.append("  - Baseline hyperparameters are already well-suited")
    report_lines.append("  - Keep using: models/random_forest_initial.joblib")
    report_lines.append("")
    report_lines.append("INSIGHT:")
    report_lines.append("  This confirms that the class imbalance problem (9 Low-score samples)")
    report_lines.append("  cannot be solved by hyperparameter optimization alone.")
    report_lines.append("  The limitation is DATA, not MODEL ARCHITECTURE.")

report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("END OF REPORT")
report_lines.append("=" * 80)

report_path = 'outputs/HYPERPARAMETER_TUNING_REPORT.txt'
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"✓ Saved report: {report_path}")
print()

# ============================================================================
# 12. FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("HYPERPARAMETER TUNING COMPLETE")
print("=" * 80)
print()
print(f"✓ Tested {total_combinations} hyperparameter combinations")
print(f"✓ Nested CV MAE: {mean_mae:.3f} ± {std_mae:.3f}")
print(f"✓ Best Test MAE: {test_mae:.3f} (Baseline: {baseline_mae:.3f})")
print(f"✓ Improvement: {improvement:+.1f}%")
print()
print(f"RECOMMENDATION: {recommendation}")
print()
print(f"✓ Model saved: {output_path}")
print(f"✓ Report saved: {report_path}")
print(f"✓ Visualization saved: {plot_path}")
print()
print("=" * 80)
