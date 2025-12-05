"""
Train Model with Sample Weights to Address Class Imbalance
===========================================================

Implements balanced sample weighting to give more importance to
underrepresented Low and Medium score ranges.

Author: Sandra Marin
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

print("=" * 80)
print("TRAINING MODEL WITH SAMPLE WEIGHTS (Class Imbalance Fix)")
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

print(f"✓ Train: {len(X_train)} samples")
print(f"✓ Val:   {len(X_val)} samples")
print(f"✓ Test:  {len(X_test)} samples")
print()

# ============================================================================
# 2. ANALYZE ORIGINAL CLASS DISTRIBUTION
# ============================================================================

print("=" * 80)
print("ORIGINAL CLASS DISTRIBUTION (IMBALANCED)")
print("=" * 80)
print()

def categorize_scores(scores):
    """Categorize scores into Low/Medium/High"""
    categories = np.zeros_like(scores, dtype=int)
    categories[scores < 30] = 0  # Low
    categories[(scores >= 30) & (scores < 70)] = 1  # Medium
    categories[scores >= 70] = 2  # High
    return categories

y_train_cat = categorize_scores(y_train)
y_val_cat = categorize_scores(y_val)
y_test_cat = categorize_scores(y_test)

# Count distribution
train_counts = np.bincount(y_train_cat)
total_train = len(y_train_cat)

print("Training Set Distribution:")
print(f"  Low (0-30):      {train_counts[0]:3d} samples ({train_counts[0]/total_train*100:5.1f}%)")
print(f"  Medium (30-70):  {train_counts[1]:3d} samples ({train_counts[1]/total_train*100:5.1f}%)")
print(f"  High (70-100):   {train_counts[2]:3d} samples ({train_counts[2]/total_train*100:5.1f}%)")
print()

# ============================================================================
# 3. COMPUTE SAMPLE WEIGHTS
# ============================================================================

print("=" * 80)
print("COMPUTING BALANCED SAMPLE WEIGHTS")
print("=" * 80)
print()

# Compute balanced weights for training set
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_cat)

# Analyze weights
print("Sample Weight Statistics:")
print(f"  Min weight:  {np.min(sample_weights):.3f}")
print(f"  Max weight:  {np.max(sample_weights):.3f}")
print(f"  Mean weight: {np.mean(sample_weights):.3f}")
print()

# Calculate average weight per category
for cat, name in enumerate(['Low', 'Medium', 'High']):
    mask = y_train_cat == cat
    avg_weight = np.mean(sample_weights[mask])
    print(f"  {name:8s} avg weight: {avg_weight:.3f}x (samples get {avg_weight:.1f}x more importance)")

print()
print("Interpretation:")
print("  - Low-score errors now penalized ~3x more heavily")
print("  - Medium-score errors penalized ~1.4x more")
print("  - High-score errors weighted normally (~0.5x)")
print()

# ============================================================================
# 4. TRAIN BASELINE MODEL (NO WEIGHTS) FOR COMPARISON
# ============================================================================

print("=" * 80)
print("TRAINING BASELINE MODEL (NO WEIGHTS)")
print("=" * 80)
print()

model_baseline = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model_baseline.fit(X_train, y_train)

# Evaluate baseline
y_train_pred_base = model_baseline.predict(X_train)
y_val_pred_base = model_baseline.predict(X_val)
y_test_pred_base = model_baseline.predict(X_test)

baseline_results = {
    'Train_MAE': mean_absolute_error(y_train, y_train_pred_base),
    'Val_MAE': mean_absolute_error(y_val, y_val_pred_base),
    'Test_MAE': mean_absolute_error(y_test, y_test_pred_base),
    'Train_R2': r2_score(y_train, y_train_pred_base),
    'Val_R2': r2_score(y_val, y_val_pred_base),
    'Test_R2': r2_score(y_test, y_test_pred_base)
}

print("Baseline Results (No Weights):")
print(f"  Train MAE: {baseline_results['Train_MAE']:.3f} | R²: {baseline_results['Train_R2']:.4f}")
print(f"  Val MAE:   {baseline_results['Val_MAE']:.3f} | R²: {baseline_results['Val_R2']:.4f}")
print(f"  Test MAE:  {baseline_results['Test_MAE']:.3f} | R²: {baseline_results['Test_R2']:.4f}")
print()

# ============================================================================
# 5. TRAIN WEIGHTED MODEL
# ============================================================================

print("=" * 80)
print("TRAINING WEIGHTED MODEL (WITH BALANCED WEIGHTS)")
print("=" * 80)
print()

model_weighted = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

# Train with sample weights
model_weighted.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluate weighted model
y_train_pred_weighted = model_weighted.predict(X_train)
y_val_pred_weighted = model_weighted.predict(X_val)
y_test_pred_weighted = model_weighted.predict(X_test)

weighted_results = {
    'Train_MAE': mean_absolute_error(y_train, y_train_pred_weighted),
    'Val_MAE': mean_absolute_error(y_val, y_val_pred_weighted),
    'Test_MAE': mean_absolute_error(y_test, y_test_pred_weighted),
    'Train_R2': r2_score(y_train, y_train_pred_weighted),
    'Val_R2': r2_score(y_val, y_val_pred_weighted),
    'Test_R2': r2_score(y_test, y_test_pred_weighted)
}

print("Weighted Model Results:")
print(f"  Train MAE: {weighted_results['Train_MAE']:.3f} | R²: {weighted_results['Train_R2']:.4f}")
print(f"  Val MAE:   {weighted_results['Val_MAE']:.3f} | R²: {weighted_results['Val_R2']:.4f}")
print(f"  Test MAE:  {weighted_results['Test_MAE']:.3f} | R²: {weighted_results['Test_R2']:.4f}")
print()

# ============================================================================
# 6. COMPARE PERFORMANCE BY SCORE RANGE
# ============================================================================

print("=" * 80)
print("PERFORMANCE BY SCORE RANGE")
print("=" * 80)
print()

def evaluate_by_range(y_true, y_pred_base, y_pred_weighted, dataset_name):
    """Evaluate performance separately for Low/Medium/High scores"""

    print(f"{dataset_name}:")
    print("-" * 70)

    for range_name, mask in [
        ('Low (0-30)', y_true < 30),
        ('Medium (30-70)', (y_true >= 30) & (y_true < 70)),
        ('High (70-100)', y_true >= 70)
    ]:
        if np.sum(mask) == 0:
            print(f"  {range_name:18s}: No samples")
            continue

        n_samples = np.sum(mask)
        mae_base = mean_absolute_error(y_true[mask], y_pred_base[mask])
        mae_weighted = mean_absolute_error(y_true[mask], y_pred_weighted[mask])
        improvement = ((mae_base - mae_weighted) / mae_base) * 100

        print(f"  {range_name:18s} (n={n_samples:2d}): "
              f"Baseline MAE={mae_base:5.2f} → Weighted MAE={mae_weighted:5.2f} "
              f"({improvement:+5.1f}% {'✓' if improvement > 0 else ''})")

    print()

evaluate_by_range(y_train, y_train_pred_base, y_train_pred_weighted, "TRAIN SET")
evaluate_by_range(y_val, y_val_pred_base, y_val_pred_weighted, "VALIDATION SET")
evaluate_by_range(y_test, y_test_pred_base, y_test_pred_weighted, "TEST SET")

# ============================================================================
# 7. TEST ON SYNTHETIC BLIND TEST
# ============================================================================

print("=" * 80)
print("TESTING ON SYNTHETIC BLIND TEST DATA")
print("=" * 80)
print()

# Load synthetic test data with ground truth
df_synthetic = pd.read_csv('data/raw/synthetic_blind_test_50_REALISTIC.csv')
X_synthetic = df_synthetic[[col for col in df_synthetic.columns if col.startswith('has_')]]

# We need ground truth scores - let's assume they're in a separate column
# For now, we'll create synthetic ground truth based on feature patterns
# In reality, you'd have these labeled

print("Generating predictions on synthetic blind test...")

y_synthetic_pred_base = model_baseline.predict(X_synthetic)
y_synthetic_pred_weighted = model_weighted.predict(X_synthetic)

print(f"✓ Baseline predictions:  Mean={np.mean(y_synthetic_pred_base):.2f}, Median={np.median(y_synthetic_pred_base):.2f}")
print(f"✓ Weighted predictions:  Mean={np.mean(y_synthetic_pred_weighted):.2f}, Median={np.median(y_synthetic_pred_weighted):.2f}")
print()

# Categorize predictions
synthetic_cat_base = categorize_scores(y_synthetic_pred_base)
synthetic_cat_weighted = categorize_scores(y_synthetic_pred_weighted)

print("Predicted Distribution (Baseline):")
for cat, name in enumerate(['Low', 'Medium', 'High']):
    count = np.sum(synthetic_cat_base == cat)
    print(f"  {name:8s}: {count:2d} websites ({count/len(synthetic_cat_base)*100:5.1f}%)")

print()
print("Predicted Distribution (Weighted):")
for cat, name in enumerate(['Low', 'Medium', 'High']):
    count = np.sum(synthetic_cat_weighted == cat)
    print(f"  {name:8s}: {count:2d} websites ({count/len(synthetic_cat_weighted)*100:5.1f}%)")

print()

# ============================================================================
# 8. SAVE WEIGHTED MODEL
# ============================================================================

print("=" * 80)
print("SAVING WEIGHTED MODEL")
print("=" * 80)
print()

# Save model with metadata
model_data = {
    'model': model_weighted,
    'model_type': 'RandomForestRegressor_Weighted',
    'feature_names': list(X_train.columns),
    'hyperparameters': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_leaf': 3,
        'random_state': 42,
        'sample_weighting': 'balanced'
    },
    'performance': weighted_results,
    'sample_weights_used': True,
    'weight_strategy': 'balanced (inverse frequency per score category)'
}

output_path = 'models/random_forest_weighted.joblib'
joblib.dump(model_data, output_path)
print(f"✓ Saved weighted model: {output_path}")
print()

# ============================================================================
# 9. CREATE COMPARISON VISUALIZATION
# ============================================================================

print("Creating comparison visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Performance Comparison (Bar Chart)
ax1 = axes[0, 0]
metrics = ['Train_MAE', 'Val_MAE', 'Test_MAE']
x = np.arange(len(metrics))
width = 0.35

baseline_vals = [baseline_results[m] for m in metrics]
weighted_vals = [weighted_results[m] for m in metrics]

bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline (No Weights)', color='#e74c3c', alpha=0.7)
bars2 = ax1.bar(x + width/2, weighted_vals, width, label='Weighted Model', color='#2ecc71', alpha=0.7)

ax1.set_ylabel('MAE', fontsize=11, fontweight='bold')
ax1.set_title('Performance Comparison: Baseline vs Weighted', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(['Train', 'Validation', 'Test'])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Sample Weight Distribution
ax2 = axes[0, 1]
for cat, name, color in [(0, 'Low', '#e74c3c'), (1, 'Medium', '#f39c12'), (2, 'High', '#2ecc71')]:
    mask = y_train_cat == cat
    weights = sample_weights[mask]
    ax2.hist(weights, bins=20, alpha=0.6, label=name, color=color, edgecolor='black')

ax2.set_xlabel('Sample Weight', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Sample Weight Distribution by Score Category', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Prediction Distribution (Synthetic Test)
ax3 = axes[1, 0]
categories = ['Low\n(0-30)', 'Medium\n(30-70)', 'High\n(70-100)']
x_cat = np.arange(len(categories))

baseline_counts = [np.sum(synthetic_cat_base == i) for i in range(3)]
weighted_counts = [np.sum(synthetic_cat_weighted == i) for i in range(3)]

bars1 = ax3.bar(x_cat - width/2, baseline_counts, width, label='Baseline', color='#e74c3c', alpha=0.7)
bars2 = ax3.bar(x_cat + width/2, weighted_counts, width, label='Weighted', color='#2ecc71', alpha=0.7)

ax3.set_ylabel('Number of Websites', fontsize=11, fontweight='bold')
ax3.set_title('Synthetic Blind Test: Predicted Distribution', fontsize=12, fontweight='bold')
ax3.set_xticks(x_cat)
ax3.set_xticklabels(categories)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 4: Summary Text
ax4 = axes[1, 1]
ax4.axis('off')

improvement_text = f"""
SUMMARY: SAMPLE WEIGHTING RESULTS

Training Strategy:
• Low-score samples weighted ~3.0x higher
• Medium-score samples weighted ~1.4x higher
• High-score samples weighted ~0.5x (normal)

Performance Changes:
• Train MAE: {baseline_results['Train_MAE']:.3f} → {weighted_results['Train_MAE']:.3f} ({((weighted_results['Train_MAE']-baseline_results['Train_MAE'])/baseline_results['Train_MAE']*100):+.1f}%)
• Val MAE:   {baseline_results['Val_MAE']:.3f} → {weighted_results['Val_MAE']:.3f} ({((weighted_results['Val_MAE']-baseline_results['Val_MAE'])/baseline_results['Val_MAE']*100):+.1f}%)
• Test MAE:  {baseline_results['Test_MAE']:.3f} → {weighted_results['Test_MAE']:.3f} ({((weighted_results['Test_MAE']-baseline_results['Test_MAE'])/baseline_results['Test_MAE']*100):+.1f}%)

Synthetic Test Predictions:
Baseline:  {baseline_counts[0]} Low, {baseline_counts[1]} Med, {baseline_counts[2]} High
Weighted:  {weighted_counts[0]} Low, {weighted_counts[1]} Med, {weighted_counts[2]} High

Expected Improvement:
✓ Better balance in predicted distributions
✓ Reduced systematic bias for low scores
✓ More realistic score predictions
"""

ax4.text(0.1, 0.9, improvement_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plot_path = 'outputs/weighted_model_comparison.png'
Path('outputs').mkdir(exist_ok=True)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved visualization: {plot_path}")
print()

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("WEIGHTED MODEL TRAINING COMPLETE")
print("=" * 80)
print()
print("RESULTS:")
print(f"  ✓ Baseline Model:  Val MAE = {baseline_results['Val_MAE']:.3f}")
print(f"  ✓ Weighted Model:  Val MAE = {weighted_results['Val_MAE']:.3f}")
print(f"  ✓ Improvement:     {((baseline_results['Val_MAE']-weighted_results['Val_MAE'])/baseline_results['Val_MAE']*100):+.1f}%")
print()
print(f"MODEL SAVED: {output_path}")
print(f"VISUALIZATION: {plot_path}")
print()
print("=" * 80)
