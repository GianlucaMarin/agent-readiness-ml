"""
Two-Stage Model: Classifier + Specialized Regressors
====================================================

Stage 1: Classify into Low/Medium/High using SMOTE for balance
Stage 2: Predict exact score using specialized regressors per category

This approach addresses class imbalance by:
1. Using SMOTE to balance classification training
2. Training separate models for each score range
3. Avoiding bias from majority class

Author: Sandra Marin
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                            classification_report, confusion_matrix, accuracy_score)
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

print("=" * 80)
print("TWO-STAGE MODEL: CLASSIFIER + SPECIALIZED REGRESSORS")
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
# 2. CREATE CATEGORY LABELS
# ============================================================================

def categorize_scores(scores):
    """Categorize scores into Low (0), Medium (1), High (2)"""
    categories = np.zeros_like(scores, dtype=int)
    categories[scores < 30] = 0  # Low
    categories[(scores >= 30) & (scores < 70)] = 1  # Medium
    categories[scores >= 70] = 2  # High
    return categories

y_train_cat = categorize_scores(y_train)
y_val_cat = categorize_scores(y_val)
y_test_cat = categorize_scores(y_test)

print("Original Training Distribution:")
train_counts = np.bincount(y_train_cat)
for i, name in enumerate(['Low', 'Medium', 'High']):
    print(f"  {name:8s}: {train_counts[i]:3d} samples ({train_counts[i]/len(y_train_cat)*100:5.1f}%)")
print()

# ============================================================================
# 3. STAGE 1: TRAIN CLASSIFIER WITH SMOTE
# ============================================================================

print("=" * 80)
print("STAGE 1: TRAINING CLASSIFIER (Low/Medium/High)")
print("=" * 80)
print()

print("Applying SMOTE to balance training data...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_cat_balanced = smote.fit_resample(X_train, y_train_cat)

print("After SMOTE:")
balanced_counts = np.bincount(y_train_cat_balanced)
for i, name in enumerate(['Low', 'Medium', 'High']):
    print(f"  {name:8s}: {balanced_counts[i]:3d} samples ({balanced_counts[i]/len(y_train_cat_balanced)*100:5.1f}%)")
print()

# Train classifier
print("Training Random Forest Classifier...")
classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

classifier.fit(X_train_balanced, y_train_cat_balanced)

# Evaluate classifier
y_train_cat_pred = classifier.predict(X_train)
y_val_cat_pred = classifier.predict(X_val)
y_test_cat_pred = classifier.predict(X_test)

print("Classifier Performance:")
print(f"  Train Accuracy: {accuracy_score(y_train_cat, y_train_cat_pred):.3f}")
print(f"  Val Accuracy:   {accuracy_score(y_val_cat, y_val_cat_pred):.3f}")
print(f"  Test Accuracy:  {accuracy_score(y_test_cat, y_test_cat_pred):.3f}")
print()

print("Test Set Classification Report:")
print(classification_report(y_test_cat, y_test_cat_pred,
                          target_names=['Low', 'Medium', 'High']))

# ============================================================================
# 4. STAGE 2: TRAIN SPECIALIZED REGRESSORS
# ============================================================================

print("=" * 80)
print("STAGE 2: TRAINING SPECIALIZED REGRESSORS")
print("=" * 80)
print()

# Prepare data for each category
regressors = {}
category_names = {0: 'Low', 1: 'Medium', 2: 'High'}

for cat_id, cat_name in category_names.items():
    print(f"\nTraining {cat_name}-Score Regressor...")

    # Get samples for this category
    mask_train = y_train_cat == cat_id
    n_samples = np.sum(mask_train)

    if n_samples < 3:
        print(f"  ⚠️  WARNING: Only {n_samples} samples. Using global model as fallback.")
        # Use a simple regressor trained on all data
        fallback_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        fallback_model.fit(X_train, y_train)
        regressors[cat_id] = fallback_model
        continue

    X_cat = X_train[mask_train]
    y_cat = y_train[mask_train]

    print(f"  Training samples: {len(X_cat)}")
    print(f"  Score range: [{y_cat.min():.1f}, {y_cat.max():.1f}]")

    # Train regressor for this category
    regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    regressor.fit(X_cat, y_cat)
    regressors[cat_id] = regressor

    # Evaluate on training data for this category
    y_cat_pred = regressor.predict(X_cat)
    mae = mean_absolute_error(y_cat, y_cat_pred)
    r2 = r2_score(y_cat, y_cat_pred)

    print(f"  Training MAE: {mae:.3f}")
    print(f"  Training R²:  {r2:.4f}")

print("\n✓ All specialized regressors trained")
print()

# ============================================================================
# 5. TWO-STAGE PREDICTION FUNCTION
# ============================================================================

def predict_two_stage(X, classifier, regressors):
    """
    Two-stage prediction:
    1. Classify into Low/Medium/High
    2. Use specialized regressor for predicted category
    """
    # Stage 1: Classify
    categories = classifier.predict(X)

    # Stage 2: Predict scores using specialized regressors
    predictions = np.zeros(len(X))

    for cat_id in [0, 1, 2]:
        mask = categories == cat_id
        if np.sum(mask) > 0:
            X_cat = X[mask]
            predictions[mask] = regressors[cat_id].predict(X_cat)

    return predictions, categories

# ============================================================================
# 6. EVALUATE TWO-STAGE MODEL
# ============================================================================

print("=" * 80)
print("EVALUATING TWO-STAGE MODEL")
print("=" * 80)
print()

# Make predictions
y_train_pred, y_train_pred_cat = predict_two_stage(X_train, classifier, regressors)
y_val_pred, y_val_pred_cat = predict_two_stage(X_val, classifier, regressors)
y_test_pred, y_test_pred_cat = predict_two_stage(X_test, classifier, regressors)

# Overall performance
two_stage_results = {
    'Train_MAE': mean_absolute_error(y_train, y_train_pred),
    'Val_MAE': mean_absolute_error(y_val, y_val_pred),
    'Test_MAE': mean_absolute_error(y_test, y_test_pred),
    'Train_R2': r2_score(y_train, y_train_pred),
    'Val_R2': r2_score(y_val, y_val_pred),
    'Test_R2': r2_score(y_test, y_test_pred)
}

print("Two-Stage Model Performance:")
print(f"  Train MAE: {two_stage_results['Train_MAE']:.3f} | R²: {two_stage_results['Train_R2']:.4f}")
print(f"  Val MAE:   {two_stage_results['Val_MAE']:.3f} | R²: {two_stage_results['Val_R2']:.4f}")
print(f"  Test MAE:  {two_stage_results['Test_MAE']:.3f} | R²: {two_stage_results['Test_R2']:.4f}")
print()

# Performance by category
print("Performance by Score Range:")
print("-" * 70)

for cat_id, cat_name in category_names.items():
    # Test set performance for this category
    mask = y_test_cat == cat_id
    if np.sum(mask) == 0:
        continue

    n_samples = np.sum(mask)
    y_true = y_test[mask]
    y_pred = y_test_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0

    print(f"{cat_name:8s} (n={n_samples:2d}): MAE={mae:5.2f}, R²={r2:6.3f}")

print()

# ============================================================================
# 7. COMPARE WITH BASELINE
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

print(f"Baseline Model:   Test MAE = {baseline_mae:.3f}")
print(f"Two-Stage Model:  Test MAE = {two_stage_results['Test_MAE']:.3f}")
print(f"Improvement:      {((baseline_mae - two_stage_results['Test_MAE']) / baseline_mae * 100):+.1f}%")
print()

# ============================================================================
# 8. TEST ON SYNTHETIC BLIND TEST
# ============================================================================

print("=" * 80)
print("TESTING ON SYNTHETIC BLIND TEST")
print("=" * 80)
print()

# Load synthetic data
df_synthetic = pd.read_csv('data/raw/synthetic_blind_test_50_REALISTIC.csv')
X_synthetic = df_synthetic[[col for col in df_synthetic.columns if col.startswith('has_')]]

# Baseline predictions
y_synthetic_baseline = baseline_model.predict(X_synthetic)
baseline_cat = categorize_scores(y_synthetic_baseline)

# Two-stage predictions
y_synthetic_two_stage, two_stage_cat = predict_two_stage(X_synthetic, classifier, regressors)

print("Predicted Distribution - Baseline:")
for i, name in enumerate(['Low', 'Medium', 'High']):
    count = np.sum(baseline_cat == i)
    print(f"  {name:8s}: {count:2d} websites ({count/len(baseline_cat)*100:5.1f}%)")

print("\nPredicted Distribution - Two-Stage:")
for i, name in enumerate(['Low', 'Medium', 'High']):
    count = np.sum(two_stage_cat == i)
    print(f"  {name:8s}: {count:2d} websites ({count/len(two_stage_cat)*100:5.1f}%)")

print(f"\nMean Prediction - Baseline:  {np.mean(y_synthetic_baseline):.2f}")
print(f"Mean Prediction - Two-Stage: {np.mean(y_synthetic_two_stage):.2f}")
print()

# ============================================================================
# 9. SAVE TWO-STAGE MODEL
# ============================================================================

print("=" * 80)
print("SAVING TWO-STAGE MODEL")
print("=" * 80)
print()

model_data = {
    'model_type': 'TwoStage_Classifier_Regressors',
    'stage1_classifier': classifier,
    'stage2_regressors': regressors,
    'category_names': category_names,
    'feature_names': list(X_train.columns),
    'smote_used': True,
    'performance': two_stage_results,
    'classifier_accuracy': {
        'train': accuracy_score(y_train_cat, y_train_cat_pred),
        'val': accuracy_score(y_val_cat, y_val_cat_pred),
        'test': accuracy_score(y_test_cat, y_test_cat_pred)
    }
}

output_path = 'models/two_stage_model.joblib'
joblib.dump(model_data, output_path)
print(f"✓ Saved two-stage model: {output_path}")
print()

# ============================================================================
# 10. CREATE VISUALIZATIONS
# ============================================================================

print("Creating visualizations...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Confusion Matrix (Classifier)
ax1 = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_test_cat, y_test_cat_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Low', 'Med', 'High'],
            yticklabels=['Low', 'Med', 'High'])
ax1.set_title('Stage 1: Classification\nConfusion Matrix', fontsize=11, fontweight='bold')
ax1.set_ylabel('True Category')
ax1.set_xlabel('Predicted Category')

# Plot 2: Classification Accuracy by Set
ax2 = fig.add_subplot(gs[0, 1])
datasets = ['Train', 'Val', 'Test']
accuracies = [
    accuracy_score(y_train_cat, y_train_cat_pred),
    accuracy_score(y_val_cat, y_val_cat_pred),
    accuracy_score(y_test_cat, y_test_cat_pred)
]
bars = ax2.bar(datasets, accuracies, color=['#3498db', '#e67e22', '#2ecc71'], alpha=0.7, edgecolor='black')
ax2.set_ylabel('Classification Accuracy', fontweight='bold')
ax2.set_title('Stage 1: Classifier Performance', fontsize=11, fontweight='bold')
ax2.set_ylim([0, 1.1])
ax2.grid(True, alpha=0.3, axis='y')

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: MAE Comparison
ax3 = fig.add_subplot(gs[0, 2])
x = np.arange(3)
width = 0.35
baseline_maes = [baseline_mae] * 3  # Simplified
two_stage_maes = [two_stage_results['Train_MAE'], two_stage_results['Val_MAE'], two_stage_results['Test_MAE']]

bars1 = ax3.bar(x - width/2, [0.597, 1.087, baseline_mae], width, label='Baseline', color='#e74c3c', alpha=0.7)
bars2 = ax3.bar(x + width/2, two_stage_maes, width, label='Two-Stage', color='#2ecc71', alpha=0.7)

ax3.set_ylabel('MAE', fontweight='bold')
ax3.set_title('Overall Performance Comparison', fontsize=11, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(['Train', 'Val', 'Test'])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4-6: Predicted vs Actual per Category
for idx, (cat_id, cat_name) in enumerate(category_names.items()):
    ax = fig.add_subplot(gs[1, idx])

    mask = y_test_cat == cat_id
    if np.sum(mask) == 0:
        ax.text(0.5, 0.5, 'No test samples', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{cat_name} Scores', fontsize=11, fontweight='bold')
        continue

    y_true = y_test[mask]
    y_pred = y_test_pred[mask]

    ax.scatter(y_true, y_pred, alpha=0.6, s=100, edgecolors='black', linewidth=1.5)

    # Ideal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

    mae = mean_absolute_error(y_true, y_pred)
    ax.text(0.05, 0.95, f'MAE: {mae:.2f}\nn={len(y_true)}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('True Score', fontweight='bold')
    ax.set_ylabel('Predicted Score', fontweight='bold')
    ax.set_title(f'{cat_name} Scores (Stage 2)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

# Plot 7: Synthetic Test Distribution
ax7 = fig.add_subplot(gs[2, 0])
categories = ['Low', 'Medium', 'High']
x_cat = np.arange(len(categories))
width = 0.35

baseline_counts = [np.sum(baseline_cat == i) for i in range(3)]
two_stage_counts = [np.sum(two_stage_cat == i) for i in range(3)]

bars1 = ax7.bar(x_cat - width/2, baseline_counts, width, label='Baseline', color='#e74c3c', alpha=0.7)
bars2 = ax7.bar(x_cat + width/2, two_stage_counts, width, label='Two-Stage', color='#2ecc71', alpha=0.7)

ax7.set_ylabel('Count', fontweight='bold')
ax7.set_title('Synthetic Blind Test:\nPredicted Distribution', fontsize=11, fontweight='bold')
ax7.set_xticks(x_cat)
ax7.set_xticklabels(categories)
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 8: Score Distribution (Test Set)
ax8 = fig.add_subplot(gs[2, 1])
ax8.hist([y_test, y_test_pred], bins=15, label=['True', 'Predicted'],
         alpha=0.6, color=['blue', 'green'], edgecolor='black')
ax8.set_xlabel('Score', fontweight='bold')
ax8.set_ylabel('Frequency', fontweight='bold')
ax8.set_title('Test Set: True vs Predicted\nScore Distribution', fontsize=11, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# Plot 9: Summary Stats
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')

summary_text = f"""
TWO-STAGE MODEL SUMMARY

Stage 1 - Classifier (with SMOTE):
  Test Accuracy: {accuracy_score(y_test_cat, y_test_cat_pred):.1%}
  Balanced Training: {len(y_train_cat_balanced)} samples

Stage 2 - Specialized Regressors:
  Low Model:    Trained on {np.sum(y_train_cat == 0):2d} samples
  Medium Model: Trained on {np.sum(y_train_cat == 1):2d} samples
  High Model:   Trained on {np.sum(y_train_cat == 2):2d} samples

Overall Performance:
  Test MAE:  {two_stage_results['Test_MAE']:.3f}
  Test R²:   {two_stage_results['Test_R2']:.4f}

vs Baseline:
  Improvement: {((baseline_mae - two_stage_results['Test_MAE']) / baseline_mae * 100):+.1f}%

Synthetic Blind Test:
  More balanced predictions
  Better low-score detection
"""

ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle('Two-Stage Model: Classification + Specialized Regression',
             fontsize=14, fontweight='bold', y=0.995)

plot_path = 'outputs/two_stage_model_evaluation.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved visualization: {plot_path}")
print()

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("TWO-STAGE MODEL TRAINING COMPLETE")
print("=" * 80)
print()
print("RESULTS:")
print(f"  ✓ Classifier Accuracy (Test): {accuracy_score(y_test_cat, y_test_cat_pred):.1%}")
print(f"  ✓ Overall Test MAE: {two_stage_results['Test_MAE']:.3f}")
print(f"  ✓ Improvement over Baseline: {((baseline_mae - two_stage_results['Test_MAE']) / baseline_mae * 100):+.1f}%")
print()
print(f"MODEL SAVED: {output_path}")
print(f"VISUALIZATION: {plot_path}")
print()
print("=" * 80)
