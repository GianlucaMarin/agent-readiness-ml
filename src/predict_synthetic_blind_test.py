"""
Synthetic Blind Test - Model Predictions
=========================================

Load synthetic test data, generate predictions with trained model,
and save results with predicted Overall_Score.

Author: Sandra Marin
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

print("=" * 80)
print("SYNTHETIC BLIND TEST - MODEL PREDICTIONS")
print("=" * 80)
print()

# ============================================================================
# 1. LOAD SYNTHETIC BLIND TEST DATA
# ============================================================================

print("Loading synthetic blind test data...")
blind_test_path = 'data/raw/synthetic_blind_test_50_websites.csv'
df = pd.read_csv(blind_test_path)

print(f"✓ Loaded: {len(df)} websites")
print(f"✓ Columns: {len(df.columns)}")
print()

# Show first few rows
print("First 3 rows (preview):")
print(df.head(3))
print()

# ============================================================================
# 2. IDENTIFY AND EXTRACT FEATURES
# ============================================================================

print("Extracting features...")

# Identify has_* feature columns
feature_cols = [col for col in df.columns if col.startswith('has_')]
print(f"✓ Found {len(feature_cols)} has_* features")

if len(feature_cols) != 41:
    print(f"⚠️ WARNING: Expected 41 features, but found {len(feature_cols)}")
    print(f"Feature columns: {feature_cols[:5]}... (showing first 5)")
print()

# Extract features
X_blind = df[feature_cols]

# Check for missing values
missing = X_blind.isnull().sum().sum()
if missing > 0:
    print(f"⚠️ WARNING: {missing} missing values found. Filling with 0...")
    X_blind = X_blind.fillna(0)
else:
    print("✓ No missing values")
print()

# ============================================================================
# 3. LOAD TRAINED MODEL
# ============================================================================

print("Loading trained model...")
model_path = 'models/random_forest_initial.joblib'
model_data = joblib.load(model_path)
model = model_data['model']

print(f"✓ Model loaded: Random Forest Regressor")
print(f"✓ Model type: {type(model).__name__}")
print(f"✓ Number of trees: {model.n_estimators}")
print(f"✓ Max depth: {model.max_depth}")
print()

# ============================================================================
# 4. GENERATE PREDICTIONS
# ============================================================================

print("Generating predictions for 50 websites...")
predictions = model.predict(X_blind)

print(f"✓ Predictions generated: {len(predictions)} values")
print()

# ============================================================================
# 5. ADD PREDICTIONS TO DATAFRAME
# ============================================================================

print("Adding predictions to dataframe...")

# Check if Overall_Score column exists
if 'Overall_Score' in df.columns:
    print("⚠️ Overall_Score column already exists. Overwriting with predictions...")
    df['Overall_Score'] = predictions
else:
    print("✓ Creating Overall_Score column with predictions...")
    df['Overall_Score'] = predictions

print()

# ============================================================================
# 6. CALCULATE SUMMARY STATISTICS
# ============================================================================

print("=" * 80)
print("PREDICTION SUMMARY STATISTICS")
print("=" * 80)
print()

mean_pred = np.mean(predictions)
median_pred = np.median(predictions)
min_pred = np.min(predictions)
max_pred = np.max(predictions)
std_pred = np.std(predictions)

print(f"Mean Predicted Score:   {mean_pred:.2f}")
print(f"Median Predicted Score: {median_pred:.2f}")
print(f"Min Predicted Score:    {min_pred:.2f}")
print(f"Max Predicted Score:    {max_pred:.2f}")
print(f"Std Predicted Score:    {std_pred:.2f}")
print()

# Score distribution
low_count = np.sum(predictions < 30)
med_count = np.sum((predictions >= 30) & (predictions < 70))
high_count = np.sum(predictions >= 70)

print("Score Distribution:")
print(f"  Low (0-30):      {low_count:2d} websites ({low_count/len(predictions)*100:.1f}%)")
print(f"  Medium (30-70):  {med_count:2d} websites ({med_count/len(predictions)*100:.1f}%)")
print(f"  High (70-100):   {high_count:2d} websites ({high_count/len(predictions)*100:.1f}%)")
print()

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

print("Saving results...")

# Save CSV with predictions
output_path = 'data/raw/synthetic_blind_test_50_websites_RESULTS.csv'
df.to_csv(output_path, index=False)
print(f"✓ Saved: {output_path}")
print()

# Save summary statistics
summary_path = 'data/raw/synthetic_blind_test_SUMMARY.txt'
with open(summary_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("SYNTHETIC BLIND TEST - PREDICTION SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Dataset: {len(predictions)} synthetic websites\n")
    f.write(f"Model: Random Forest Regressor (random_forest_initial.joblib)\n")
    f.write(f"Date: December 2025\n\n")

    f.write("SUMMARY STATISTICS:\n")
    f.write(f"  Mean:   {mean_pred:.2f}\n")
    f.write(f"  Median: {median_pred:.2f}\n")
    f.write(f"  Min:    {min_pred:.2f}\n")
    f.write(f"  Max:    {max_pred:.2f}\n")
    f.write(f"  Std:    {std_pred:.2f}\n\n")

    f.write("SCORE DISTRIBUTION:\n")
    f.write(f"  Low (0-30):     {low_count:2d} websites ({low_count/len(predictions)*100:.1f}%)\n")
    f.write(f"  Medium (30-70): {med_count:2d} websites ({med_count/len(predictions)*100:.1f}%)\n")
    f.write(f"  High (70-100):  {high_count:2d} websites ({high_count/len(predictions)*100:.1f}%)\n\n")

    f.write("TOP 10 HIGHEST PREDICTED SCORES:\n")
    top_10 = df.nlargest(10, 'Overall_Score')[['Website_Name', 'Overall_Score']]
    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        f.write(f"  {idx:2d}. {row['Website_Name']:30s} - {row['Overall_Score']:.2f}\n")

    f.write("\nBOTTOM 10 LOWEST PREDICTED SCORES:\n")
    bottom_10 = df.nsmallest(10, 'Overall_Score')[['Website_Name', 'Overall_Score']]
    for idx, (_, row) in enumerate(bottom_10.iterrows(), 1):
        f.write(f"  {idx:2d}. {row['Website_Name']:30s} - {row['Overall_Score']:.2f}\n")

print(f"✓ Saved: {summary_path}")
print()

# ============================================================================
# 8. CREATE VISUALIZATIONS
# ============================================================================

print("Creating visualizations...")

# Visualization 1: Score Distribution Histogram
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Histogram with categories
ax1 = axes[0, 0]
bins = np.arange(0, 105, 5)
counts, edges, patches = ax1.hist(predictions, bins=bins, edgecolor='black', linewidth=1.5)

# Color bars by category
for i, patch in enumerate(patches):
    if edges[i] < 30:
        patch.set_facecolor('#e74c3c')  # Red for Low
    elif edges[i] < 70:
        patch.set_facecolor('#f39c12')  # Orange for Medium
    else:
        patch.set_facecolor('#2ecc71')  # Green for High

ax1.axvline(mean_pred, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_pred:.1f}')
ax1.axvline(median_pred, color='red', linestyle='--', linewidth=2, label=f'Median: {median_pred:.1f}')
ax1.set_xlabel('Predicted Overall Score', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Websites', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Predicted Scores', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Subplot 2: Category Bar Chart
ax2 = axes[0, 1]
categories = ['Low\n(0-30)', 'Medium\n(30-70)', 'High\n(70-100)']
category_counts = [low_count, med_count, high_count]
colors = ['#e74c3c', '#f39c12', '#2ecc71']

bars = ax2.bar(categories, category_counts, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax2.set_ylabel('Number of Websites', fontsize=12, fontweight='bold')
ax2.set_title('Score Category Distribution', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, count in zip(bars, category_counts):
    height = bar.get_height()
    percentage = count / len(predictions) * 100
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(count)}\n({percentage:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Subplot 3: Box Plot
ax3 = axes[1, 0]
bp = ax3.boxplot([predictions], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][0].set_alpha(0.7)
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2)

# Add category shading
ax3.axhspan(0, 30, alpha=0.1, color='red', label='Low')
ax3.axhspan(30, 70, alpha=0.1, color='orange', label='Medium')
ax3.axhspan(70, 100, alpha=0.1, color='green', label='High')

ax3.set_ylabel('Predicted Overall Score', fontsize=12, fontweight='bold')
ax3.set_title('Score Distribution (Box Plot)', fontsize=13, fontweight='bold')
ax3.set_xticklabels(['All Websites'])
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# Subplot 4: Top/Bottom 10
ax4 = axes[1, 1]
top_5 = df.nlargest(5, 'Overall_Score')
bottom_5 = df.nsmallest(5, 'Overall_Score')

# Combine and sort
combined = pd.concat([bottom_5, top_5]).sort_values('Overall_Score')
y_pos = np.arange(len(combined))
colors_bar = ['#e74c3c' if score < 30 else '#f39c12' if score < 70 else '#2ecc71'
              for score in combined['Overall_Score']]

bars = ax4.barh(y_pos, combined['Overall_Score'], color=colors_bar,
                edgecolor='black', linewidth=1.5, alpha=0.8)
ax4.set_yticks(y_pos)
ax4.set_yticklabels([name[:20] for name in combined['Website_Name']], fontsize=9)
ax4.set_xlabel('Predicted Overall Score', fontsize=12, fontweight='bold')
ax4.set_title('Top 5 Highest & Bottom 5 Lowest Scores', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, score in zip(bars, combined['Overall_Score']):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2.,
             f'{score:.1f}', ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plot_path = 'data/raw/synthetic_blind_test_PREDICTIONS_PLOT.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {plot_path}")
print()

# ============================================================================
# 9. FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("PREDICTION COMPLETE")
print("=" * 80)
print()
print(f"✓ Processed {len(predictions)} synthetic websites")
print(f"✓ Predictions saved to: {output_path}")
print(f"✓ Summary saved to: {summary_path}")
print(f"✓ Visualization saved to: {plot_path}")
print()
print("QUICK STATS:")
print(f"  Average Score: {mean_pred:.2f}")
print(f"  Score Range:   {min_pred:.2f} - {max_pred:.2f}")
print(f"  High Quality:  {high_count}/{len(predictions)} websites ({high_count/len(predictions)*100:.1f}%)")
print()
print("=" * 80)
