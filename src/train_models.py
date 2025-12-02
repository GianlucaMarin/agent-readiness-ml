"""
Model Training Script for Agent Readiness ML
Train Random Forest and XGBoost models
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost, skip if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    print(f"⚠️  XGBoost not available: {str(e)[:100]}")
    print("   Will train only Random Forest")
    XGBOOST_AVAILABLE = False

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

print("=" * 70)
print("AGENT READINESS ML: MODEL TRAINING")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/7] Loading prepared data...")

X_train = pd.read_csv('data/processed/X_train.csv')
X_val = pd.read_csv('data/processed/X_val.csv')
X_test = pd.read_csv('data/processed/X_test.csv')

y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

print(f"✓ Data loaded:")
print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Val:   {X_val.shape[0]} samples")
print(f"  Test:  {X_test.shape[0]} samples (held out)")

# ============================================================================
# STEP 2: BASELINE
# ============================================================================
print("\n[2/7] Calculating baseline performance...")

baseline_pred = np.full(len(y_val), y_train.mean())
baseline_mae = mean_absolute_error(y_val, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
baseline_r2 = r2_score(y_val, baseline_pred)

print(f"✓ Baseline (predict mean={y_train.mean():.2f}):")
print(f"  MAE:  {baseline_mae:.2f}")
print(f"  RMSE: {baseline_rmse:.2f}")
print(f"  R²:   {baseline_r2:.4f}")
print(f"  🎯 Goal: Beat MAE < {baseline_mae:.2f}")

# ============================================================================
# STEP 3: TRAIN RANDOM FOREST
# ============================================================================
print("\n[3/7] Training Random Forest...")

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train, y_train)

# Predictions
rf_train_pred = rf_model.predict(X_train)
rf_val_pred = rf_model.predict(X_val)

# Metrics
rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_val_mae = mean_absolute_error(y_val, rf_val_pred)
rf_val_r2 = r2_score(y_val, rf_val_pred)

print(f"✓ Random Forest trained:")
print(f"  Train: MAE={rf_train_mae:.2f}, R²={rf_train_r2:.4f}")
print(f"  Val:   MAE={rf_val_mae:.2f}, R²={rf_val_r2:.4f}")
improvement_rf = ((baseline_mae - rf_val_mae) / baseline_mae) * 100
print(f"  Improvement: {improvement_rf:.1f}%")

# Visualize RF predictions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(y_train, rf_train_pred, alpha=0.6, s=50)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Score')
axes[0].set_ylabel('Predicted Score')
axes[0].set_title(f'RF: Training (MAE={rf_train_mae:.2f})')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_val, rf_val_pred, alpha=0.6, s=50, color='orange')
axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Score')
axes[1].set_ylabel('Predicted Score')
axes[1].set_title(f'RF: Validation (MAE={rf_val_mae:.2f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/rf_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()
print("  📊 Saved: outputs/rf_predictions_vs_actual.png")

# RF Feature importance
rf_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

top_20_rf = rf_importances.head(20).copy()
top_20_rf['Feature_Short'] = top_20_rf['Feature'].str.replace('has_', '')

plt.figure(figsize=(12, 8))
plt.barh(range(len(top_20_rf)), top_20_rf['Importance'], color='steelblue', edgecolor='black')
plt.yticks(range(len(top_20_rf)), top_20_rf['Feature_Short'])
plt.xlabel('Feature Importance')
plt.title('Random Forest: Top 20 Features')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('outputs/rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  📊 Saved: outputs/rf_feature_importance.png")

# ============================================================================
# STEP 4: TRAIN XGBOOST (if available)
# ============================================================================
if XGBOOST_AVAILABLE:
    print("\n[4/7] Training XGBoost...")

    try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        xgb_model.fit(X_train, y_train)

        # Predictions
        xgb_train_pred = xgb_model.predict(X_train)
        xgb_val_pred = xgb_model.predict(X_val)

        # Metrics
        xgb_train_mae = mean_absolute_error(y_train, xgb_train_pred)
        xgb_train_r2 = r2_score(y_train, xgb_train_pred)
        xgb_val_mae = mean_absolute_error(y_val, xgb_val_pred)
        xgb_val_r2 = r2_score(y_val, xgb_val_pred)

        print(f"✓ XGBoost trained:")
        print(f"  Train: MAE={xgb_train_mae:.2f}, R²={xgb_train_r2:.4f}")
        print(f"  Val:   MAE={xgb_val_mae:.2f}, R²={xgb_val_r2:.4f}")
        improvement_xgb = ((baseline_mae - xgb_val_mae) / baseline_mae) * 100
        print(f"  Improvement: {improvement_xgb:.1f}%")

        # Visualize XGB predictions
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].scatter(y_train, xgb_train_pred, alpha=0.6, s=50, color='green')
        axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Score')
        axes[0].set_ylabel('Predicted Score')
        axes[0].set_title(f'XGB: Training (MAE={xgb_train_mae:.2f})')
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(y_val, xgb_val_pred, alpha=0.6, s=50, color='purple')
        axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Score')
        axes[1].set_ylabel('Predicted Score')
        axes[1].set_title(f'XGB: Validation (MAE={xgb_val_mae:.2f})')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('outputs/xgb_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  📊 Saved: outputs/xgb_predictions_vs_actual.png")

        # XGB Feature importance
        xgb_importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        top_20_xgb = xgb_importances.head(20).copy()
        top_20_xgb['Feature_Short'] = top_20_xgb['Feature'].str.replace('has_', '')

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_20_xgb)), top_20_xgb['Importance'], color='purple', edgecolor='black')
        plt.yticks(range(len(top_20_xgb)), top_20_xgb['Feature_Short'])
        plt.xlabel('Feature Importance')
        plt.title('XGBoost: Top 20 Features')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('outputs/xgb_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  📊 Saved: outputs/xgb_feature_importance.png")

        XGBOOST_TRAINED = True
    except Exception as e:
        print(f"  ⚠️  XGBoost training failed: {str(e)}")
        XGBOOST_TRAINED = False
else:
    print("\n[4/7] Skipping XGBoost (not available)")
    XGBOOST_TRAINED = False

# ============================================================================
# STEP 5: MODEL COMPARISON
# ============================================================================
print("\n[5/7] Comparing models...")

if XGBOOST_TRAINED:
    comparison = pd.DataFrame({
        'Model': ['Baseline', 'Random Forest', 'XGBoost'],
        'MAE': [baseline_mae, rf_val_mae, xgb_val_mae],
        'R²': [baseline_r2, rf_val_r2, xgb_val_r2]
    })

    best_model_name = comparison.iloc[1:]['MAE'].idxmin()
    best_model_name = comparison.iloc[best_model_name]['Model']
    best_mae = comparison.iloc[1:]['MAE'].min()
else:
    comparison = pd.DataFrame({
        'Model': ['Baseline', 'Random Forest'],
        'MAE': [baseline_mae, rf_val_mae],
        'R²': [baseline_r2, rf_val_r2]
    })
    best_model_name = 'Random Forest'
    best_mae = rf_val_mae

print("✓ Comparison table:")
print(comparison.to_string(index=False))
print(f"\n🏆 Best model: {best_model_name} (MAE={best_mae:.2f})")

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(comparison['Model'], comparison['MAE'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(comparison)], edgecolor='black')
axes[0].set_ylabel('MAE (lower is better)')
axes[0].set_title('Mean Absolute Error')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(comparison['Model'], comparison['R²'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(comparison)], edgecolor='black')
axes[1].set_ylabel('R² (higher is better)')
axes[1].set_title('R² Score')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  📊 Saved: outputs/model_comparison.png")

# ============================================================================
# STEP 6: ERROR ANALYSIS
# ============================================================================
print("\n[6/7] Performing error analysis...")

rf_residuals = y_val - rf_val_pred

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.scatter(rf_val_pred, rf_residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Score')
ax.set_ylabel('Residual (Actual - Predicted)')
ax.set_title(f'Random Forest: Residual Plot')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/residual_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("  📊 Saved: outputs/residual_plots.png")

# Find worst predictions
rf_errors = np.abs(y_val - rf_val_pred)
worst_5_idx = np.argsort(rf_errors)[-5:][::-1]

print("✓ Top 5 worst RF predictions:")
for idx in worst_5_idx:
    print(f"  Actual={y_val[idx]:.1f}, Predicted={rf_val_pred[idx]:.1f}, Error={rf_errors[idx]:.1f}")

# ============================================================================
# STEP 7: SAVE MODELS
# ============================================================================
print("\n[7/7] Saving models...")

# Save Random Forest
rf_metadata = {
    'model': rf_model,
    'hyperparameters': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_leaf': 3,
        'random_state': 42
    },
    'performance': {
        'train_mae': rf_train_mae,
        'train_r2': rf_train_r2,
        'val_mae': rf_val_mae,
        'val_r2': rf_val_r2
    },
    'feature_names': list(X_train.columns)
}
joblib.dump(rf_metadata, 'models/random_forest_initial.joblib')
print(f"✓ Saved: models/random_forest_initial.joblib")

if XGBOOST_TRAINED:
    xgb_metadata = {
        'model': xgb_model,
        'hyperparameters': {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 8,
            'random_state': 42
        },
        'performance': {
            'train_mae': xgb_train_mae,
            'train_r2': xgb_train_r2,
            'val_mae': xgb_val_mae,
            'val_r2': xgb_val_r2
        },
        'feature_names': list(X_train.columns)
    }
    joblib.dump(xgb_metadata, 'models/xgboost_initial.joblib')
    print(f"✓ Saved: models/xgboost_initial.joblib")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"\n🎯 Baseline MAE: {baseline_mae:.2f}")
print(f"🌲 Random Forest MAE: {rf_val_mae:.2f} ({improvement_rf:.1f}% improvement)")
if XGBOOST_TRAINED:
    print(f"🚀 XGBoost MAE: {xgb_val_mae:.2f} ({improvement_xgb:.1f}% improvement)")
print(f"\n🏆 Best Model: {best_model_name}")
print("\n📊 Generated outputs:")
print("  • rf_predictions_vs_actual.png")
print("  • rf_feature_importance.png")
if XGBOOST_TRAINED:
    print("  • xgb_predictions_vs_actual.png")
    print("  • xgb_feature_importance.png")
print("  • model_comparison.png")
print("  • residual_plots.png")
print("  • random_forest_initial.joblib")
if XGBOOST_TRAINED:
    print("  • xgboost_initial.joblib")
print("\n" + "=" * 70)
