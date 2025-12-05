# Experimental Models (NICHT für Production)

Dieser Ordner enthält experimentelle Modelle, die **nicht für Production verwendet werden sollen**.

## ❌ Fehlgeschlagene Experimente

### 1. `random_forest_weighted.joblib`
- **Ansatz:** Sample Weighting mit balanced class weights
- **Test MAE:** 0,705 (-10,9% vs. Baseline)
- **Problem:** Trade-off - Medium verbessert, aber Low verschlechtert
- **Status:** ❌ Fehlgeschlagen - schlechter als Baseline

### 2. `two_stage_model.joblib`
- **Ansatz:** SMOTE-balanced Classifier + Specialized Regressors
- **Test MAE:** 0,967 (-52,1% vs. Baseline)
- **Problem:** Classifier-Bias trotz 100% Accuracy, SMOTE löst Grundproblem nicht
- **Status:** ❌❌ Kritisch fehlgeschlagen - deutlich schlechter als Baseline

## ✅ Production Model

Das einzige für Production empfohlene Modell ist:

**`models/random_forest_initial.joblib`** (Baseline)
- Test MAE: 0,636
- R²: 0,992
- Status: ✅ Best Performance

## Warum sind die Experimente fehlgeschlagen?

**Root Cause:** Nur 9 Low-Score-Trainings-Samples (7,3%)
- Keine ML-Technik kann fehlende Daten kompensieren
- SMOTE interpoliert, schafft aber keine neuen Feature-Patterns
- Sample Weighting verschiebt Bias, löst ihn aber nicht

**Lesson Learned:**
> "Simple is better than complex when data is limited."

## Dokumentation

Vollständige Analyse siehe:
- [outputs/MODEL_COMPARISON_SUMMARY.txt](../../outputs/MODEL_COMPARISON_SUMMARY.txt)
- [outputs/MODEL_COMPARISON_FINAL.png](../../outputs/MODEL_COMPARISON_FINAL.png)
- [README.md - Section 4.12](../../README.md#412-two-stage-model-experiment-smote--specialized-regressors-fehlgeschlagen)
