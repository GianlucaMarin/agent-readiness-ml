# ML Models

## ✅ Production Model

### `random_forest_initial.joblib` ← **USE THIS**

**Das offizielle Production-Modell für Website Quality Score Prediction.**

**Performance:**
- Test MAE: 0,636
- Test R²: 0,992
- Validation MAE: 1,09

**Verwendung:**
```python
import joblib
model_data = joblib.load('models/random_forest_initial.joblib')
model = model_data['model']
predictions = model.predict(X)
```

**Verwendet in:**
- `src/evaluate_test_set.py`
- `src/predict_synthetic_blind_test.py`

**Status:** ✅ **Production-Ready**

---

## ❌ Experimental Models (NICHT verwenden!)

Der Ordner `experiments/` enthält fehlgeschlagene Modell-Experimente:

### 1. `experiments/random_forest_weighted.joblib`
- Sample Weighting Ansatz
- Test MAE: 0,705 (-10,9% vs. Baseline)
- Status: ❌ Fehlgeschlagen

### 2. `experiments/two_stage_model.joblib`
- SMOTE + Specialized Regressors
- Test MAE: 0,967 (-52,1% vs. Baseline)
- Status: ❌❌ Kritisch fehlgeschlagen

**Diese Modelle sind nur für Dokumentationszwecke vorhanden.**

Siehe [experiments/README.md](experiments/README.md) für Details.

---

## Model Vergleich

| Modell | Location | Test MAE | Status |
|--------|----------|----------|--------|
| **Baseline** | `random_forest_initial.joblib` | **0,636** | ✅ Production |
| Sample Weighted | `experiments/random_forest_weighted.joblib` | 0,705 | ❌ Deprecated |
| Two-Stage | `experiments/two_stage_model.joblib` | 0,967 | ❌ Deprecated |

---

## Hinweise für Entwickler

**Production Deployments:**
- Verwende **ausschließlich** `random_forest_initial.joblib`
- Lade das Modell mit `joblib.load()`
- Modell erwartet 41 `has_*` Features

**Bekannte Limitationen:**
- Überschätzt Low-Scores um durchschnittlich +3,2 Punkte
- Empfohlen nur für High-Score-Bereich (Score > 60)
- Class Imbalance: Training-Daten sind 65% High-Scores

**Dokumentation:**
- Vollständiges README: [../README.md](../README.md)
- Model Comparison: [../outputs/MODEL_COMPARISON_SUMMARY.txt](../outputs/MODEL_COMPARISON_SUMMARY.txt)
