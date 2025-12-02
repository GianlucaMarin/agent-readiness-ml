# Agent Readiness ML Model

Machine Learning project for predicting agent readiness scores based on website and organizational features.

## Project Overview

This project uses machine learning to score and predict agent readiness for websites. The model analyzes 41 features across 178 websites to provide accurate readiness assessments.

### Dataset
- **Samples**: 178 websites
- **Features**: 41 features including:
  - Technical capabilities
  - Content quality metrics
  - Organizational readiness indicators
  - Infrastructure assessments
- **Target**: Agent Readiness Score

### Model Performance
- **Mean Absolute Error (MAE)**: TBD
- **R² Score**: TBD
- **RMSE**: TBD

## Project Structure

```
agent-readiness-ml/
├── data/
│   ├── raw/          # Original Excel/CSV data files
│   └── processed/    # Preprocessed and cleaned data
├── models/           # Saved ML models (.joblib files)
├── notebooks/        # Jupyter notebooks for exploration
├── src/              # Python source code
│   └── __init__.py
├── outputs/          # Generated plots, reports, results
├── .gitignore        # Git ignore rules
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd "Ml Agent Ready"
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add Your Data

Place your data files in the `data/raw/` directory:
- Excel files (.xlsx)
- CSV files (.csv)

## Usage

### Running Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to the `notebooks/` directory to explore data and train models.

### Training Models

```python
# Example usage (to be implemented)
from src import train_model

model = train_model.train()
```

## Development

### Adding New Features

1. Add new feature engineering code to `src/`
2. Update preprocessing pipeline
3. Retrain models and evaluate performance

### Model Evaluation

All model evaluation metrics and plots are saved to `outputs/`.

## Dependencies

See [requirements.txt](requirements.txt) for full list of dependencies.

Key libraries:
- **pandas**: Data manipulation
- **scikit-learn**: ML algorithms
- **xgboost**: Gradient boosting
- **matplotlib/seaborn**: Visualization
- **openpyxl**: Excel file handling

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## License

TBD

## Contact

TBD
