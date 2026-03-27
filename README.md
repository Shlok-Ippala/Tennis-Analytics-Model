# Tennis Match Prediction

Predicts ATP tennis match outcomes using pre-match features (rankings, rolling form, surface history). Combines statistical feature engineering with machine learning (Logistic Regression, Random Forest, XGBoost).

## Pipeline

```
Raw Data (data/tennis_atp-master/)
    ↓
[1] python sql/create_match_level_csv.py
    → Cleans, transforms, engineers all features
    → Output: data/clean/atp_matches_features.csv
    ↓
[2] python src/model.py
    → Trains LR + Random Forest + XGBoost
    → Output: data/output/*.csv
    ↓
[3] Power BI Dashboard
    → Connects to data/clean/ + data/output/
```

## Project Structure

```
tennis-win-analysis/
├── sql/create_match_level_csv.py   # DuckDB ETL pipeline
├── src/model.py                    # Modeling + evaluation + export
├── data/
│   ├── tennis_atp-master/          # Raw ATP match CSVs (1968-2024)
│   ├── clean/                      # Feature-engineered match data
│   └── output/                     # Model results for Power BI
├── archive/notebooks/              # Legacy analysis notebooks
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Generate features

```bash
python sql/create_match_level_csv.py
```

Loads raw ATP match CSVs (2005-2024), creates match-level records with:
- Delta features (rank, age, height, rank points, match stats)
- Rolling 52-week win percentages (overall + surface-specific)
- Rolling 10-week win percentages
- Missing value indicators

Output: `data/clean/atp_matches_features.csv`

### Step 2: Train models and export results

```bash
python src/model.py
```

Trains three models on pre-match features (train: ≤2021, test: ≥2022):
- Logistic Regression (interpretable baseline)
- Random Forest
- XGBoost

Exports to `data/output/`:
| File | Contents |
|------|----------|
| `model_metrics.csv` | Accuracy, ROC AUC, log loss per model |
| `feature_importances.csv` | Feature importance/coefficients per model |
| `predictions.csv` | Test set predictions with match context |
| `confusion_matrices.csv` | TP/FP/TN/FN per model |
| `calibration_data.csv` | Predicted vs actual probability bins |

### Step 3: Power BI Dashboard

See [docs/powerbi_guide.md](docs/powerbi_guide.md) for a full step-by-step setup guide covering:
- Importing all 6 CSVs into Power BI Desktop
- 5 dashboard pages: Match Exploration, Model Comparison, Feature Importance, Predictions, Calibration
- Formatting, themes, and refresh workflow

## Data Source

ATP match data from [Jeff Sackmann's tennis_atp repository](https://github.com/JeffSackmann/tennis_atp), covering professional singles matches from 1968-2024.
