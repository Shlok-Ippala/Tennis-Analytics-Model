"""
Predictive modeling pipeline for ATP tennis match outcomes.

Reads the feature-engineered CSV from the SQL pipeline, trains Logistic Regression,
Random Forest, and XGBoost models, evaluates them, and exports results as CSVs
for Power BI consumption.

Usage: python src/model.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "clean" / "atp_matches_features.csv"
OUTPUT_DIR = ROOT / "data" / "output"

# Time-based split boundaries
TRAIN_END_YEAR = 2021  # Train: <= 2021
TEST_START_YEAR = 2022  # Test: >= 2022

# Pre-match features only (available before the match is played)
BASELINE_DELTAS = ["delta_rank", "delta_rank_points", "delta_age", "delta_ht"]
BASELINE_MISSING = [f"{col}_missing" for col in BASELINE_DELTAS]
WIN_PCT_FEATURES = ["delta_win_pct_52w", "delta_win_pct_52w_surface", "delta_win_pct_10w"]
HISTORY_INDICATORS = [
    "has_history_A", "has_history_B",
    "has_surface_history_A", "has_surface_history_B",
    "has_10w_history_A", "has_10w_history_B",
]

FEATURE_COLS = BASELINE_DELTAS + BASELINE_MISSING + WIN_PCT_FEATURES + HISTORY_INDICATORS
TARGET_COL = "target"


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        print(f"Error: input file not found: {INPUT_PATH}", file=sys.stderr)
        print("Run sql/create_match_level_csv.py first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df):,} matches from {INPUT_PATH.name}")
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_mask = df["year"] <= TRAIN_END_YEAR
    test_mask = df["year"] >= TEST_START_YEAR

    X_train = df.loc[train_mask, FEATURE_COLS].copy()
    X_test = df.loc[test_mask, FEATURE_COLS].copy()
    y_train = df.loc[train_mask, TARGET_COL].copy()
    y_test = df.loc[test_mask, TARGET_COL].copy()

    # Impute missing values with 0 (missing indicators capture the missingness)
    for col in BASELINE_DELTAS + WIN_PCT_FEATURES:
        X_train[col] = X_train[col].fillna(0)
        X_test[col] = X_test[col].fillna(0)

    print(f"Train: {len(X_train):,} matches (<= {TRAIN_END_YEAR})")
    print(f"Test:  {len(X_test):,} matches (>= {TEST_START_YEAR})")
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    return X_train, X_test, y_train, y_test


def train_models(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> dict[str, dict]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            min_child_weight=20, random_state=42, n_jobs=-1,
            eval_metric="logloss",
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
        else:
            # Tree models don't need scaling but it doesn't hurt
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
        print(f"  {name} trained successfully")

    return results


def evaluate_models(
    results: dict[str, dict], y_test: pd.Series
) -> dict[str, dict]:
    metrics = {}
    for name, res in results.items():
        y_pred = res["y_pred"]
        y_prob = res["y_prob"]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        ll = log_loss(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        metrics[name] = {
            "accuracy": acc,
            "roc_auc": auc,
            "log_loss": ll,
            "confusion_matrix": cm,
        }
        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  ROC AUC:  {auc:.4f}")
        print(f"  Log Loss: {ll:.4f}")

    return metrics


def export_results(
    results: dict[str, dict],
    metrics: dict[str, dict],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Model metrics
    metrics_rows = []
    for name, m in metrics.items():
        metrics_rows.append({
            "model": name,
            "accuracy": round(m["accuracy"], 4),
            "roc_auc": round(m["roc_auc"], 4),
            "log_loss": round(m["log_loss"], 4),
        })
    pd.DataFrame(metrics_rows).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    print(f"\nSaved model_metrics.csv")

    # 2. Feature importances
    importance_rows = []
    for name, res in results.items():
        model = res["model"]
        if hasattr(model, "coef_"):
            importances = model.coef_[0]
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            continue
        for feat, imp in zip(FEATURE_COLS, importances):
            importance_rows.append({
                "model": name,
                "feature": feat,
                "importance": round(float(imp), 6),
            })
    pd.DataFrame(importance_rows).to_csv(OUTPUT_DIR / "feature_importances.csv", index=False)
    print(f"Saved feature_importances.csv")

    # 3. Predictions (test set with match context)
    test_mask = df["year"] >= TEST_START_YEAR
    pred_df = df.loc[test_mask, [
        "tourney_name", "tourney_date", "surface", "round", "year",
        "player_A_name", "player_B_name", "player_A_rank", "player_B_rank", "target",
    ]].copy().reset_index(drop=True)

    for name, res in results.items():
        safe_name = name.lower().replace(" ", "_")
        pred_df[f"prob_{safe_name}"] = np.round(res["y_prob"], 4)
        pred_df[f"pred_{safe_name}"] = res["y_pred"]

    pred_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    print(f"Saved predictions.csv ({len(pred_df):,} rows)")

    # 4. Confusion matrices
    cm_rows = []
    for name, m in metrics.items():
        cm = m["confusion_matrix"]
        cm_rows.append({
            "model": name,
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        })
    pd.DataFrame(cm_rows).to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False)
    print(f"Saved confusion_matrices.csv")

    # 5. Calibration data
    cal_rows = []
    for name, res in results.items():
        prob_true, prob_pred = calibration_curve(y_test, res["y_prob"], n_bins=10, strategy="uniform")
        for pt, pp in zip(prob_true, prob_pred):
            cal_rows.append({
                "model": name,
                "predicted_probability": round(float(pp), 4),
                "actual_probability": round(float(pt), 4),
            })
    pd.DataFrame(cal_rows).to_csv(OUTPUT_DIR / "calibration_data.csv", index=False)
    print(f"Saved calibration_data.csv")


def main() -> None:
    print("=" * 60)
    print("Tennis Match Prediction Pipeline")
    print("=" * 60)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    results = train_models(X_train, y_train, X_test)
    metrics = evaluate_models(results, y_test)
    export_results(results, metrics, X_test, y_test, df)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")
    print("Ready for Power BI consumption.")


if __name__ == "__main__":
    main()
