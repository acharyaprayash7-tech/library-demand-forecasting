"""
Model Evaluation & Final Selection
--------------------------------------
Re-trains all three models (same time-aware split as train_model.py),
checks for overfitting (train vs test gap), inspects residuals for
the top candidate, and saves the final selected model + metadata
for use by the prediction pipeline (Phase 11).
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES_PATH = "data/processed/library_demand_features.csv"
ID_COLS = ["record_id", "date", "book_id", "book_title", "category"]
TARGET_COL = "target_next_month_demand"
RANDOM_SEED = 42

# ----------------------------------------------------------------
# 1. LOAD DATA + SAME TIME-AWARE SPLIT AS train_model.py
# ----------------------------------------------------------------
df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
df = df.sort_values(["date", "book_id"]).reset_index(drop=True)

unique_dates = sorted(df["date"].unique())
n_dates = len(unique_dates)
test_month_count = max(1, int(round(n_dates * 0.2)))
split_date = unique_dates[-test_month_count]

train_df = df[df["date"] < split_date].copy()
test_df = df[df["date"] >= split_date].copy()

feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET_COL]]

X_train = train_df[feature_cols].astype(float)
y_train = train_df[TARGET_COL].astype(float)
X_test = test_df[feature_cols].astype(float)
y_test = test_df[TARGET_COL].astype(float)

print("=" * 60)
print("RE-TRAINING ALL MODELS FOR EVALUATION")
print("=" * 60)


def metrics_for(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


models = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        random_state=RANDOM_SEED, n_jobs=-1
    ),
    "xgboost": XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, n_jobs=-1
    ),
}

fitted_models = {}
overfit_report = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_metrics = metrics_for(y_train, train_preds)
    test_metrics = metrics_for(y_test, test_preds)

    rmse_gap = train_metrics["RMSE"] - test_metrics["RMSE"]
    overfit_report[name] = {
        "train": train_metrics,
        "test": test_metrics,
        "train_test_rmse_gap": round(rmse_gap, 4),
    }

    print(f"\n[{name}]")
    print(f"  Train -> MAE {train_metrics['MAE']}, RMSE {train_metrics['RMSE']}, R2 {train_metrics['R2']}")
    print(f"  Test  -> MAE {test_metrics['MAE']}, RMSE {test_metrics['RMSE']}, R2 {test_metrics['R2']}")
    print(f"  Train/Test RMSE gap: {rmse_gap:.4f} "
          f"(large negative gap can indicate overfitting - train RMSE much lower than test RMSE)")

# ----------------------------------------------------------------
# 2. RESIDUAL ANALYSIS FOR EACH MODEL (on test set)
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("RESIDUAL ANALYSIS (test set)")
print("=" * 60)

residual_summary = {}
for name, model in fitted_models.items():
    preds = model.predict(X_test)
    residuals = y_test.values - preds
    residual_summary[name] = {
        "mean_residual": round(float(np.mean(residuals)), 4),
        "std_residual": round(float(np.std(residuals)), 4),
        "max_abs_residual": round(float(np.max(np.abs(residuals))), 4),
    }
    print(f"[{name}] mean residual: {residual_summary[name]['mean_residual']} "
          f"(close to 0 is good - means no strong systematic bias)")

# ----------------------------------------------------------------
# 3. SELECT FINAL MODEL
#    Rule: lowest test RMSE, UNLESS it shows a much larger
#    train/test gap than alternatives (sign of overfitting) -
#    in that case we prefer the more stable model.
#    This decision is printed transparently for the report.
# ----------------------------------------------------------------
test_rmse_ranking = sorted(
    overfit_report.items(), key=lambda kv: kv[1]["test"]["RMSE"]
)

print("\n" + "=" * 60)
print("FINAL MODEL SELECTION")
print("=" * 60)
for name, info in test_rmse_ranking:
    print(f"{name}: test RMSE={info['test']['RMSE']}, "
          f"train/test gap={info['train_test_rmse_gap']}")

best_model_name = test_rmse_ranking[0][0]
best_model = fitted_models[best_model_name]

print(f"\nSELECTED MODEL: {best_model_name}")
print("(Selection based on lowest test RMSE among the trained models. "
      "Review the train/test gap above manually - if the top model shows "
      "a much larger gap than the runner-up, consider that trade-off "
      "in your report even though this script picks by RMSE.)")

# ----------------------------------------------------------------
# 4. SAVE FINAL MODEL + METADATA (for Phase 11 prediction pipeline)
# ----------------------------------------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/demand_model.pkl")
print(f"\nSaved final model to: models/demand_model.pkl")

metadata = {
    "selected_model": best_model_name,
    "feature_columns": feature_cols,
    "split_date": str(pd.Timestamp(split_date).date()),
    "evaluation": overfit_report,
    "residuals": residual_summary,
}

with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved model metadata to: models/model_metadata.json")
print("\nThis metadata file will be used by src/prediction.py (Phase 11)")
print("to know exactly which features the saved model expects, in order.")