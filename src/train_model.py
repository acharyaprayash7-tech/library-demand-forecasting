"""
Model Training - Baseline, Random Forest, and XGBoost Comparison
--------------------------------------------------------------------
Loads the engineered feature table, performs a TIME-AWARE train/test
split (chronological, not random), and trains three models:
  1. Linear Regression (baseline)
  2. Random Forest Regressor
  3. XGBoost Regressor
All three are evaluated on the SAME split for a fair comparison.
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES_PATH = "data/processed/library_demand_features.csv"
ID_COLS = ["record_id", "date", "book_id", "book_title", "category"]
TARGET_COL = "target_next_month_demand"
RANDOM_SEED = 42

# ----------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------
df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
df = df.sort_values(["date", "book_id"]).reset_index(drop=True)

print("=" * 60)
print("TIME-AWARE TRAIN/TEST SPLIT")
print("=" * 60)

# ----------------------------------------------------------------
# 2. TIME-AWARE SPLIT (chronological, NOT random)
# ----------------------------------------------------------------
unique_dates = sorted(df["date"].unique())
n_dates = len(unique_dates)
test_month_count = max(1, int(round(n_dates * 0.2)))
split_date = unique_dates[-test_month_count]

train_df = df[df["date"] < split_date].copy()
test_df = df[df["date"] >= split_date].copy()

print(f"Total unique months in feature table : {n_dates}")
print(f"Split date (first month of test set) : {pd.Timestamp(split_date).date()}")
print(f"Training rows : {len(train_df)}")
print(f"Testing rows  : {len(test_df)}")

if len(test_df) == 0 or len(train_df) == 0:
    raise ValueError("Split produced an empty train or test set - check date range in feature table.")

# ----------------------------------------------------------------
# 3. SEPARATE FEATURES / TARGET
# ----------------------------------------------------------------
feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET_COL]]

X_train = train_df[feature_cols].astype(float)
y_train = train_df[TARGET_COL].astype(float)
X_test = test_df[feature_cols].astype(float)
y_test = test_df[TARGET_COL].astype(float)

print(f"\nNumber of features used: {len(feature_cols)}")


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n[{name}]")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R2   : {r2:.4f}")
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


all_metrics = {}

# ----------------------------------------------------------------
# 4. MODEL 1: LINEAR REGRESSION (baseline)
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL 1: LINEAR REGRESSION (BASELINE)")
print("=" * 60)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
all_metrics["linear_regression"] = evaluate("Linear Regression", y_test, lr_preds)

# ----------------------------------------------------------------
# 5. MODEL 2: RANDOM FOREST REGRESSOR
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL 2: RANDOM FOREST REGRESSOR")
print("=" * 60)
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=3,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
all_metrics["random_forest"] = evaluate("Random Forest", y_test, rf_preds)

# Feature importance (useful for the report)
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)
print("\nTop 10 Random Forest feature importances:")
print(importances.head(10).to_string())

# ----------------------------------------------------------------
# 6. MODEL 3: XGBOOST REGRESSOR
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL 3: XGBOOST REGRESSOR")
print("=" * 60)
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
all_metrics["xgboost"] = evaluate("XGBoost", y_test, xgb_preds)

# ----------------------------------------------------------------
# 7. COMPARISON TABLE
# ----------------------------------------------------------------
comparison_df = pd.DataFrame(all_metrics).T
comparison_df = comparison_df.sort_values("RMSE")  # lower RMSE = better, at top

print("\n" + "=" * 60)
print("MODEL COMPARISON (sorted by RMSE, best first)")
print("=" * 60)
print(comparison_df.to_string())

best_model_name = comparison_df.index[0]
print(f"\nBest model by RMSE: {best_model_name}")

# ----------------------------------------------------------------
# 8. SAVE RESULTS
# ----------------------------------------------------------------
os.makedirs("models", exist_ok=True)

results = {
    "split_date": str(pd.Timestamp(split_date).date()),
    "train_rows": len(train_df),
    "test_rows": len(test_df),
    "feature_count": len(feature_cols),
    "feature_columns": feature_cols,
    "models": all_metrics,
    "best_model_by_rmse": best_model_name,
}

with open("models/model_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved comparison results to: models/model_comparison.json")
print("NOTE: Final model selection (Phase 10) will consider RMSE, MAE, R2,")
print("      and practical factors together - not RMSE alone.")