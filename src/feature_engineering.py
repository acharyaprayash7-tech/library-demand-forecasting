"""
Feature Engineering
---------------------
Builds ML-ready features from the cleaned dataset while avoiding
data leakage. Target = next month's borrowed_count for each book.

Leakage rule: every feature must be something we would actually know
at prediction time (i.e., BEFORE the month we're predicting).
"""

import pandas as pd
import numpy as np
import os

df = pd.read_csv("data/processed/library_demand_clean.csv", parse_dates=["date"])
df = df.sort_values(["book_id", "date"]).reset_index(drop=True)

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)
print(f"Starting rows: {len(df)}")

# ----------------------------------------------------------------
# 1. TARGET VARIABLE: next month's borrowed_count, per book
# ----------------------------------------------------------------
df["target_next_month_demand"] = df.groupby("book_id")["borrowed_count"].shift(-1)

# ----------------------------------------------------------------
# 2. LAG FEATURES (past values only - safe, no leakage)
# ----------------------------------------------------------------
df["lag_1_borrowed"] = df.groupby("book_id")["borrowed_count"].shift(1)
df["lag_2_borrowed"] = df.groupby("book_id")["borrowed_count"].shift(2)

# ----------------------------------------------------------------
# 3. ROLLING AVERAGE (past 3 months INCLUDING current, shifted so
#    it never includes the target month itself)
# ----------------------------------------------------------------
df["rolling_3_avg_borrowed"] = (
    df.groupby("book_id")["borrowed_count"]
      .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
)
# Note: this rolling average includes the CURRENT month's borrowed_count,
# which is fine because "current month" is fully observed before we
# predict "next month" - this is not future information.

# ----------------------------------------------------------------
# 4. TIME TREND INDEX (0, 1, 2, ... per book's chronological order)
# ----------------------------------------------------------------
df["month_index"] = df.groupby("book_id").cumcount()

# ----------------------------------------------------------------
# 5. CATEGORICAL ENCODING
#    Using simple label/one-hot encoding - documented for the report.
# ----------------------------------------------------------------
df["exam_period"] = df["exam_period"].astype(int)
df["holiday_indicator"] = df["holiday_indicator"].astype(int)
df["semester_encoded"] = df["semester"].map({"Odd": 0, "Even": 1})

category_dummies = pd.get_dummies(df["category"], prefix="cat")
df = pd.concat([df, category_dummies], axis=1)

# ----------------------------------------------------------------
# 6. DROP ROWS WITH NO VALID TARGET (last month per book) OR
#    INSUFFICIENT LAG HISTORY (first 2 months per book)
# ----------------------------------------------------------------
before_drop = len(df)
df_features = df.dropna(subset=["target_next_month_demand", "lag_1_borrowed", "lag_2_borrowed"])
after_drop = len(df_features)

print(f"Rows dropped (no future target / insufficient lag history): {before_drop - after_drop}")
print(f"Rows remaining for modeling: {after_drop}")

# ----------------------------------------------------------------
# 7. FINAL FEATURE LIST (documented explicitly)
# ----------------------------------------------------------------
feature_cols = (
    ["lag_1_borrowed", "lag_2_borrowed", "rolling_3_avg_borrowed",
     "month_index", "exam_period", "holiday_indicator", "semester_encoded",
     "available_copies", "reservation_count", "renewal_count",
     "publication_year"]
    + list(category_dummies.columns)
)

id_cols = ["record_id", "date", "book_id", "book_title", "category"]
target_col = "target_next_month_demand"

final_df = df_features[id_cols + feature_cols + [target_col]].reset_index(drop=True)

print("-" * 60)
print(f"Feature columns ({len(feature_cols)}):")
for c in feature_cols:
    print(f"  - {c}")
print(f"Target column: {target_col}")

# ----------------------------------------------------------------
# 8. SAVE
# ----------------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
final_df.to_csv("data/processed/library_demand_features.csv", index=False)
print("-" * 60)
print("Saved feature table to: data/processed/library_demand_features.csv")
print(f"Final shape: {final_df.shape}")