"""
Data Preprocessing
-------------------
Cleans the raw synthetic library dataset and saves a processed version.
Defensive by design: checks are written to catch real-world issues
(missing values, duplicates, bad types, outliers) even though our
current synthetic data is already clean.
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/library_demand_data.csv"
PROCESSED_PATH = "data/processed/library_demand_clean.csv"

print("Loading raw dataset...")
df = pd.read_csv(RAW_PATH)
initial_rows = len(df)

# ----------------------------------------------------------------
# 1. MISSING VALUES
# ----------------------------------------------------------------
missing_before = df.isnull().sum().sum()
print(f"Missing values found: {missing_before}")

numeric_cols = ["available_copies", "borrowed_count", "returned_count",
                 "renewal_count", "reservation_count", "publication_year"]
for col in numeric_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  Filled missing '{col}' with median ({median_val})")

categorical_cols = ["book_title", "author", "category", "department", "semester"]
for col in categorical_cols:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"  Filled missing '{col}' with mode ({mode_val})")

# ----------------------------------------------------------------
# 2. DUPLICATES
# ----------------------------------------------------------------
dupes = df.duplicated().sum()
print(f"Duplicate rows found: {dupes}")
if dupes > 0:
    df = df.drop_duplicates()
    print(f"  Removed {dupes} duplicate rows")

# ----------------------------------------------------------------
# 3. DATA TYPES
# ----------------------------------------------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
bad_dates = df["date"].isnull().sum()
if bad_dates > 0:
    print(f"  WARNING: {bad_dates} rows had unparseable dates and were dropped")
    df = df.dropna(subset=["date"])

int_cols = ["record_id", "publication_year", "available_copies",
            "borrowed_count", "returned_count", "renewal_count",
            "reservation_count", "exam_period", "holiday_indicator"]
for col in int_cols:
    df[col] = df[col].astype(int)

df["category"] = df["category"].astype("category")
df["department"] = df["department"].astype("category")
df["semester"] = df["semester"].astype("category")

# ----------------------------------------------------------------
# 4. LOGICAL VALIDATION / OUTLIER CHECKS
# (flag, don't blindly delete - these are checks, not blind trimming)
# ----------------------------------------------------------------
negative_counts = (df[["borrowed_count", "returned_count", "renewal_count",
                        "reservation_count", "available_copies"]] < 0).sum().sum()
print(f"Negative count values found: {negative_counts}")
if negative_counts > 0:
    for col in ["borrowed_count", "returned_count", "renewal_count",
                "reservation_count", "available_copies"]:
        df[col] = df[col].clip(lower=0)
    print("  Clipped negative values to 0")

extreme_high = df[df["borrowed_count"] > df["borrowed_count"].quantile(0.999)]
print(f"Extreme high borrowed_count rows (top 0.1%): {len(extreme_high)} "
      f"(kept - genuine high demand, not treated as errors)")

# ----------------------------------------------------------------
# 5. SORT (important for later lag-feature correctness)
# ----------------------------------------------------------------
df = df.sort_values(["book_id", "date"]).reset_index(drop=True)

# ----------------------------------------------------------------
# 6. SAVE
# ----------------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)

print("=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)
print(f"Rows before : {initial_rows}")
print(f"Rows after  : {len(df)}")
print(f"Columns     : {list(df.columns)}")
print(f"Dtypes:\n{df.dtypes}")
print(f"Saved cleaned dataset to: {PROCESSED_PATH}")