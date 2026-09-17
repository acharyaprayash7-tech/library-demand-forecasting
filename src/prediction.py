"""
Prediction Pipeline
----------------------
Reusable function to predict next-month demand for a given book,
classify its demand level, and recommend additional copies.

This module does NOT recompute feature engineering - it reads the
already-engineered feature table and uses each book's most recent
row (which contains correctly computed lag/rolling features).
"""

import pandas as pd
import numpy as np
import joblib
import json
import math

MODEL_PATH = "models/demand_model.pkl"
METADATA_PATH = "models/model_metadata.json"
FEATURES_PATH = "data/processed/library_demand_features.csv"

# ----------------------------------------------------------------
# LOAD ONCE (module-level) - avoids reloading the model on every call
# ----------------------------------------------------------------
_model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as f:
    _metadata = json.load(f)

_feature_cols = _metadata["feature_columns"]

_features_df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
_features_df = _features_df.sort_values(["book_id", "date"])

# ----------------------------------------------------------------
# DEMAND LEVEL THRESHOLDS
# Computed from the actual historical distribution of the target
# variable across ALL books (tertiles: bottom third = Low,
# middle third = Medium, top third = High). Documented, not guessed.
# ----------------------------------------------------------------
_target_values = _features_df["target_next_month_demand"]
_low_cutoff = _target_values.quantile(1 / 3)
_high_cutoff = _target_values.quantile(2 / 3)


def classify_demand_level(predicted_value: float) -> str:
    """Classify a predicted demand value into Low/Medium/High
    using tertile cutoffs from historical data."""
    if predicted_value <= _low_cutoff:
        return "Low"
    elif predicted_value <= _high_cutoff:
        return "Medium"
    else:
        return "High"


def get_available_book_ids():
    """Returns the list of all book_ids known to the system."""
    return sorted(_features_df["book_id"].unique().tolist())


def predict_demand(book_id: str) -> dict:
    """
    Predicts next-month demand for a given book_id.

    Returns a dictionary with:
      - book_id, book_title, category
      - current_stock (available_copies)
      - predicted_demand
      - demand_level (Low/Medium/High)
      - recommended_additional_copies
      - recommendation_note (explains this is a suggestion, not a rule)

    Raises ValueError if the book_id is not found.
    """
    book_rows = _features_df[_features_df["book_id"] == book_id]

    if book_rows.empty:
        raise ValueError(
            f"book_id '{book_id}' not found in the feature table. "
            f"Use get_available_book_ids() to see valid IDs."
        )

    # Most recent row for this book = most up-to-date lag/rolling features
    latest_row = book_rows.sort_values("date").iloc[-1]

    # Build the feature vector in the EXACT order the model was trained on
    X = latest_row[_feature_cols].astype(float).values.reshape(1, -1)

    predicted_demand = float(_model.predict(X)[0])
    predicted_demand = max(predicted_demand, 0.0)  # demand can't be negative

    demand_level = classify_demand_level(predicted_demand)

    current_stock = int(latest_row["available_copies"])
    shortfall = predicted_demand - current_stock
    recommended_additional_copies = max(0, math.ceil(shortfall))

    return {
        "book_id": book_id,
        "book_title": latest_row["book_title"],
        "category": latest_row["category"],
        "current_stock": current_stock,
        "predicted_demand": round(predicted_demand, 1),
        "demand_level": demand_level,
        "recommended_additional_copies": recommended_additional_copies,
        "recommendation_note": (
            "This is a data-driven SUGGESTION based on predicted demand vs. "
            "current stock. It does not account for budget, shelf space, or "
            "whether the book is being phased out of the syllabus - final "
            "purchasing decisions should involve human judgment."
        ),
    }


# ----------------------------------------------------------------
# QUICK MANUAL TEST (only runs when this file is executed directly)
# ----------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("PREDICTION PIPELINE - MANUAL TEST")
    print("=" * 60)
    print(f"Demand level cutoffs -> Low: <= {_low_cutoff:.2f}, "
          f"Medium: <= {_high_cutoff:.2f}, High: > {_high_cutoff:.2f}")

    sample_ids = get_available_book_ids()[:5]
    print(f"\nTesting on 5 sample books: {sample_ids}\n")

    for bid in sample_ids:
        result = predict_demand(bid)
        print(f"[{result['book_id']}] {result['book_title']}")
        print(f"   Category: {result['category']}")
        print(f"   Current stock: {result['current_stock']}")
        print(f"   Predicted demand: {result['predicted_demand']}")
        print(f"   Demand level: {result['demand_level']}")
        print(f"   Recommended additional copies: {result['recommended_additional_copies']}")
        print("-" * 60)

    # Test the error path with an invalid book_id
    print("\nTesting invalid book_id handling:")
    try:
        predict_demand("NOT_A_REAL_BOOK")
    except ValueError as e:
        print(f"   Correctly raised ValueError: {e}")