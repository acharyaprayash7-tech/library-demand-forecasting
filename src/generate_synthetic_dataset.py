"""
Synthetic Library Book Demand Dataset Generator
-------------------------------------------------
This script generates a SYNTHETIC dataset simulating monthly book
borrowing activity in a college library. It is used because real
library data was not available for this academic project.

Each row = one book's aggregated activity for one month.

Run this script once to create:
    data/raw/library_demand_data.csv
"""

import pandas as pd
import numpy as np

# ----------------------------------------------------------------
# 1. SETTINGS (change these if you want a bigger/smaller dataset)
# ----------------------------------------------------------------
RANDOM_SEED = 42
NUM_MONTHS = 30                     # ~2.5 years of history
BOOKS_PER_CATEGORY = 12
START_DATE = "2022-01-01"

np.random.seed(RANDOM_SEED)

# ----------------------------------------------------------------
# 2. CATEGORY -> DEPARTMENT MAPPING
#    Each category also gets an "exam_boost" - how much extra
#    demand it gets during exam months (technical subjects spike
#    harder than general/literature subjects).
# ----------------------------------------------------------------
CATEGORIES = {
    "Computer Science":   {"department": "CSE",     "exam_boost": 1.8},
    "Data Science":       {"department": "CSE",     "exam_boost": 1.9},
    "Electronics":        {"department": "ECE",     "exam_boost": 1.6},
    "Mechanical":         {"department": "MECH",    "exam_boost": 1.5},
    "Civil":              {"department": "CIVIL",   "exam_boost": 1.4},
    "Mathematics":        {"department": "SCIENCE", "exam_boost": 1.7},
    "Physics":            {"department": "SCIENCE", "exam_boost": 1.5},
    "Chemistry":          {"department": "SCIENCE", "exam_boost": 1.4},
    "Business Management":{"department": "MBA",     "exam_boost": 1.3},
    "English Literature": {"department": "HUMANITIES","exam_boost": 1.1},
}

AUTHOR_FIRST = ["A.", "R.", "S.", "K.", "M.", "P.", "N.", "V.", "J.", "D."]
AUTHOR_LAST = ["Sharma", "Kumar", "Reddy", "Iyer", "Nair", "Bose",
               "Rao", "Singh", "Gupta", "Menon", "Khan", "Verma"]

TITLE_TEMPLATES = [
    "Introduction to {cat}", "Fundamentals of {cat}",
    "{cat}: A Practical Approach", "Advanced {cat}",
    "Principles of {cat}", "{cat} for Engineers",
    "Modern {cat}", "Applied {cat}", "{cat} Concepts and Applications",
    "Essentials of {cat}", "{cat}: Theory and Practice",
    "Understanding {cat}",
]

# ----------------------------------------------------------------
# 3. BUILD BOOK MASTER LIST
#    Each book gets:
#      - base_popularity : how much it's borrowed on average
#      - trend_type       : rising / falling / stable over time
#      - trend_slope      : how strong that trend is
# ----------------------------------------------------------------
books = []
book_counter = 1

for category, info in CATEGORIES.items():
    for i in range(BOOKS_PER_CATEGORY):
        book_id = f"BK{book_counter:04d}"
        title = TITLE_TEMPLATES[i % len(TITLE_TEMPLATES)].format(cat=category)
        author = f"{np.random.choice(AUTHOR_FIRST)} {np.random.choice(AUTHOR_LAST)}"

        base_popularity = np.random.gamma(shape=3.0, scale=6.0)  # skewed: most books modest, a few very popular
        trend_type = np.random.choice(
            ["rising", "falling", "stable"], p=[0.3, 0.2, 0.5]
        )
        trend_slope = {
            "rising":  np.random.uniform(0.4, 1.2),
            "falling": -np.random.uniform(0.3, 0.9),
            "stable":  np.random.uniform(-0.05, 0.05),
        }[trend_type]

        books.append({
            "book_id": book_id,
            "book_title": title,
            "author": author,
            "category": category,
            "department": info["department"],
            "exam_boost": info["exam_boost"],
            "publication_year": int(np.random.randint(1995, 2024)),
            "available_copies": int(np.random.randint(15, 70)),
            "base_popularity": base_popularity,
            "trend_slope": trend_slope,
        })
        book_counter += 1

books_df = pd.DataFrame(books)

# ----------------------------------------------------------------
# 4. BUILD THE MONTHLY TIMELINE
# ----------------------------------------------------------------
months = pd.date_range(start=START_DATE, periods=NUM_MONTHS, freq="MS")  # month start


def get_semester(month_num: int) -> str:
    """Jan-Jun = Even semester, Jul-Dec = Odd semester (simplified)."""
    return "Even" if month_num <= 6 else "Odd"


def is_exam_period(month_num: int) -> int:
    """Exam months: April, May, October, November (typical semester ends)."""
    return 1 if month_num in [4, 5, 10, 11] else 0


def is_holiday(month_num: int) -> int:
    """Holiday/break months: June, December."""
    return 1 if month_num in [6, 12] else 0


# ----------------------------------------------------------------
# 5. GENERATE MONTHLY DEMAND RECORDS
# ----------------------------------------------------------------
records = []
record_id = 1
prev_borrowed = {}  # book_id -> last month's borrowed_count (for returns lag)

for month_index, date in enumerate(months):
    month_num = date.month
    semester = get_semester(month_num)
    exam_flag = is_exam_period(month_num)
    holiday_flag = is_holiday(month_num)

    for _, book in books_df.iterrows():
        # --- Trend component: popularity drifts up/down over time ---
        trend_effect = 1 + (book["trend_slope"] * month_index / NUM_MONTHS)
        trend_effect = max(trend_effect, 0.1)  # never let it go negative/zero

        # --- Seasonal component ---
        seasonal_multiplier = 1.0
        if exam_flag:
            seasonal_multiplier *= book["exam_boost"]
        if holiday_flag:
            seasonal_multiplier *= 0.4  # much lower activity during breaks

        # --- Expected demand before noise ---
        expected_demand = book["base_popularity"] * trend_effect * seasonal_multiplier
        expected_demand = max(expected_demand, 0.5)

        # --- Add realistic noise (Poisson matches "count" data well) ---
        borrowed_count = int(np.random.poisson(lam=expected_demand))

        # --- Returned count: based on LAST month's borrows (a real lag) ---
        last_month_borrowed = prev_borrowed.get(book["book_id"], borrowed_count)
        returned_count = int(
            np.random.binomial(n=max(last_month_borrowed, 0), p=0.85)
        )

        # --- Renewals: a fraction of this month's borrows ---
        renewal_count = int(np.random.binomial(n=borrowed_count, p=0.15))

        # --- Reservations: rise when demand exceeds available stock ---
        unmet_demand = max(0, borrowed_count - book["available_copies"])
        reservation_count = int(
            np.random.poisson(lam=unmet_demand * 0.5 + 0.3)
        )

        records.append({
            "record_id": record_id,
            "date": date.strftime("%Y-%m-%d"),
            "book_id": book["book_id"],
            "book_title": book["book_title"],
            "author": book["author"],
            "category": book["category"],
            "department": book["department"],
            "publication_year": book["publication_year"],
            "available_copies": book["available_copies"],
            "borrowed_count": borrowed_count,
            "returned_count": returned_count,
            "renewal_count": renewal_count,
            "reservation_count": reservation_count,
            "semester": semester,
            "exam_period": exam_flag,
            "holiday_indicator": holiday_flag,
        })

        prev_borrowed[book["book_id"]] = borrowed_count
        record_id += 1

df = pd.DataFrame(records)

# ----------------------------------------------------------------
# 6. VALIDATION CHECKS
# ----------------------------------------------------------------
print("=" * 60)
print("DATASET GENERATION SUMMARY")
print("=" * 60)
print(f"Total records        : {len(df)}")
print(f"Unique books          : {df['book_id'].nunique()}")
print(f"Date range            : {df['date'].min()} to {df['date'].max()}")
print(f"Missing values total  : {df.isnull().sum().sum()}")
print(f"Duplicate rows        : {df.duplicated().sum()}")
print("-" * 60)
print("Borrowed count stats:")
print(df["borrowed_count"].describe())
print("-" * 60)
print("Sample rows:")
print(df.head(5).to_string(index=False))

# ----------------------------------------------------------------
# 7. SAVE TO CSV
# ----------------------------------------------------------------
output_path = "data/raw/library_demand_data.csv"
df.to_csv(output_path, index=False)
print("-" * 60)
print(f"Saved dataset to: {output_path}")

# Also save a synthetic-data notice alongside it
notice = (
    "SYNTHETIC DATASET NOTICE\n"
    "-------------------------\n"
    "This dataset (library_demand_data.csv) is artificially generated "
    "for academic demonstration purposes. It is NOT real library data.\n"
    "It was created using src/generate_synthetic_dataset.py with a fixed "
    "random seed (42) for reproducibility.\n"
)
with open("data/raw/SYNTHETIC_DATA_NOTICE.txt", "w") as f:
    f.write(notice)
print("Saved notice to: data/raw/SYNTHETIC_DATA_NOTICE.txt")