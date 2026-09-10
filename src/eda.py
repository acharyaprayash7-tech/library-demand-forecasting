"""
Exploratory Data Analysis
---------------------------
Generates charts and printed findings from the cleaned dataset.
Saves charts as PNG files in notebooks/eda_charts/.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/processed/library_demand_clean.csv", parse_dates=["date"])

OUT_DIR = "notebooks/eda_charts"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print(f"Records: {len(df)}, Books: {df['book_id'].nunique()}, "
      f"Categories: {df['category'].nunique()}")
print(f"Average monthly borrowed_count: {df['borrowed_count'].mean():.2f}")
print(f"Median monthly borrowed_count : {df['borrowed_count'].median():.2f}")

# ------------------------------------------------------------
# 1. Overall monthly demand trend
# ------------------------------------------------------------
monthly_total = df.groupby("date")["borrowed_count"].sum()
plt.figure(figsize=(10, 5))
plt.plot(monthly_total.index, monthly_total.values, marker="o")
plt.title("Total Monthly Borrow Demand (All Books)")
plt.xlabel("Month")
plt.ylabel("Total Borrowed Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_monthly_demand_trend.png")
plt.close()
print("Saved: 01_monthly_demand_trend.png")

# ------------------------------------------------------------
# 2. Category-wise average demand
# ------------------------------------------------------------
category_avg = df.groupby("category")["borrowed_count"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
category_avg.plot(kind="bar", color="steelblue")
plt.title("Average Monthly Demand by Category")
plt.ylabel("Avg Borrowed Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_category_avg_demand.png")
plt.close()
print("Saved: 02_category_avg_demand.png")
print("\nTop category by avg demand:", category_avg.index[0],
      f"({category_avg.iloc[0]:.2f})")
print("Lowest category by avg demand:", category_avg.index[-1],
      f"({category_avg.iloc[-1]:.2f})")

# ------------------------------------------------------------
# 3. Top 10 most borrowed books (total over all time)
# ------------------------------------------------------------
top_books = df.groupby("book_title")["borrowed_count"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
top_books.sort_values().plot(kind="barh", color="darkgreen")
plt.title("Top 10 Most Borrowed Books (Total)")
plt.xlabel("Total Borrowed Count")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_top10_books.png")
plt.close()
print("Saved: 03_top10_books.png")

# ------------------------------------------------------------
# 4. Exam vs non-exam period demand
# ------------------------------------------------------------
exam_avg = df.groupby("exam_period")["borrowed_count"].mean()
plt.figure(figsize=(6, 5))
exam_avg.plot(kind="bar", color=["gray", "orangered"])
plt.title("Avg Demand: Exam Period vs Non-Exam Period")
plt.xticks([0, 1], ["Non-Exam", "Exam Period"], rotation=0)
plt.ylabel("Avg Borrowed Count")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_exam_vs_nonexam.png")
plt.close()
print("Saved: 04_exam_vs_nonexam.png")
print(f"\nExam period avg demand: {exam_avg[1]:.2f}")
print(f"Non-exam avg demand   : {exam_avg[0]:.2f}")
print(f"Demand increase during exams: "
      f"{((exam_avg[1] - exam_avg[0]) / exam_avg[0] * 100):.1f}%")

# ------------------------------------------------------------
# 5. Demand distribution
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(df["borrowed_count"], bins=30, color="teal", edgecolor="black")
plt.title("Distribution of Monthly Borrowed Count")
plt.xlabel("Borrowed Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_demand_distribution.png")
plt.close()
print("Saved: 05_demand_distribution.png")

# ------------------------------------------------------------
# 6. Individual book trend example (one rising, one falling book)
# ------------------------------------------------------------
book_totals = df.groupby("book_id")["borrowed_count"].sum()
first_half = df[df["date"] < df["date"].median()].groupby("book_id")["borrowed_count"].mean()
second_half = df[df["date"] >= df["date"].median()].groupby("book_id")["borrowed_count"].mean()
change = (second_half - first_half).sort_values()

falling_book = change.index[0]
rising_book = change.index[-1]

plt.figure(figsize=(10, 5))
for book_id, label, color in [(rising_book, "Rising Example", "green"),
                                (falling_book, "Falling Example", "red")]:
    subset = df[df["book_id"] == book_id].sort_values("date")
    plt.plot(subset["date"], subset["borrowed_count"], marker="o",
              label=f"{label}: {subset['book_title'].iloc[0]}", color=color)
plt.title("Example Rising vs Falling Demand Books")
plt.xlabel("Month")
plt.ylabel("Borrowed Count")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_rising_vs_falling_example.png")
plt.close()
print("Saved: 06_rising_vs_falling_example.png")

print("\nAll EDA charts saved to:", OUT_DIR)