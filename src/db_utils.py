"""
Database Utilities (Optional Enhancement)
---------------------------------------------
Migrates the cleaned CSV dataset into a SQLite database and provides
a helper to load it back as a DataFrame, as a drop-in alternative to
reading library_demand_clean.csv directly.
"""

import sqlite3
import pandas as pd
import os

DB_PATH = "data/library.db"
CSV_PATH = "data/processed/library_demand_clean.csv"
TABLE_NAME = "borrowing_records"


def migrate_csv_to_sqlite():
    """One-time migration: CSV -> SQLite table."""
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Migrated {len(df)} rows into {DB_PATH} (table: {TABLE_NAME})")


def load_from_sqlite() -> pd.DataFrame:
    """Load the full table back as a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn, parse_dates=["date"])
    conn.close()
    return df


if __name__ == "__main__":
    migrate_csv_to_sqlite()
    check_df = load_from_sqlite()
    print(f"Verification: loaded back {len(check_df)} rows from SQLite")
    print(check_df.head(3).to_string(index=False))