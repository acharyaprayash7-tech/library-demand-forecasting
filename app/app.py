"""
Library Book Demand Forecasting System - Web Dashboard
----------------------------------------------------------
Streamlit multi-page dashboard. Run with:
    streamlit run app/app.py
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.prediction import predict_demand, get_available_book_ids

# ----------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Library Demand Forecasting",
    page_icon="📚",
    layout="wide",
)

# ----------------------------------------------------------------
# DATA LOADING (cached)
# ----------------------------------------------------------------
@st.cache_data
def load_clean_data():
    return pd.read_csv("data/processed/library_demand_clean.csv", parse_dates=["date"])


@st.cache_data
def load_feature_data():
    return pd.read_csv("data/processed/library_demand_features.csv", parse_dates=["date"])

with st.spinner("Loading library data..."):
    df = load_clean_data()
    features_df = load_feature_data()

# ----------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------
st.sidebar.title("📚 Library Demand System")
st.sidebar.caption("AI & Data Science Academic Project")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "🔮 Demand Forecasting", "📖 Book Analysis",
     "📦 Inventory Recommendation", "📊 Analytics"],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard uses a SYNTHETIC dataset generated for academic "
    "demonstration. See data/raw/SYNTHETIC_DATA_NOTICE.txt for details."
)


def book_label_map():
    """Maps 'BK0001 - Introduction to Data Science' -> 'BK0001' for select boxes."""
    latest = df.sort_values("date").groupby("book_id").last().reset_index()
    labels = {
        f"{row.book_id} - {row.book_title}": row.book_id
        for row in latest.itertuples()
    }
    return labels


# ==================================================================
# PAGE: DASHBOARD (HOME)
# ==================================================================
if page == "🏠 Dashboard":
    st.title("📚 Library Book Demand Forecasting Dashboard")
    st.markdown("Overview of library borrowing activity and demand patterns.")

    total_books = df["book_id"].nunique()
    total_transactions = int(df["borrowed_count"].sum())
    most_popular = df.groupby("book_title")["borrowed_count"].sum().idxmax()
    top_category = df.groupby("category")["borrowed_count"].mean().idxmax()
    avg_monthly_demand = df.groupby("date")["borrowed_count"].sum().mean()

    book_totals = df.groupby("book_id")["borrowed_count"].sum()
    high_demand_cutoff = book_totals.quantile(2 / 3)
    num_high_demand_books = int((book_totals > high_demand_cutoff).sum())

    latest_per_book = df.sort_values("date").groupby("book_id").last()
    avg_demand_per_book = df.groupby("book_id")["borrowed_count"].mean()
    low_stock_high_demand = (
        latest_per_book["available_copies"] < avg_demand_per_book
    ).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Books", total_books)
    col2.metric("Total Transactions", f"{total_transactions:,}")
    col3.metric("Most Popular Book", most_popular)
    col4.metric("Highest Demand Category", top_category)

    col5, col6, col7 = st.columns(3)
    col5.metric("Avg Monthly Demand", f"{avg_monthly_demand:.0f}")
    col6.metric("High-Demand Books", num_high_demand_books)
    col7.metric("Low-Stock/High-Demand Books", int(low_stock_high_demand))

    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Monthly Demand Trend")
        monthly = df.groupby("date")["borrowed_count"].sum().reset_index()
        fig = px.line(monthly, x="date", y="borrowed_count", markers=True)
        fig.update_layout(xaxis_title="Month", yaxis_title="Total Borrowed")
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.subheader("Category-wise Average Demand")
        cat_avg = (
            df.groupby("category")["borrowed_count"].mean()
              .sort_values(ascending=False).reset_index()
        )
        fig2 = px.bar(cat_avg, x="category", y="borrowed_count")
        fig2.update_layout(xaxis_title="Category", yaxis_title="Avg Borrowed")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 10 Most Borrowed Books")
    top10 = (
        df.groupby("book_title")["borrowed_count"].sum()
          .sort_values(ascending=False).head(10).reset_index()
    )
    fig3 = px.bar(top10.sort_values("borrowed_count"),
                  x="borrowed_count", y="book_title", orientation="h")
    fig3.update_layout(xaxis_title="Total Borrowed", yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

# ==================================================================
# PAGE: DEMAND FORECASTING
# ==================================================================
elif page == "🔮 Demand Forecasting":
    st.title("🔮 Demand Forecasting")
    st.markdown("Select a book to see its predicted future demand.")

    labels = book_label_map()
    selected_label = st.selectbox("Select a book", sorted(labels.keys()))
    book_id = labels[selected_label]

    try:
        result = predict_demand(book_id)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Book", result["book_title"])
        c2.metric("Category", result["category"])
        c3.metric("Demand Level", result["demand_level"])

        c4, c5, c6 = st.columns(3)
        c4.metric("Current Stock", result["current_stock"])
        c5.metric("Predicted Demand (next month)", result["predicted_demand"])
        c6.metric("Recommended Additional Copies",
                  result["recommended_additional_copies"])

        if result["recommended_additional_copies"] > 0:
            st.warning(
                f"📈 Predicted demand ({result['predicted_demand']}) exceeds "
                f"current stock ({result['current_stock']}). "
                f"Recommended additional copies: **{result['recommended_additional_copies']}**"
            )
        else:
            st.success("✅ Current stock appears sufficient for predicted demand.")

        st.caption(result["recommendation_note"])

        st.markdown("---")
        st.subheader("Historical Demand")
        history = df[df["book_id"] == book_id].sort_values("date")
        fig = px.line(history, x="date", y="borrowed_count", markers=True,
                      title=f"Borrowing History - {result['book_title']}")
        st.plotly_chart(fig, use_container_width=True)

    except ValueError as e:
        st.error(str(e))

# ==================================================================
# PAGE: BOOK ANALYSIS
# ==================================================================
elif page == "📖 Book Analysis":
    st.title("📖 Book Analysis")
    st.markdown("Deep dive into a specific book's borrowing pattern.")

    labels = book_label_map()
    selected_label = st.selectbox("Search / select a book", sorted(labels.keys()))
    book_id = labels[selected_label]

    book_history = df[df["book_id"] == book_id].sort_values("date")
    book_info = book_history.iloc[-1]

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Title", book_info["book_title"])
    c2.metric("Category", book_info["category"])
    c3.metric("Author", book_info["author"])
    c4.metric("Available Copies", int(book_info["available_copies"]))

    avg_demand = book_history["borrowed_count"].mean()
    recent_demand = book_history["borrowed_count"].iloc[-3:].mean()

    try:
        result = predict_demand(book_id)
        predicted = result["predicted_demand"]
        trend_label = result["demand_level"]
    except ValueError:
        predicted = None
        trend_label = "N/A"

    c5, c6, c7 = st.columns(3)
    c5.metric("Average Historical Demand", f"{avg_demand:.1f}")
    c6.metric("Recent (Last 3 Months) Avg", f"{recent_demand:.1f}")
    c7.metric("Predicted Demand", predicted if predicted is not None else "N/A")

    st.markdown("---")
    st.subheader("Monthly Demand Graph")
    fig = px.bar(book_history, x="date", y="borrowed_count",
                 title=f"Monthly Borrowed Count - {book_info['book_title']}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Reservations Over Time (unmet demand signal)")
    fig2 = px.line(book_history, x="date", y="reservation_count", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

# ==================================================================
# PAGE: INVENTORY RECOMMENDATION
# ==================================================================
elif page == "📦 Inventory Recommendation":
    st.title("📦 Inventory Recommendation")
    st.markdown(
        "Books flagged for review, based on predicted demand vs. current stock. "
        "These are suggestions, not mandates."
    )

    all_ids = get_available_book_ids()

    rows = []
    for bid in all_ids:
        try:
            r = predict_demand(bid)
            if r["recommended_additional_copies"] > 0:
                recommendation = "🔴 Purchase more"
            elif r["demand_level"] == "Low":
                recommendation = "🟡 Low demand"
            else:
                recommendation = "🟢 Stock sufficient"

            rows.append({
                "Book": r["book_title"],
                "Category": r["category"],
                "Current Stock": r["current_stock"],
                "Predicted Demand": r["predicted_demand"],
                "Demand Level": r["demand_level"],
                "Recommendation": recommendation,
            })
        except ValueError:
            continue

    inventory_df = pd.DataFrame(rows)

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        category_filter = st.multiselect(
            "Filter by category",
            options=sorted(inventory_df["Category"].unique()),
        )
    with filter_col2:
        rec_filter = st.multiselect(
            "Filter by recommendation",
            options=sorted(inventory_df["Recommendation"].unique()),
        )

    filtered = inventory_df.copy()
    if category_filter:
        filtered = filtered[filtered["Category"].isin(category_filter)]
    if rec_filter:
        filtered = filtered[filtered["Recommendation"].isin(rec_filter)]

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.caption(
        f"Showing {len(filtered)} of {len(inventory_df)} books. "
        "Recommendations are generated automatically from predicted demand "
        "vs. current stock, and should be reviewed by library staff before "
        "purchasing."
    )

# ==================================================================
# PAGE: ANALYTICS
# ==================================================================
elif page == "📊 Analytics":
    st.title("📊 Analytics")
    st.markdown("Deeper trends across the whole collection.")

    tab1, tab2, tab3 = st.tabs(
        ["Category Trends", "Top / Low Demand Books", "Demand Distribution"]
    )

    with tab1:
        st.subheader("Monthly Demand by Category")
        cat_monthly = (
            df.groupby(["date", "category"])["borrowed_count"].sum().reset_index()
        )
        fig = px.line(cat_monthly, x="date", y="borrowed_count", color="category")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        book_totals = (
            df.groupby("book_title")["borrowed_count"].sum().sort_values()
        )
        with col1:
            st.subheader("Top 10 Books")
            fig2 = px.bar(book_totals.tail(10).reset_index(),
                          x="borrowed_count", y="book_title", orientation="h")
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            st.subheader("Bottom 10 Books (Low Demand)")
            fig3 = px.bar(book_totals.head(10).reset_index(),
                          x="borrowed_count", y="book_title", orientation="h")
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.subheader("Distribution of Monthly Borrowed Count")
        fig4 = px.histogram(df, x="borrowed_count", nbins=30)
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Historical vs Predicted Demand (sample of 15 books)")
        sample_ids = get_available_book_ids()[:15]
        comp_rows = []
        for bid in sample_ids:
            try:
                r = predict_demand(bid)
                hist_avg = df[df["book_id"] == bid]["borrowed_count"].mean()
                comp_rows.append({
                    "Book": r["book_title"],
                    "Historical Avg": round(hist_avg, 1),
                    "Predicted Next Month": r["predicted_demand"],
                })
            except ValueError:
                continue
        comp_df = pd.DataFrame(comp_rows)
        fig5 = px.bar(comp_df, x="Book", y=["Historical Avg", "Predicted Next Month"],
                      barmode="group")
        fig5.update_xaxes(tickangle=45)
        st.plotly_chart(fig5, use_container_width=True)