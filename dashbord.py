import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import os
import time
import psycopg2
import sqlalchemy


st.set_page_config(
    page_title="Customer Segmentation Analytics",
    layout="wide"
)


# DATABASE CONNECTION

DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL, poolclass=NullPool)


# CLUSTER NAMES (defined before loaders so all functions can use it)
 
cluster_names = {
    "behavioral": {
        0: 'Recent High-Value Customers',
        1: 'Loyal High-Spenders',
        2: 'Low Engagement Customers'
    },
    "aggregated": {
        0: 'Low-Value New Customer',
        1: 'High-Value Loyal Customer',
        2: 'Low-Value Mid-Tenure',
        3: 'New Individual Customer',
        4: 'High-Value Mid-Tenure',
        5: 'Premium Customer',
        6: 'Mid-Value Overseas',
        7: 'New Overseas Customer'
    }
}


# DATA LOADERS
 
@st.cache_data(ttl=300)
def load_data():
    """Load deduplicated data - latest record per customer per model only."""
    query = text("SELECT * FROM final_customer_clusters")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
 
    df["scored_at"] = pd.to_datetime(df["scored_at"])
 
    # Keep only the latest record per customer per model
    df = (
        df.sort_values("scored_at", ascending=False)
        .drop_duplicates(subset=["row_id", "model_used"], keep="first")
    )
 
    df["cluster_description"] = df.apply(
        lambda x: cluster_names.get(x["model_used"], {}).get(x["cluster"], "Unknown"),
        axis=1
    )
 
    return df
 
 
@st.cache_data(ttl=300)
def load_full_history():
    """Load full history for cluster movement detection."""
    query = text("SELECT * FROM final_customer_clusters")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
 
    df["scored_at"] = pd.to_datetime(df["scored_at"])
 
    df["cluster_description"] = df.apply(
        lambda x: cluster_names.get(x["model_used"], {}).get(x["cluster"], "Unknown"),
        axis=1
    )
 
    return df
 
 
@st.cache_data(ttl=300)
def load_model_runs():
    """Load model run history."""
    query = text("SELECT * FROM model_runs ORDER BY run_timestamp DESC")
    with engine.connect() as conn:
        return pd.read_sql(query, conn)
 
 
def load_raw_features(columns):
    """Load selected columns from raw customer table with retry on SSL disconnect."""
    if not columns:
        return pd.DataFrame()
    cols_str = ", ".join(["row_id"] + columns)
    query = text(f"SELECT {cols_str} FROM raw_customer_events")
 
    retries = 3
    delay = 2
    attempt = 0
    while attempt < retries:
        try:
            with engine.connect() as conn:
                df_raw = pd.read_sql(query, conn)
            return df_raw
        except (psycopg2.OperationalError, sqlalchemy.exc.OperationalError):
            attempt += 1
            st.warning(f"Database connection failed. Retrying {attempt}/{retries}...")
            time.sleep(delay)
    st.error("Failed to connect to raw customer table after multiple attempts.")
    return pd.DataFrame()
 

# LOAD DATA
 
df = load_data()
runs = load_model_runs()
 
# SIDEBAR FILTERS

st.sidebar.title("Dashboard Filters")

# Handle empty database
if df.empty:
    st.warning("No data available yet. Please ingest data to view the dashboard.")
    st.stop()

models = st.sidebar.multiselect(
    "Selected Model",
    df["model_used"].unique(),
    default=df["model_used"].unique()
)

clusters = st.sidebar.multiselect(
    "Selected Cluster",
    df["cluster_description"].unique(),
    default=df["cluster_description"].unique()
)

min_date = df["scored_at"].min()
max_date = df["scored_at"].max()

# Handle NaT dates from empty tables
if pd.isna(min_date) or pd.isna(max_date):
    default_start = datetime.today().date() - timedelta(days=30)
    default_end = datetime.today().date()
else:
    default_start = min_date.date()
    default_end = max_date.date()

start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    [default_start, default_end]
)

df_filtered = df.copy()
df_filtered = df_filtered[df_filtered["model_used"].isin(models)]
df_filtered = df_filtered[df_filtered["cluster_description"].isin(clusters)]

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)

df_filtered = df_filtered[
    (df_filtered["scored_at"] >= start_date) &
    (df_filtered["scored_at"] < end_date)
]

# HEADER METRICS
 
st.title("Customer Segmentation Analytics Dashboard")
 
total_customers = df_filtered["row_id"].nunique()
num_clusters = df_filtered["cluster"].nunique()
 
latest_model = (
    df_filtered.sort_values("scored_at", ascending=False)["model_used"].iloc[0]
    if not df_filtered.empty else "N/A"
)
 
last_scored = (
    df_filtered["scored_at"].max()
    if not df_filtered.empty else pd.Timestamp.now()
)
 
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", total_customers)
col2.metric("Active Clusters", num_clusters)
col3.metric("Latest Model Used", latest_model)
col4.metric("Last Scoring Date", last_scored.strftime("%Y-%m-%d"))
 
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Cluster Analysis",
    "Model Monitoring",
    "Customer Drill-Down"
])

with tab1:
    colA, colB = st.columns(2)
    with colA:
         st.subheader("Customer Distribution by Cluster")
 
    cluster_dist = (
        df_filtered.groupby("cluster_description")
        .size()
        .reset_index(name="count")
    )
 
    fig1 = px.bar(
        cluster_dist,
        x="cluster_description",
        y="count",
        color="cluster_description",
        text="count"
    )
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
    with colB:
        st.subheader("Cluster Scoring Trend")
 
    trend = (
        df_filtered.groupby("scored_at")
        .size()
        .reset_index(name="count")
    )
 
    fig3 = px.line(
        trend,
        x="scored_at",
        y="count",
        markers=True
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    colC, colD = st.columns(2)
    with colC:
         st.subheader("Customer Cluster Movement")
 
    # Use full history (not deduplicated) to detect movement
    movement_df = load_full_history().sort_values(["row_id", "scored_at"])
 
    movement_df["previous_cluster"] = movement_df.groupby("row_id")["cluster_description"].shift(1)
    movement_df["moved"] = movement_df["previous_cluster"] != movement_df["cluster_description"]
 
    movement_changes = movement_df[
        (movement_df["moved"] == True) &
        (movement_df["previous_cluster"].notna())
    ]
 
    if not movement_changes.empty:
        st.write("Customers who changed clusters")
 
        st.dataframe(
            movement_changes[[
                "row_id",
                "previous_cluster",
                "cluster_description",
                "scored_at",
                "model_used"
            ]],
            use_container_width=True
        )
 
        movement_summary = (
            movement_changes.groupby(["previous_cluster", "cluster_description"])
            .size()
            .reset_index(name="count")
        )
 
        fig_move = px.sunburst(
            movement_summary,
            path=["previous_cluster", "cluster_description"],
            values="count",
            title="Cluster Movement Flow"
        )
        st.plotly_chart(fig_move, use_container_width=True)
 
    else:
        st.info("No cluster movement detected yet.")
    with colD:
        st.subheader("Cluster Profiles (All Models)")
 
raw_features = {
    "behavioral": ["lifetime_value", "membership", "total_purchases", "days_since_last_purchase"],
    "aggregated": ["total_revenue", "monthly_fee", "tenure_months"],
}
 
cluster_profiles_list = []
 
for model in df_filtered["model_used"].unique():
    features = raw_features.get(model, [])
    df_raw = load_raw_features(features)
 
    if df_raw.empty:
        continue
 
    df_full = df_filtered[df_filtered["model_used"] == model].merge(
        df_raw, on="row_id", how="left"
    )
 
    features = [f for f in features if f in df_full.columns]
    if not features:
        continue
 
    profile = (
        df_full.groupby("cluster_description")[features]
        .agg(["mean", "median", "min", "max"])
        .round(2)
    )
    profile.columns = ["_".join(col).strip() for col in profile.columns.values]
    profile.reset_index(inplace=True)
    profile["model_used"] = model
 
    cluster_profiles_list.append(profile)
 
if cluster_profiles_list:
    cluster_profiles_df = pd.concat(cluster_profiles_list, ignore_index=True)
    st.dataframe(cluster_profiles_df, use_container_width=True)
 
    csv_profiles = cluster_profiles_df.to_csv(index=False)
    st.download_button(
        "Download Cluster Profiles",
        csv_profiles,
        file_name="cluster_profiles_all_models.csv"
    )
else:
    st.info("No cluster profile data available for the selected filters.")

with tab3:
     st.subheader("Model Performance Monitoring")
 
     if not runs.empty:
        fig4 = px.line(
            runs,
            x="run_timestamp",
            y="silhouette_score",
            color="model_name",
            markers=True,
            title="Model Quality Over Time"
        )
        st.plotly_chart(fig4, use_container_width=True)
 
        st.subheader("Recent Model Runs")
        st.dataframe(
            runs.sort_values("run_timestamp", ascending=False),
            use_container_width=True
        )
    else:
        st.info("No model monitoring data available.")

with tab4:
    st.subheader("Customer Drill-Down Analytics")
 
selected_cluster = st.selectbox(
    "Select Cluster to Explore",
    df_filtered["cluster_description"].unique()
)
 
cluster_customers = df_filtered[
    df_filtered["cluster_description"] == selected_cluster
].copy()
 
model_for_cluster = cluster_customers["model_used"].iloc[0] if not cluster_customers.empty else None
 
if model_for_cluster and model_for_cluster in raw_features:
    features = raw_features[model_for_cluster]
    df_raw = load_raw_features(features)
 
    if not df_raw.empty:
        cluster_customers = cluster_customers.merge(df_raw, on="row_id", how="left")
 
display_cols = ["row_id", "scored_at"] + [
    c for c in cluster_customers.columns
    if c not in ["cluster", "model_used", "cluster_description", "scored_at", "row_id"]
]
 
st.write(f"**{len(cluster_customers)}** customers in **{selected_cluster}**")
 
numeric_features = [
    c for c in display_cols
    if c not in ["row_id", "scored_at"]
    and c in cluster_customers.columns
    and pd.api.types.is_numeric_dtype(cluster_customers[c])
]
 
if numeric_features:
    summary_cols = st.columns(len(numeric_features))
    for i, feat in enumerate(numeric_features):
        col_data = cluster_customers[feat].dropna()
        if col_data.empty:
            summary_cols[i].metric(
                label=feat.replace("_", " ").title(),
                value="N/A",
                delta="no data"
            )
        else:
            mean_val = col_data.mean()
            median_val = col_data.median()
            summary_cols[i].metric(
                label=feat.replace("_", " ").title(),
                value=f"{mean_val:,.1f}",
                delta=f"median: {median_val:,.1f}"
            )
else:
    st.info("No feature data available for this cluster.")
 
st.divider()
 
safe_display_cols = [c for c in display_cols if c in cluster_customers.columns]
 
st.dataframe(
    cluster_customers[safe_display_cols],
    use_container_width=True
)
 
csv = cluster_customers[safe_display_cols].to_csv(index=False)
st.download_button(
    "Download Cluster Data",
    csv,
    file_name=f"{selected_cluster.replace(' ', '_').lower()}_customers.csv"
)


 
