# Customer Analytic Dashboard

Interactive dashboard for visualizing customer segmentation clusters generated from machine learning models.

Machine Learning Pipeline Architecture:

Raw Data
   ↓
Data Ingestion using FAST API(Python)
   ↓
Validation Layer (PostgreSQL Functions)
   ↓
Cleaning Layer (SQL Views)
   ↓
Feature Engineering
   ↓
Clustering Models using FAST API(Python ML)
   ↓
Results Stored in Supabase tables for clusters
   ↓
final_customer_clusters View
   ↓
Streamlit Analytics Dashboard

Features:
- Behavioral and Aggregated clustering models
- Cluster distribution visualization
- Model usage analytics
- Time trend analysis
- Interactive filtering
- cluster profiles
- model perfomance

Built with:
- Python
- Streamlit
- Plotly
- Supabase PostgreSQL for data warehousing
- FAST API for ingestion and models-for it to be accessed by supabase edge functions
- Jupyter Notebook for models training
DASHBOARD LINK => https://dashboardapp-m5wiqnithnevhjfrzbvcd6.streamlit.app/
