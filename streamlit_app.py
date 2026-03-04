import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os

from dotenv import load_dotenv
from openai import OpenAI

from regime_detection import run_regime_engine, get_latest_regime


# --------------------------------------------------
# Load environment
# --------------------------------------------------

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="AI Business Intelligence Platform",
    layout="wide"
)

st.title("AI Regime Detection & Business Intelligence")


DATASET = "regime_detection_dataset.parquet"


# --------------------------------------------------
# Load Data
# --------------------------------------------------

raw_df = pd.read_parquet(DATASET)

results, shap_df, meta_features = run_regime_engine(DATASET)

latest = results.iloc[-1]


# --------------------------------------------------
# Tabs
# --------------------------------------------------

dashboard_tab, chatbot_tab = st.tabs(["Dashboard", "AI Chatbot"])


# ==================================================
# DASHBOARD
# ==================================================

with dashboard_tab:

    st.header("Business Regime Intelligence")


    # ----------------------------------------------
    # Metrics
    # ----------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Regime",
        latest["regime_label"]
    )

    col2.metric(
        "Confidence",
        round(latest["confidence_score"], 2)
    )

    col3.metric(
        "Latest Revenue",
        f"${int(latest['total_revenue']):,}"
    )


    st.divider()


    # ----------------------------------------------
    # Revenue Trend
    # ----------------------------------------------

    st.subheader("Revenue Trend")

    fig_rev = px.line(
        results,
        x="week_start",
        y="total_revenue",
        markers=True
    )

    st.plotly_chart(fig_rev, use_container_width=True)


    # ----------------------------------------------
    # Cost Trend
    # ----------------------------------------------

    st.subheader("Unit Cost Trend")

    fig_cost = px.line(
        results,
        x="week_start",
        y="unit_cost",
        markers=True
    )

    st.plotly_chart(fig_cost, use_container_width=True)


    # ----------------------------------------------
    # Discount Trend
    # ----------------------------------------------

    st.subheader("Discount Trend")

    fig_discount = px.line(
        results,
        x="week_start",
        y="discount_pct",
        markers=True
    )

    st.plotly_chart(fig_discount, use_container_width=True)


    st.divider()


    # ----------------------------------------------
    # Regime Timeline
    # ----------------------------------------------

    st.subheader("Regime Timeline")

    fig_regime = px.scatter(
        results,
        x="week_start",
        y="confidence_score",
        color="regime_label",
        size="confidence_score"
    )

    st.plotly_chart(fig_regime, use_container_width=True)


    st.divider()


    # ----------------------------------------------
    # SHAP Feature Importance
    # ----------------------------------------------

    st.subheader("Feature Importance (SHAP)")

    shap_mean = shap_df.abs().mean().sort_values(ascending=False)

    fig_shap = px.bar(
        shap_mean,
        title="Top Features Driving Regime Detection"
    )

    st.plotly_chart(fig_shap, use_container_width=True)


    st.divider()


    # ----------------------------------------------
    # Correlation Heatmap
    # ----------------------------------------------

    st.subheader("Feature Correlation Heatmap")

    corr = raw_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10,6))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)


    st.divider()


    # ----------------------------------------------
    # Data Table
    # ----------------------------------------------

    st.subheader("Regime Detection Results")

    st.dataframe(results)



# ==================================================
# CHATBOT
# ==================================================

with chatbot_tab:

    st.header("AI Analytics Chatbot")

    question = st.text_input(
        "Ask a question about business trends"
    )

    if question:

        context = get_latest_regime(DATASET)

        prompt = f"""
You are an AI business intelligence assistant.

Context:

Regime: {context['regime_label']}
Confidence: {context['confidence_score']}
Revenue: {context['total_revenue']}
Cost: {context['unit_cost']}
Discount: {context['discount_pct']}
IsolationForest: {context['iso_anomaly']}
HMM State: {context['hmm_state']}
Bayesian Cluster: {context['bayesian_cluster']}

Answer the user's question using this context.
"""

        response = client.chat.completions.create(

            model="gpt-4o-mini",

            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ]

        )

        st.write(response.choices[0].message.content)