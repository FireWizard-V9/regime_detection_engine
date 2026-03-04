import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.mixture import BayesianGaussianMixture

from statsmodels.tsa.seasonal import STL
import ruptures as rpt

from hmmlearn.hmm import GaussianHMM
import lightgbm as lgb

import shap


def run_regime_engine(dataset_path):

    # --------------------------------
    # LOAD DATA
    # --------------------------------

    df = pd.read_parquet(dataset_path)

    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start").reset_index(drop=True)


    # --------------------------------
    # FEATURE SELECTION
    # --------------------------------

    features = [

        "unit_cost",
        "total_revenue",
        "discount_pct",

        "channel_count",
        "region_count",
        "sku_count",

        "cost_pct_change",
        "cost_volatility_4w",

        "revenue_pct_change",
        "revune_volatility_4w",

        "discount_volatility_4w",
        "discount_zscore",

        "channel_diff",
        "region_diff",
        "sku_diff"
    ]

    X = df[features].copy()


    # --------------------------------
    # FEATURE SCALING
    # --------------------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # --------------------------------
    # MODEL 1 — ISOLATION FOREST
    # --------------------------------

    iso = IsolationForest(
        n_estimators=400,
        contamination=0.10,
        random_state=42
    )

    df["iso_anomaly"] = iso.fit_predict(X_scaled)
    df["iso_anomaly"] = df["iso_anomaly"].map({1:0, -1:1})


    # --------------------------------
    # MODEL 2 — CHANGE POINT DETECTION
    # --------------------------------

    signal = df["total_revenue"].values

    algo = rpt.Pelt(model="rbf").fit(signal)
    breakpoints = algo.predict(pen=10)

    df["change_point"] = 0

    for bp in breakpoints[:-1]:
        df.loc[bp, "change_point"] = 1


    # --------------------------------
    # MODEL 3 — STL DECOMPOSITION
    # --------------------------------

    stl = STL(df["total_revenue"], period=4)
    result = stl.fit()

    df["trend"] = result.trend
    df["seasonal"] = result.seasonal
    df["residual"] = result.resid


    # --------------------------------
    # MODEL 4 — BAYESIAN GAUSSIAN MIXTURE
    # --------------------------------

    bgm = BayesianGaussianMixture(
        n_components=5,
        covariance_type="full",
        random_state=42
    )

    df["bayesian_cluster"] = bgm.fit_predict(X_scaled)


    # --------------------------------
    # MODEL 5 — HIDDEN MARKOV MODEL
    # --------------------------------

    hmm_features = df[
        [
            "cost_pct_change",
            "revenue_pct_change",
            "discount_zscore",
            "cost_volatility_4w"
        ]
    ].values

    hmm_model = GaussianHMM(
        n_components=4,
        covariance_type="full",
        n_iter=200,
        random_state=42
    )

    hmm_model.fit(hmm_features)

    df["hmm_state"] = hmm_model.predict(hmm_features)

    state_probs = hmm_model.predict_proba(hmm_features)

    for i in range(state_probs.shape[1]):
        df[f"hmm_prob_{i}"] = state_probs[:, i]


    # --------------------------------
    # MODEL 6 — LIGHTGBM META MODEL
    # --------------------------------

    meta_features = df[
        [
            "cost_pct_change",
            "revenue_pct_change",
            "discount_zscore",
            "cost_volatility_4w",
            "revune_volatility_4w",
            "channel_diff",
            "sku_diff"
        ]
    ]

    y_meta = df["iso_anomaly"]

    meta_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )

    meta_model.fit(meta_features, y_meta)

    df["meta_signal"] = meta_model.predict(meta_features)


    # --------------------------------
    # SHAP EXPLAINABILITY
    # --------------------------------

    explainer = shap.TreeExplainer(meta_model)

    shap_values = explainer.shap_values(meta_features)

    shap_df = pd.DataFrame(
        shap_values,
        columns=meta_features.columns
    )


    # --------------------------------
    # BUSINESS RULE SIGNALS
    # --------------------------------

    df["supply_signal"] = (df["cost_pct_change"] > 0.15).astype(int)

    df["demand_signal"] = (df["revenue_pct_change"] < -0.20).astype(int)

    df["competition_signal"] = (df["discount_zscore"] > 2).astype(int)

    df["structure_signal"] = (
        (df["channel_diff"] > 0) |
        (df["region_diff"] > 0) |
        (df["sku_diff"] > 0)
    ).astype(int)


    # --------------------------------
    # FINAL ENSEMBLE CLASSIFIER
    # --------------------------------

    def classify(row):

        if row["supply_signal"]:
            return "SUPPLY_SHOCK"

        if row["demand_signal"]:
            return "DEMAND_SHOCK"

        if row["competition_signal"]:
            return "COMPETITIVE_PRESSURE"

        if row["structure_signal"]:
            return "STRUCTURAL_CHANGE"

        if row["iso_anomaly"]:
            return "ANOMALOUS_MARKET"

        return "NORMAL"


    df["regime_label"] = df.apply(classify, axis=1)


    # --------------------------------
    # CONFIDENCE SCORE
    # --------------------------------

    df["confidence_score"] = (

        df[
            [
                "supply_signal",
                "demand_signal",
                "competition_signal",
                "structure_signal",
                "iso_anomaly",
                "change_point",
                "meta_signal"
            ]
        ].sum(axis=1)

    ) / 7


    # --------------------------------
    # FINAL OUTPUT TABLE
    # --------------------------------

    results = df[
        [
            "week_start",
            "regime_label",
            "confidence_score",
            "unit_cost",
            "total_revenue",
            "discount_pct",
            "iso_anomaly",
            "change_point",
            "hmm_state",
            "bayesian_cluster"
        ]
    ]

    return results, shap_df, meta_features



def get_latest_regime(dataset_path):

    results, shap_df, meta_features = run_regime_engine(dataset_path)

    latest = results.iloc[-1]

    shap_latest = shap_df.iloc[-1]

    top_features = shap_latest.abs().sort_values(ascending=False).head(3)

    explanation = {
        "top_drivers": list(top_features.index),
        "impact_scores": list(top_features.values)
    }

    output = latest.to_dict()

    output["shap_explanation"] = explanation

    return output