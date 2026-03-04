

&nbsp;               DATA LAYER

&nbsp;       ┌──────────────────────────┐

&nbsp;       │ sales / financial / inv  │

&nbsp;       └─────────────┬────────────┘

&nbsp;                     │

&nbsp;                     ▼

&nbsp;           FEATURE ENGINEERING

&nbsp;       ┌──────────────────────────┐

&nbsp;       │ regime\_detection\_dataset │

&nbsp;       └─────────────┬────────────┘

&nbsp;                     │

&nbsp;                     ▼

&nbsp;           REGIME DETECTION ENGINE

&nbsp;       ┌──────────────────────────────────┐

&nbsp;       │                                  │

&nbsp;       │  Statistical Models              │

&nbsp;       │  ├ CUSUM drift                   │

&nbsp;       │  ├ STL decomposition             │

&nbsp;       │  └ Change-point detection        │

&nbsp;       │                                  │

&nbsp;       │  ML Models                       │

&nbsp;       │  ├ Isolation Forest              │

&nbsp;       │  ├ Bayesian Gaussian Mixture     │

&nbsp;       │  └ LightGBM meta classifier      │

&nbsp;       │                                  │

&nbsp;       │  Probabilistic Model             │

&nbsp;       │  └ Hidden Markov Model (HMM)     │

&nbsp;       │                                  │

&nbsp;       │  Ensemble Voting Layer           │

&nbsp;       │                                  │

&nbsp;       │  Explainability Layer            │

&nbsp;       │  └ SHAP feature attribution      │

&nbsp;       │                                  │

&nbsp;       └─────────────┬────────────────────┘

&nbsp;                     │

&nbsp;                     ▼

&nbsp;             INSIGHT OUTPUT TABLE

&nbsp;       ┌─────────────────────────────┐

&nbsp;       │ week | regime | confidence  │

&nbsp;       │ drivers | anomaly\_score     │

&nbsp;       └─────────────┬───────────────┘

&nbsp;                     │

&nbsp;       ┌─────────────┴─────────────┐

&nbsp;       ▼                           ▼

&nbsp; STREAMLIT DASHBOARD        OPENAI CHATBOT

&nbsp; Charts / analytics         reasoning + Q\&A





--------------------------------------------------

Time-Series Signals

&nbsp;     │

&nbsp;     ▼

Feature Engineering Layer

&nbsp;     │

&nbsp;     ▼

Signal Detection Models

&nbsp;  ├─ Rolling variance detection

&nbsp;  ├─ STL decomposition

&nbsp;  ├─ CUSUM drift detection

&nbsp;  ├─ Bayesian change point detection

&nbsp;     │

&nbsp;     ▼

Voting / Ensemble Logic

&nbsp;     │

&nbsp;     ▼

Regime Classification











-------------------------------------------









Regime Dataset

&nbsp;     │

Feature Standardization

&nbsp;     │

Signal Detectors

&nbsp;├ Supply Shock Detector

&nbsp;├ Demand Shock Detector

&nbsp;├ Competitive Pressure Detector

&nbsp;├ Structural Change Detector

&nbsp;     │

Change-Point Models

&nbsp;├ CUSUM

&nbsp;├ Bayesian Change Point

&nbsp;├ STL anomaly

&nbsp;     │

Ensemble Voting Model

&nbsp;     │

Regime Classification

&nbsp;     │

Explainability Layer

&nbsp;├ SHAP

&nbsp;└ Feature Attribution





----------------------------------------------------

signals for the four regimes

Supply Shock



Detect abnormal cost changes.

Signals:



* avg\_unit\_cost
* cost\_pct\_change
* cost\_volatility



Demand Shock



Detect sudden demand drops.

Signals:



* total\_revenue
* revenue\_pct\_change
* revenue\_volatility
* Competitive Pressure



Detect discount spikes.

Signals:



* avg\_discount
* discount\_volatility
* discount\_zscore
* Structural Change



Detect structural expansion.

Signals:



* channel\_count
* region\_count
* sku\_count











CUSUM Drift: It tracks small changes over time. If your costs go up by 0.1% every week, CUSUM is the alarm that eventually says, "Hey, all these tiny increases have added up to a massive shift!"



STL Decomposition: It breaks your data into three parts: Trend (the long-term move), Seasonality (holiday spikes), and Remainder (random noise). It's used to see the real trend without getting distracted by Christmas sales.



Change-point Detection: It looks for a "Before and After" moment. For example, "The day the pandemic started, the data broke and never went back to normal."



Isolation Forest: Imagine a forest of trees. Normal data points travel deep into the trees. "Outliers" (anomalies) get caught early. It’s used to find weird weeks that don't fit any pattern.



Bayesian Gaussian Mixture: This is a Clustering tool. It looks at your data and says, "I see three distinct types of weeks here. I'll label them Type A, B, and C." It automatically discovers your "Regimes" (e.g., Stable, Volatile, or High-Growth).



LightGBM Meta Classifier: This acts like the Head Judge. It takes the results from all other models and makes the final decision on which regime we are in.



Hidden Markov Model (HMM): This is the "Gold Standard" for regime detection. It assumes that there is a hidden state (the Regime) that we can't see, but we can see its effects (Sales/Costs).



Example: You can't see "Inflation" directly on a receipt, but you see prices rising. HMM calculates the probability that we have moved from a "Low Inflation" state to a "High Inflation" state.
