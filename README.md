# E-Commerce Sales Analysis & Customer Intelligence

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-blueviolet)

A full-stack data science project applied to an e-commerce dataset — from raw data ingestion to
production-ready machine learning pipelines. Built as a portfolio piece demonstrating end-to-end
analytical and engineering skills.

---

## Project Structure

```
.
├── salesAnalysis_dx1_improved.ipynb   # Main notebook (17 sections, 237 cells)
├── data/
│   ├── customers.csv                  # Customer demographics
│   ├── products.csv                   # Product catalog
│   └── transactions.csv               # Purchase history
├── models/
│   └── clv_xgb_pipeline.pkl           # Serialised CLV pipeline (joblib)
├── requirements.txt
└── README.md
```

---

## Key Results

| Analysis | Finding |
|---|---|
| **Pareto / Lorenz** | Gini > 0.6 — top ~20% of customers drive ~80% of revenue |
| **Anomaly detection** | October category 1 drop: −80% sessions, stable prices → supply failure, not demand |
| **A/B Test (gender)** | Transaction-level p < 0.001 was a **false positive** (pseudo-replication); user-level p = 0.18 → no effect |
| **Churn model** | Random Forest outperforms Logistic Regression on AUC & recall; recency is the dominant signal |
| **CLV — BG/NBD** | Probabilistic 12-month forecast per customer using `lifetimes` |
| **CLV — XGBoost** | ML regression pipeline, TimeSeriesSplit CV, serialised for production scoring |

---

## Sections Overview

| # | Section | Techniques |
|---|---|---|
| 1 | Introduction & Objectives | — |
| 2 | Data Preparation | Pandas, multi-source merge |
| 3 | Data Cleaning & Validation | Missing values, referential integrity, type checks |
| 4 | Feature Engineering | Date features, session aggregation |
| 5 | Exploratory Data Analysis | Histograms, basket distribution |
| 6 | Revenue Analysis | Pareto/Lorenz, Gini coefficient, top-client heatmap |
| 7 | Anomaly Detection | Time-series breakdown, root-cause analysis |
| 8 | Customer Analysis | Revenue/purchase distribution by age & gender |
| 9 | KPI Dashboard | Aggregated metrics |
| 10 | A/B Testing | Mann-Whitney U, Cohen's d, pseudo-replication correction |
| 11 | Correlation & Features | Pearson correlation matrix |
| 12 | Business Insights | 5 actionable recommendations |
| 13 | Final Conclusion | Executive summary across all sections |
| 14 | RFM Segmentation | Recency/Frequency/Monetary scoring, 5 segments |
| 15 | Customer Clustering | KMeans, elbow + silhouette selection |
| 16 | Churn Prediction | Logistic Regression vs Random Forest, ROC, CV |
| 17 | **CLV Prediction** | **BG/NBD + Gamma-Gamma, XGBoost Pipeline, Gold/Silver/Bronze tiers** |

---

## Installation

```bash
git clone https://github.com/<your-username>/ecommerce-sales-analysis.git
cd ecommerce-sales-analysis

pip install -r requirements.txt

jupyter notebook salesAnalysis_dx1_improved.ipynb
```

> **Data**: Place the three CSV files (`customers.csv`, `products.csv`, `transactions.csv`)
> inside a `./data/` folder before running the notebook.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Statistics | `scipy`, `statsmodels` |
| Machine Learning | `scikit-learn`, `xgboost` |
| CLV (probabilistic) | `lifetimes` (BG/NBD + Gamma-Gamma) |
| Model serialisation | `joblib` |

---

## Methodology Highlights

### Temporal CLV Validation
The CLV model uses a strict **calibration / holdout split** (75% / 25% of the timeline)
to avoid data leakage. Features are computed exclusively from the calibration window;
the holdout window provides the ground-truth revenue target.

### Pseudo-replication Correction (A/B Test)
The transaction-level gender test returned p < 0.001 — a classic false positive caused
by non-independent observations (multiple rows per customer). The analysis corrects this
by re-running at user level and session level, where the dependency is removed.

### Dual CLV Approach
| Model | Use case |
|---|---|
| BG/NBD + Gamma-Gamma | Customers with few purchases — no labelled target required |
| XGBoost Pipeline | Established customers — leverages full feature set |

The XGBoost pipeline wraps preprocessing and modelling in a single `sklearn.Pipeline`
object, serialised with `joblib` for straightforward production deployment.

---

## Business Roadmap

| Priority | Action | Model |
|---|---|---|
| 1 | Retain Gold-tier + high-churn-score customers | CLV × Churn |
| 2 | Reactivate At-Risk RFM segment | Churn model |
| 3 | Upsell Loyal Customers | RFM segmentation |
| 4 | Audit October category 1 supply chain | Anomaly detection |
| 5 | Redirect gender-targeted budget to RFM tiers | A/B test |

---

## Author

**Zobir Zeghoud**
[LinkedIn](https://linkedin.com/in/) · [GitHub](https://github.com/)

---

## License

MIT — free to use and adapt with attribution.
