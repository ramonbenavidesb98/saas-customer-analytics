# SaaS Customer & Revenue Analytics

An end-to-end analytics project analyzing customer behavior, revenue trends, and churn across a B2B SaaS business. Covers MRR decomposition, cohort retention analysis, acquisition channel performance, customer segmentation, and a churn prediction model.

Built to demonstrate production-ready analytics skills applicable to SaaS, fintech, e-commerce, and any data-driven business with recurring revenue or customer lifecycle data.

---

## Business Questions Answered

| Question | Method |
|---|---|
| How is MRR growing — and what's driving it? | MRR trend + event decomposition |
| Which plans retain customers best? | Churn rate & tenure by plan |
| Which acquisition channels bring the highest-value customers? | Channel performance analysis |
| Are newer cohorts retaining better than older ones? | Cohort retention heatmap |
| Which customers are most at risk of churning? | Gradient Boosting churn model (AUC: 0.827) |
| How should we segment our customer base? | RFM-style segmentation |

---

## Key Findings

- **Total MRR: $5.26M** across 1,438 active customers
- **Enterprise plan churn is 4-5x lower** than Starter — upsell path is the highest-leverage retention lever
- **Referral channel** produces the longest customer tenure and lowest churn — worth investing in
- **Cohort retention stabilizes around month 6** — customers who survive 6 months have significantly higher LTV
- **Churn prediction model (AUC 0.827)** — NPS score, tenure, and plan are the strongest predictors
- **17% of active customers classified as "At Risk"** based on low NPS — actionable intervention target

---

## Project Structure

```
saas-analytics/
├── data/
│   ├── generate_data.py       # Synthetic dataset generator
│   ├── customers.csv          # 2,500 customer records
│   └── mrr_events.csv         # 33,577 monthly MRR events
├── sql/
│   └── metrics.sql            # MRR summary, churn by plan, cohort base,
│                              # channel performance, NRR, segmentation views
├── analysis/
│   └── saas_analysis.py       # Full Python analysis + model
├── outputs/
│   ├── 01_mrr_trend.png
│   ├── 02_mrr_breakdown.png
│   ├── 03_plan_analysis.png
│   ├── 04_cohort_retention.png
│   ├── 05_channel_analysis.png
│   ├── 06_segmentation.png
│   └── 07_churn_model.png
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Generation | Python (Pandas, NumPy) |
| SQL Metrics | Standard SQL (MySQL-compatible) — MRR views, cohort queries, NRR |
| Analysis | Python — Pandas, Matplotlib, Seaborn |
| Prediction Model | scikit-learn — Gradient Boosting Classifier |
| Visualizations | Matplotlib, Seaborn |

---

## Visualizations

### MRR Trend (Jan 2022 — Dec 2024)
![MRR Trend](01_mrr_trend.png)

### MRR Composition by Event Type
![MRR Breakdown](02_mrr_breakdown.png)

### Plan Performance: Churn Rate & Average MRR
![Plan Analysis](03_plan_analysis.png)

### Cohort Retention Heatmap
![Cohort Retention](04_cohort_retention.png)

### Acquisition Channel Performance
![Channel Analysis](05_channel_analysis.png)

### Customer Segmentation
![Segmentation](06_segmentation.png)

### Churn Prediction Model
![Churn Model](07_churn_model.png)

---

## How to Run

```bash
# 1. Generate data
python data/generate_data.py

# 2. Run full analysis
python analysis/saas_analysis.py
```

**Requirements:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---

## About

Built by **Ramon Benavides** — analytics and operations professional with nearly 3 years of experience building forecasting models, monitoring systems, and data pipelines.

- [Portfolio](https://ramonbenavidesb98.github.io)
- [LinkedIn](https://linkedin.com/in/ramonbenavides)
- [GitHub](https://github.com/ramonbenavidesb98)
