-- ============================================================
-- SaaS Customer & Revenue Analytics
-- SQL Metric Definitions
-- Author: Ramon Benavides
-- ============================================================

-- ── 1. MONTHLY RECURRING REVENUE (MRR) SUMMARY ──────────────
-- Calculates total MRR, new MRR, expansion, contraction,
-- and churned MRR per month.

CREATE VIEW mrr_summary AS
SELECT
    month,
    SUM(CASE WHEN event_type = 'recurring'   THEN mrr ELSE 0 END) AS recurring_mrr,
    SUM(CASE WHEN event_type = 'new'         THEN mrr ELSE 0 END) AS new_mrr,
    SUM(CASE WHEN event_type = 'expansion'   THEN mrr ELSE 0 END) AS expansion_mrr,
    SUM(CASE WHEN event_type = 'contraction' THEN mrr ELSE 0 END) AS contraction_mrr,
    SUM(CASE WHEN event_type = 'churned'     THEN mrr ELSE 0 END) AS churned_mrr,
    SUM(mrr)                                                        AS total_mrr,
    COUNT(DISTINCT customer_id)                                     AS active_customers
FROM mrr_events
GROUP BY month
ORDER BY month;


-- ── 2. CUSTOMER CHURN RATE BY PLAN ──────────────────────────
-- Monthly churn rate = churned customers / active customers at start of month

CREATE VIEW churn_by_plan AS
SELECT
    plan,
    COUNT(*)                                          AS total_customers,
    SUM(churned)                                      AS churned_customers,
    ROUND(AVG(churned) * 100, 2)                      AS churn_rate_pct,
    ROUND(AVG(tenure_months), 1)                      AS avg_tenure_months,
    ROUND(AVG(mrr), 0)                                AS avg_mrr,
    -- Customer Lifetime Value estimate: avg_mrr / monthly_churn_rate
    ROUND(AVG(mrr) / NULLIF(AVG(churned), 0), 0)     AS estimated_ltv
FROM customers
GROUP BY plan
ORDER BY avg_mrr;


-- ── 3. MRR BY ACQUISITION CHANNEL ────────────────────────────
-- Revenue quality by channel — which channels bring the best customers?

CREATE VIEW channel_performance AS
SELECT
    channel,
    COUNT(*)                             AS total_customers,
    SUM(churned)                         AS churned_customers,
    ROUND(AVG(churned) * 100, 2)         AS churn_rate_pct,
    ROUND(AVG(mrr), 0)                   AS avg_mrr,
    ROUND(AVG(tenure_months), 1)         AS avg_tenure_months,
    SUM(mrr * tenure_months)             AS total_revenue_generated
FROM customers
GROUP BY channel
ORDER BY total_revenue_generated DESC;


-- ── 4. COHORT RETENTION BASE ──────────────────────────────────
-- Assigns each customer a signup cohort (year-month) for retention analysis

CREATE VIEW cohort_base AS
SELECT
    customer_id,
    DATE_FORMAT(signup_date, '%Y-%m')    AS cohort_month,
    tenure_months,
    churned,
    plan,
    mrr
FROM customers;


-- ── 5. COHORT RETENTION MATRIX ────────────────────────────────
-- Retention rate at each month for each signup cohort

CREATE VIEW cohort_retention AS
SELECT
    cohort_month,
    tenure_months                               AS months_since_signup,
    COUNT(*)                                    AS customers_remaining,
    ROUND(COUNT(*) * 100.0 /
        FIRST_VALUE(COUNT(*)) OVER (
            PARTITION BY cohort_month
            ORDER BY tenure_months
        ), 1)                                   AS retention_rate_pct
FROM cohort_base
WHERE tenure_months >= 0
GROUP BY cohort_month, tenure_months
ORDER BY cohort_month, tenure_months;


-- ── 6. HIGH-VALUE CUSTOMER SEGMENTATION ──────────────────────
-- RFM-style segmentation: identifies at-risk vs champion customers

CREATE VIEW customer_segments AS
SELECT
    customer_id,
    plan,
    mrr,
    tenure_months,
    churned,
    nps_score,
    CASE
        WHEN churned = 0 AND tenure_months >= 12 AND mrr >= 399  THEN 'Champion'
        WHEN churned = 0 AND tenure_months >= 6  AND mrr >= 149  THEN 'Loyal'
        WHEN churned = 0 AND tenure_months <= 3                  THEN 'New'
        WHEN churned = 0 AND nps_score < 0                       THEN 'At Risk'
        WHEN churned = 1 AND tenure_months <= 3                  THEN 'Early Churn'
        WHEN churned = 1                                          THEN 'Lost'
        ELSE 'Active'
    END AS segment
FROM customers;


-- ── 7. NET REVENUE RETENTION (NRR) ───────────────────────────
-- NRR > 100% means expansion revenue exceeds churn
-- Calculated per month using MRR events

CREATE VIEW net_revenue_retention AS
SELECT
    m1.month,
    SUM(m1.mrr)                                              AS starting_mrr,
    SUM(m2.mrr)                                              AS ending_mrr,
    ROUND(SUM(m2.mrr) * 100.0 / NULLIF(SUM(m1.mrr), 0), 1) AS nrr_pct
FROM mrr_events m1
JOIN mrr_events m2
    ON  m1.customer_id = m2.customer_id
    AND m2.month = DATE_FORMAT(DATE_ADD(STR_TO_DATE(CONCAT(m1.month, '-01'), '%Y-%m-%d'), INTERVAL 1 MONTH), '%Y-%m')
WHERE m1.event_type != 'churned'
GROUP BY m1.month
ORDER BY m1.month;
