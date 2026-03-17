"""
SaaS Customer & Revenue Analytics — Data Generator
Generates a realistic synthetic SaaS dataset with customer lifecycle,
subscription plans, MRR, and churn events.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N_CUSTOMERS = 2500
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 12, 31)

PLANS = {
    "Starter":    {"mrr": 49,  "churn_rate": 0.055},
    "Growth":     {"mrr": 149, "churn_rate": 0.035},
    "Business":   {"mrr": 399, "churn_rate": 0.022},
    "Enterprise": {"mrr": 999, "churn_rate": 0.010},
}

PLAN_WEIGHTS = [0.40, 0.30, 0.20, 0.10]
CHANNELS = ["Organic", "Paid Search", "Referral", "Direct", "Partner"]
CHANNEL_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]
INDUSTRIES = ["SaaS", "E-commerce", "Fintech", "Healthcare", "Media", "Logistics", "Education"]

def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

customers = []
for i in range(1, N_CUSTOMERS + 1):
    plan = np.random.choice(list(PLANS.keys()), p=PLAN_WEIGHTS)
    channel = np.random.choice(CHANNELS, p=CHANNEL_WEIGHTS)
    industry = random.choice(INDUSTRIES)
    signup_date = random_date(START_DATE, END_DATE - timedelta(days=30))

    base_churn = PLANS[plan]["churn_rate"]
    # Adjust churn by channel
    channel_mod = {"Referral": 0.7, "Partner": 0.8, "Organic": 0.9, "Direct": 1.0, "Paid Search": 1.2}
    adjusted_churn = base_churn * channel_mod[channel]

    # Determine if churned
    months_active = max(1, (END_DATE - signup_date).days // 30)
    churned = False
    churn_date = None
    for m in range(months_active):
        if random.random() < adjusted_churn:
            churned = True
            churn_date = signup_date + timedelta(days=30 * (m + 1))
            break

    end_date = churn_date if churned else None
    tenure_months = ((churn_date or END_DATE) - signup_date).days // 30

    # Seats (expansion revenue proxy)
    base_seats = {"Starter": 1, "Growth": 3, "Business": 8, "Enterprise": 20}
    seats = max(1, int(np.random.normal(base_seats[plan], base_seats[plan] * 0.3)))

    mrr = PLANS[plan]["mrr"] * seats

    # NPS score (churned customers tend lower)
    nps_base = 45 if not churned else 20
    nps = int(np.clip(np.random.normal(nps_base, 20), -100, 100))

    customers.append({
        "customer_id": f"CUST{i:04d}",
        "signup_date": signup_date.date(),
        "churn_date": churn_date.date() if churn_date else None,
        "plan": plan,
        "channel": channel,
        "industry": industry,
        "seats": seats,
        "mrr": mrr,
        "tenure_months": tenure_months,
        "churned": int(churned),
        "nps_score": nps,
        "country": np.random.choice(["US", "UK", "CA", "AU", "DE", "FR", "Other"],
                                     p=[0.50, 0.15, 0.10, 0.07, 0.07, 0.06, 0.05])
    })

df_customers = pd.DataFrame(customers)

# Generate monthly MRR events
mrr_events = []
for _, row in df_customers.iterrows():
    signup = pd.to_datetime(row["signup_date"])
    end = pd.to_datetime(row["churn_date"]) if row["churn_date"] else pd.to_datetime(END_DATE)
    current = signup.replace(day=1)
    prev_mrr = 0
    while current <= end.replace(day=1):
        # Small expansion/contraction noise
        noise = np.random.choice([-1, 0, 0, 0, 1], p=[0.05, 0.35, 0.35, 0.15, 0.10])
        seat_change = max(1, row["seats"] + noise)
        month_mrr = PLANS[row["plan"]]["mrr"] * seat_change

        event_type = "new" if prev_mrr == 0 else (
            "expansion" if month_mrr > prev_mrr else
            "contraction" if month_mrr < prev_mrr else
            "recurring"
        )
        if current.replace(day=1) == end.replace(day=1) and row["churned"]:
            event_type = "churned"
            month_mrr = 0

        mrr_events.append({
            "customer_id": row["customer_id"],
            "month": current.strftime("%Y-%m"),
            "plan": row["plan"],
            "channel": row["channel"],
            "mrr": month_mrr,
            "event_type": event_type,
        })
        prev_mrr = month_mrr
        current = (current + timedelta(days=32)).replace(day=1)

df_mrr = pd.DataFrame(mrr_events)

df_customers.to_csv("/home/claude/saas_project/data/customers.csv", index=False)
df_mrr.to_csv("/home/claude/saas_project/data/mrr_events.csv", index=False)

print(f"Customers: {len(df_customers)}")
print(f"MRR events: {len(df_mrr)}")
print(f"Churn rate: {df_customers['churned'].mean():.1%}")
print(f"Avg MRR: ${df_customers['mrr'].mean():.0f}")
print(f"Total MRR (active): ${df_customers[df_customers['churned']==0]['mrr'].sum():,.0f}")
