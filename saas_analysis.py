"""
SaaS Customer & Revenue Analytics
End-to-end analysis: MRR, Churn, Cohort Retention, Prediction Model
Author: Ramon Benavides
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                              ConfusionMatrixDisplay, roc_curve)
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('/home/claude/saas_project/outputs', exist_ok=True)

# ── Style ────────────────────────────────────────────────────
NAVY  = "#0B1C2C"
TEAL  = "#00A589"
MID   = "#5A6A7A"
LIGHT = "#D0D8DF"
AMBER = "#F59E0B"
RED   = "#EF4444"
PLAN_COLORS = {
    "Starter": "#94A3B8", "Growth": TEAL,
    "Business": NAVY, "Enterprise": AMBER
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': MID, 'xtick.color': MID, 'ytick.color': MID,
    'axes.titleweight': 'bold', 'axes.titlecolor': NAVY,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.color': LIGHT,
})

# ── Load Data ────────────────────────────────────────────────
df = pd.read_csv('/home/claude/saas_project/data/customers.csv', parse_dates=['signup_date', 'churn_date'])
mrr = pd.read_csv('/home/claude/saas_project/data/mrr_events.csv')

print(f"Dataset: {len(df):,} customers | {len(mrr):,} MRR events")
print(f"Date range: {df['signup_date'].min().date()} → 2024-12-31")
print(f"Overall churn rate: {df['churned'].mean():.1%}")
print(f"Active MRR: ${df[df['churned']==0]['mrr'].sum():,.0f}\n")


# ════════════════════════════════════════════════════════════════
# 1. MRR TREND
# ════════════════════════════════════════════════════════════════
monthly = mrr[mrr['event_type'] != 'churned'].groupby('month')['mrr'].sum().reset_index()
monthly['month_dt'] = pd.to_datetime(monthly['month'])
monthly = monthly.sort_values('month_dt')
monthly['mrr_m'] = monthly['mrr'] / 1_000_000

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(monthly['month_dt'], monthly['mrr_m'], alpha=0.15, color=TEAL)
ax.plot(monthly['month_dt'], monthly['mrr_m'], color=TEAL, lw=2.5)
ax.set_title("Monthly Recurring Revenue (MRR) — Jan 2022 to Dec 2024", pad=15)
ax.set_ylabel("MRR ($M)")
ax.set_xlabel("")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}M"))
peak = monthly['mrr_m'].max()
peak_month = monthly.loc[monthly['mrr_m'].idxmax(), 'month_dt']
ax.annotate(f"Peak: ${peak:.1f}M", xy=(peak_month, peak),
            xytext=(10, -20), textcoords='offset points',
            fontsize=9, color=NAVY, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.2))
plt.tight_layout()
plt.savefig('/home/claude/saas_project/outputs/01_mrr_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ MRR trend chart saved")


# ════════════════════════════════════════════════════════════════
# 2. MRR BREAKDOWN BY EVENT TYPE
# ════════════════════════════════════════════════════════════════
event_monthly = mrr.groupby(['month', 'event_type'])['mrr'].sum().reset_index()
event_pivot = event_monthly.pivot(index='month', columns='event_type', values='mrr').fillna(0)
event_pivot.index = pd.to_datetime(event_pivot.index)
event_pivot = event_pivot.sort_index()

fig, ax = plt.subplots(figsize=(12, 5))
cols_order = [c for c in ['new', 'recurring', 'expansion', 'contraction'] if c in event_pivot.columns]
colors = [TEAL, '#94A3B8', AMBER, RED][:len(cols_order)]
event_pivot[cols_order].plot(kind='area', stacked=True, ax=ax,
                              color=colors, alpha=0.85, legend=True)
ax.set_title("MRR Composition by Event Type", pad=15)
ax.set_ylabel("MRR ($)")
ax.set_xlabel("")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.legend(title="", loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig('/home/claude/saas_project/outputs/02_mrr_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ MRR breakdown chart saved")


# ════════════════════════════════════════════════════════════════
# 3. CHURN RATE BY PLAN
# ════════════════════════════════════════════════════════════════
plan_stats = df.groupby('plan').agg(
    total=('customer_id', 'count'),
    churned=('churned', 'sum'),
    avg_mrr=('mrr', 'mean'),
    avg_tenure=('tenure_months', 'mean')
).reset_index()
plan_stats['churn_rate'] = plan_stats['churned'] / plan_stats['total']
plan_order = ['Starter', 'Growth', 'Business', 'Enterprise']
plan_stats['plan'] = pd.Categorical(plan_stats['plan'], categories=plan_order, ordered=True)
plan_stats = plan_stats.sort_values('plan')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Churn rate
bars = axes[0].bar(plan_stats['plan'], plan_stats['churn_rate'] * 100,
                    color=[PLAN_COLORS[p] for p in plan_stats['plan']], width=0.55)
axes[0].set_title("Churn Rate by Plan")
axes[0].set_ylabel("Churn Rate (%)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
for bar, val in zip(bars, plan_stats['churn_rate']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1%}", ha='center', fontsize=9, color=NAVY, fontweight='bold')

# Avg MRR by plan
bars2 = axes[1].bar(plan_stats['plan'], plan_stats['avg_mrr'],
                     color=[PLAN_COLORS[p] for p in plan_stats['plan']], width=0.55)
axes[1].set_title("Average MRR by Plan")
axes[1].set_ylabel("Avg MRR ($)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
for bar, val in zip(bars2, plan_stats['avg_mrr']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f"${val:,.0f}", ha='center', fontsize=9, color=NAVY, fontweight='bold')

plt.suptitle("Plan Performance Analysis", fontsize=13, fontweight='bold', color=NAVY, y=1.02)
plt.tight_layout()
plt.savefig('/home/claude/saas_project/outputs/03_plan_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plan analysis chart saved")


# ════════════════════════════════════════════════════════════════
# 4. COHORT RETENTION HEATMAP
# ════════════════════════════════════════════════════════════════
df['cohort'] = df['signup_date'].dt.to_period('M').astype(str)
cohort_data = []
for cohort, group in df.groupby('cohort'):
    n_start = len(group)
    for m in range(0, 13):
        survived = (group['tenure_months'] > m).sum()
        cohort_data.append({'cohort': cohort, 'month': m, 'retention': survived / n_start})

cohort_df = pd.DataFrame(cohort_data)
cohort_pivot = cohort_df.pivot(index='cohort', columns='month', values='retention')
cohort_pivot = cohort_pivot.sort_index().tail(18)  # last 18 cohorts
cohort_pivot.columns = [f"M{c}" for c in cohort_pivot.columns]

fig, ax = plt.subplots(figsize=(14, 8))
mask = cohort_pivot.isna()
sns.heatmap(cohort_pivot, annot=True, fmt='.0%', cmap='YlGn',
            vmin=0, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Retention Rate'}, mask=mask,
            annot_kws={'size': 8})
ax.set_title("Cohort Retention Heatmap — Monthly Retention by Signup Cohort", pad=15, fontsize=13)
ax.set_xlabel("Months Since Signup")
ax.set_ylabel("Signup Cohort")
plt.tight_layout()
plt.savefig('/home/claude/saas_project/outputs/04_cohort_retention.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Cohort retention heatmap saved")


# ════════════════════════════════════════════════════════════════
# 5. ACQUISITION CHANNEL ANALYSIS
# ════════════════════════════════════════════════════════════════
channel_stats = df.groupby('channel').agg(
    customers=('customer_id', 'count'),
    churn_rate=('churned', 'mean'),
    avg_mrr=('mrr', 'mean'),
    avg_tenure=('tenure_months', 'mean')
).reset_index().sort_values('avg_tenure', ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
ch_colors = [TEAL if i == 0 else LIGHT for i in range(len(channel_stats))]

axes[0].barh(channel_stats['channel'], channel_stats['avg_tenure'],
             color=ch_colors, height=0.55)
axes[0].set_title("Avg Tenure (Months)")
axes[0].set_xlabel("Months")

ch_colors2 = [RED if v > channel_stats['churn_rate'].median() else TEAL
              for v in channel_stats['churn_rate']]
axes[1].barh(channel_stats['channel'], channel_stats['churn_rate'] * 100,
             color=ch_colors2, height=0.55)
axes[1].set_title("Churn Rate (%)")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

axes[2].barh(channel_stats['channel'], channel_stats['avg_mrr'],
             color=ch_colors, height=0.55)
axes[2].set_title("Avg MRR ($)")
axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

plt.suptitle("Acquisition Channel Performance", fontsize=13, fontweight='bold', color=NAVY)
plt.tight_layout()
plt.savefig('/home/claude/saas_project/outputs/05_channel_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Channel analysis chart saved")


# ════════════════════════════════════════════════════════════════
# 6. CUSTOMER SEGMENTATION
# ════════════════════════════════════════════════════════════════
def segment(row):
    if row['churned'] == 1 and row['tenure_months'] <= 3:
        return 'Early Churn'
    elif row['churned'] == 1:
        return 'Lost'
    elif row['tenure_months'] >= 12 and row['mrr'] >= 399:
        return 'Champion'
    elif row['tenure_months'] >= 6 and row['mrr'] >= 149:
        return 'Loyal'
    elif row['tenure_months'] <= 3:
        return 'New'
    elif row['nps_score'] < 0:
        return 'At Risk'
    else:
        return 'Active'

df['segment'] = df.apply(segment, axis=1)
seg_counts = df['segment'].value_counts()
seg_mrr = df[df['churned']==0].groupby('segment')['mrr'].sum()

seg_colors = {
    'Champion': AMBER, 'Loyal': TEAL, 'Active': '#94A3B8',
    'New': '#60A5FA', 'At Risk': RED, 'Early Churn': '#F97316', 'Lost': '#374151'
}

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
colors_list = [seg_colors.get(s, MID) for s in seg_counts.index]
wedges, texts, autotexts = axes[0].pie(
    seg_counts.values, labels=seg_counts.index, autopct='%1.1f%%',
    colors=colors_list, startangle=140, pctdistance=0.82,
    wedgeprops=dict(width=0.6))
for t in autotexts: t.set_fontsize(8)
axes[0].set_title("Customer Distribution by Segment")

active_seg_mrr = seg_mrr.sort_values(ascending=False)
bar_colors = [seg_colors.get(s, MID) for s in active_seg_mrr.index]
axes[1].bar(active_seg_mrr.index, active_seg_mrr.values / 1000,
            color=bar_colors, width=0.55)
axes[1].set_title("MRR by Segment (Active Customers)")
axes[1].set_ylabel("MRR ($K)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
plt.xticks(rotation=20, ha='right')

plt.suptitle("Customer Segmentation Analysis", fontsize=13, fontweight='bold', color=NAVY)
plt.tight_layout()
plt.savefig('/home/claude/saas_project/outputs/06_segmentation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Segmentation chart saved")


# ════════════════════════════════════════════════════════════════
# 7. CHURN PREDICTION MODEL
# ════════════════════════════════════════════════════════════════
print("\n── Churn Prediction Model ──")

features = ['mrr', 'tenure_months', 'seats', 'nps_score']
cat_features = ['plan', 'channel', 'industry', 'country']

df_model = df.copy()
le = LabelEncoder()
for col in cat_features:
    df_model[col + '_enc'] = le.fit_transform(df_model[col].astype(str))

feature_cols = features + [c + '_enc' for c in cat_features]
X = df_model[feature_cols]
y = df_model['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {auc:.3f}")
print(classification_report(y_test, y_pred))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, color=TEAL, lw=2, label=f'AUC = {auc:.3f}')
axes[0].plot([0,1],[0,1], '--', color=LIGHT, lw=1)
axes[0].fill_between(fpr, tpr, alpha=0.1, color=TEAL)
axes[0].set_title("ROC Curve — Churn Prediction")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(frameon=False)

# Feature Importance
importances = pd.Series(model.feature_importances_, index=feature_cols)
top_features = importances.sort_values(ascending=True).tail(8)
feat_labels = [f.replace('_enc', '').replace('_', ' ').title() for f in top_features.index]
colors_fi = [TEAL if v == top_features.max() else '#94A3B8' for v in top_features.values]
axes[1].barh(feat_labels, top_features.values, color=colors_fi, height=0.55)
axes[1].set_title("Feature Importance")
axes[1].set_xlabel("Importance Score")

# Churn probability distribution
churned_probs = y_prob[y_test == 1]
active_probs  = y_prob[y_test == 0]
axes[2].hist(active_probs,  bins=30, alpha=0.6, color=TEAL,  label='Active',  density=True)
axes[2].hist(churned_probs, bins=30, alpha=0.6, color=RED,   label='Churned', density=True)
axes[2].set_title("Predicted Churn Probability Distribution")
axes[2].set_xlabel("Churn Probability")
axes[2].set_ylabel("Density")
axes[2].legend(frameon=False)

plt.suptitle("Churn Prediction Model — Gradient Boosting Classifier", fontsize=13,
             fontweight='bold', color=NAVY)
plt.tight_layout()
plt.savefig('/home/claude/saas_project/outputs/07_churn_model.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Churn model chart saved")


# ════════════════════════════════════════════════════════════════
# 8. EXECUTIVE SUMMARY METRICS
# ════════════════════════════════════════════════════════════════
active = df[df['churned'] == 0]
total_mrr = active['mrr'].sum()
total_customers = len(df)
active_customers = len(active)
churn_rate = df['churned'].mean()
avg_ltv = (df['mrr'] * df['tenure_months']).mean()

print("\n── Executive Summary ──")
print(f"Total Customers:   {total_customers:,}")
print(f"Active Customers:  {active_customers:,}")
print(f"Total MRR:         ${total_mrr:,.0f}")
print(f"Overall Churn:     {churn_rate:.1%}")
print(f"Avg LTV:           ${avg_ltv:,.0f}")
print(f"Model AUC:         {auc:.3f}")
print("\n✓ All charts saved to /home/claude/saas_project/outputs/")
