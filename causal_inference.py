# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:05:35 2026

@author: RauchD
"""
# %%
# imports
import os
os.getcwd()
os.chdir(r"C:\data exercises\classification")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu, skew
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone
from statsmodels.stats.proportion import proportions_ztest


# %%
# generate fake clicked / not-clicked data
# idea here:
# - people with certain traits are more likely to click the link
# - those same traits also lead to better outcomes
# - so before matching, clickers look better
# - after matching, differences should shrink a lot



np.random.seed(1234)

n_total = 14000

# ----------------------------
# make one base population first
# ----------------------------
device_type = np.random.choice(
    ["Mobile", "Desktop"],
    size=n_total,
    p=[0.68, 0.32]
)

user_visit_type = np.random.choice(
    ["New", "Returning"],
    size=n_total,
    p=[0.55, 0.45]
)

home_purchase_status = np.random.choice(
    ["Own", "Rent", "Other"],
    size=n_total,
    p=[0.58, 0.37, 0.05]
)

residence_type = np.random.choice(
    ["Single Family", "Townhome", "Condo", "Apartment"],
    size=n_total,
    p=[0.42, 0.18, 0.17, 0.23]
)

construction_type = np.random.choice(
    ["Frame", "Brick", "Masonry", "Mixed"],
    size=n_total,
    p=[0.46, 0.24, 0.17, 0.13]
)

year_built = np.random.randint(1945, 2023, size=n_total)

square_footage = np.clip(
    np.random.normal(2100, 650, size=n_total),
    500,
    6000
)

q2b_density = np.clip(
    np.random.normal(0.52, 0.18, size=n_total),
    0.01,
    1.5
)

replacement_amount = np.clip(
    50000
    + square_footage * 115
    + (2025 - year_built) * 350
    + np.random.normal(0, 45000, size=n_total),
    80000,
    1200000
)

base = pd.DataFrame({
    "HOME_PURCHASE_STATUS": home_purchase_status,
    "DEVICE_TYPE": device_type,
    "USER_VISIT_TYPE": user_visit_type,
    "YEAR_BUILT": year_built,
    "Q2B_DENSITY": q2b_density,
    "RESIDENCE_TYPE": residence_type,
    "CONSTRUCTION_TYPE": construction_type,
    "REPLACEMENT_AMOUNT_NUMBER": replacement_amount,
    "SQUARE_FOOTAGE": square_footage
})

# ----------------------------
# numeric versions just to drive click propensity + outcomes
# ----------------------------
own_num = (base["HOME_PURCHASE_STATUS"] == "Own").astype(int)
desktop_num = (base["DEVICE_TYPE"] == "Desktop").astype(int)
returning_num = (base["USER_VISIT_TYPE"] == "Returning").astype(int)
single_family_num = (base["RESIDENCE_TYPE"] == "Single Family").astype(int)
brick_num = (base["CONSTRUCTION_TYPE"] == "Brick").astype(int)

replacement_std = (base["REPLACEMENT_AMOUNT_NUMBER"] - base["REPLACEMENT_AMOUNT_NUMBER"].mean()) / base["REPLACEMENT_AMOUNT_NUMBER"].std()
sqft_std = (base["SQUARE_FOOTAGE"] - base["SQUARE_FOOTAGE"].mean()) / base["SQUARE_FOOTAGE"].std()
year_std = (base["YEAR_BUILT"] - base["YEAR_BUILT"].mean()) / base["YEAR_BUILT"].std()
density_std = (base["Q2B_DENSITY"] - base["Q2B_DENSITY"].mean()) / base["Q2B_DENSITY"].std()

# ----------------------------
# make link-click more likely for "better" users
# this is the confounding part
# ----------------------------
click_score = (
    -0.25
    + 0.70 * returning_num
    + 0.40 * desktop_num
    + 0.55 * own_num
    + 0.20 * single_family_num
    + 0.12 * brick_num
    + 0.52 * replacement_std
    + 0.40 * sqft_std
    - 0.10 * density_std
    + 0.05 * year_std
    + np.random.normal(0, 0.55, size=n_total)
)

click_prob = 1 / (1 + np.exp(-click_score))
group = np.random.binomial(1, click_prob, size=n_total)

base["Type"] = np.where(group == 1, "link click", "Not link click")
base["Group"] = group.astype(bool)

# ----------------------------
# outcomes depend on the SAME covariates, not on click itself
# so raw difference exists, but matched difference should fade
# ----------------------------

# latent "quality / value" score
value_score = (
    0.75 * own_num
    + 0.35 * desktop_num
    + 0.60 * returning_num
    + 0.15 * single_family_num
    + 0.08 * brick_num
    + 0.70 * replacement_std
    + 0.45 * sqft_std
    - 0.12 * density_std
    + np.random.normal(0, 0.55, size=n_total)
)

# no real causal effect from clicking
# if you want a tiny residual effect, change 0.00 to something like 0.03
true_click_effect = 0.00

# very skewed revenue-ish outcomes
base["TOTAL_RPA"] = np.exp(
    4.35 + 0.34 * value_score + true_click_effect * group + np.random.normal(0, 0.85, size=n_total)
)

base["TOTAL_RPU"] = np.exp(
    4.55 + 0.30 * value_score + true_click_effect * group + np.random.normal(0, 0.82, size=n_total)
)

base["TOTAL_REVENUE"] = np.exp(
    4.95 + 0.31 * value_score + true_click_effect * group + np.random.normal(0, 0.88, size=n_total)
)

# clicks / count outcomes
purchase_click_lambda = np.exp(
    -0.15 + 0.32 * value_score + true_click_effect * group
)
base["UNIQUE_PURCHASE_CLICKS"] = np.random.poisson(np.clip(purchase_click_lambda, 0.05, 12))

bridge_click_lambda = np.exp(
    0.65 + 0.28 * value_score + true_click_effect * group
)
base["UNIQUE_BRIDGE_CLICKS"] = np.random.poisson(np.clip(bridge_click_lambda, 0.10, 20))

# applicant / customer flags
app_logit = (
    -0.55
    + 0.85 * value_score
    + true_click_effect * group
    + np.random.normal(0, 0.25, size=n_total)
)
app_prob = 1 / (1 + np.exp(-app_logit))
base["IS_SITE_APPLICANT"] = np.where(np.random.binomial(1, app_prob) == 1, "Yes", "No")

cust_logit = (
    -1.05
    + 0.72 * value_score
    + true_click_effect * group
    + np.random.normal(0, 0.25, size=n_total)
)
cust_prob = 1 / (1 + np.exp(-cust_logit))
base["IS_SITE_CUSTOMER"] = np.where(np.random.binomial(1, cust_prob) == 1, "Yes", "No")

# ----------------------------
# split into the 2 files your script expects
# ----------------------------
clicked_link = base[base["Type"] == "link click"].drop(columns=["Group"]).copy()
not_clicked_link = base[base["Type"] == "Not link click"].drop(columns=["Group"]).copy()

clicked_link.to_csv("clicked_link.csv", index=False)
not_clicked_link.to_csv("not_clicked_link.csv", index=False)

print("clicked_link shape:", clicked_link.shape)
print("not_clicked_link shape:", not_clicked_link.shape)

print("\nMean TOTAL_RPA before matching:")
print(base.groupby("Type")["TOTAL_RPA"].mean())

print("\nMean applicant rate before matching:")
tmp_app = base.assign(site_applicant_num=(base["IS_SITE_APPLICANT"] == "Yes").astype(int))
print(tmp_app.groupby("Type")["site_applicant_num"].mean())

print(clicked_link.head())
print(not_clicked_link.head())


# %%
# combine both groups
matched = pd.concat([not_clicked_link, clicked_link], ignore_index=True)

# boolean flag for treatment group
matched["Group"] = matched["Type"] == "link click"

print(matched.head())
print(matched.dtypes)


# %%
# fill missing values
# doing something close to the R code here
for col in matched.columns:
    if pd.api.types.is_numeric_dtype(matched[col]):
        matched[col] = matched[col].fillna(0)
    elif pd.api.types.is_bool_dtype(matched[col]):
        matched[col] = matched[col].fillna(False)
    else:
        matched[col] = matched[col].fillna("0")


# %%
# turn yes/no fields into numeric flags
matched["site_applicant_num"] = np.where(matched["IS_SITE_APPLICANT"].astype(str).str.strip().str.lower() == "yes", 1, 0)
matched["site_customer_num"] = np.where(matched["IS_SITE_CUSTOMER"].astype(str).str.strip().str.lower() == "yes", 1, 0)


# %%
# quick summary - avg RPA by group
summary_rpa_group = (
    matched.groupby("Group", as_index=False)["TOTAL_RPA"]
    .mean()
    .rename(columns={"TOTAL_RPA": "mean_TOTAL_RPA"})
)

print(summary_rpa_group)

plt.figure(figsize=(7, 5))
sns.barplot(data=summary_rpa_group, x="Group", y="mean_TOTAL_RPA")
plt.xlabel("Group")
plt.ylabel("Average total RPA")
plt.title("Total RPA by group")
plt.show()


# %%
# avg RPA by type
summarized_rpa = (
    matched.groupby("Type", as_index=False)["TOTAL_RPA"]
    .mean()
    .rename(columns={"TOTAL_RPA": "mean_Total_RPA"})
)

print(summarized_rpa)


# %%
# naive test before matching
# this is basically the "raw difference" version, which is not really the right comparison
link_click_rpa_raw = matched.loc[matched["Type"] == "link click", "TOTAL_RPA"]
not_click_rpa_raw = matched.loc[matched["Type"] == "Not link click", "TOTAL_RPA"]

raw_rpa_test = mannwhitneyu(link_click_rpa_raw, not_click_rpa_raw, alternative="two-sided")
print("Raw TOTAL_RPA Mann-Whitney test:")
print(raw_rpa_test)


# %%
# propensity score matching setup
# using something similar in spirit to MatchIt nearest-neighbor matching
match_cols = [
    "HOME_PURCHASE_STATUS",
    "DEVICE_TYPE",
    "USER_VISIT_TYPE",
    "YEAR_BUILT",
    "Q2B_DENSITY",
    "RESIDENCE_TYPE",
    "CONSTRUCTION_TYPE",
    "REPLACEMENT_AMOUNT_NUMBER",
    "SQUARE_FOOTAGE",
]

# keep only columns needed for matching
match_df = matched.copy()

X = match_df[match_cols]
y = match_df["Group"].astype(int)

# split numeric vs categorical
numeric_features = [c for c in match_cols if pd.api.types.is_numeric_dtype(match_df[c])]
categorical_features = [c for c in match_cols if c not in numeric_features]

# preprocess:
# - median for numeric
# - most frequent for categorical
# - one-hot encode categoricals
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
    ]
)

ps_model = Pipeline([
    ("prep", preprocessor),
    ("logit", LogisticRegression(max_iter=2000, random_state=1234))
])

ps_model.fit(X, y)

# propensity score = probability of being in click group
match_df["propensity_score"] = ps_model.predict_proba(X)[:, 1]

print(match_df[["Type", "Group", "propensity_score"]].head())


# %%
# nearest-neighbor matching on propensity score
treated = match_df[match_df["Group"]].copy()
control = match_df[~match_df["Group"]].copy()

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[["propensity_score"]])

distances, indices = nn.kneighbors(treated[["propensity_score"]])

matched_control = control.iloc[indices.flatten()].copy()
matched_control.index = treated.index  # line up rows a bit more cleanly

df_match = pd.concat([treated, matched_control], axis=0).reset_index(drop=True)

print(df_match["Type"].value_counts())
print(df_match.shape)


# %%
# summaries after matching

summarized_rpa = (
    df_match.groupby("Type", as_index=False)["TOTAL_RPA"]
    .mean()
    .rename(columns={"TOTAL_RPA": "mean_Total_RPA"})
)
print("\nMatched mean TOTAL_RPA")
print(summarized_rpa)

summarized_rpu = (
    df_match.groupby("Type", as_index=False)["TOTAL_RPU"]
    .mean()
    .rename(columns={"TOTAL_RPU": "mean_Total_RPU"})
)
print("\nMatched mean TOTAL_RPU")
print(summarized_rpu)

summarized_replacement = (
    df_match.groupby("Type", as_index=False)["REPLACEMENT_AMOUNT_NUMBER"]
    .mean()
    .rename(columns={"REPLACEMENT_AMOUNT_NUMBER": "mean_REPLACEMENT_AMOUNT_NUMBER"})
)
print("\nMatched mean replacement amount")
print(summarized_replacement)

summarized_site_applicant_num = (
    df_match.groupby("Type", as_index=False)["site_applicant_num"]
    .mean()
    .rename(columns={"site_applicant_num": "mean_site_applicant_num"})
)
print("\nMatched mean site applicant rate")
print(summarized_site_applicant_num)


# %%
# boxplots
plt.figure(figsize=(7, 5))
sns.boxplot(data=df_match, x="Type", y="TOTAL_RPA")
plt.title("TOTAL_RPA by type (matched)")
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(data=df_match, x="Type", y="site_applicant_num")
plt.title("Applicant flag by type (matched)")
plt.show()


# %%
# matched tests
# using Mann-Whitney because this is the closest quick Python equivalent to wilcox.test here

def run_mwu(df, outcome, group_col="Type", treat_label="link click", control_label="Not link click"):
    treat_vals = df.loc[df[group_col] == treat_label, outcome]
    control_vals = df.loc[df[group_col] == control_label, outcome]
    result = mannwhitneyu(treat_vals, control_vals, alternative="two-sided")
    return result

link_rpa = run_mwu(df_match, "TOTAL_RPA")
print("\nTOTAL_RPA test")
print(link_rpa)

link_rpu = run_mwu(df_match, "TOTAL_RPU")
print("\nTOTAL_RPU test")
print(link_rpu)

link_bridge = run_mwu(df_match, "UNIQUE_PURCHASE_CLICKS")
print("\nUNIQUE_PURCHASE_CLICKS test")
print(link_bridge)

link_revenue = run_mwu(df_match, "TOTAL_REVENUE")
print("\nTOTAL_REVENUE test")
print(link_revenue)

link_app = run_mwu(df_match, "site_applicant_num")
print("\nsite_applicant_num test")
print(link_app)


# %%
# skewness check
app_skew = skew(df_match["site_applicant_num"], bias=False)
print("\nSkewness of site_applicant_num:")
print(app_skew)


# %%
# counts for proportion test
link_click_apps = df_match[(df_match["site_applicant_num"] == 1) & (df_match["Type"] == "link click")].shape[0]
not_click_apps = df_match[(df_match["site_applicant_num"] == 1) & (df_match["Type"] == "Not link click")].shape[0]

link_click_n = df_match[df_match["Type"] == "link click"].shape[0]
not_click_n = df_match[df_match["Type"] == "Not link click"].shape[0]

print("\nApplicants in link click group:", link_click_apps)
print("Applicants in not link click group:", not_click_apps)
print("Total in link click group:", link_click_n)
print("Total in not link click group:", not_click_n)

z_stat, p_value = proportions_ztest(
    count=[link_click_apps, not_click_apps],
    nobs=[link_click_n, not_click_n]
)

print("\nProportion z-test for applicant rate")
print("z-stat:", z_stat)
print("p-value:", p_value)


# %%
# histogram of bridge clicks by type
g = sns.FacetGrid(df_match, row="Type", height=4, aspect=2)
g.map_dataframe(sns.histplot, x="UNIQUE_BRIDGE_CLICKS", bins=30, edgecolor="black")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Distribution of UNIQUE_BRIDGE_CLICKS by type")
plt.show()
