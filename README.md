# Causal Inference in a Product Funnel (Propensity Score Matching in R & Python)

## TL;DR

* Users who clicked a new feature looked more valuable at first (higher revenue per user)
* After matching similar users, that effect disappeared
* The difference was driven by *who clicked*, not the feature itself
* In practice, this would have led to a bad rollout decision

Main takeaway: don’t trust raw A/B comparisons when users self-select into treatment.

---

## What this is

This project looks at a pretty common product question:

> “Did this feature actually improve outcomes, or are better users just more likely to use it?”

I used propensity score matching to try to separate those two things.

The same analysis is implemented in both **R and Python** — partly to sanity check results, and partly to understand how the workflows differ.

---

## Real-world context

This is based on a real A/B test I worked on.

A new feature was shown to ~10% of users. Within that group, users could choose whether to engage with it.

Initial results looked promising:

* users who engaged had higher revenue per user

That led to a proposal to roll the feature out more broadly.

The issue was that engagement wasn’t random — the users choosing to click were different to begin with.

Using statistical matching, I compared those users to similar users who didn’t engage.

After matching:

* the revenue lift disappeared
* in some cases, similar non-clickers performed better

So the original result was mostly selection bias.

Because of that, the rollout was delayed and the feature continued as a test. Conversion rates improved later, and the feature was rolled out in a better state.

---

## Data + setup

Simulated version of a product funnel dataset with:

* user attributes (device, visit type, property characteristics, etc.)
* behavioral outcomes (revenue, clicks, conversion flags)

In the original work:

* data came from SQL (clickstream level)
* modeling was done in R
* results were shared in Tableau

This repo recreates that workflow in both R and Python.

---

## Approach

High level:

1. Combine users who clicked the feature and those who didn’t
2. Estimate a propensity score (probability of clicking) based on user characteristics
3. Match users with similar scores across groups
4. Recompare outcomes after matching

The goal is to approximate:

> “What would have happened if these same users had *not* clicked?”

---

## What I saw

Before matching:

* clickers had higher revenue per user
* looked like a clear win

After matching:

* differences were much smaller
* often not statistically significant

So:

* the feature itself wasn’t driving the outcome
* user selection was

---

## R vs Python

I implemented the same idea in both:

* **R**: `MatchIt`, more direct matching workflow
* **Python**: logistic regression + nearest-neighbor matching

Results were broadly consistent, but the workflow is definitely smoother in R for this type of problem.

---

## Why this matters

This comes up all the time in product work:

* features that look good because power users adopt them
* misleading “uplift” from non-random behavior
* pressure to ship based on early metrics

This is a simple example, but the pattern is very real.

---

## Repo structure

```text
.
├── r/
│   └── causal_matching.R
├── python/
│   └── causal_matching.py
├── data/
│   └── (simulated csvs)
```

---

## Note on data

The dataset used here is synthetic and designed to mirror the structure and behavior of the original problem.

In the real analysis, data came from clickstream logs and internal product metrics. Since that data can’t be shared, I generated a synthetic version that preserves:

* the same feature structure (user/device/home characteristics)
* similar distributions (skewed revenue, conversion rates, etc.)
* the key pattern of interest (selection bias before matching, reduced differences after matching)

This makes the analysis reproducible while keeping the focus on the modeling and decision-making rather than the specific data.

---

## How to run

### Python

```bash
conda activate classification
python causal_matching.py
```

### R

Run the script in RStudio:

```r
source("causal_matching.R")
```

---

## Takeaway

For problems like this:

* raw comparisons can be misleading
* matching / causal methods help sanity check results
* model choice matters less than *how you frame the question*

If I saw this in production again, I’d treat early uplift from opt-in behavior pretty skeptically until I could control for who is opting in.

---

## If I kept going

* use doubly robust methods / causal forests
* test sensitivity to different matching strategies
* run the same analysis on a larger dataset

---
