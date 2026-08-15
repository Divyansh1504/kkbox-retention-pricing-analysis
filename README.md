# KKBox Retention & Pricing Analysis

A retention and pricing analysis on a real subscription business, built as a portfolio piece
for Business Analyst / Operations Analyst / Consulting applications.

**This is not a churn-prediction project.** Nearly every public repo on this dataset stacks
classifiers and reports AUC. This one asks a business question instead: *where is this
subscription business losing revenue, which levers actually retain subscribers, and what
should the company do about it?* A predictive model appears in [notebook 05](notebooks/05_churn_model.ipynb),
as one section near the end — not the centerpiece.

> **Status:** the analysis code in `src/` and `notebooks/` is complete and has been checked for
> syntax and logical consistency. The
> [Recommendation Memo](#recommendation-memo) section is a template — run notebooks `01`–`05` in
> order, then replace the bracketed placeholders with the actual output before treating this as
> a finished writeup.

## Data

**Source:** [WSDM - KKBox's Churn Prediction Challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge)
(Kaggle). KKBox is a music streaming subscription service in Southeast Asia.

The raw data is **not included in this repo** (see [Data handling](#data-handling)). To
reproduce the analysis:

1. Log into Kaggle and open the
   [competition rules page](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/rules) —
   **you must accept the rules before you're able to download anything**, including via the API.
2. Download these four files from the
   [competition's data page](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data)
   and place them, extracted, directly in `data/`:

   | File | Contents |
   |---|---|
   | `transactions.csv` | Subscription and billing history, v1 — full per-user transaction history, 2015-01-01 to 2017-02-28 |
   | `transactions_v2.csv` | Subscription and billing history, v2 — supplement extending coverage to 2017-03-31 |
   | `members_v3.csv` | User demographics (city, age, gender, registration date) |
   | `train_v2.csv` | Churn labels |

   Filenames must match exactly — `src/data_load.py` looks for these names in `data/`.

### Why v1 *and* v2 — this is easy to get wrong

The competition ships two transaction files, and using only one is a trap:

- **`transactions_v2.csv` alone is not enough.** It looks self-contained — same columns, a
  plausible row count — but it's concentrated almost entirely around the labeling window (75% of
  its rows are from March 2017 alone) and gives a median of **one transaction per user**. There's
  no per-user history in it to reconstruct renewal cycles from.
- **`transactions.csv` (v1) has the real history**: median 5 transactions/user, mean 9.1, volume
  spread evenly across every month from 2015-01 to 2017-02 — but it stops at 2017-02-28, two days
  short of the data window this analysis needs.
- **They don't overlap** — verified row-by-row: zero exact-duplicate rows between the two files
  — so `src/data_load.py` concatenates both to get the full 2015-01-01 to 2017-03-31 ledger with
  no double-counting. Using v2 in isolation (an easy mistake, since it's the smaller, more
  "recent-looking" file) silently breaks any multi-cycle retention analysis: over 80% of periods
  come out unresolvable, and the tiny sliver that does resolve is almost entirely cancellations,
  not a representative sample.

### Why not `user_logs_v2.csv`

The competition also offers `user_logs_v2.csv` — daily listening-event logs, **30GB+** on disk.
This repo does not use it. That's a deliberate scoping decision, not an omission: this is a
billing/retention/pricing analysis built on subscription and transaction records, not a
listening-behavior study, and listening events aren't needed to answer the business question.
Excluding it also keeps the other files (under 2GB combined) small enough that this analysis
runs on a laptop, which a 30GB dependency would rule out.

### Data boundaries

- Coverage is roughly **January 2015 to March 2017**.
- Training labels (`train_v2.csv`) cover subscriptions expiring **February 2017**.
- **Churn is defined as: no renewal within 30 days of membership expiry** — this repo's own
  period-reconstruction (`src/cohorts.py`) uses the same 30-day window, applied to every
  transaction in the data, not just the labeled snapshot.
- All prices are in **New Taiwan dollars (NT$)**.

This is Kaggle competition data. If you plan to redistribute anything derived from it, review
the [competition rules](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/rules) first.

### Key columns (`transactions.csv` / `transactions_v2.csv`, same schema)

`msno` (user id), `payment_method_id`, `payment_plan_days`, `plan_list_price`,
`actual_amount_paid`, `is_auto_renew`, `transaction_date`, `membership_expire_date`, `is_cancel`.

## Repo structure

```
notebooks/   01_data_prep, 02_retention_cohorts, 03_pricing_discounts,
             04_segmentation, 05_churn_model
src/         data_load.py, cohorts.py, pricing.py, plotting.py
data/        gitignored — see "Data" above for how to populate it
figures/     key charts, exported as PNG when the notebooks are run
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
# place the three CSVs in data/ — see "Data" above
jupyter notebook
```

Run the notebooks in order — each later notebook loads a parquet cache written by
`01_data_prep.ipynb`, so it has to run first.

## Data handling

- The `data/` directory is gitignored entirely except this directory's own `README.md`. Raw
  CSVs and the parquet cache built by notebook 01 are never committed.
- `kaggle.json` and any other credential file are gitignored explicitly — committing a Kaggle
  API token is a real credential leak, not a hypothetical one.

## Methodology

`transactions.csv` + `transactions_v2.csv` combined hold one row per billing transaction, and
most users have many rows across their tenure. `src/cohorts.py` turns that into an ordered
sequence of membership **periods** per user, and for each period asks: did the next transaction
start within 30 days of this one's expiry (renewed), was it flagged `is_cancel` (voluntary
cancellation), or did neither happen (silent lapse)? Periods too close to the end of the data
window to know the answer yet are marked **censored** and excluded from every rate calculation
downstream — without this, recent cohorts would look like they churn faster purely because
there hasn't been time to observe their renewals, not because they actually churn more.

### Three retention metrics — read this before trusting any number in notebook 02

An earlier draft of this repo conflated three different metrics under the word "retention" and
reported a "cycle 6 retention" of 94.3% vs. 97.7% that was actually none of them — a conditional
renewal rate computed over an already-survivor-selected population. That number is gone from
every notebook, chart, and this file. In its place, three explicitly distinct metrics:

- **Survival (the headline metric).** The fraction of a cohort's transacting population still
  CONTINUOUSLY subscribed through cycle N — every period from 1 to N-1 renewed, no lapse or
  cancellation anywhere in the chain. This is what "retention" means everywhere in this repo
  unless stated otherwise (`cohorts.cohort_survival_curve`). It correctly excludes users whose
  fate is still censored/pending rather than counting them as failures.
- **Reach (reactivation-inclusive, reported alongside survival, never as a substitute).** The
  fraction that EVER made it to an Nth transaction by any path, including a user who lapsed for
  months and came back. Always >= survival for the same population. The gap between the two
  (`reactivation_gap` in `cohorts.cohort_survival_and_reach`) is itself a finding: it quantifies
  how much of a cohort's apparent longevity is win-back rather than uninterrupted renewal.
- **Revenue retained/lost.** A different question again — total resolved-period revenue across a
  user's *entire* tenure (every cycle mixed), split by whether each bill renewed. It is not
  cycle-gated and does not reconcile numerically with the survival curve — notebook 02 stress-tests
  this explicitly rather than assuming the two agree.

The naive per-user "did they reach cycle N" check (a handful of atypical fast-cycling users
technically reaching cycle 6 within days) also needed a calendar-time eligibility rule instead of
a per-user reachability check — see `cohorts.cohort_calendar_eligible` — otherwise a cohort can
pass the eligibility bar via a handful of outliers while the rest of it hasn't had time yet,
producing a fake cliff to 0% in exactly the cohorts the eligibility filter was meant to protect.

### The central finding: acquisition mix, not cohort age, drives the retention trend

The headline (uncontrolled) survival comparison shows recent cohorts retaining WORSE than older
ones. That reverses once the population is split by whether a user's first transaction was
trial/promotional (short plan and/or a near-zero first payment): among non-trial, full-price
signups specifically, RECENT cohorts retain significantly BETTER. Recent cohorts simply include a
much larger share of trial signups, which convert to long-term subscribers at a very low rate —
blending that into the aggregate drags the headline number down even though real paying
subscribers are doing better than ever. See notebook 02's trial-mix section and notebook 04's
direct characterization of the 7-day plan / payment method 35 trial mechanism.

### Leakage guard in the predictive model (notebook 05)

The obvious way to build a feature table — join each user's most recent transaction to their
label — leaks. For a churned user, the most recent transaction is often the cancellation itself,
or a transaction whose `membership_expire_date` falls exactly in the labeled expiry window; both
are downstream of the outcome, not predictors of it. Notebook 05 demonstrates this concretely
(a naive `is_cancel`-from-latest-transaction feature alone separates churn at 78.7% vs. 4.4%)
before fixing it: a hard cutoff, **`CUTOFF_DATE = 2017-02-01`** (the start of the labeled expiry
month), restricts every feature to transactions strictly before it, and reports feature
importance both with and without the leaky features so the gap is visible rather than asserted.
`membership_expire_date` gets the same treatment — only pre-cutoff values are used as features.

Every claimed effect in the notebooks is either backed by a significance test (chi-square,
Mann-Whitney U, two-proportion z-test, or a logistic regression coefficient with a p-value) or
explicitly labeled descriptive-only. Several confounds get particular attention because they'd
otherwise flip the sign of the headline finding:

- **Discount targeting** (notebook 03): retention offers may be targeted at users already
  flagged as at-risk, which would make a naive discount-vs-retention correlation look negative
  even if discounting itself helps.
- **Discount depth conflates two mechanisms** (notebook 03): naive "discount depth" silently
  averages together `free_grant` (paid NT$0 on a real, priced plan — 65% of everything a naive
  calculation calls "discounted," renews at ~58%, and IS the at-risk retention-save mechanism)
  and `genuine_discount` (paid something, clustered tightly around one ~20%-off price point,
  renews at ~99%, and is *not* targeted at at-risk users — its share actually drops after a prior
  cancellation). These have opposite signals and were never on one dose-response curve; treating
  them as one "discount depth" scale produces the misleading naive chart. `src/pricing.py`'s
  `add_price_category` splits them explicitly.
- **`plan_list_price` / `payment_plan_days` collinearity** (notebook 04): correlated at r=0.96 —
  longer plans cost more, almost mechanically. `plan_list_price` is dropped from the regression
  entirely rather than kept in with a caveat; a caveat doesn't make an unstable, uninterpretable
  coefficient (an earlier draft got -11.6 per NT$1000) trustworthy. `payment_plan_days` is kept
  as the plan-length control.
- **7-day plans and payment method 35 are trial mechanisms, not underperforming paid products**
  (notebook 04): 98.7% of 7-day-plan periods and 99.9998% of payment-method-35 periods are paid
  NT$0. Their low "renewal rate" is a trial-to-paid conversion rate, a different business metric
  with different benchmarks — not evidence that a paid product is failing.
- **Auto-renew / plan length / payment method more broadly** (notebook 04): users who opt into
  these are self-selected, likely lower-churn-propensity populations to begin with. Associations
  are reported as descriptive segmentation, not as levers proven to work if pulled on a different
  user.

## Recommendation Memo

*Template — replace the bracketed items below with the actual output of notebooks 02–05, in
priority order, before publishing this repo as a finished piece.*

| # | Recommendation | Expected impact | Confidence |
|---|---|---|---|
| 1 | `[e.g. target retention offers at the segment identified in 04_segmentation.ipynb as carrying the largest revenue-at-risk]` | `[NT$ / % of at-risk revenue, from the segment table]` | `[High/Medium/Low — state why, e.g. "high: large sample, effect survives stratification"]` |
| 2 | `[finding from 03_pricing_discounts.ipynb — only include if the adjusted (confound-controlled) result supports it]` | `[...]` | `[...]` |
| 3 | `[finding from 02_retention_cohorts.ipynb — voluntary-cancel vs. silent-lapse mix, and which one to address]` | `[...]` | `[...]` |

Confidence should reflect what notebooks 02–04 actually established: a descriptive association
is not the same confidence level as a result that survived a stratified or confound-adjusted
check. Where a section produced a null or weak result, say so here too — a memo that
recommends against acting on a specific lever, because the data doesn't support it, is a more
credible piece of work than one that finds a story everywhere it looks.
