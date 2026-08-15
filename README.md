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
explicitly labeled descriptive-only. Two confounds get particular attention because they'd
otherwise flip the sign of the headline finding:

- **Discounting** (notebook 03): discounts may be targeted at users already flagged as at-risk
  (retention offers), which would make a naive discount-vs-retention correlation look negative
  even if discounting itself helps. The notebook tests for this targeting directly rather than
  reporting the naive correlation as an effect.
- **Auto-renew / plan length / payment method** (notebook 04): users who opt into these are
  self-selected, likely lower-churn-propensity populations to begin with. Associations are
  reported as descriptive segmentation, not as levers proven to work if pulled on a different
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
