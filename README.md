# KKBox Retention & Pricing Analysis

A retention and pricing analysis on a real subscription business, built as a portfolio piece
for Business Analyst / Operations Analyst / Consulting applications.

**This is not a churn-prediction project.** Nearly every public repo on this dataset stacks
classifiers and reports AUC. This one asks a business question instead: *where is this
subscription business losing revenue, which levers actually retain subscribers, and what
should the company do about it?* A predictive model appears in [notebook 05](notebooks/05_churn_model.ipynb),
as one section near the end — not the centerpiece.

> **Status:** the analysis code in `src/` and `notebooks/` is complete and has been checked for
> syntax and logical consistency, but has not yet been executed end-to-end against the full
> dataset (that requires the ~GB-scale files described below, downloaded locally). The
> [Recommendation Memo](#recommendation-memo) section is a template — run notebooks `01`–`05` in
> order, then replace the bracketed placeholders with the actual output before treating this as
> a finished writeup.

## Data source

[WSDM - KKBox's Churn Prediction Challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge)
(Kaggle). KKBox is a music streaming subscription service in Southeast Asia.

This project uses **only**:

| File | Contents |
|---|---|
| `transactions_v2.csv` | Subscription and billing history — one row per transaction |
| `members_v3.csv` | User demographics (city, age, gender, registration date) |
| `train_v2.csv` | Churn labels |

`user_logs_v2.csv` (30GB+ of raw listening-behavior events) is **deliberately excluded** — it's
irrelevant to a billing/retention analysis, and including it would make this repo impossible to
run on a laptop.

### Data boundaries

- Coverage is roughly **January 2015 to March 2017**.
- Training labels (`train_v2.csv`) cover subscriptions expiring **February 2017**.
- **Churn is defined as: no renewal within 30 days of membership expiry** — this repo's own
  period-reconstruction (`src/cohorts.py`) uses the same 30-day window, applied to every
  transaction in the data, not just the labeled snapshot.
- All prices are in **New Taiwan dollars (NT$)**.

This is Kaggle competition data. If you plan to redistribute anything derived from it, review
the [competition rules](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/rules) first.

### Key columns (`transactions_v2.csv`)

`msno` (user id), `payment_method_id`, `payment_plan_days`, `plan_list_price`,
`actual_amount_paid`, `is_auto_renew`, `transaction_date`, `membership_expire_date`, `is_cancel`.

## Repo structure

```
notebooks/   01_data_prep, 02_retention_cohorts, 03_pricing_discounts,
             04_segmentation, 05_churn_model
src/         data_load.py, cohorts.py, pricing.py, plotting.py
data/        gitignored — see "Getting the data" below
figures/     key charts, exported as PNG when the notebooks are run
```

## Getting the data

The raw CSVs are never committed to this repo (see [Data handling](#data-handling)). To
reproduce the analysis:

1. **Create a Kaggle API token.** Go to [kaggle.com/settings](https://www.kaggle.com/settings) →
   "Create New Token". This downloads a `kaggle.json` file.
2. **Place the token** at `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` and `KAGGLE_KEY` as
   environment variables instead).
3. **Accept the competition rules.** Open
   [the competition's rules page](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/rules)
   while logged in and accept them. This step is easy to miss — without it, the Kaggle API
   returns a bare `403 Forbidden` with no explanation, even with valid credentials.
4. **Run the download script** from the repo root:

   ```bash
   python data/download_data.py
   ```

   It downloads and extracts only the three files this analysis needs, skips files that are
   already present, and prints the row count and date range of each file when done.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python data/download_data.py
jupyter notebook
```

Run the notebooks in order — each later notebook loads a parquet cache written by
`01_data_prep.ipynb`, so it has to run first.

## Data handling

- The `data/` directory is gitignored entirely except `download_data.py` and this
  directory's own `README.md`. Raw CSVs, extracted archives, and the parquet cache built by
  notebook 01 are never committed.
- `kaggle.json` and any other credential file are gitignored explicitly — committing a Kaggle
  API token is a real credential leak, not a hypothetical one.

## Methodology

`transactions_v2.csv` holds one row per billing transaction, and most users have many rows
across their tenure. `src/cohorts.py` turns that into an ordered sequence of membership
**periods** per user, and for each period asks: did the next transaction start within 30 days of
this one's expiry (renewed), was it flagged `is_cancel` (voluntary cancellation), or did neither
happen (silent lapse)? Periods too close to the end of the data window to know the answer yet
are marked **censored** and excluded from every rate calculation downstream — without this,
recent cohorts would look like they churn faster purely because there hasn't been time to
observe their renewals, not because they actually churn more.

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
