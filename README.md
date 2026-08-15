# KKBox Retention & Pricing Analysis

A retention and pricing analysis on a real subscription business, built as a portfolio piece
for Business Analyst / Operations Analyst / Consulting applications.

**This is not a churn-prediction project.** Nearly every public repo on this dataset stacks
classifiers and reports AUC. This one asks a business question instead: *where is this
subscription business losing revenue, which levers actually retain subscribers, and what
should the company do about it?* A predictive model appears in [notebook 05](notebooks/05_churn_model.ipynb),
as one section near the end — not the centerpiece.

> **Status:** all five notebooks have been executed end-to-end against the full dataset. The
> [Recommendation Memo](#recommendation-memo) below reflects the actual output, not a template.

## Key findings

The two strongest results in this repo — read these first.

### 1. Acquisition mix reverses the retention trend (Simpson's paradox)

The uncontrolled, cohort-age comparison says recent cohorts retain **worse**: cumulative
survival to the 6th subscription cycle is 48.7% for cohorts registered in 2014 vs. 44.4% for
cohorts registered in 2016 (z=23.68, p=5.2e-124, on 118,918 and 218,319 evaluated users). Taken
at face value, that's a "the product is getting worse at keeping people" story.

**It reverses once acquisition mix is controlled for.** Splitting each era by whether the user's
first transaction was trial/promotional (a short plan and/or a near-zero first payment) shows:

| First transaction | 2014 cohorts | 2016 cohorts | z | p |
|---|---|---|---|---|
| Trial | 6.20% | 2.98% | 26.81 | 2.2e-158 |
| **Non-trial (paying)** | 67.34% | **74.24%** | -34.21 | 1.7e-256 |

Among real, paying subscribers, **recent cohorts retain significantly better**. The aggregate
decline is a composition effect: recent cohorts simply include a much larger share of trial
signups, a population that converts to long-term subscribers at a low rate almost by
definition, and blending that into the aggregate drags the headline number down even though the
paying-subscriber product is doing better than ever.

This was stress-tested, not just asserted: a 2016 registrant on a long plan can't resolve 6
cycles before the March 2017 data cutoff, so the late-era non-trial sample eligible for
comparison skews toward monthly subscribers relative to the 2014 sample. Restricting **both**
eras to exactly the same 30-day plan (removing that skew entirely rather than just noting it)
gives 67.7% -> 75.6% — the finding survives strict matching and gets slightly stronger, not
weaker. See `notebooks/02_retention_cohorts.ipynb` for the full derivation, including the
reversal, the matched re-test, and a separate finding that trial-to-paid conversion itself is
declining (6.20% -> 2.98%, a 52% relative drop) — a distinct problem from "there are more
trials now," addressed separately in the memo below.

### 2. The obvious way to build the churn model leaks — demonstrated, then fixed

Joining each user's most recent transaction to their churn label is the standard approach for
this dataset, and it's wrong: for a churned user, that transaction is often the cancellation
itself. `notebooks/05_churn_model.ipynb` demonstrates this concretely before fixing it —
`is_cancel` on the latest-ever transaction alone separates churn at 84.0% vs. 6.6%, and a model
built on that kind of feature reaches **AUC 0.981**. A model built with a hard time cutoff
(features strictly before the label window, no post-hoc transaction allowed in) drops to **AUC
0.875** — still a strong, legitimate result using only plan/pricing/demographic data, but the
0.106 AUC gap is the leaked model's "skill" evaporating once the label stops leaking through the
features. Both models are reported side by side with their feature-importance rankings so the
gap is visible, not just asserted.

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

See [Key findings](#key-findings) above for the acquisition-mix reversal (Simpson's paradox) and
the leakage demonstration — the two most load-bearing results in this repo, including the
plan-length-matched robustness check and the separate trial-conversion-decline finding.

### Leakage guard in the predictive model (notebook 05)

The obvious way to build a feature table — join each user's most recent transaction to their
label — leaks. For a churned user, the most recent transaction is often the cancellation itself,
or a transaction whose `membership_expire_date` falls exactly in the labeled expiry window; both
are downstream of the outcome, not predictors of it. Notebook 05 demonstrates this concretely
(a naive `is_cancel`-from-latest-transaction feature alone separates churn at 84.0% vs. 6.6%)
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
  `add_price_category` splits them explicitly. Checked and rejected: `genuine_discount` is not an
  annual-plan commitment effect in disguise (a pay-12-get-14 structure would mechanically inflate
  a per-decision renewal rate) — 99.998% of `genuine_discount` periods are on the same 30-day plan
  as the comparison groups, so the ~99% renewal is a same-plan-length price effect, not a
  plan-length artifact.
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

**To leadership — what this analysis found and what to do about it.**

### Finding 1: Paying-subscriber retention is improving, not declining

Non-trial subscriber survival to the 6th billing cycle rose from **67.3% (2014 cohorts) to
74.2% (2016 cohorts)** — confirmed under strict plan-length matching (67.7% → 75.6%), so this
isn't a censoring artifact. The naive, unsegmented number says the opposite (48.7% → 44.4%)
purely because recent cohorts include more trial signups; see Finding 2.

**Confidence: high** that the reversal is real — huge samples, p < 1e-120, survives matching.
**Directional only** on *why* paying retention improved; no causal driver is in this data.

**Recommend:** don't let the aggregate number drive a "retention is declining" narrative or
budget cut — the paying product is healthier than the headline suggests. Lost revenue in this
population is **83% silent lapse vs. 17% voluntary cancellation**, and auto-renew subscribers
retain at 95.6% vs. 68.7% manual-renew — so prioritize renewal reminders, grace periods, and
auto-renew opt-in campaigns over reactive save-offer campaigns. **Expected impact: largest —**
manual-renew, ≤31-day-plan subscribers carry NT$107M in lost revenue in the sampled cohorts, the
natural first target.

### Finding 2: Trial-to-paid conversion has collapsed, independent of trial volume

Trial-first-transaction survival to cycle 6 fell from **6.20% to 2.98%** — a **52% relative
decline** (z=26.81, p=2.2e-158) — while trial share of the transacting population grew from
28.5% to 38.6% over the same period. These are two different problems.

**Confidence: high** that the conversion *rate* fell, not just trial count. **Directional only**
on cause — this data can't separate an acquisition-channel/targeting shift from a trial-quality
change.

**Recommend:** audit the trial funnel and its traffic sources directly; don't scale trial volume
further without first understanding why conversion halved. **Expected impact: second** — a large
and growing segment, so even a partial conversion recovery compounds.

### Finding 3: Genuine discounting looks like a real, low-risk lever; free "goodwill" grants don't

A genuine ~20%-off discount correlates with **98.9% renewal vs. 89.4% at full price** (adjusted
odds ratio 7.87 vs. full price, controlling for plan length, auto-renew, and prior cancellation).
Free/comp'd grants — paid NT$0, the other 65% of what a naive "discount" scale includes — renew
at only **58%**.

**Confidence: medium-high** for the association — at-risk targeting is ruled out for genuine
discounts (their share *drops* after a prior cancellation) and so is an annual-plan artifact
(99.998% on the same 30-day plan as the comparison group). **Not causal:** who receives or takes
a genuine discount still isn't random, so this is the best-adjusted observational estimate this
data supports, not a proven price effect.

**Recommend:** pilot a randomized ~20%-off offer on a held-out segment before committing pricing
strategy to it. Don't expand free/comp'd grants as a retention tool — they're already a reactive,
at-risk-targeted mechanism, and they save only 58% of the accounts they're given to while
forfeiting 100% of that revenue on the rest. **Expected impact: third, but directly testable** —
unlike Findings 1-2, this one can be validated causally with a controlled experiment before any
wider rollout.
