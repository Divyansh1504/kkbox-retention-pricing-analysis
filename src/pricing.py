"""Discounting and pricing analysis.

The central question: does discounting buy retention, or just give up margin?

Two separate confounds get tested here, not one:

1. **Targeting.** Retention offers are disproportionately given to users the
   business already suspects are about to leave. If that's happening, deep
   discounts will correlate with LOWER apparent retention even if the
   discount itself has a small positive effect, because the discounted
   population was riskier to begin with.
2. **Discount depth is not one thing.** Naive "discount depth" (list price
   minus paid amount, as a percentage) silently conflates two very different
   transaction types: a GENUINE partial discount (the user paid something,
   just less than list) and a FREE / comp'd grant (the user paid nothing at
   all on an otherwise real, priced plan). In this data these turn out to
   have very different renewal signatures — see `add_price_category` — so
   treating "discount depth" as one continuous, dose-response scale silently
   averages together two different business mechanisms with opposite
   signals.

Every function below that reports a discount-retention relationship is
paired with a check for one or both of these, and results are labeled
`descriptive` vs `adjusted` throughout — never asserted as causal outright.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DISCOUNT_BINS = [-np.inf, 0.001, 0.10, 0.25, 0.50, np.inf]
DISCOUNT_LABELS = ["0% (list price)", "0-10%", "10-25%", "25-50%", "50%+"]


def add_discount_bucket(periods: pd.DataFrame) -> pd.DataFrame:
    out = periods.copy()
    out["discount_bucket"] = pd.cut(
        out["discount_depth"].clip(lower=0), bins=DISCOUNT_BINS, labels=DISCOUNT_LABELS
    )
    return out


def add_price_category(periods: pd.DataFrame) -> pd.DataFrame:
    """Splits priced periods (plan_list_price > 0) into three clean,
    non-overlapping categories that naive `discount_depth` bucketing
    conflates:

    - `full_price`: paid at (or within 0.1% of) list price.
    - `genuine_discount`: paid something, but less than list price. In this
      dataset these cluster tightly around ~20% off — a fixed promotional
      price point, not a continuum — and renew at ~99%.
    - `free_grant`: paid exactly NT$0 on an otherwise real, priced plan. Not
      a discount in any normal pricing sense — this is 65% of what naive
      `discount_depth > 0` bucketing would call "discounted," and it renews
      at ~58%, dragging down every high-discount-depth bucket in a way that
      has nothing to do with genuine discounting.
    """
    out = periods[periods["plan_list_price"] > 0].copy()
    out["price_category"] = np.select(
        [out["discount_depth"] <= 0.001, out["actual_amount_paid"] == 0],
        ["full_price", "free_grant"],
        default="genuine_discount",
    )
    return out


def add_prior_cancel_flag(periods: pd.DataFrame) -> pd.DataFrame:
    """Flag periods immediately following a voluntary cancellation on the
    user's previous transaction — a proxy for 'this user had already shown a
    churn signal before this discount was applied'."""
    out = periods.sort_values(["msno", "period_seq"]).copy()
    out["prior_is_cancel"] = out.groupby("msno", observed=True)["is_cancel"].shift(1).fillna(0)
    return out


def renewal_rate_by_discount_bucket(periods: pd.DataFrame) -> pd.DataFrame:
    """Naive, unadjusted renewal rate by discount depth bucket, plus a
    chi-square test of independence. This is DESCRIPTIVE ONLY — see module
    docstring on the retention-offer confound before treating the direction
    of this relationship as an effect of discounting."""
    df = add_discount_bucket(periods)
    table = pd.crosstab(df["discount_bucket"], df["renewed"])
    chi2, p, dof, _ = stats.chi2_contingency(table)

    summary = (
        df.groupby("discount_bucket", observed=True)
        .agg(n_periods=("renewed", "size"), renewal_rate=("renewed", "mean"))
        .reset_index()
    )
    return summary, {"chi2": chi2, "p_value": p, "dof": dof}


def renewal_rate_by_discount_bucket_stratified(
    periods: pd.DataFrame, min_group_size: int = 200
) -> pd.DataFrame:
    """Same comparison as `renewal_rate_by_discount_bucket`, but computed
    within (payment_plan_days, plan_list_price) strata — i.e. holding the
    plan itself fixed and only varying what the user actually paid on it.
    This removes confounding from users self-selecting into different plan
    types, but does NOT remove confounding from within-plan discount
    targeting (see `discount_depth_by_prior_cancel`)."""
    df = add_discount_bucket(periods)
    grouped = df.groupby(
        ["payment_plan_days", "plan_list_price", "discount_bucket"], observed=True
    ).agg(n_periods=("renewed", "size"), renewal_rate=("renewed", "mean"))
    grouped = grouped.reset_index()

    strata_size = df.groupby(
        ["payment_plan_days", "plan_list_price"], observed=True
    ).size()
    valid_strata = strata_size[strata_size >= min_group_size].index
    grouped = grouped.set_index(["payment_plan_days", "plan_list_price"])
    grouped = grouped.loc[grouped.index.isin(valid_strata)].reset_index()
    return grouped


def discount_depth_by_prior_cancel(periods: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Tests whether discount depth is systematically higher for periods that
    follow a voluntary cancellation on the user's prior transaction — direct
    evidence for (or against) the retention-offer targeting confound.

    Uses Mann-Whitney U rather than a t-test: discount depth is zero-inflated
    (most transactions are at list price) and not normally distributed.
    """
    df = add_prior_cancel_flag(periods)
    with_prior_cancel = df.loc[df["prior_is_cancel"] == 1, "discount_depth"].dropna()
    without_prior_cancel = df.loc[df["prior_is_cancel"] == 0, "discount_depth"].dropna()

    u_stat, p_value = stats.mannwhitneyu(
        with_prior_cancel, without_prior_cancel, alternative="two-sided"
    )

    summary = pd.DataFrame(
        {
            "group": ["prior_voluntary_cancel", "no_prior_cancel"],
            "n": [len(with_prior_cancel), len(without_prior_cancel)],
            "median_discount_depth": [
                with_prior_cancel.median(),
                without_prior_cancel.median(),
            ],
            "mean_discount_depth": [
                with_prior_cancel.mean(),
                without_prior_cancel.mean(),
            ],
        }
    )
    return summary, {"u_statistic": u_stat, "p_value": p_value}


def fit_retention_logit(periods: pd.DataFrame):
    """Logistic regression of renewal on discount depth, controlling for plan
    length, list price, auto-renew status, and the prior-cancel proxy.

    This partially adjusts for the confounds above but does not establish
    causality: discount depth is still not randomly assigned, and unobserved
    factors driving both "was offered a discount" and "was going to churn
    anyway" (e.g. a support-flagged at-risk status we don't have) can remain.
    Report the coefficient with that caveat attached, not as an isolated
    number.
    """
    import statsmodels.api as sm

    df = add_prior_cancel_flag(periods).dropna(subset=["discount_depth"]).copy()
    X = pd.DataFrame(
        {
            "discount_depth": df["discount_depth"],
            "payment_plan_days": df["payment_plan_days"],
            "plan_list_price": df["plan_list_price"] / 1000.0,  # rescale for solver stability
            "is_auto_renew": df["is_auto_renew"],
            "prior_is_cancel": df["prior_is_cancel"],
        }
    )
    X = sm.add_constant(X)
    y = df["renewed"].astype(int)

    model = sm.Logit(y, X).fit(disp=False)
    return model
