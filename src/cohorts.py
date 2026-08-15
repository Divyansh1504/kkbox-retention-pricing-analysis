"""Cohort construction, membership-period reconstruction, and retention curves.

transactions_v2.csv holds one row per billing transaction, and most users have
many rows across their tenure (mean ~10 transactions/user). We turn that into
an ordered sequence of membership "periods" per user, then ask, for each
period, whether the *next* period started within 30 days of this one's expiry
(renewed) or not (lapsed) — matching the competition's own churn definition.

A period's outcome can only be known if there was enough calendar time left in
the data window to observe a renewal. Periods too close to the end of the
observation window are marked `censored` and must be excluded from
renewal/lapse rates — including them would bias recent cohorts toward
looking like they churn more, purely as an artifact of not having caught up
with the data cutoff yet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RENEWAL_WINDOW_DAYS = 30


def build_membership_periods(
    transactions: pd.DataFrame, observation_end: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Collapse raw transaction rows into one ordered sequence of periods per user.

    Same-day multiple transactions (e.g. a cancellation followed by a new
    signup) are kept as separate consecutive periods rather than merged —
    merging them would require guessing intent that isn't in the data.

    Parameters
    ----------
    observation_end : the last date the data can be trusted to have captured
        a renewal, if it happened. Defaults to the max transaction_date in
        the input, which is the standard right-censoring boundary for data
        collected up to a fixed date.
    """
    df = (
        transactions.drop_duplicates()
        .sort_values(["msno", "transaction_date", "membership_expire_date"])
        .reset_index(drop=True)
    )

    observation_end = observation_end or df["transaction_date"].max()

    grouped = df.groupby("msno", observed=True)
    df["period_seq"] = grouped.cumcount()
    df["n_periods"] = grouped["period_seq"].transform("size")
    df["is_last_period"] = df["period_seq"] == (df["n_periods"] - 1)
    df["next_transaction_date"] = grouped["transaction_date"].shift(-1)

    gap_days = (df["next_transaction_date"] - df["membership_expire_date"]).dt.days
    df["gap_to_next_days"] = gap_days

    df["censored"] = df["is_last_period"] & (
        (df["membership_expire_date"] + pd.Timedelta(days=RENEWAL_WINDOW_DAYS))
        > observation_end
    )
    df["renewed"] = (~df["is_last_period"]) & (gap_days <= RENEWAL_WINDOW_DAYS)
    df["lapsed"] = (~df["censored"]) & (~df["renewed"])

    df["discount_amount"] = df["plan_list_price"] - df["actual_amount_paid"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["discount_depth"] = np.where(
            df["plan_list_price"] > 0,
            df["discount_amount"] / df["plan_list_price"],
            np.nan,
        )

    outcome = np.select(
        [df["censored"], df["renewed"], df["is_cancel"] == 1],
        ["censored", "renewed", "voluntary_cancel"],
        default="lapsed_no_renewal",
    )
    df["outcome"] = outcome

    return df


def assign_cohort(members: pd.DataFrame) -> pd.DataFrame:
    """Tag each member with their registration cohort month.

    Stored as a month-start Timestamp rather than a pandas Period — Period
    columns don't round-trip cleanly through parquet, and a Timestamp sorts
    and displays just as well for a monthly cohort label.
    """
    out = members.copy()
    out["cohort_month"] = out["registration_init_time"].dt.to_period("M").dt.to_timestamp()
    return out


def cohort_sizes(members_with_cohort: pd.DataFrame) -> pd.Series:
    """Registered members per cohort month — includes members who never
    subscribed at all. NOT the right denominator for a retention rate; use
    `_transacting_cohort_sizes` for that. Kept for callers that genuinely want
    registration counts (e.g. reporting overall registered-population growth).
    """
    return members_with_cohort.groupby("cohort_month")["msno"].nunique()


def _transacting_cohort_sizes(
    periods: pd.DataFrame, members_with_cohort: pd.DataFrame
) -> pd.Series:
    """Members per cohort month who ever appear in `periods` at least once —
    i.e. ever had a subscription transaction. Most registered members in
    `members_v3.csv` never subscribed at all (registration doesn't require
    payment), so `cohort_sizes` massively overstates the population a
    retention rate should be measured against."""
    merged = periods.merge(
        members_with_cohort[["msno", "cohort_month"]], on="msno", how="inner"
    )
    return merged.groupby("cohort_month")["msno"].nunique()


def cohort_retention_curve(
    periods: pd.DataFrame, members_with_cohort: pd.DataFrame, max_cycle: int = 12
) -> pd.DataFrame:
    """Fraction of each registration cohort's SUBSCRIBERS (not all registrants)
    still renewing at each subscription cycle.

    The x-axis is subscription CYCLE number (1st period, 2nd period, ...), not
    calendar month: payment_plan_days varies across users (30 / 90 / 365-day
    plans), so a fixed cycle count doesn't correspond to a fixed calendar
    tenure across everyone. Reaching cycle k only requires a row existing at
    that period_seq, so this uses the full periods table (censoring doesn't
    apply here — it's a fact about the data, not an inferred outcome).

    The denominator is each cohort's TRANSACTING population (members who ever
    had at least one subscription transaction), not its full registered
    population — most registered members in this dataset never subscribed at
    all, and dividing by them would conflate registration-to-subscription
    conversion with subscription retention, two different business questions.
    At cycle 1 this makes every cohort start at 100% by construction, which is
    correct: everyone counted in the denominator transacted at least once.

    Cohorts registered close to the data cutoff will show artificially low
    retention at higher cycle numbers simply because they haven't had time to
    reach them yet — filter to a common comparison horizon before charting.
    """
    merged = periods.merge(
        members_with_cohort[["msno", "cohort_month"]], on="msno", how="inner"
    )
    merged["cycle"] = merged["period_seq"] + 1

    reached = (
        merged[merged["cycle"] <= max_cycle]
        .groupby(["cohort_month", "cycle"])["msno"]
        .nunique()
        .rename("n_reached")
        .reset_index()
    )
    sizes = _transacting_cohort_sizes(periods, members_with_cohort)
    reached["cohort_size"] = reached["cohort_month"].map(sizes)
    reached["retention_rate"] = reached["n_reached"] / reached["cohort_size"]
    return reached


def cohort_revenue_summary(
    periods: pd.DataFrame, members_with_cohort: pd.DataFrame
) -> pd.DataFrame:
    """Revenue collected per cohort, split by whether that period renewed,
    lapsed without renewal, or ended in a voluntary cancellation.

    Censored periods (outcome unknown) are excluded — their revenue isn't
    lost or retained, it's simply not yet resolved.
    """
    merged = periods.merge(
        members_with_cohort[["msno", "cohort_month"]], on="msno", how="inner"
    )
    resolved = merged[merged["outcome"] != "censored"]
    summary = (
        resolved.groupby(["cohort_month", "outcome"])["actual_amount_paid"]
        .sum()
        .unstack(fill_value=0)
    )
    return summary


def cohort_calendar_eligible(
    members_with_cohort: pd.DataFrame,
    cycle: int,
    observation_end: pd.Timestamp,
    cycle_length_days: int = 30,
) -> pd.Index:
    """Cohort months old enough, in calendar time, to plausibly have reached
    `cycle` renewal cycles by `observation_end` — using the dominant 30-day
    plan length as the reference cycle length (the large majority of
    transactions are on 30-day plans; see notebook 01).

    This is a stricter, calendar-time-based alternative to
    `cohort_max_reachable_cycle`: asking whether ANY single user in a cohort
    reached a given cycle lets a handful of atypical fast-cycling users
    (short plans, rapid repeat transactions) pass the bar long before the
    cohort as a whole has had time to — which then biases that cohort's
    reported retention at that cycle toward only its most unusual members,
    not a representative slice. Use this for any comparison across cohorts of
    different ages; `cohort_max_reachable_cycle` remains useful as a
    per-user, data-driven diagnostic on its own.
    """
    sizes = cohort_sizes(members_with_cohort)
    min_age = pd.DateOffset(days=cycle * cycle_length_days)
    return sizes.index[sizes.index + min_age <= observation_end]


def cohort_max_reachable_cycle(
    periods: pd.DataFrame, members_with_cohort: pd.DataFrame
) -> pd.Series:
    """For each cohort, the highest subscription cycle that has at least one
    *resolved* (non-censored) outcome. Use this to decide which cohorts have
    had enough elapsed time to be fairly included in a retention-curve
    comparison — younger cohorts simply haven't lived long enough to reach
    higher cycle numbers, and including them makes recent cohorts look like
    they churn faster than they actually do."""
    merged = periods.merge(
        members_with_cohort[["msno", "cohort_month"]], on="msno", how="inner"
    )
    merged["cycle"] = merged["period_seq"] + 1
    resolved = merged[merged["outcome"] != "censored"]
    return resolved.groupby("cohort_month")["cycle"].max()


def cohort_outcome_counts(
    periods: pd.DataFrame, members_with_cohort: pd.DataFrame
) -> pd.DataFrame:
    """Period counts per cohort split by outcome (renewed / voluntary_cancel /
    lapsed_no_renewal). Same shape as `cohort_revenue_summary` but counting
    periods instead of summing revenue — use this to see whether voluntary
    cancellation and silent lapse are shifting in relative frequency across
    cohorts, independent of how much revenue each period carried."""
    merged = periods.merge(
        members_with_cohort[["msno", "cohort_month"]], on="msno", how="inner"
    )
    resolved = merged[merged["outcome"] != "censored"]
    return (
        resolved.groupby(["cohort_month", "outcome"]).size().unstack(fill_value=0)
    )


def period_level_outcomes(
    periods: pd.DataFrame, members_with_cohort: pd.DataFrame
) -> pd.DataFrame:
    """Non-censored periods joined to cohort — the base table most downstream
    retention/pricing/segmentation analysis should start from."""
    merged = periods.merge(
        members_with_cohort[["msno", "cohort_month", "city", "bd", "gender", "registered_via"]],
        on="msno",
        how="inner",
    )
    return merged[merged["outcome"] != "censored"].copy()
