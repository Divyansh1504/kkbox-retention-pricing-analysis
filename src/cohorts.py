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


def cohort_survival_curve(
    periods: pd.DataFrame,
    members_with_cohort: pd.DataFrame,
    max_cycle: int = 12,
    group_col: str = "cohort_month",
) -> pd.DataFrame:
    """HEADLINE retention metric for this repo: true cumulative survival —
    the fraction of each group's transacting population still CONTINUOUSLY
    subscribed through cycle N. "Continuous" means periods 1..N-1 were ALL
    renewed, with no lapse or voluntary cancellation anywhere in the chain.
    A user who lapses and later reactivates does NOT count as a survivor
    here — see `cohort_survival_and_reach` for the more lenient,
    reactivation-inclusive "reach" number reported alongside this one, and
    the gap between the two (itself a reportable quantity: how much of
    "reach" is win-back rather than continuous renewal).

    `group_col` defaults to `cohort_month` (registration cohort) but can be
    any column present in `members_with_cohort` — e.g. a combined
    cohort-era x acquisition-channel label, to test whether a retention
    difference between two groups survives controlling for a third factor.

    Right-censoring is handled by EXCLUSION, not by counting a still-pending
    subscription as a failure. Every user's *last* period always has
    `renewed == False` by construction (renewed requires a next period to
    exist), so a user's first non-renewed period is either a confirmed
    failure (lapsed_no_renewal / voluntary_cancel) or, if it's their very
    last period and still inside its renewal window, `censored` —
    unresolved, not a failure. Users whose first failure is a censored
    period before cycle N are excluded from cycle N's denominator entirely,
    matching the censoring discipline used everywhere else in this module.
    """
    merged = periods.merge(
        members_with_cohort[["msno", group_col]], on="msno", how="inner"
    )
    not_renewed = merged.loc[
        ~merged["renewed"], ["msno", group_col, "period_seq", "outcome"]
    ]
    # every user has >=1 row here: their last period always has renewed == False
    first_failure = (
        not_renewed.sort_values("period_seq")
        .groupby("msno", as_index=False)
        .first()
        .rename(columns={"period_seq": "first_failure_seq"})
    )

    rows = []
    for cycle in range(1, max_cycle + 1):
        threshold = cycle - 1  # survived cycle N iff periods 0..N-2 (indices) all renewed
        survived_mask = first_failure["first_failure_seq"] >= threshold
        excluded_mask = (~survived_mask) & (first_failure["outcome"] == "censored")
        eval_pop = first_failure.loc[~excluded_mask].copy()
        eval_pop["survived"] = eval_pop["first_failure_seq"] >= threshold

        g = eval_pop.groupby(group_col)["survived"]
        cycle_summary = pd.DataFrame({"n_survived": g.sum(), "n_evaluated": g.size()})
        cycle_summary["cycle"] = cycle
        rows.append(cycle_summary.reset_index())

    out = pd.concat(rows, ignore_index=True)
    out["survival_rate"] = out["n_survived"] / out["n_evaluated"]
    return out


def cohort_survival_and_reach(
    periods: pd.DataFrame,
    members_with_cohort: pd.DataFrame,
    max_cycle: int = 12,
    group_col: str = "cohort_month",
) -> pd.DataFrame:
    """Survival (headline) and reach (reactivation-inclusive), computed on
    the SAME evaluated population at each cycle so the two are directly
    comparable.

    Survival = continuously renewed through cycle N, no lapse or
    cancellation anywhere in the chain (see `cohort_survival_curve` for the
    standalone version, used for the full-history heatmap). Reach = ever
    made it to an Nth transaction, by any path, including a user who lapsed
    for months and later reactivated — more lenient, reported alongside
    survival, never as a substitute.

    `group_col` defaults to `cohort_month` (registration cohort) but can be
    any column present in `members_with_cohort` — e.g. a combined
    cohort-era x acquisition-channel label, to test whether a retention
    difference between two groups survives controlling for a third factor.

    These two use different denominators when computed separately (survival
    excludes users whose fate is still censored/pending; reach doesn't need
    to, since "does a period at cycle N exist" has no ambiguity) —
    subtracting one rate from the other computed independently can come out
    negative, which looks like a contradiction but is really just a
    population mismatch. This function avoids that trap by computing both
    against one shared, correctly censoring-aware population per cycle, so:
    `reactivation_gap = reach_rate - survival_rate` is the share of the
    evaluated cohort that reached cycle N via at least one win-back
    reactivation along the way, rather than continuous renewal.
    """
    merged = periods.merge(
        members_with_cohort[["msno", group_col]], on="msno", how="inner"
    )
    user_info = merged.groupby("msno").agg(
        **{group_col: (group_col, "first")}, max_period_seq=("period_seq", "max")
    )

    not_renewed = merged.loc[~merged["renewed"], ["msno", "period_seq", "outcome"]]
    first_failure = (
        not_renewed.sort_values("period_seq")
        .groupby("msno")
        .first()
        .rename(columns={"period_seq": "first_failure_seq"})
    )
    user_info = user_info.join(first_failure[["first_failure_seq", "outcome"]])

    rows = []
    for cycle in range(1, max_cycle + 1):
        threshold = cycle - 1
        survived_mask = user_info["first_failure_seq"] >= threshold
        excluded_mask = (~survived_mask) & (user_info["outcome"] == "censored")
        eval_pop = user_info.loc[~excluded_mask].copy()
        eval_pop["survived"] = eval_pop["first_failure_seq"] >= threshold
        eval_pop["reached"] = eval_pop["max_period_seq"] >= threshold

        g = eval_pop.groupby(group_col)
        cycle_summary = pd.DataFrame(
            {
                "n_survived": g["survived"].sum(),
                "n_reached": g["reached"].sum(),
                "n_evaluated": g["survived"].size(),
            }
        )
        cycle_summary["cycle"] = cycle
        rows.append(cycle_summary.reset_index())

    out = pd.concat(rows, ignore_index=True)
    out["survival_rate"] = out["n_survived"] / out["n_evaluated"]
    out["reach_rate"] = out["n_reached"] / out["n_evaluated"]
    out["reactivation_gap"] = out["reach_rate"] - out["survival_rate"]
    return out


def first_transaction_trial_flag(
    periods: pd.DataFrame, plan_days_threshold: int = 14, price_threshold: float = 10.0
) -> pd.DataFrame:
    """Flags each user's FIRST transaction as trial/promotional if its
    `payment_plan_days` is short (<= plan_days_threshold) or its
    `actual_amount_paid` is near-zero (<= price_threshold NT$). One row per
    msno, for joining into any cohort-level split.

    These thresholds are a starting operational definition for testing
    whether acquisition mix (trial vs. full-price signups) explains a
    retention difference between cohorts — notebooks 03/04 characterize the
    short-plan and low-price populations in more depth and may refine this.
    """
    first_txn = (
        periods.sort_values(["msno", "period_seq"])
        .groupby("msno", as_index=False)
        .first()[["msno", "payment_plan_days", "actual_amount_paid", "plan_list_price"]]
    )
    first_txn["is_trial_first_txn"] = (first_txn["payment_plan_days"] <= plan_days_threshold) | (
        first_txn["actual_amount_paid"] <= price_threshold
    )
    return first_txn


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
