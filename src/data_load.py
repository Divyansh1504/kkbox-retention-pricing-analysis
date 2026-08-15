"""Load the three raw KKBox CSVs with correct dtypes and parsed dates.

Kept deliberately dumb: this module reads files as-is and fixes dtypes/date
parsing only. Filtering, dedup, and other analytical decisions belong in
notebooks/01_data_prep.ipynb where they can be documented and inspected,
not hidden in a library function.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

TRANSACTIONS_DTYPES = {
    "msno": "category",
    "payment_method_id": "int16",
    "payment_plan_days": "int16",
    "plan_list_price": "int32",
    "actual_amount_paid": "int32",
    "is_auto_renew": "int8",
    "is_cancel": "int8",
}

MEMBERS_DTYPES = {
    "msno": "category",
    "city": "int16",
    "bd": "int16",
    "gender": "category",
    "registered_via": "int16",
}

TRAIN_DTYPES = {
    "msno": "category",
    "is_churn": "int8",
}


def _parse_yyyymmdd(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y%m%d")


def _read_transactions_file(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=TRANSACTIONS_DTYPES)
    df["transaction_date"] = _parse_yyyymmdd(df["transaction_date"])
    df["membership_expire_date"] = _parse_yyyymmdd(df["membership_expire_date"])
    return df


def load_transactions(
    v1_path: Path | str = DATA_DIR / "transactions.csv",
    v2_path: Path | str = DATA_DIR / "transactions_v2.csv",
) -> pd.DataFrame:
    """Full transaction history: `transactions.csv` (v1, 2015-01-01 -> 2017-02-28,
    ~9 transactions/user on average) concatenated with `transactions_v2.csv`
    (v2, the supplement extending coverage to 2017-03-31).

    v2 is NOT a resend of v1 — verified zero exact-duplicate rows between the
    two files across their full overlap — so this is a safe concatenation, not
    a merge that needs conflict resolution. `.drop_duplicates()` still runs to
    catch the small number of duplicate rows that exist *within* each file on
    its own (~3.3K), unrelated to the v1/v2 combination itself.

    v2 alone covers only the Feb-Mar 2017 label window (median 1 transaction/user)
    and is not enough on its own to reconstruct per-user subscription history —
    see the README's Data section for why both files are required.
    """
    v1 = _read_transactions_file(v1_path)
    v2 = _read_transactions_file(v2_path)
    combined = pd.concat([v1, v2], ignore_index=True).drop_duplicates()
    return combined.sort_values(["msno", "transaction_date", "membership_expire_date"]).reset_index(
        drop=True
    )


def load_members(path: Path | str = DATA_DIR / "members_v3.csv") -> pd.DataFrame:
    df = pd.read_csv(path, dtype=MEMBERS_DTYPES)
    df["registration_init_time"] = _parse_yyyymmdd(df["registration_init_time"])
    return df


def load_train(path: Path | str = DATA_DIR / "train_v2.csv") -> pd.DataFrame:
    return pd.read_csv(path, dtype=TRAIN_DTYPES)
