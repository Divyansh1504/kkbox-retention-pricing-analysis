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


def load_transactions(path: Path | str = DATA_DIR / "transactions_v2.csv") -> pd.DataFrame:
    df = pd.read_csv(path, dtype=TRANSACTIONS_DTYPES)
    df["transaction_date"] = _parse_yyyymmdd(df["transaction_date"])
    df["membership_expire_date"] = _parse_yyyymmdd(df["membership_expire_date"])
    return df


def load_members(path: Path | str = DATA_DIR / "members_v3.csv") -> pd.DataFrame:
    df = pd.read_csv(path, dtype=MEMBERS_DTYPES)
    df["registration_init_time"] = _parse_yyyymmdd(df["registration_init_time"])
    return df


def load_train(path: Path | str = DATA_DIR / "train_v2.csv") -> pd.DataFrame:
    return pd.read_csv(path, dtype=TRAIN_DTYPES)
