"""
Download the three KKBox files this analysis needs from the Kaggle
"WSDM - KKBox's Churn Prediction Challenge" competition, and extract them.

Deliberately excludes user_logs_v2.csv (30GB+ of listening-behavior data
that this analysis does not use — see README.md for why).

Usage:
    python data/download_data.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

COMPETITION = "kkbox-churn-prediction-challenge"
DATA_DIR = Path(__file__).resolve().parent

# csv filename -> archive filename on Kaggle
FILES = {
    "transactions_v2.csv": "transactions_v2.csv.7z",
    "members_v3.csv": "members_v3.csv.7z",
    "train_v2.csv": "train_v2.csv.7z",
}

# columns to scan for a date range per file (stored as YYYYMMDD ints in this dataset)
DATE_COLUMNS = {
    "transactions_v2.csv": ["transaction_date", "membership_expire_date"],
    "members_v3.csv": ["registration_init_time"],
    "train_v2.csv": [],
}


def check_credentials() -> None:
    """Fail fast, before importing kaggle, with an actionable message.

    `import kaggle` authenticates at import time and raises an opaque
    IOError if it can't find credentials, so we check first ourselves.
    """
    has_env = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    has_file = (config_dir / "kaggle.json").exists()

    if has_env or has_file:
        return

    sys.exit(
        "\nNo Kaggle API credentials found.\n\n"
        "This script needs a Kaggle API token to download competition data:\n"
        "  1. Go to https://www.kaggle.com/settings -> 'Create New Token'.\n"
        "     This downloads kaggle.json.\n"
        f"  2. Place it at {config_dir / 'kaggle.json'}\n"
        "     (or set KAGGLE_USERNAME and KAGGLE_KEY as environment variables instead).\n"
        "  3. IMPORTANT: also open\n"
        f"     https://www.kaggle.com/competitions/{COMPETITION}/rules\n"
        "     and accept the competition rules while logged in. Kaggle's API returns a\n"
        "     bare 403 Forbidden, with no useful message, if you have valid credentials\n"
        "     but haven't accepted the rules yet — this step is easy to miss.\n"
    )


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    import py7zr

    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=dest_dir)


def download_file(api, filename: str, archive_name: str) -> None:
    target_csv = DATA_DIR / filename
    if target_csv.exists():
        print(f"  {filename} already present, skipping.")
        return

    archive_path = DATA_DIR / archive_name
    if not archive_path.exists():
        print(f"  Downloading {archive_name} ...")
        try:
            api.competition_download_file(
                COMPETITION, archive_name, path=str(DATA_DIR), force=False, quiet=False
            )
        except Exception as exc:
            message = str(exc)
            if "403" in message:
                sys.exit(
                    f"\nKaggle API returned 403 Forbidden while downloading {archive_name}.\n"
                    "This almost always means you have not accepted the competition rules yet.\n"
                    f"Go to https://www.kaggle.com/competitions/{COMPETITION}/rules, log in,\n"
                    "and accept them, then re-run this script.\n"
                )
            if "404" in message:
                sys.exit(
                    f"\nKaggle API returned 404 Not Found for '{archive_name}' in competition "
                    f"'{COMPETITION}'.\n"
                    "The file name or competition slug may have changed since this script was "
                    "written — check\n"
                    f"https://www.kaggle.com/competitions/{COMPETITION}/data and update FILES / "
                    "COMPETITION above rather than guessing at alternatives.\n"
                )
            raise

    if not archive_path.exists():
        # some Kaggle datasets serve the plain .csv directly instead of a .7z archive
        return

    print(f"  Extracting {archive_name} ...")
    extract_archive(archive_path, DATA_DIR)
    archive_path.unlink()


def summarize_csv(path: Path, date_columns: list[str]) -> None:
    import pandas as pd

    if not path.exists():
        print(f"  {path.name}: not found, skipping summary.")
        return

    row_count = 0
    mins: dict[str, object] = {c: None for c in date_columns}
    maxs: dict[str, object] = {c: None for c in date_columns}
    churn_sum = 0
    has_churn_col = path.name == "train_v2.csv"

    usecols = (date_columns + (["is_churn"] if has_churn_col else [])) or None
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=1_000_000):
        row_count += len(chunk)
        for c in date_columns:
            col_min, col_max = chunk[c].min(), chunk[c].max()
            mins[c] = col_min if mins[c] is None else min(mins[c], col_min)
            maxs[c] = col_max if maxs[c] is None else max(maxs[c], col_max)
        if has_churn_col:
            churn_sum += chunk["is_churn"].sum()

    print(f"  {path.name}: {row_count:,} rows")
    for c in date_columns:
        print(f"    {c} range: {mins[c]} -> {maxs[c]}  (YYYYMMDD)")
    if has_churn_col and row_count:
        print(f"    is_churn positive rate: {churn_sum / row_count:.4f}")


def main() -> None:
    check_credentials()

    import kaggle  # safe now that credentials are confirmed to exist

    api = kaggle.api
    api.authenticate()

    print(f"Downloading data for competition '{COMPETITION}'\n")
    for filename, archive_name in FILES.items():
        download_file(api, filename, archive_name)

    print("\nSummary:")
    for filename in FILES:
        summarize_csv(DATA_DIR / filename, DATE_COLUMNS[filename])

    print(
        "\nDone. user_logs_v2.csv was NOT downloaded — this analysis doesn't use it "
        "and it's 30GB+."
    )


if __name__ == "__main__":
    main()
