# data/

This directory is gitignored except for this file and `download_data.py`.

Run `python data/download_data.py` from the repo root to populate it with:

- `transactions_v2.csv`
- `members_v3.csv`
- `train_v2.csv`

See the main [README](../README.md#getting-the-data) for Kaggle credential setup.
`user_logs_v2.csv` is never downloaded — this analysis doesn't use it, and it's 30GB+.
