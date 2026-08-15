# data/

This directory is gitignored except for this file. Populate it manually with four files
downloaded from Kaggle (see the main [README](../README.md#data) for the competition link,
the rules-acceptance step required before downloading, and why v1 *and* v2 are both needed):

- `transactions.csv`
- `transactions_v2.csv`
- `members_v3.csv`
- `train_v2.csv`

Filenames must match exactly — `src/data_load.py` looks for these names here.
`user_logs_v2.csv` is never used — this analysis doesn't need it, and it's 30GB+.
