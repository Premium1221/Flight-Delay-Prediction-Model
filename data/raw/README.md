# Raw Data (Not Versioned)

Drop any source datasets required for experimentation or model training into
this folder. Examples include:

- Monthly BTS On-Time extracts (`*.zip`, `*.csv`)
- Eurocontrol route extracts (`eurocontrol-rnd-extract/…`)
- Derived samples such as `flights_sample_3m.csv`

Nothing inside `data/raw/` is tracked by git. If you need to keep a short sample
in the repo, place it under `data/reference/` instead.
