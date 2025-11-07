# Data Layout

This repository does not bundle raw data. The folders under `data/` describe the
expected inputs so you can download only what you need before running the
notebooks or training new models.

| Folder            | Description                                                                 | Source / Notes |
|-------------------|-----------------------------------------------------------------------------|----------------|
| `raw/`            | Unversioned CSV/ZIP dumps from BTS On‑Time, Eurocontrol, or other providers. | Copy the original files here. Anything in `raw/` is git‑ignored. |
| `reference/`      | Small lookup tables that are safe to keep in git (e.g., `L_MONTHS.csv`).     | Add additional reference tables as needed. |

Typical BTS extracts that have already been used locally:

- `Jan_2019_ontime.csv`, `Jan_2020_ontime.csv`
- `On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2024_1.csv`
- `ONTIME.TD.202412.REL01.10FEB2025/ontime.td.202412.asc`
- Delay-cause supplemental files such as `ot_delaycause1_DL/Airline_Delay_Cause.csv`

Keep large intermediary artifacts (joined tables, feature matrices, etc.) out of
git by storing them under `data/interim/` or `data/processed/` (create the
folders when needed).

### Getting the BTS On-Time Dataset

1. Visit https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236
2. Select the months/carriers you need and download the ZIP.
3. Extract the CSVs into `data/raw/` and update the notebooks to point to those paths.

### Licensing

The BTS data is public-domain but may require attribution. Eurocontrol data is
subject to their research license—review their terms before redistributing.
