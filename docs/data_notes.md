# Data notes

## Source and version

| | |
|---|---|
| Dataset | freMTPL2 (`freMTPL2freq`, `freMTPL2sev`) |
| Package | [CASdatasets](https://github.com/dutangc/CASdatasets) 1.2-1, C. Dutang and A. Charpentier, GPL (≥ 2) |
| Pinned commit | `227fb56b8734bdb7c0327a41180e01d2ddaeaf26` |
| Files | `data/freMTPL2freq.rda` (SHA-256 `82c85985…15a3`), `data/freMTPL2sev.rda` (SHA-256 `78e6e501…b14b3`) |
| Original source | an unnamed French private insurer, policies observed mostly over one year, about 2011–2013 (package documentation) |

`src/mtpl/data.py` downloads both files from that commit, refuses to continue
if a checksum differs, reads them without R (the `rdata` package), and loads
them into DuckDB as `policies_raw` and `claims_raw`.

### Why the version matters

CASdatasets replaced both freMTPL2 files on 2024-09-17 (package 1.2-0, per its
`NEWS.md`). Most tutorials use an older copy:

| | This repo (CASdatasets 1.2-1) | Older copy (OpenML 41214/41215, Kaggle, Hugging Face) |
|---|---|---|
| Policies | 677,991 | 678,013 (scikit-learn documentation) |
| Claim records | 26,444 | 26,639 (Hugging Face dataset card) |
| Policies whose `ClaimNb` disagrees with the claim records | 0 (checked) | 9,117 (Hugging Face dataset card) |

The older-copy figures are as reported by those sources; they were not
re-computed here because the build environment could not reach OpenML. To
check them yourself, load the OpenML tables into the same database and run
the same SQL:

```python
# Not part of the pipeline. Requires network access to openml.org.
import duckdb
from sklearn.datasets import fetch_openml
from mtpl import data

freq = fetch_openml(data_id=41214, as_frame=True).frame
sev = fetch_openml(data_id=41215, as_frame=True).frame
con = duckdb.connect()
con.register("f", freq); con.register("s", sev)
con.execute("CREATE TABLE policies_raw AS SELECT * FROM f")
con.execute("CREATE TABLE claims_raw AS SELECT * FROM s")
data.clean_in_db(con)
print(con.sql("SELECT * FROM reconciliation_summary"))
```

## Columns

`freMTPL2freq`, one row per policy:

| Column | Meaning | Used as |
|---|---|---|
| IDpol | policy ID, links to the claims table | key |
| ClaimNb | number of claims in the exposure period | target |
| Exposure | years insured during the period | offset |
| VehPower | vehicle power class, 4–15 | banded: 4, 5, 6, 7, 8, 9–10, 11+ |
| VehAge | vehicle age in years | banded: 0, 1–2, 3–5, 6–9, 10–12, 13–15, 16–19, 20+ |
| DrivAge | driver age in years (18+) | banded: 18–20 … 80+ |
| BonusMalus | French claims-history coefficient, 50–350; under 100 is a bonus | banded: 50, 51–59, … 125+ |
| VehBrand | car brand, anonymised (B1–B14, 11 values) | as is |
| VehGas | Diesel or Regular | as is |
| Area | density class, A (rural) to F (urban centre) | not used (aliased with Density) |
| Density | inhabitants per km² of the driver's town | banded on a log scale |
| Region | French region (1970–2015 boundaries), 22 values | as is |

`freMTPL2sev`, one row per claim: `IDpol`, `ClaimAmount`. Some amounts are
flat-rate settlements under the French IRSA-IDA convention between insurers
(package documentation).

## Quirks found, with the rule applied

Counts below come from `reports/tables/cleaning_log.csv` and
`reports/tables/data_profile.csv` (built by `sql/02_clean.sql` and
`sql/05_data_profile.sql`).

1. **A policy ID written as `1e+05`.** `IDpol` is stored as an R factor whose
   text labels include `"1e+05"` for policy 100000. A direct text-to-integer
   cast fails. The loader parses the labels as numbers, checks they are
   whole and unique, then stores integers.
2. **Claim counts of 5 to 16 on eight policies.** Seven of them: driver aged
   52–53, bonus-malus 50, brand B1, Languedoc-Roussillon, exposures 0.07–0.41
   years, 66 claims between them. Capped at 4 claims per policy (39 claims
   removed, 0.15%).
3. **Exposure above one year on 1,224 policies** (maximum 2.01). Capped at 1.
4. **Repeated risk profiles.** 108,662 groups covering 257,908 policies share
   all nine rating fields. In 79% of groups the policy IDs are within 100 of
   each other (median gap 2), and the average exposure is 0.33 years against
   0.53 for all policies. Handled by splitting train and holdout by profile.
5. **Bonus-malus sits on the claim-free ladder** (50, 51, 54, 57, 60, 64, 68,
   72, 76, 80, 85, 90, 95, 100) for 91% of exposure. Values off the ladder
   usually follow an at-fault claim. Kept as is; see decision 14 for the timing
   question.
6. **Area and Density carry the same information** (correlation 0.97 between
   the area code and log density). Area is dropped.

## Claim amounts (input for the severity model)

From `reports/tables/data_profile.csv`, all 26,444 claims, uncapped:

| | |
|---|---|
| Median | €1,172 |
| Mean | €2,266 |
| 99th percentile | €16,451 |
| Maximum | €4,075,401 |
| Claims on the five most common exact amounts (1,204; 1,128.12; 1,172; 1,128; 602) | 42% of claims, 21% of cost |
| Claims above €100,000 | 41 |
| Cost above €100,000 on those claims | 17.7% of all claim cost |

Two consequences for phase 3 of the roadmap: the distribution is a point
mass of convention amounts plus a long tail, not one smooth Gamma; and 41
claims carry almost a fifth of the cost, so large claims have to be capped
and their excess loaded back separately or they will dominate any fit.
