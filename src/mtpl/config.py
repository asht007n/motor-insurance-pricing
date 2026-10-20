"""Project settings: data source, paths, cleaning thresholds and rating bands.

Everything a reviewer might question (where the data came from, what was
capped, how the bands were cut) lives here so it can be read in one place.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "mtpl.duckdb"
SQL_DIR = ROOT / "sql"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"

# --------------------------------------------------------------------------
# Data source, pinned to one commit of the CASdatasets repository so the
# numbers in the README can be reproduced exactly. Older copies of freMTPL2
# (OpenML 41214/41215, Kaggle, Hugging Face) are a different version: they
# have 678,013 policies, 26,639 claim rows and ~9k policies whose ClaimNb
# disagrees with the severity table. The reconciliation step handles either.
# --------------------------------------------------------------------------
CAS_COMMIT = "227fb56b8734bdb7c0327a41180e01d2ddaeaf26"  # CASdatasets 1.2-1
_CAS_RAW = f"https://raw.githubusercontent.com/dutangc/CASdatasets/{CAS_COMMIT}/data"
SOURCES = {
    "freMTPL2freq": {
        "url": f"{_CAS_RAW}/freMTPL2freq.rda",
        "sha256": "82c8598513d9fa78226b6c7d271a9940eca4b8083e9e32320a75d377bdfe15a3",
    },
    "freMTPL2sev": {
        "url": f"{_CAS_RAW}/freMTPL2sev.rda",
        "sha256": "78e6e5016ae046603b37e88ff8ad8ca327f5d59f827c1f75d140388de09b14b3",
    },
}

# --------------------------------------------------------------------------
# Cleaning thresholds (see docs/modeling_decisions.md for the reasoning)
# --------------------------------------------------------------------------
MAX_CLAIMS_PER_POLICY = 4   # 8 policies have 5-16 claims; 7 of them share one profile
MAX_EXPOSURE_YEARS = 1.0    # data is described as observed "mostly on one year"

# --------------------------------------------------------------------------
# Train / holdout split
# --------------------------------------------------------------------------
SEED = 2026
HOLDOUT_SHARE = 0.20
# Columns that define a "risk profile". Policies with identical values are
# kept on the same side of the split (see modeling_decisions.md, decision 5).
PROFILE_COLS = [
    "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand",
    "VehGas", "Area", "Density", "Region",
]

# --------------------------------------------------------------------------
# Rating bands. Edges are left-closed: [lo, hi). The last band is open-ended.
# Cut from the one-way analysis so each band has enough exposure to give a
# stable relativity and the shape (e.g. the young-driver slope) is kept.
# --------------------------------------------------------------------------
BANDS = {
    "DrivAge": {
        "name": "driver_age",
        "edges": [18, 21, 23, 26, 30, 40, 50, 60, 70, 80],
        "labels": ["18-20", "21-22", "23-25", "26-29", "30-39", "40-49",
                   "50-59", "60-69", "70-79", "80+"],
    },
    "BonusMalus": {
        "name": "bonus_malus",
        "edges": [50, 51, 60, 70, 80, 90, 100, 110, 125],
        "labels": ["50", "51-59", "60-69", "70-79", "80-89", "90-99",
                   "100-109", "110-124", "125+"],
    },
    "VehAge": {
        "name": "vehicle_age",
        "edges": [0, 1, 3, 6, 10, 13, 16, 20],
        "labels": ["0", "1-2", "3-5", "6-9", "10-12", "13-15", "16-19", "20+"],
    },
    "VehPower": {
        "name": "vehicle_power",
        "edges": [4, 5, 6, 7, 8, 9, 11],
        "labels": ["4", "5", "6", "7", "8", "9-10", "11+"],
    },
    "Density": {
        "name": "density",
        "edges": [1, 20, 50, 150, 400, 1000, 3000, 8000],
        "labels": ["1-19", "20-49", "50-149", "150-399", "400-999",
                   "1000-2999", "3000-7999", "8000+"],
    },
}

# Categorical factors used as they come.
CATEGORICAL = {
    "VehBrand": "vehicle_brand",
    "VehGas": "fuel",
    "Region": "region",
}

# Final rating factors, in the order they appear in the rating table.
# Area is left out on purpose: it is a coarse banding of Density
# (correlation 0.97 with log density), so including both would split one
# effect across two aliased factors and make neither relativity readable.
RATING_FACTORS = [
    "driver_age", "bonus_malus", "vehicle_age", "vehicle_power",
    "vehicle_brand", "fuel", "density", "region",
]

FACTOR_LABELS = {
    "driver_age": "Driver age",
    "bonus_malus": "Bonus-malus",
    "vehicle_age": "Vehicle age",
    "vehicle_power": "Vehicle power",
    "vehicle_brand": "Vehicle brand",
    "fuel": "Fuel",
    "density": "Population density (per km²)",
    "region": "Region",
}
