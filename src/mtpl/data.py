"""Download the pinned freMTPL2 files, verify them, and load them into DuckDB.

Raw files are never committed. `make data` (or `python -m mtpl.data`)
re-creates everything under data/ from the pinned source.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

from mtpl import config


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raw(force: bool = False) -> dict[str, Path]:
    """Download the two .rda files and check their SHA-256 against config."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, src in config.SOURCES.items():
        path = config.RAW_DIR / f"{name}.rda"
        if force or not path.exists():
            print(f"Downloading {name} ...")
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(src["url"], timeout=120) as resp, tmp.open("wb") as out:
                shutil.copyfileobj(resp, out)
            tmp.replace(path)
        digest = _sha256(path)
        if digest != src["sha256"]:
            raise ValueError(
                f"{path.name}: checksum {digest} does not match the pinned "
                f"{src['sha256']}. The source changed; do not use these numbers."
            )
        paths[name] = path
    return paths


def read_rda(path: Path) -> pd.DataFrame:
    """Read a single-object .rda file without needing R installed."""
    import rdata  # imported here so tests do not need it

    objects = rdata.read_rda(str(path))
    if len(objects) != 1:
        raise ValueError(f"Expected one object in {path}, found {list(objects)}")
    return next(iter(objects.values())).reset_index(drop=True)


def parse_policy_id(ids: pd.Series) -> pd.Series:
    """Turn the IDpol factor labels into integers.

    IDpol is stored as an R factor whose labels are text, and R wrote policy
    100000 as "1e+05". Casting the text straight to int fails on that label,
    so parse as a number first and check nothing was lost on the way.
    """
    num = pd.to_numeric(ids.astype(str), errors="raise")
    if (num != num.round()).any():
        raise ValueError("Non-integer policy IDs found")
    return num.astype("int64")


def to_parquet(paths: dict[str, Path]) -> dict[str, Path]:
    """Convert the .rda files to typed parquet (fast to re-read)."""
    out = {}
    freq = read_rda(paths["freMTPL2freq"])
    freq["IDpol"] = parse_policy_id(freq["IDpol"])
    if freq["IDpol"].duplicated().any():
        raise ValueError("Duplicate policy IDs in freMTPL2freq")
    for col in ["ClaimNb", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]:
        freq[col] = freq[col].astype("int64")
    for col in ["VehBrand", "VehGas", "Area", "Region"]:
        freq[col] = freq[col].astype(str)
    out["policies"] = config.RAW_DIR / "policies.parquet"
    freq.to_parquet(out["policies"], index=False)

    sev = read_rda(paths["freMTPL2sev"])
    sev["IDpol"] = parse_policy_id(sev["IDpol"])
    out["claims"] = config.RAW_DIR / "claims.parquet"
    sev.to_parquet(out["claims"], index=False)
    return out


def run_sql_file(con: duckdb.DuckDBPyConnection, name: str) -> None:
    """Execute every statement in sql/<name>."""
    con.execute((config.SQL_DIR / name).read_text())


def build_database(parquet: dict[str, Path], db_path: Path = config.DB_PATH) -> duckdb.DuckDBPyConnection:
    """Load raw tables into a DuckDB database file and build the cleaned table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE OR REPLACE TABLE policies_raw AS SELECT * FROM read_parquet(?)",
        [str(parquet["policies"])],
    )
    con.execute(
        "CREATE OR REPLACE TABLE claims_raw AS SELECT * FROM read_parquet(?)",
        [str(parquet["claims"])],
    )
    clean_in_db(con)
    return con


def clean_in_db(con: duckdb.DuckDBPyConnection) -> None:
    """Run the reconciliation and cleaning SQL against policies_raw / claims_raw."""
    con.execute(f"SET VARIABLE max_claims = {int(config.MAX_CLAIMS_PER_POLICY)}")
    con.execute(f"SET VARIABLE max_exposure = {float(config.MAX_EXPOSURE_YEARS)}")
    run_sql_file(con, "01_reconcile.sql")
    run_sql_file(con, "02_clean.sql")


def main() -> None:
    paths = download_raw()
    parquet = to_parquet(paths)
    con = build_database(parquet)
    print(con.sql("SELECT * FROM cleaning_log ORDER BY step_order"))
    con.close()


if __name__ == "__main__":
    main()
