"""Price a driver from the published rating table, factor by factor.

The rating table (reports/tables/relativities.csv) plus the base frequency is
the whole model: expected claims per year = base x one relativity per factor.
This module uses only those files, not the fitted statsmodels object, which is
the point: anyone with the table can reproduce every price.

Example:
    python -m mtpl.rating --age 22 --bonus-malus 90 --vehicle-age 4 --power 6 \\
        --brand B12 --fuel Diesel --density 2500 --region Ile-de-France
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from mtpl import config, features


def load_rating_table():
    rel = pd.read_csv(config.TABLE_DIR / "relativities.csv", dtype={"level": str})
    metrics = json.loads((config.REPORTS_DIR / "metrics.json").read_text())
    return float(metrics["model"]["base_frequency"]), rel


def rate_frame(df: pd.DataFrame, rel: pd.DataFrame, base_freq: float) -> np.ndarray:
    """Vectorised: expected claims per policy-year for every row of a banded frame."""
    lookup = {(r.factor, str(r.level)): r.glm_relativity for r in rel.itertuples()}
    rate = np.full(len(df), base_freq)
    for f in rel.factor.unique():
        rate *= df[f].astype(str).map(lambda lev, f=f: lookup[(f, lev)]).to_numpy(dtype=float)
    return rate


def rate_driver(profile: dict, rel: pd.DataFrame, base_freq: float) -> pd.DataFrame:
    """Step-by-step breakdown for one driver given raw fields (DrivAge, BonusMalus, ...)."""
    row = features.add_rating_factors(pd.DataFrame([profile]))
    steps = [{"factor": "base", "level": "", "relativity": 1.0, "frequency": base_freq}]
    freq = base_freq
    for f in config.RATING_FACTORS:
        lev = str(row[f].iloc[0])
        match = rel[(rel.factor == f) & (rel.level.astype(str) == lev)]
        if match.empty:
            raise ValueError(f"No relativity for {f}={lev}")
        r = float(match.glm_relativity.iloc[0])
        freq *= r
        steps.append({"factor": f, "level": lev, "relativity": r, "frequency": freq})
    return pd.DataFrame(steps)


def main() -> None:
    p = argparse.ArgumentParser(description="Expected claim frequency from the rating table")
    p.add_argument("--age", type=int, required=True, help="driver age, 18+")
    p.add_argument("--bonus-malus", type=int, required=True, help="50 (best) to 350")
    p.add_argument("--vehicle-age", type=int, required=True)
    p.add_argument("--power", type=int, required=True, help="vehicle power class, 4-15")
    p.add_argument("--brand", required=True, help="B1..B14")
    p.add_argument("--fuel", required=True, choices=["Diesel", "Regular"])
    p.add_argument("--density", type=int, required=True, help="inhabitants per km²")
    p.add_argument("--region", required=True)
    a = p.parse_args()
    profile = {"DrivAge": a.age, "BonusMalus": a.bonus_malus, "VehAge": a.vehicle_age,
               "VehPower": a.power, "VehBrand": a.brand, "VehGas": a.fuel,
               "Density": a.density, "Region": a.region}
    base, rel = load_rating_table()
    out = rate_driver(profile, rel, base)
    print(out.to_string(index=False, formatters={"relativity": "{:.3f}".format,
                                                 "frequency": "{:.4f}".format}))
    print(f"\nExpected claims per year: {out.frequency.iloc[-1]:.4f} "
          f"(about 1 claim every {1 / out.frequency.iloc[-1]:.0f} years)")


if __name__ == "__main__":
    main()
