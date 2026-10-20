"""End-to-end run: data -> clean -> split -> fit -> validate -> reports.

    python -m mtpl.pipeline          # everything (downloads data on first run)
    python -m mtpl.pipeline --no-nb  # skip the negative binomial check (peak RAM ~1.5 GB instead of ~3.2 GB)

Every number quoted in the README is written to reports/metrics.json or a CSV
in reports/tables/ by this script.
"""

from __future__ import annotations

import argparse
import gc
import json
import time

import duckdb
import numpy as np
import pandas as pd

from mtpl import config, data, evaluate, features, frequency, plots, rating

# Example drivers for the README: same car and address (all at base levels),
# so the only differences are the driver's age and bonus-malus.
EXAMPLES = {
    "new driver, 19, bonus-malus 100": {"DrivAge": 19, "BonusMalus": 100},
    "experienced driver, 45, bonus-malus 50": {"DrivAge": 45, "BonusMalus": 50},
    "driver, 45, bonus-malus 100": {"DrivAge": 45, "BonusMalus": 100},
}
EXAMPLE_CAR = {"VehAge": 7, "VehPower": 6, "VehBrand": "B1", "VehGas": "Regular",
               "Density": 100, "Region": "Centre"}


def cross_validate(df: pd.DataFrame, groups: np.ndarray, levels: dict, k: int = 5,
                   seed: int = config.SEED) -> pd.DataFrame:
    """Grouped k-fold CV: how much do the holdout metrics move with the split?

    One holdout is one draw. Refitting on k different splits (risk profiles
    kept together, as in the main split) shows the range a reader should
    expect, so a later challenger model is not credited for split noise.
    """
    rng = np.random.default_rng(seed)
    fold = (rng.permutation(groups.max() + 1) % k)[groups]
    df = df[config.RATING_FACTORS + ["ClaimNb", "Exposure"]]
    rows = []
    for i in range(k):
        tr, te = df[fold != i], df[fold == i]
        m, _ = frequency.fit_frequency(tr, features.choose_base_levels(tr), levels)
        y, e = te.ClaimNb.to_numpy(float), te.Exposure.to_numpy()
        mu = m.predict_claims(te)
        mu0 = tr.ClaimNb.sum() / tr.Exposure.sum() * e
        rows.append({"fold": i + 1, "policies": len(te), "claims": int(y.sum()),
                     "deviance_explained": evaluate.deviance_explained(y, mu, mu0),
                     "gini": evaluate.gini(y, e, mu / e),
                     "actual_to_expected": float(y.sum() / mu.sum())})
        del tr, te, m
        gc.collect()
    return pd.DataFrame(rows)


def _ensure_database() -> duckdb.DuckDBPyConnection:
    if not config.DB_PATH.exists():
        paths = data.download_raw()
        parquet = data.to_parquet(paths)
        data.build_database(parquet).close()
    con = duckdb.connect(str(config.DB_PATH))
    data.clean_in_db(con)  # cheap; keeps the cleaned table in sync with config
    return con


def run(with_nb: bool = True) -> dict:
    t0 = time.time()
    for d in (config.FIG_DIR, config.TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    con = _ensure_database()

    con.sql("SELECT * FROM cleaning_log ORDER BY step_order").df().to_csv(
        config.TABLE_DIR / "cleaning_log.csv", index=False)
    con.sql("SELECT * FROM reconciliation_summary").df().to_csv(
        config.TABLE_DIR / "reconciliation_summary.csv", index=False)
    data.run_sql_file(con, "05_data_profile.sql")
    profile = con.sql("SELECT * FROM data_profile").df()
    profile.to_csv(config.TABLE_DIR / "data_profile.csv", index=False)

    df = con.sql("SELECT * FROM policies_clean ORDER BY IDpol").df()
    df = features.add_rating_factors(df)
    df["is_holdout"] = features.split_by_profile(df)
    groups = features.profile_groups(df)
    # Keep only what the model and reports use, so later copies stay small.
    df = df[["IDpol", "is_holdout", "ClaimNb", "ClaimNb_reconciled", "Exposure"]
            + config.RATING_FACTORS].copy()
    train = df[~df.is_holdout]

    levels = {f: frequency.level_order(df[f]) for f in config.RATING_FACTORS}
    base = features.choose_base_levels(train)
    model, diag = frequency.fit_frequency(train, base, levels)
    print(f"fitted in {time.time() - t0:.0f}s", flush=True)

    df["pred_rate"] = model.predict_rate(df)
    df["pred_claims"] = df["pred_rate"] * df["Exposure"]
    train, hold = df[~df.is_holdout], df[df.is_holdout]

    # Back to the database for the SQL summaries.
    md = df[["IDpol", "is_holdout", "Exposure", "ClaimNb", "pred_rate", "pred_claims"]].copy()
    for f in config.RATING_FACTORS:
        md[f] = df[f].astype(str)
    con.register("md_view", md)
    con.execute("CREATE OR REPLACE TABLE model_data AS SELECT * FROM md_view")
    con.unregister("md_view")
    data.run_sql_file(con, "03_factor_summary.sql")
    data.run_sql_file(con, "04_lift.sql")
    lift = con.sql("SELECT * FROM lift_by_decile").df()
    fsum = con.sql("SELECT * FROM factor_summary").df()

    # ---- Holdout metrics ---------------------------------------------------
    y, mu, e = hold.ClaimNb.to_numpy(float), hold.pred_claims.to_numpy(), hold.Exposure.to_numpy()
    train_freq = train.ClaimNb.sum() / train.Exposure.sum()
    mu_null = train_freq * e
    dev_model, dev_null = evaluate.poisson_deviance(y, mu), evaluate.poisson_deviance(y, mu_null)
    gini = evaluate.gini(y, e, mu / e)
    gini_ci = evaluate.bootstrap_gini(y, e, mu / e, n_boot=200, seed=config.SEED)
    d2 = 1 - dev_model / dev_null
    d2_ci = evaluate.bootstrap_deviance_explained(y, mu, mu_null, n_boot=200, seed=config.SEED)
    lx, ly = evaluate.lorenz_curve(y, e, mu / e)

    # A/E by factor level on the holdout, with a z-score that allows for dispersion.
    phi = max(model.dispersion["pearson_phi"], 1.0)
    ae = fsum[fsum["sample"] == "holdout"].copy()
    ae["z"] = (ae.claims - ae.expected_claims) / np.sqrt(ae.expected_claims * phi)
    ae.to_csv(config.TABLE_DIR / "holdout_ae_by_factor.csv", index=False)
    tr_ae = fsum[fsum["sample"] == "train"]
    lift.to_csv(config.TABLE_DIR / "lift_by_decile.csv", index=False)

    # ---- Rating table ------------------------------------------------------
    rel = frequency.relativity_table(model, train)
    rel.to_csv(config.TABLE_DIR / "relativities.csv", index=False)
    base_freq = frequency.base_frequency(model)
    table_rate = rating.rate_frame(hold, rel, base_freq)
    max_rate_gap = float(np.max(np.abs(table_rate / hold.pred_rate.to_numpy() - 1)))

    examples = {}
    for name, driver in EXAMPLES.items():
        steps = rating.rate_driver({**EXAMPLE_CAR, **driver}, rel, base_freq)
        examples[name] = {
            "annual_frequency": float(steps.frequency.iloc[-1]),
            "multiple_of_base": float(steps.frequency.iloc[-1] / base_freq),
            "steps": steps.to_dict(orient="records"),
        }

    # ---- Negative binomial check ------------------------------------------
    nb = None
    if with_nb:
        alpha = model.dispersion["cameron_trivedi_alpha"]
        nb_params = frequency.fit_negative_binomial(train, model, alpha)
        X = frequency.design_matrix(hold, base, levels)
        mu_nb = np.exp(X.to_numpy() @ nb_params[X.columns].to_numpy()) * e
        rel_ratio = np.exp(nb_params - model.params).drop("const")
        nb = {
            "alpha": alpha,
            "holdout_deviance_explained": float(1 - evaluate.poisson_deviance(y, mu_nb) / dev_null),
            "max_relativity_change": float(np.max(np.abs(rel_ratio - 1))),
            "median_relativity_change": float(np.median(np.abs(rel_ratio - 1))),
        }
        print(f"NB check done at {time.time() - t0:.0f}s", flush=True)

    # ---- Stability across splits -------------------------------------------
    cv = cross_validate(df, groups, levels)
    cv.to_csv(config.TABLE_DIR / "cv_folds.csv", index=False)
    cv_summary = {c: {"mean": float(cv[c].mean()), "min": float(cv[c].min()), "max": float(cv[c].max())}
                  for c in ["deviance_explained", "gini", "actual_to_expected"]}
    print(f"CV done at {time.time() - t0:.0f}s", flush=True)

    # ---- Figures -----------------------------------------------------------
    plots.one_way_vs_glm(rel, config.FIG_DIR / "one_way_vs_glm.png")
    plots.lift_chart(lift, config.FIG_DIR / "lift_holdout.png")
    plots.lorenz_chart(lx, ly, gini, gini_ci, config.FIG_DIR / "lorenz_holdout.png")
    plots.forest(rel, config.FIG_DIR / "relativities.png")

    cl = pd.read_csv(config.TABLE_DIR / "cleaning_log.csv")
    metrics = {
        "data": {
            "source": f"CASdatasets @ {config.CAS_COMMIT}",
            "policies": int(cl.policies.iloc[0]),
            "claims_raw": int(cl.claims.iloc[0]),
            "exposure_years_raw": float(cl.exposure_years.iloc[0]),
            "policies_with_claims": int((df.ClaimNb_reconciled > 0).sum()),
            "share_policies_no_claim": float((df.ClaimNb_reconciled == 0).mean()),
            "count_mismatches": int(cl.rows_affected.iloc[2]),
            "orphan_claim_records": int(cl.rows_affected.iloc[1]),
            "policies_claims_capped": int(cl.rows_affected.iloc[3]),
            "claims_after_cap": int(cl.claims.iloc[3]),
            "policies_exposure_capped": int(cl.rows_affected.iloc[4]),
            "exposure_years_clean": float(cl.exposure_years.iloc[4]),
            "risk_profiles": int(groups.max() + 1),
            "policies_sharing_a_profile": int(pd.Series(groups).duplicated(keep=False).sum()),
            "profile": dict(zip(profile.metric, profile.value.astype(float))),
        },
        "split": {
            "seed": config.SEED,
            "train_policies": int(len(train)), "holdout_policies": int(len(hold)),
            "train_exposure": float(train.Exposure.sum()), "holdout_exposure": float(hold.Exposure.sum()),
            "train_claims": int(train.ClaimNb.sum()), "holdout_claims": int(hold.ClaimNb.sum()),
        },
        "model": {
            **diag,
            "base_levels": base,
            "base_frequency": base_freq,
            "train_overall_frequency": float(train_freq),
            "train_max_level_ae_gap": float(np.max(np.abs(tr_ae.actual_to_expected - 1))),
            "rating_table_max_relative_gap": max_rate_gap,
        },
        "holdout": {
            "mean_poisson_deviance_model": dev_model / len(y),
            "mean_poisson_deviance_null": dev_null / len(y),
            "deviance_explained": d2,
            "deviance_explained_ci95": list(d2_ci),
            "gini": gini,
            "gini_ci95": list(gini_ci),
            "actual_claims": float(y.sum()),
            "expected_claims": float(mu.sum()),
            "actual_to_expected": float(y.sum() / mu.sum()),
            "lift_top_vs_bottom_decile": float(lift.actual_freq.iloc[-1] / lift.actual_freq.iloc[0]),
            "top_decile_share_of_claims": float(lift.actual_claims.iloc[-1] / lift.actual_claims.sum()),
            "lift_decile_ae_min": float(lift.actual_to_expected.min()),
            "lift_decile_ae_max": float(lift.actual_to_expected.max()),
            "factor_levels_checked": int(len(ae)),
            "factor_levels_abs_z_over_2": int((ae.z.abs() > 2).sum()),
        },
        "cross_validation_5fold": cv_summary,
        "negative_binomial_check": nb,
        "examples": examples,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    con.close()
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-nb", action="store_true", help="skip the negative binomial check")
    a = p.parse_args()
    run(with_nb=not a.no_nb)


if __name__ == "__main__":
    main()
