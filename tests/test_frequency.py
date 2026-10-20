"""The GLM on simulated data where the true relativities are known."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from mtpl import frequency, rating

FACTORS = ["age", "car"]
TRUE = {"age": {"young": 2.0, "mid": 1.0, "old": 0.8},
        "car": {"small": 1.0, "big": 1.3}}
BASE_FREQ = 0.07


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(11)
    n = 200_000
    df = pd.DataFrame({
        "age": rng.choice(["young", "mid", "old"], n, p=[0.2, 0.5, 0.3]),
        "car": rng.choice(["small", "big"], n, p=[0.6, 0.4]),
        "Exposure": rng.uniform(0.05, 1.0, n),
    })
    rate = BASE_FREQ * df.age.map(TRUE["age"]) * df.car.map(TRUE["car"])
    df["ClaimNb"] = rng.poisson(rate * df.Exposure)
    base = {"age": "mid", "car": "small"}
    levels = {"age": ["young", "mid", "old"], "car": ["small", "big"]}
    model, diag = frequency.fit_frequency(df, base, levels, FACTORS)
    return df, model, diag, base, levels


def test_recovers_true_relativities(sim):
    df, model, *_ = sim
    rel = frequency.relativity_table(model, df).set_index(["factor", "level"])
    for f, levels in TRUE.items():
        for lev, r in levels.items():
            assert rel.loc[(f, lev), "glm_relativity"] == pytest.approx(r, rel=0.06)
            assert rel.loc[(f, lev), "ci_low"] <= r <= rel.loc[(f, lev), "ci_high"] or r == 1.0
    assert frequency.base_frequency(model) == pytest.approx(BASE_FREQ, rel=0.06)


def test_cell_fit_matches_policy_level_fit(sim):
    df, model, _, base, levels = sim
    X = frequency.design_matrix(df, base, levels, FACTORS)
    full = sm.GLM(df.ClaimNb.to_numpy(float), X, family=sm.families.Poisson(),
                  offset=np.log(df.Exposure.to_numpy())).fit()
    np.testing.assert_allclose(model.params[X.columns], full.params[X.columns], rtol=1e-8, atol=1e-10)


def test_balance_property_on_training_data(sim):
    # With a log link and an intercept, fitted claims equal actual claims in
    # total and within every level of every factor.
    df, model, diag, *_ = sim
    mu = model.predict_claims(df)
    assert diag["train_claims_fitted"] == pytest.approx(diag["train_claims_actual"])
    for f in FACTORS:
        by = pd.DataFrame({"f": df[f], "y": df.ClaimNb, "mu": mu}).groupby("f")[["y", "mu"]].sum()
        np.testing.assert_allclose(by.y, by.mu, rtol=1e-8)


def test_rating_table_reproduces_model_exactly(sim):
    df, model, *_ = sim
    rel = frequency.relativity_table(model, df)
    from_table = rating.rate_frame(df, rel, frequency.base_frequency(model))
    np.testing.assert_allclose(from_table, model.predict_rate(df), rtol=1e-12)


def test_no_overdispersion_detected_in_pure_poisson_data(sim):
    *_, diag, _, _ = sim
    assert diag["pearson_phi"] == pytest.approx(1.0, abs=0.05)
    assert abs(diag["cameron_trivedi_t"]) < 3
