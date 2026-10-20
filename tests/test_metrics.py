import numpy as np
import pytest

from mtpl import evaluate


def test_deviance_is_zero_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert evaluate.poisson_deviance(y, y) == pytest.approx(0.0)


def test_deviance_handles_zero_claims():
    # For y = 0 the unit deviance is 2 * mu.
    assert evaluate.poisson_deviance(np.array([0.0]), np.array([0.3])) == pytest.approx(0.6)


def test_true_mean_beats_flat_mean_on_simulated_data():
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.02, 0.3, 50_000)
    y = rng.poisson(mu)
    flat = np.full_like(mu, mu.mean())
    assert evaluate.deviance_explained(y, mu, flat) > 0


def test_gini_is_zero_when_every_prediction_is_the_same():
    rng = np.random.default_rng(2)
    claims = rng.poisson(0.1, 1000).astype(float)
    exposure = rng.uniform(0.1, 1, 1000)
    rate = np.full(1000, 0.07)
    assert evaluate.gini(claims, exposure, rate) == pytest.approx(0.0, abs=1e-12)


def test_gini_rewards_correct_ranking_and_ignores_scale():
    rng = np.random.default_rng(3)
    true_rate = rng.uniform(0.01, 0.4, 20_000)
    exposure = rng.uniform(0.1, 1, 20_000)
    claims = rng.poisson(true_rate * exposure).astype(float)
    g = evaluate.gini(claims, exposure, true_rate)
    assert g > 0.2
    assert evaluate.gini(claims, exposure, 3 * true_rate) == pytest.approx(g)
    assert evaluate.gini(claims, exposure, -true_rate) < 0  # reversed ranking


def test_lorenz_curve_runs_from_origin_to_one():
    x, y = evaluate.lorenz_curve(np.array([0, 1, 2.0]), np.array([1, 1, 1.0]), np.array([0.1, 0.2, 0.3]))
    assert (x[0], y[0]) == (0.0, 0.0)
    assert x[-1] == pytest.approx(1.0) and y[-1] == pytest.approx(1.0)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(4)
    rate = rng.uniform(0.01, 0.4, 5_000)
    exposure = np.ones(5_000)
    claims = rng.poisson(rate).astype(float)
    lo, hi = evaluate.bootstrap_gini(claims, exposure, rate, n_boot=100, seed=0)
    assert lo < evaluate.gini(claims, exposure, rate) < hi
