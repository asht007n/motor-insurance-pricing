"""Holdout metrics: Poisson deviance, ordered-Lorenz Gini, bootstrap intervals."""

from __future__ import annotations

import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Total Poisson deviance: 2 * sum(y*log(y/mu) - (y - mu)), with 0*log(0) = 0.

    This is the loss the Poisson GLM minimises, so it is the fair yardstick for
    comparing any two frequency models on the same policies. Lower is better.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if (mu <= 0).any():
        raise ValueError("Predicted claims must be positive")
    return float(np.sum(_unit_deviance(y, mu)))


def _unit_deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Per-policy Poisson deviance contributions."""
    safe_y = np.where(y > 0, y, 1.0)
    return 2.0 * (np.where(y > 0, y * np.log(safe_y / mu), 0.0) - (y - mu))


def deviance_explained(y: np.ndarray, mu: np.ndarray, mu_null: np.ndarray) -> float:
    """Share of the null model's deviance removed by the model (D-squared)."""
    return 1.0 - poisson_deviance(y, mu) / poisson_deviance(y, mu_null)


def _lorenz_points(claims: np.ndarray, exposure: np.ndarray, rate: np.ndarray,
                   weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative exposure share (x) and claim share (y), riskiest last.

    Policies with the same predicted rate are pooled into one step, so the
    curve does not depend on how ties happen to be ordered. A banded GLM
    gives many identical rates, so this matters.
    """
    w = np.ones_like(exposure, dtype=float) if weights is None else weights
    # Round away float noise so rates that are equal in the rating table
    # count as ties however they were computed.
    rate = np.round(rate, 12)
    order = np.argsort(rate, kind="stable")
    r, c, e = rate[order], claims[order] * w[order], exposure[order] * w[order]
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    c_g = np.add.reduceat(c, starts)
    e_g = np.add.reduceat(e, starts)
    x = np.r_[0.0, np.cumsum(e_g) / e_g.sum()]
    yv = np.r_[0.0, np.cumsum(c_g) / c_g.sum()]
    return x, yv


def gini(claims, exposure, rate, weights=None) -> float:
    """Gini from the ordered Lorenz curve, exposure on x and claims on y.

    Sort policies from lowest to highest predicted rate. If the model ranks
    risk well, the safest 50% of exposure produces far less than 50% of
    claims, the curve bows below the diagonal, and the Gini is large.
    A model that ranks at random scores about 0. It measures ranking only:
    multiplying every prediction by 2 leaves it unchanged.
    """
    x, yv = _lorenz_points(np.asarray(claims, float), np.asarray(exposure, float),
                           np.asarray(rate, float), weights)
    area = np.trapezoid(yv, x) if hasattr(np, "trapezoid") else np.trapz(yv, x)
    return float(1.0 - 2.0 * area)


def lorenz_curve(claims, exposure, rate) -> tuple[np.ndarray, np.ndarray]:
    return _lorenz_points(np.asarray(claims, float), np.asarray(exposure, float), np.asarray(rate, float))


def bootstrap_gini(claims, exposure, rate, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval for the Gini by resampling holdout policies.

    Uses Poisson(1) weights per policy, which behaves like resampling with
    replacement but avoids re-sorting on every draw.
    """
    rng = np.random.default_rng(seed)
    claims, exposure, rate = (np.asarray(a, float) for a in (claims, exposure, rate))
    draws = [gini(claims, exposure, rate, rng.poisson(1.0, size=len(claims)).astype(float))
             for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_deviance_explained(y, mu, mu_null, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval for D-squared, same Poisson-weight resampling."""
    rng = np.random.default_rng(seed)
    y, mu, mu_null = (np.asarray(a, float) for a in (y, mu, mu_null))
    d_mod, d_null = _unit_deviance(y, mu), _unit_deviance(y, mu_null)
    draws = []
    for _ in range(n_boot):
        w = rng.poisson(1.0, size=len(y))
        draws.append(1.0 - (w @ d_mod) / (w @ d_null))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)
