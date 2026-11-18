"""Poisson GLM for claim frequency, its dispersion checks and its rating table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from mtpl import config
from mtpl.evaluate import _unit_deviance


def level_order(series: pd.Series) -> list[str]:
    """Display order for a factor's levels: band order, else natural sort (B2 before B10)."""
    if isinstance(series.dtype, pd.CategoricalDtype) and series.cat.ordered:
        return [str(x) for x in series.cat.categories]
    levels = {str(x) for x in series.unique()}

    def key(s: str):
        digits = "".join(ch for ch in s if ch.isdigit())
        return (s.rstrip("0123456789"), int(digits) if digits else -1, s)

    return sorted(levels, key=key)


def design_matrix(df: pd.DataFrame, base: dict[str, str], levels: dict[str, list[str]],
                  factors: list[str] = config.RATING_FACTORS) -> pd.DataFrame:
    """Intercept plus one 0/1 column per non-base level (treatment coding).

    Built by hand rather than with a formula so the column names are exactly
    the rows of the rating table: "driver_age=18-20" and so on.
    """
    cols = {"const": np.ones(len(df))}
    for f in factors:
        values = df[f].astype(str).to_numpy()
        for lev in levels[f]:
            if lev == base[f]:
                continue
            cols[f"{f}={lev}"] = (values == lev).astype(np.float64)
    return pd.DataFrame(cols, index=df.index)


@dataclass
class FrequencyModel:
    base: dict[str, str]
    levels: dict[str, list[str]]
    params: pd.Series          # log-scale coefficients
    bse_quasi: pd.Series       # standard errors scaled by the Pearson dispersion
    dispersion: dict
    factors: list[str]

    def predict_rate(self, df: pd.DataFrame) -> np.ndarray:
        """Expected claims per policy-year: exp(intercept + one coefficient per factor).

        Looks each level's coefficient up directly instead of building the
        0/1 design matrix, which gives the same answer with far less memory.
        """
        eta = np.full(len(df), float(self.params["const"]))
        for f in self.factors:
            coef = {lev: 0.0 if lev == self.base[f] else float(self.params[f"{f}={lev}"])
                    for lev in self.levels[f]}
            add = df[f].astype(str).map(coef).to_numpy(dtype=float)
            if np.isnan(add).any():
                raise ValueError(f"Unknown level in {f}; it has no relativity")
            eta += add
        return np.exp(eta)

    def predict_claims(self, df: pd.DataFrame) -> np.ndarray:
        """Expected claims over each policy's own exposure."""
        return self.predict_rate(df) * df["Exposure"].to_numpy()


def cameron_trivedi(y: np.ndarray, mu: np.ndarray) -> dict:
    """Regression-based overdispersion test (Cameron & Trivedi 1990).

    Under Poisson, E[(y - mu)^2 - y] = 0. Under NB2 it equals alpha * mu^2.
    Regressing ((y - mu)^2 - y) / mu on mu with no intercept estimates alpha;
    a large positive t-statistic is evidence of overdispersion.
    """
    z = ((y - mu) ** 2 - y) / mu
    ols = sm.OLS(z, mu).fit()
    return {"alpha": float(ols.params[0]), "t_stat": float(ols.tvalues[0])}


def aggregate_cells(df: pd.DataFrame, factors: list[str] = config.RATING_FACTORS) -> pd.DataFrame:
    """Pool policies that share every rating factor into one row (a rating cell).

    For a Poisson model this loses nothing: the log-likelihood depends on a
    cell only through its total claims and total exposure, so fitting on
    cells gives exactly the same coefficients and standard errors as fitting
    on individual policies, with about a third of the rows and memory.
    """
    return (df.groupby(factors, observed=True)
              .agg(ClaimNb=("ClaimNb", "sum"), Exposure=("Exposure", "sum"))
              .reset_index())


def fit_frequency(train: pd.DataFrame, base: dict[str, str], levels: dict[str, list[str]],
                  factors: list[str] = config.RATING_FACTORS) -> tuple[FrequencyModel, dict]:
    """Fit the Poisson GLM with a log link and log(Exposure) offset.

    The fit runs on rating cells; the dispersion checks run on individual
    policies, because overdispersion is a statement about policy-level
    variation that pooling would hide.
    """
    cells = aggregate_cells(train, factors)
    Xc = design_matrix(cells, base, levels, factors)
    glm = sm.GLM(cells["ClaimNb"].to_numpy(float), Xc.to_numpy(), family=sm.families.Poisson(),
                 offset=np.log(cells["Exposure"].to_numpy()))
    # Newton's method gives the same estimates as the default IRLS (checked to
    # 1e-13) at about a third of the peak memory, which matters on a laptop.
    res = glm.fit(method="newton", maxiter=100, tol=1e-12)
    params = pd.Series(res.params, index=Xc.columns)
    bse = pd.Series(res.bse, index=Xc.columns)
    converged = bool(res.mle_retvals.get("converged", False))

    model = FrequencyModel(base=base, levels=levels, params=params,
                           bse_quasi=bse.copy(), dispersion={}, factors=factors)

    # Policy-level dispersion checks.
    y = train["ClaimNb"].to_numpy(dtype=float)
    mu = model.predict_claims(train)
    n, p = len(y), Xc.shape[1]
    x2 = (y - mu) ** 2 / mu
    pearson_phi = float(x2.sum() / (n - p))
    unit_dev = _unit_deviance(y, mu)
    deviance_phi = float(unit_dev.sum() / (n - p))
    ct = cameron_trivedi(y, mu)
    # How fragile is the Pearson estimate? A claim on a policy insured for a
    # few days has a tiny mu and a huge (y - mu)^2 / mu.
    short = train["Exposure"].to_numpy() < 0.05
    top100_share = float(np.sort(x2)[-100:].sum() / x2.sum())
    phi_excl_short = float(x2[~short].sum() / ((~short).sum() - p))

    # Quasi-Poisson: same coefficients, standard errors scaled by sqrt(phi).
    # Only inflate, never shrink: phi < 1 would make the relativities look
    # more certain than the Poisson model itself says they are.
    model.bse_quasi = bse * np.sqrt(max(pearson_phi, 1.0))
    model.dispersion = {
        "pearson_phi": pearson_phi,
        "deviance_phi": deviance_phi,
        "cameron_trivedi_alpha": ct["alpha"],
        "cameron_trivedi_t": ct["t_stat"],
        "pearson_share_from_top100_policies": top100_share,
        "pearson_phi_excluding_exposure_under_0.05": phi_excl_short,
        "policies_exposure_under_0.05": int(short.sum()),
    }
    diagnostics = {
        "n_train_policies": int(n),
        "n_train_cells": int(len(cells)),
        "n_params": int(p),
        "converged": converged,
        "train_deviance": float(unit_dev.sum()),
        **model.dispersion,
        "train_claims_actual": float(y.sum()),
        "train_claims_fitted": float(mu.sum()),
    }
    return model, diagnostics


def fit_negative_binomial(train: pd.DataFrame, model: FrequencyModel, alpha: float) -> pd.Series:
    """NB2 GLM at policy level with alpha fixed at the Cameron-Trivedi estimate.

    Used only as a check: if the relativities barely move, the Poisson mean
    model is fine and overdispersion only matters for the standard errors.
    Newton's method keeps the extra memory for this fit under about 2 GB.
    """
    X = design_matrix(train, model.base, model.levels, model.factors)
    y = train["ClaimNb"].to_numpy(dtype=float)
    offset = np.log(train["Exposure"].to_numpy())
    res = sm.GLM(y, X.to_numpy(), family=sm.families.NegativeBinomial(alpha=alpha),
                 offset=offset).fit(method="newton", maxiter=50)
    return pd.Series(res.params, index=X.columns)


def relativity_table(model: FrequencyModel, train: pd.DataFrame) -> pd.DataFrame:
    """One row per factor level: GLM relativity with 95% CI next to the one-way view.

    The one-way relativity is the level's observed frequency divided by the
    base level's observed frequency. Where it differs from the GLM relativity,
    the factor is correlated with something else in the model (for example,
    young drivers also tend to have a high bonus-malus).
    """
    rows = []
    z = 1.959964
    for f in model.factors:
        g = train.groupby(f, observed=True).agg(exposure=("Exposure", "sum"), claims=("ClaimNb", "sum"))
        g.index = g.index.astype(str)
        g["observed_freq"] = g["claims"] / g["exposure"]
        base_freq = g.loc[model.base[f], "observed_freq"]
        for lev in model.levels[f]:
            name = f"{f}={lev}"
            is_base = lev == model.base[f]
            seen = lev in g.index
            coef = 0.0 if is_base else float(model.params[name])
            se = 0.0 if is_base else float(model.bse_quasi[name])
            rows.append({
                "factor": f,
                "level": lev,
                "is_base": is_base,
                "exposure_years": float(g.loc[lev, "exposure"]) if seen else 0.0,
                "claims": int(g.loc[lev, "claims"]) if seen else 0,
                "observed_freq": float(g.loc[lev, "observed_freq"]) if seen else np.nan,
                "one_way_relativity": float(g.loc[lev, "observed_freq"] / base_freq) if seen else np.nan,
                "glm_coef": coef,
                "glm_se": se,
                "glm_relativity": float(np.exp(coef)),
                "ci_low": float(np.exp(coef - z * se)),
                "ci_high": float(np.exp(coef + z * se)),
            })
    return pd.DataFrame(rows)


def base_frequency(model: FrequencyModel) -> float:
    """Annual claim frequency for a driver at every base level."""
    return float(np.exp(model.params["const"]))
