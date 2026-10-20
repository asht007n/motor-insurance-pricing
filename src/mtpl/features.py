"""Turn raw policy fields into rating factors and split train / holdout."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from mtpl import config


def band(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Categorical:
    """Cut a numeric column into left-closed bands; the last band is open-ended.

    Values below the first edge are an error rather than silently dropped,
    because a policy with no band would get no price.
    """
    if len(labels) != len(edges):
        raise ValueError("Need one label per left edge")
    if (values < edges[0]).any():
        raise ValueError(f"Values below the first band edge {edges[0]}")
    bins = list(edges) + [np.inf]
    out = pd.cut(values, bins=bins, labels=labels, right=False)
    if out.isna().any():
        raise ValueError("Some values did not fall in any band")
    return out


def add_rating_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add one categorical column per rating factor (names in config.RATING_FACTORS)."""
    out = df.copy()
    for raw_col, spec in config.BANDS.items():
        out[spec["name"]] = band(out[raw_col], spec["edges"], spec["labels"])
    for raw_col, name in config.CATEGORICAL.items():
        out[name] = pd.Categorical(out[raw_col].astype(str))
    return out


def profile_groups(df: pd.DataFrame) -> np.ndarray:
    """Integer id per distinct risk profile (identical values on PROFILE_COLS)."""
    return df.groupby(config.PROFILE_COLS, sort=True, observed=True).ngroup().to_numpy()


def split_by_profile(df: pd.DataFrame, holdout_share: float = config.HOLDOUT_SHARE,
                     seed: int = config.SEED) -> pd.Series:
    """Return a boolean Series: True for holdout rows.

    Policies with an identical risk profile go to the same side, so a driver
    whose year was split into several policy records cannot be in both the
    training data and the holdout.
    """
    groups = profile_groups(df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_share, random_state=seed)
    _, test_idx = next(splitter.split(df, groups=groups))
    is_holdout = np.zeros(len(df), dtype=bool)
    is_holdout[test_idx] = True
    return pd.Series(is_holdout, index=df.index, name="is_holdout")


def choose_base_levels(train: pd.DataFrame, factors: list[str] = config.RATING_FACTORS) -> dict[str, str]:
    """Base (relativity = 1.00) level for each factor: the level with most exposure.

    The base level is where the intercept sits. Every relativity is measured
    against it, so a large base keeps their standard errors small, and the
    base driver is a typical one rather than an edge case.
    """
    base = {}
    for f in factors:
        expo = train.groupby(f, observed=True)["Exposure"].sum()
        base[f] = str(expo.idxmax())
    return base
