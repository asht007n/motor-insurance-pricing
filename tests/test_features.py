import numpy as np
import pandas as pd
import pytest

from mtpl import config, features


def _band(col, values):
    spec = config.BANDS[col]
    return list(features.band(pd.Series(values), spec["edges"], spec["labels"]).astype(str))


def test_driver_age_band_edges():
    assert _band("DrivAge", [18, 20, 21, 29, 30, 79, 80, 100]) == [
        "18-20", "18-20", "21-22", "26-29", "30-39", "70-79", "80+", "80+"]


def test_bonus_malus_band_edges():
    assert _band("BonusMalus", [50, 51, 59, 60, 100, 124, 125, 350]) == [
        "50", "51-59", "51-59", "60-69", "100-109", "110-124", "125+", "125+"]


def test_density_band_edges():
    assert _band("Density", [1, 19, 20, 7999, 8000, 27000]) == [
        "1-19", "1-19", "20-49", "3000-7999", "8000+", "8000+"]


def test_value_below_first_band_is_an_error():
    with pytest.raises(ValueError):
        _band("DrivAge", [17])


def _toy_policies(n=2_000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "VehPower": rng.integers(4, 8, n), "VehAge": rng.integers(0, 5, n),
        "DrivAge": rng.integers(18, 30, n), "BonusMalus": 50,
        "VehBrand": rng.choice(["B1", "B2"], n), "VehGas": "Diesel",
        "Area": "C", "Density": 100, "Region": "Centre",
        "Exposure": rng.uniform(0.1, 1, n),
    })


def test_identical_profiles_stay_on_the_same_side_of_the_split():
    df = _toy_policies()
    df["is_holdout"] = features.split_by_profile(df, holdout_share=0.2, seed=1)
    per_profile = df.groupby(config.PROFILE_COLS)["is_holdout"].nunique()
    assert (per_profile == 1).all()
    assert 0.1 < df.is_holdout.mean() < 0.3


def test_split_does_not_depend_on_row_order():
    df = _toy_policies()
    a = features.split_by_profile(df, seed=5)
    shuffled = df.sample(frac=1, random_state=0)
    b = features.split_by_profile(shuffled, seed=5).reindex(df.index)
    assert (a == b).all()


def test_base_level_is_the_largest_exposure_level():
    df = pd.DataFrame({"f": ["a", "b", "b", "c"], "Exposure": [5.0, 1.0, 1.0, 2.0]})
    assert features.choose_base_levels(df, ["f"]) == {"f": "a"}
