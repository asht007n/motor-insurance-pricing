-- 03_factor_summary.sql
-- One row per sample x rating factor x level: exposure, claims, observed
-- frequency, and actual vs expected (A/E) once predictions are attached.
-- UNPIVOT turns the eight factor columns into (factor, level) pairs so one
-- GROUP BY covers every factor; the window gives each level's share of
-- exposure within its own factor.

CREATE OR REPLACE TABLE factor_summary AS
WITH long AS (
    UNPIVOT (
        SELECT
            CASE WHEN is_holdout THEN 'holdout' ELSE 'train' END AS sample,
            Exposure, ClaimNb, pred_claims,
            driver_age, bonus_malus, vehicle_age, vehicle_power,
            vehicle_brand, fuel, density, region
        FROM model_data
    )
    ON driver_age, bonus_malus, vehicle_age, vehicle_power,
       vehicle_brand, fuel, density, region
    INTO NAME factor VALUE level
),
agg AS (
    SELECT
        sample,
        factor,
        level,
        COUNT(*)         AS policies,
        SUM(Exposure)    AS exposure_years,
        SUM(ClaimNb)     AS claims,
        SUM(pred_claims) AS expected_claims
    FROM long
    GROUP BY sample, factor, level
)
SELECT
    sample,
    factor,
    level,
    policies,
    exposure_years,
    exposure_years / SUM(exposure_years) OVER (PARTITION BY sample, factor) AS exposure_share,
    claims,
    expected_claims,
    claims / exposure_years          AS observed_freq,
    claims / NULLIF(expected_claims, 0) AS actual_to_expected
FROM agg
ORDER BY sample, factor, level;
