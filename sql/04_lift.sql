-- 04_lift.sql
-- Lift table on the holdout: rank policies by predicted claim rate, cut into
-- ten groups of equal EXPOSURE (not equal policy count, because a policy
-- insured for one month should not weigh the same as one insured all year),
-- then compare actual and expected claims in each group.
--
-- A good model shows (1) a steep rise in actual frequency from decile 1 to
-- 10, which is segmentation, and (2) A/E close to 1.00 in every decile,
-- which is calibration.

CREATE OR REPLACE TABLE lift_by_decile AS
WITH holdout AS (
    SELECT IDpol, Exposure, ClaimNb, pred_claims, pred_rate
    FROM model_data
    WHERE is_holdout
),
ranked AS (
    SELECT
        *,
        SUM(Exposure) OVER (ORDER BY pred_rate, IDpol
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            / SUM(Exposure) OVER () AS cum_exposure_share
    FROM holdout
),
deciled AS (
    SELECT *, LEAST(CAST(CEIL(cum_exposure_share * 10) AS INTEGER), 10) AS decile
    FROM ranked
)
SELECT
    decile,
    COUNT(*)                          AS policies,
    SUM(Exposure)                     AS exposure_years,
    SUM(ClaimNb)                      AS actual_claims,
    SUM(pred_claims)                  AS expected_claims,
    SUM(ClaimNb) / SUM(pred_claims)   AS actual_to_expected,
    SUM(ClaimNb) / SUM(Exposure)      AS actual_freq,
    SUM(pred_claims) / SUM(Exposure)  AS predicted_freq
FROM deciled
GROUP BY decile
ORDER BY decile;
