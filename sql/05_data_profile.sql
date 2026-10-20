-- 05_data_profile.sql
-- Facts about the data quoted in the README and docs, computed here so each
-- one can be traced to a query. Output: data_profile(metric, value).

CREATE OR REPLACE TABLE data_profile AS
WITH
-- French bonus-malus falls 5% per claim-free year from 100, rounded down:
-- 100, 95, 90, 85, 80, 76, 72, 68, 64, 60, 57, 54, 51, 50. Values off this
-- ladder mean an at-fault claim at some point.
ladder AS (
    SELECT SUM(Exposure) FILTER (WHERE BonusMalus IN
               (50, 51, 54, 57, 60, 64, 68, 72, 76, 80, 85, 90, 95, 100))
           / SUM(Exposure) AS share
    FROM policies_clean
),
-- Policies sharing every one of the nine rating fields with another policy.
profiles AS (
    SELECT
        COUNT(*)                     AS n,
        SUM(Exposure)                AS expo,
        MAX(IDpol) - MIN(IDpol)      AS id_span
    FROM policies_clean
    GROUP BY VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Area, Density, Region
    HAVING COUNT(*) > 1
),
-- The most common exact claim amounts (flat-rate settlements).
amount_counts AS (
    SELECT ClaimAmount, COUNT(*) AS n,
           ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC, ClaimAmount) AS rk
    FROM claims_clean
    GROUP BY ClaimAmount
),
sev AS (
    SELECT
        COUNT(*)                                        AS n,
        SUM(ClaimAmount)                                AS total,
        MEDIAN(ClaimAmount)                             AS med,
        AVG(ClaimAmount)                                AS mean,
        QUANTILE_CONT(ClaimAmount, 0.99)                AS p99,
        MAX(ClaimAmount)                                AS mx,
        COUNT(*) FILTER (WHERE ClaimAmount > 100000)    AS n_over_100k,
        SUM(GREATEST(ClaimAmount - 100000, 0))          AS excess_over_100k
    FROM claims_clean
)
SELECT 'bm_exposure_share_on_claim_free_ladder' AS metric, share AS value FROM ladder
-- Area (A..F) against log density: why Area is left out of the model.
UNION ALL SELECT 'corr_area_code_log_density', CORR(ASCII(Area), LN(Density)) FROM policies_clean
UNION ALL SELECT 'repeated_profile_groups', COUNT(*) FROM profiles
UNION ALL SELECT 'repeated_profile_policies', SUM(n) FROM profiles
UNION ALL SELECT 'repeated_profile_share_groups_ids_within_100', AVG((id_span < 100)::INT) FROM profiles
UNION ALL SELECT 'repeated_profile_median_id_span', MEDIAN(id_span) FROM profiles
UNION ALL SELECT 'repeated_profile_mean_exposure', SUM(expo) / SUM(n) FROM profiles
UNION ALL SELECT 'all_policies_mean_exposure', AVG(Exposure) FROM policies_clean
UNION ALL SELECT 'claims', n FROM sev
UNION ALL SELECT 'claim_amount_median', med FROM sev
UNION ALL SELECT 'claim_amount_mean', mean FROM sev
UNION ALL SELECT 'claim_amount_p99', p99 FROM sev
UNION ALL SELECT 'claim_amount_max', mx FROM sev
UNION ALL SELECT 'claims_on_5_most_common_amounts_share',
       (SELECT SUM(n) FROM amount_counts WHERE rk <= 5) / (SELECT n FROM sev)
UNION ALL SELECT 'cost_on_5_most_common_amounts_share',
       (SELECT SUM(n * ClaimAmount) FROM amount_counts WHERE rk <= 5) / (SELECT total FROM sev)
UNION ALL SELECT 'claims_over_100k', n_over_100k FROM sev
UNION ALL SELECT 'cost_share_above_100k_excess', excess_over_100k / total FROM sev;
