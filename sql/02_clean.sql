-- 02_clean.sql
-- Build the modelling table and log every rule with the rows it touched.
-- Thresholds come from src/mtpl/config.py via SET VARIABLE.

-- Rule 1: the severity table is the source of truth for claim counts,
--         because every row there carries an amount.
-- Rule 2: cap claim counts at max_claims (likely duplicated or fleet records).
-- Rule 3: cap exposure at max_exposure years (data is described as one year).
CREATE OR REPLACE TABLE policies_clean AS
SELECT
    p.IDpol,
    r.n_claim_records                                   AS ClaimNb_reconciled,
    LEAST(r.n_claim_records, getvariable('max_claims')) AS ClaimNb,
    p.Exposure                                          AS Exposure_raw,
    LEAST(p.Exposure, getvariable('max_exposure'))      AS Exposure,
    r.total_claim_amount                                AS ClaimAmount,
    p.* EXCLUDE (IDpol, ClaimNb, Exposure)
FROM policies_raw AS p
JOIN reconciliation AS r USING (IDpol);

-- Individual claims for the severity model (orphans removed, amounts uncapped).
CREATE OR REPLACE TABLE claims_clean AS
SELECT s.*
FROM claims_raw AS s
SEMI JOIN policies_raw AS p USING (IDpol);

-- State of the data after each step.
CREATE OR REPLACE TABLE cleaning_log AS
WITH raw AS (
    SELECT COUNT(*) AS policies, SUM(ClaimNb) AS claims, SUM(Exposure) AS exposure
    FROM policies_raw
)
SELECT 1 AS step_order,
       'Raw policy table' AS step,
       NULL::BIGINT AS rows_affected,
       policies, claims, exposure AS exposure_years
FROM raw
UNION ALL
SELECT 2,
       'Drop claim records with no matching policy',
       (SELECT COUNT(*) FROM orphan_claims),
       policies, claims, exposure
FROM raw
UNION ALL
SELECT 3,
       'Set ClaimNb to the number of claim records',
       (SELECT COUNT(*) FILTER (WHERE is_mismatch) FROM reconciliation),
       policies,
       (SELECT SUM(n_claim_records) FROM reconciliation),
       exposure
FROM raw
UNION ALL
SELECT 4,
       'Cap ClaimNb at ' || getvariable('max_claims'),
       COUNT(*) FILTER (WHERE ClaimNb_reconciled > ClaimNb),
       COUNT(*), SUM(ClaimNb), SUM(Exposure_raw)
FROM policies_clean
UNION ALL
SELECT 5,
       'Cap Exposure at ' || getvariable('max_exposure') || ' year',
       COUNT(*) FILTER (WHERE Exposure_raw > Exposure),
       COUNT(*), SUM(ClaimNb), SUM(Exposure)
FROM policies_clean;
