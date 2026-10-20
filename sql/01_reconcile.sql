-- 01_reconcile.sql
-- Does the claim count on each policy agree with the claim records in the
-- severity table? A pure premium is frequency x severity, so both tables
-- have to describe the same claims or the two models will not add up.

CREATE OR REPLACE TABLE claims_per_policy AS
SELECT
    IDpol,
    COUNT(*)         AS n_claim_records,
    SUM(ClaimAmount) AS total_claim_amount
FROM claims_raw
GROUP BY IDpol;

CREATE OR REPLACE TABLE reconciliation AS
SELECT
    p.IDpol,
    p.ClaimNb                                   AS claimnb_recorded,
    COALESCE(c.n_claim_records, 0)              AS n_claim_records,
    COALESCE(c.total_claim_amount, 0)           AS total_claim_amount,
    p.ClaimNb <> COALESCE(c.n_claim_records, 0) AS is_mismatch
FROM policies_raw AS p
LEFT JOIN claims_per_policy AS c USING (IDpol);

-- Claim records that point at a policy that is not in the policy table.
CREATE OR REPLACE TABLE orphan_claims AS
SELECT s.*
FROM claims_raw AS s
ANTI JOIN policies_raw AS p USING (IDpol);

-- One row per kind of agreement or disagreement.
CREATE OR REPLACE TABLE reconciliation_summary AS
SELECT
    CASE
        WHEN NOT is_mismatch AND claimnb_recorded = 0 THEN 'match: no claims'
        WHEN NOT is_mismatch                          THEN 'match: has claims'
        WHEN n_claim_records = 0                      THEN 'mismatch: count but no claim records'
        WHEN claimnb_recorded > n_claim_records       THEN 'mismatch: count above claim records'
        ELSE                                               'mismatch: count below claim records'
    END                   AS status,
    COUNT(*)              AS policies,
    SUM(claimnb_recorded) AS claims_on_policy_table,
    SUM(n_claim_records)  AS claim_records
FROM reconciliation
GROUP BY ALL
ORDER BY status;
