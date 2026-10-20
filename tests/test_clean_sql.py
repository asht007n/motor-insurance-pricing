"""The reconciliation SQL on a toy database with every kind of mismatch.

The pinned CASdatasets copy reconciles perfectly, so these cases only show up
in older copies of freMTPL2 (OpenML, Kaggle). The rules still need testing.
"""

import duckdb
import pandas as pd
import pytest

from mtpl import data

POLICIES = pd.DataFrame({
    "IDpol":    [1, 2, 3, 4, 5, 6],
    "ClaimNb":  [0, 1, 2, 0, 6, 0],
    "Exposure": [1.0, 0.5, 1.0, 0.8, 0.3, 1.5],
    "Region":   ["Centre"] * 6,
})
CLAIMS = pd.DataFrame({
    # policy 2: count 1, no records; policy 3: count 2, one record;
    # policy 4: count 0, one record; policy 5: six records; 99: orphan
    "IDpol":       [3, 4, 5, 5, 5, 5, 5, 5, 99],
    "ClaimAmount": [1200.0, 800.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 50.0],
})


@pytest.fixture
def con():
    c = duckdb.connect()
    c.register("p", POLICIES)
    c.register("s", CLAIMS)
    c.execute("CREATE TABLE policies_raw AS SELECT * FROM p")
    c.execute("CREATE TABLE claims_raw AS SELECT * FROM s")
    data.clean_in_db(c)
    yield c
    c.close()


def test_claim_counts_follow_the_severity_table_then_cap(con):
    got = con.sql("SELECT IDpol, ClaimNb_reconciled, ClaimNb FROM policies_clean ORDER BY IDpol").df()
    assert got.ClaimNb_reconciled.tolist() == [0, 0, 1, 1, 6, 0]
    assert got.ClaimNb.tolist() == [0, 0, 1, 1, 4, 0]


def test_exposure_is_capped_at_one_year(con):
    got = con.sql("SELECT Exposure FROM policies_clean ORDER BY IDpol").df().Exposure.tolist()
    assert got == [1.0, 0.5, 1.0, 0.8, 0.3, 1.0]


def test_cleaning_log_counts_each_rule(con):
    log = con.sql("SELECT step_order, rows_affected FROM cleaning_log ORDER BY step_order").df()
    affected = dict(zip(log.step_order, log.rows_affected))
    assert affected[2] == 1  # orphan claim record
    assert affected[3] == 3  # policies 2, 3 and 4 disagree with the severity table
    assert affected[4] == 1  # policy 5 capped
    assert affected[5] == 1  # policy 6 capped


def test_reconciliation_summary_names_each_mismatch(con):
    s = con.sql("SELECT status, policies FROM reconciliation_summary").df()
    got = dict(zip(s.status, s.policies))
    assert got["mismatch: count but no claim records"] == 1
    assert got["mismatch: count above claim records"] == 1
    assert got["mismatch: count below claim records"] == 1


def test_orphans_are_kept_out_of_the_claims_table(con):
    assert con.sql("SELECT COUNT(*) FROM claims_clean WHERE IDpol = 99").fetchone()[0] == 0


def test_scientific_notation_policy_id_is_parsed():
    ids = pd.Series(["99999", "1e+05", "100001"])
    assert data.parse_policy_id(ids).tolist() == [99999, 100000, 100001]
