"""Leakage and alignment checks for Project B."""
import numpy as np
import pandas as pd
import pytest
from src.mos.data import vintage, DELAY, TZ


def test_vintage_uses_day2_when_day1_not_published():
    city = "dallas"
    v = pd.DatetimeIndex(["2025-07-02 23:00"], tz="UTC")        # 18:00 CDT on T
    issue = pd.DatetimeIndex(["2025-07-01 23:00"], tz="UTC")    # 18:00 CDT on T-1
    idx = pd.MultiIndex.from_tuples([(city, v[0])], names=["city", "time"])
    tab = pd.DataFrame({"temperature_2m_previous_day1": [1.0], "temperature_2m_previous_day2": [2.0]}, index=idx)
    # day1 run = floor6(v-24h) = 2025-07-01 18Z, + 6h = 07-02 00Z > issue -> day2
    assert vintage(tab, city, v, issue, "gfs_global", "temperature_2m")[0] == 2.0
    v2 = pd.DatetimeIndex(["2025-07-02 06:00"], tz="UTC")
    idx2 = pd.MultiIndex.from_tuples([(city, v2[0])], names=["city", "time"])
    tab2 = pd.DataFrame({"temperature_2m_previous_day1": [1.0], "temperature_2m_previous_day2": [2.0]}, index=idx2)
    # day1 run = 07-01 06Z + 6h = 12Z <= 23Z -> day1
    assert vintage(tab2, city, v2, issue, "gfs_global", "temperature_2m")[0] == 1.0


@pytest.fixture(scope="module")
def table():
    from src.mos.features import add
    return add()


def test_targets_are_local_day_T(table):
    loc = table["v"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    assert (loc == table["T"]).all()
    n = table.groupby(["station", "T"]).size()
    assert set(n.unique()) <= {23, 24, 25}


def test_issue_and_leads(table):
    assert (table["lead"] >= 6).all() and (table["lead"] <= 31).all()


def test_rolling_bias_only_past(table):
    t = table[(table["station"] == "DAL") & (table["hour"] == 15)].sort_values("T").reset_index(drop=True)
    i = t.index[t["bias30"].notna()][40]
    T = t.loc[i, "T"]
    w = t[(t["T"] >= T - pd.Timedelta(days=31)) & (t["T"] <= T - pd.Timedelta(days=2))]
    assert np.isclose((w["bl"] - w["y"]).mean(), t.loc[i, "bias30"], equal_nan=False)
