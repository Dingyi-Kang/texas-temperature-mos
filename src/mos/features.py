"""Derived, issue-safe features on top of data.build().

  bl          blend = mean(GFS, ECMWF) where ECMWF exists, else GFS (load project's blend)
  bl_last     blend at the latest known observation time; err_last = ob_last - bl_last
  bias30      mean(bl - y) over the same station and local hour on days T-31..T-2 (complete days
              known at the issue; >= 10 days required)
  bias7       same over T-8..T-2
  err_prev    ob_prev minus the blend issued for that time (same hour, latest known day)
  spread      |GFS - ECMWF|; dep_last = dew-point depression at the latest observation
  sin/cos     day-of-year harmonics
"""
import numpy as np
import pandas as pd
from .data import build


def add(d=None):
    d = build() if d is None else d
    d = d.copy()
    d["bl"] = np.where(d["e_t"].isna(), d["g_t"], (d["g_t"] + d["e_t"]) / 2)
    d["bl_last"] = np.where(d["e_last"].isna(), d["g_last"], (d["g_last"] + d["e_last"]) / 2)
    d["err_last"] = d["ob_last"] - d["bl_last"]
    d["sin"] = np.sin(2 * np.pi * d["doy"] / 365.25); d["cos"] = np.cos(2 * np.pi * d["doy"] / 365.25)
    # rolling bias by station x local hour over complete days T-31..T-2
    e = d[["station", "hour", "T"]].copy(); e["err"] = d["bl"] - d["y"]
    daily = e.groupby(["station", "hour", "T"])["err"].mean()
    out = []
    for (st, h), s in daily.groupby(level=[0, 1]):
        s = s.droplevel([0, 1])
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        r = s.rolling(30, min_periods=10).mean().shift(2)               # window ends at T-2
        r7 = s.rolling(7, min_periods=3).mean().shift(2)
        out.append(pd.DataFrame({"station": st, "hour": h, "T": r.index, "bias30": r.values, "bias7": r7.values}))
    b = pd.concat(out)
    d = d.merge(b, on=["station", "hour", "T"], how="left")
    # error of the latest known same-hour forecast (v-1 day if observed before the issue, else v-2 days)
    k = d.set_index(["station", "v"])["bl"]
    last = d["issue"] - pd.Timedelta(hours=1)
    v1 = d["v"] - pd.Timedelta(days=1); v2 = d["v"] - pd.Timedelta(days=2)
    vp = v1.where(v1 <= last, v2)
    blp = k.reindex(pd.MultiIndex.from_arrays([d["station"], vp])).values
    d["err_prev"] = d["ob_prev"] - blp
    d["spread"] = (d["g_t"] - d["e_t"]).abs()
    d["dep_last"] = d["ob_last"] - d["ob_td_last"]
    d["sid"] = d["station"].astype("category").cat.codes
    return d
