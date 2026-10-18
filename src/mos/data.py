"""Builds the Project B model table (PROTOCOL.md).

One row per (station, valid_utc) for every top-of-hour UTC valid time of local day T
(America/Chicago), T in 2021-04-01..2026-09-30. Issue = 18:00 local on T-1.
Columns
  y            observed temperature (C), ASOS METAR hh:53 -> valid hh+1:00
  g_* / e_*    GFS / ECMWF point forecasts at the city point, vintage per the load-project rule
               (previous_day1 if floor6(v-24h)+delay <= issue, else previous_day2)
  ob_last      latest observation with valid <= issue-1h; g_last / e_last forecast for that time
  ob_td_last   dew point of that latest observation
  ob_prev      observation at the same UTC hour one day earlier if known at the issue, else two
  ob_max1      max observation of local day T-1 known at the issue
  mav_t        NWS MAV 12Z run of T-1 temperature (3-hourly valid times only; baseline only)
  mav18_t      NWS MAV 18Z run of T-1 (released ~22Z, before the issue; amendment B-1)
Cached at data/processed/table.parquet.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from .stations import CITY_STATION, STATION_CITY

TZ = "America/Chicago"
P1 = Path.home() / "ML_forecast/data/processed"
DELAY = {"gfs_global": 6, "ecmwf_ifs025": 8}
VARS = {"temperature_2m": "t", "dew_point_2m": "td", "cloud_cover": "cc", "shortwave_radiation": "sw",
        "wind_speed_10m": "ws"}
CACHE = Path("data/processed/table.parquet")


def f2c(f):
    return (f - 32) * 5 / 9


def obs():
    rows = [pd.read_csv(f) for f in sorted(glob.glob("data/raw/asos/*.csv"))]
    a = pd.concat(rows, ignore_index=True)
    a["valid"] = pd.to_datetime(a["valid"]).dt.tz_localize("UTC")
    a = a[a["valid"].dt.minute.between(45, 59)]                 # routine report near hh:53
    a["v"] = a["valid"].dt.ceil("h")
    a["t"] = f2c(a["tmpf"]); a["td"] = f2c(a["dwpf"])
    a = a.dropna(subset=["t"]).sort_values("valid").drop_duplicates(["station", "v"], keep="last")
    return a.set_index(["station", "v"])[["t", "td"]]


def mav():
    rows = [pd.read_csv(f, usecols=["runtime", "ftime", "station", "tmp", "n_x"]) for f in sorted(glob.glob("data/raw/mav/*.csv"))]
    m = pd.concat(rows, ignore_index=True)
    m["runtime"] = pd.to_datetime(m["runtime"]).dt.tz_localize("UTC")
    m["v"] = pd.to_datetime(m["ftime"]).dt.tz_localize("UTC")
    m = m[m["runtime"].dt.hour.isin([12, 18])]
    m["station"] = m["station"].str[1:]
    m["mav_t"] = f2c(m["tmp"]); m["mav_nx"] = f2c(m["n_x"])
    return m.drop_duplicates(["station", "runtime", "v"]).set_index(["station", "runtime", "v"])[["mav_t", "mav_nx"]]


def vintage(tab, city, v, issue, model, var):
    s1 = tab[f"{var}_previous_day1"].xs(city, level="city") if f"{var}_previous_day1" in tab else None
    if s1 is None:
        return np.full(len(v), np.nan)
    s2 = tab[f"{var}_previous_day2"].xs(city, level="city")
    d = pd.Timedelta(hours=DELAY[model])
    ok1 = (v - pd.Timedelta(hours=24)).floor("6h") + d <= issue
    ok2 = (v - pd.Timedelta(hours=48)).floor("6h") + d <= issue
    a1, a2 = s1.reindex(v).values, s2.reindex(v).values
    out = np.where(ok1 & ~np.isnan(a1), a1, a2)
    return np.where(ok2 | (ok1 & ~np.isnan(a1)), out, np.nan)


def build(start="2021-04-01", end="2026-09-30"):
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    O = obs()
    M = mav()
    G = pd.read_parquet(P1 / "weather_fcst_gfs_global.parquet")
    E = pd.read_parquet(P1 / "weather_fcst_ecmwf_ifs025.parquet")
    days = pd.date_range(start, end, freq="D")
    parts = []
    for st, city in STATION_CITY.items():
        vs, Ts = [], []
        for T in days:
            a = T.tz_localize(TZ).tz_convert("UTC"); b = (T + pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
            r = pd.date_range(a, b, freq="h", inclusive="left")
            vs.append(r); Ts.append(np.repeat(T, len(r)))
        v = pd.DatetimeIndex(np.concatenate([x.values for x in vs])).tz_localize("UTC")
        T = pd.DatetimeIndex(np.concatenate(Ts))
        issue = pd.DatetimeIndex((T - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).tz_localize(TZ)).tz_convert("UTC")
        d = pd.DataFrame({"station": st, "city": city, "T": T, "v": v, "issue": issue})
        ost = O.xs(st, level="station")
        d["y"] = ost["t"].reindex(v).values
        for model, tab, p in [("gfs_global", G, "g"), ("ecmwf_ifs025", E, "e")]:
            for var, short in VARS.items():
                d[f"{p}_{short}"] = vintage(tab, city, v, issue, model, var)
        # observations known at the issue
        last = issue - pd.Timedelta(hours=1)
        tser = ost["t"].sort_index()
        pos = tser.index.searchsorted(last, side="right") - 1
        d["ob_last"] = tser.values[pos]
        d["ob_last_age"] = (last - tser.index[pos]).total_seconds() / 3600
        d.loc[d["ob_last_age"] > 6, "ob_last"] = np.nan
        tdser = ost["td"].sort_index()
        d["ob_td_last"] = tdser.values[pos]
        d.loc[d["ob_last_age"] > 6, "ob_td_last"] = np.nan
        lastv = pd.DatetimeIndex(last)
        d["g_last"] = vintage(G, city, lastv, issue, "gfs_global", "temperature_2m")
        d["e_last"] = vintage(E, city, lastv, issue, "ecmwf_ifs025", "temperature_2m")
        p1 = v - pd.Timedelta(days=1)
        prev = np.where(p1 <= last, ost["t"].reindex(p1).values, ost["t"].reindex(v - pd.Timedelta(days=2)).values)
        d["ob_prev"] = prev
        # max of T-1 up to the issue
        tm1 = (T - pd.Timedelta(days=1))
        mx = {}
        for t0 in np.unique(tm1):
            t0 = pd.Timestamp(t0)
            a = t0.tz_localize(TZ).tz_convert("UTC"); b = (t0 + pd.Timedelta(hours=17)).tz_localize(TZ).tz_convert("UTC")
            w = tser[(tser.index >= a) & (tser.index <= b)]
            mx[t0] = w.max() if len(w) else np.nan
        d["ob_max1"] = tm1.map(mx).values
        rt = pd.DatetimeIndex((T - pd.Timedelta(days=1)).tz_localize("UTC") + pd.Timedelta(hours=12))
        mi = pd.MultiIndex.from_arrays([np.repeat(st, len(d)), rt, v])
        mm = M.reindex(mi)
        d["mav_t"] = mm["mav_t"].values; d["mav_nx"] = mm["mav_nx"].values
        # 18Z run (released ~22Z): available before the 23Z (CDT) / 00Z (CST) issue; stronger reference
        mm18 = M.reindex(pd.MultiIndex.from_arrays([np.repeat(st, len(d)), rt + pd.Timedelta(hours=6), v]))
        d["mav18_t"] = mm18["mav_t"].values
        parts.append(d)
        print(st, len(d), "y missing", round(d["y"].isna().mean(), 4), flush=True)
    out = pd.concat(parts, ignore_index=True)
    lv = out["v"].dt.tz_convert(TZ)
    out["hour"] = lv.dt.hour; out["doy"] = out["T"].dt.dayofyear; out["lead"] = (out["v"] - out["issue"]).dt.total_seconds() / 3600
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    return out


if __name__ == "__main__":
    t = build()
    print(t.shape)
    print(t.groupby(t["T"].dt.year)[["y", "g_t", "e_t", "g_td", "ob_last", "ob_prev", "mav_t", "mav_nx"]].apply(lambda x: x.notna().mean().round(3)))
