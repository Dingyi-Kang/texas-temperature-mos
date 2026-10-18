"""Daytime max (NWS definition 7am-7pm LST = 13Z-01Z) vs the MAV max forecast (n_x
at 00Z). Max of hourly predictions vs observed max of hourly observations, 13Z..00Z. The nighttime min
(01Z-14Z) starts on local day T-1, outside our forecast range, so it is not compared.
Usage: python -m src.mos.dailymax EXP [--holdout]"""
import sys
import numpy as np
import pandas as pd
from .compare import load
from .metrics import W


def main():
    p = load(sys.argv[1], "--holdout" in sys.argv)
    for col in ["mav_nx"]:
        rows = []
        for (st, T), g in p.groupby(["station", "T"]):
            g = g.set_index("v").sort_index()
            nx = g["mav_nx"].dropna()
            nx = nx[nx.index.hour == 0]
            if not len(nx):
                continue
            v = nx.index[0]; w = g.loc[v - pd.Timedelta(hours=11): v]
            if len(w) < 12 or w["y"].isna().any():
                continue
            rows.append((st, g["city"].iloc[0], T, w["y"].max(), w["pred"].max(), nx.iloc[0]))
        d = pd.DataFrame(rows, columns=["station", "city", "T", "obs", "model", "mav"])
        w = d["city"].map(W)
        f = lambda c: float(((d[c] - d["obs"]).abs() * w).sum() / w.sum())
        print(f"{sys.argv[1]}: daytime max, days x stations = {len(d)}; weighted MAE model {f('model'):.3f} C, MAV 12Z {f('mav'):.3f} C; "
              f"bias model {float((d['model'] - d['obs']).mean()):+.3f}, MAV {float((d['mav'] - d['obs']).mean()):+.3f}")


if __name__ == "__main__":
    main()
