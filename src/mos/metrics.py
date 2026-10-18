"""Project B metrics (PROTOCOL.md). Errors in deg C."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

W = json.loads(Path("configs/city_weights.json").read_text())


def wmae(df, col="pred"):
    w = df["city"].map(W)
    return float((w * (df[col] - df["y"]).abs()).sum() / w.sum())


def summarize(df, col="pred"):
    df = df[df["y"].notna() & df[col].notna()]
    e = df[col] - df["y"]
    m = {"wmae": wmae(df, col), "mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
         "bias": float(e.mean()), "n": int(len(df))}
    ext = (df["y"] >= 38) | (df["y"] <= -5)
    m["mae_extreme"] = float(e[ext].abs().mean()) if ext.any() else float("nan")
    m["n_extreme"] = int(ext.sum())
    for st, g in df.groupby("station"):
        m[f"mae_{st}"] = float((g[col] - g["y"]).abs().mean())
    sub = df[df["mav_t"].notna()]
    if len(sub):
        m["wmae_3h"] = wmae(sub, col); m["wmae_3h_mav"] = wmae(sub, "mav_t")
    if "mav18_t" in df:
        s18 = df[df["mav18_t"].notna()]
        if len(s18):
            m["wmae_3h18"] = wmae(s18, col); m["wmae_3h_mav18"] = wmae(s18, "mav18_t")
    return m


def daily_err(df, col="pred"):
    df = df[df["y"].notna() & df[col].notna()]
    w = df["city"].map(W)
    return ((df[col] - df["y"]).abs() * w).groupby(df["T"]).sum() / w.groupby(df["T"]).sum()


def block_bootstrap_diff(a, b, block=7, n=2000, seed=0):
    idx = a.index.intersection(b.index)
    d = (a.reindex(idx) - b.reindex(idx)).values
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(d) / block)); stats = []
    for _ in range(n):
        st = rng.integers(0, len(d) - block + 1, nb)
        stats.append(np.concatenate([d[s:s + block] for s in st])[:len(d)].mean())
    return float(d.mean()), (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))
