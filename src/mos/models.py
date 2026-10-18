"""Project B models. Each: fit(train_df) -> self; predict(df) -> deg C array."""
import numpy as np
import pandas as pd


class Col:
    """Raw column (R0 GFS, R1 ECMWF with GFS fallback, R2 blend)."""
    def __init__(self, col, fallback=None):
        self.col, self.fb = col, fallback
    def fit(self, tr): return self
    def predict(self, df):
        v = df[self.col].values
        return np.where(np.isnan(v), df[self.fb].values, v) if self.fb else v


class RollingBias:
    """Blend minus the 30-day station x hour mean error (feature bias30, issue-safe)."""
    def fit(self, tr): return self
    def predict(self, df):
        return df["bl"].values - np.nan_to_num(df["bias30"].values)


class LinearMOS:
    """Ridge per station x local-hour group on the blend, recent error and seasonality."""
    def __init__(self, feats=("bl", "err_last", "ob_prev", "sin", "cos"), hour_groups=24, alpha=1.0, start=None):
        self.feats, self.hg, self.alpha, self.start = list(feats), hour_groups, alpha, start
    def _key(self, df):
        return df["station"] + "_" + (df["hour"] * self.hg // 24).astype(str)
    def fit(self, tr):
        from sklearn.linear_model import Ridge
        tr = tr if self.start is None else tr[tr["T"] >= self.start]
        tr = tr.dropna(subset=self.feats + ["y"])
        self.m = {}
        for k, g in tr.groupby(self._key(tr)):
            X = g[self.feats].values
            self.m[k] = Ridge(alpha=self.alpha).fit(X, g["y"].values - g["bl"].values)
        return self
    def predict(self, df):
        out = df["bl"].values.copy()
        X = df[self.feats].copy()
        X = X.fillna({"err_last": 0.0, "ob_prev": df["bl"]}).values
        for k, idx in pd.Series(np.arange(len(df))).groupby(self._key(df).values):
            if k in self.m:
                ok = ~np.isnan(X[idx.values]).any(1)
                out[idx.values[ok]] += self.m[k].predict(X[idx.values][ok])
        return out


class LGBMResid:
    """Global LightGBM on the residual y - blend (L1 loss), station id as categorical."""
    FEATS = ["bl", "g_t", "e_t", "g_td", "e_td", "g_cc", "e_cc", "g_sw", "e_sw", "g_ws", "e_ws", "ob_last", "err_last",
             "ob_prev", "ob_max1", "bias30", "hour", "sin", "cos", "sid", "lead"]
    def __init__(self, feats=None, params=None, start=None, seeds=(0,)):
        self.feats = feats or self.FEATS
        self.params = dict(objective="l1", learning_rate=0.03, n_estimators=1500, num_leaves=63, min_child_samples=50,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=16, verbose=-1)
        self.params.update(params or {})
        self.start, self.seeds = start, seeds
    def fit(self, tr):
        import lightgbm as lgb
        tr = tr if self.start is None else tr[tr["T"] >= self.start]
        tr = tr[tr["y"].notna() & tr["bl"].notna()]
        self.ms = [lgb.LGBMRegressor(**{**self.params, "random_state": s}).fit(
            tr[self.feats], tr["y"] - tr["bl"], categorical_feature=["sid"]) for s in self.seeds]
        return self
    def predict(self, df):
        return df["bl"].values + np.mean([m.predict(df[self.feats]) for m in self.ms], axis=0)


class Average:
    def __init__(self, *members): self.members = members
    def fit(self, tr):
        for m in self.members: m.fit(tr)
        return self
    def predict(self, df): return np.mean([m.predict(df) for m in self.members], axis=0)
