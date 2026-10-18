"""Rolling-origin backtest for Project B (PROTOCOL.md).
Usage: python -m src.mos.backtest EXP [EXP ...]            # validation
       python -m src.mos.backtest --holdout EXP [...]     # needs configs/final_B.yaml, once per exp
"""
import argparse, datetime as dt, json, subprocess, sys, time
from pathlib import Path
import pandas as pd
from .features import add
from .metrics import summarize
from .experiments import EXPERIMENTS

REG = Path("experiments/registry.jsonl")
VAL_FOLDS = ["2024-07-01", "2024-10-01", "2025-01-01", "2025-04-01"]; VAL_END = "2025-06-30"
HOLD_FOLDS = ["2025-07-01", "2025-10-01", "2026-01-01", "2026-04-01", "2026-07-01"]; HOLD_END = "2026-09-30"
TRAIN_START = "2021-04-01"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def folds(starts, end):
    ends = [pd.Timestamp(s) - pd.Timedelta(days=1) for s in starts[1:]] + [pd.Timestamp(end)]
    return [(pd.Timestamp(s), e) for s, e in zip(starts, ends)]


def run(df, spec, fl):
    preds = []
    for i, (S, E) in enumerate(fl):
        tr = df[(df["T"] >= TRAIN_START) & (df["T"] <= S - pd.Timedelta(days=2)) & df["y"].notna()]
        te = df[(df["T"] >= S) & (df["T"] <= E)].copy()
        assert tr["v"].max() <= te["issue"].min() - pd.Timedelta(hours=1)
        m = spec["model"]().fit(tr)
        te["pred"] = m.predict(te); te["fold"] = i
        preds.append(te[["station", "city", "T", "v", "hour", "lead", "y", "pred", "mav_t", "mav18_t", "mav_nx", "fold"]])
    return pd.concat(preds, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("exps", nargs="+"); ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args(argv)
    if a.holdout:
        import yaml
        cfg = Path("configs/final_B.yaml")
        if not cfg.exists():
            sys.exit("Holdout locked: create configs/final_B.yaml first.")
        allowed = yaml.safe_load(cfg.read_text())["holdout_set"]
        done = {json.loads(l)["experiment"] for l in REG.open() if json.loads(l)["phase"] == "holdout"} if REG.exists() else set()
        bad = [e for e in a.exps if e not in allowed or e in done]
        if bad:
            sys.exit(f"Not allowed on holdout: {bad}")
    df = add()
    fl = folds(HOLD_FOLDS, HOLD_END) if a.holdout else folds(VAL_FOLDS, VAL_END)
    REG.parent.mkdir(exist_ok=True)
    for name in a.exps:
        spec = EXPERIMENTS[name]; t0 = time.time()
        run_id = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}{'_HOLDOUT' if a.holdout else ''}"
        rec = {"run_id": run_id, "experiment": name, "phase": "holdout" if a.holdout else "validation",
               "hypothesis": spec["hypothesis"], "config": spec.get("config", {}), "git": git_commit()}
        try:
            p = run(df, spec, fl)
            rdir = Path("experiments/runs") / run_id; rdir.mkdir(parents=True, exist_ok=True)
            p.to_parquet(rdir / "predictions.parquet", index=False)
            m = summarize(p); rec.update(status="ok", metrics=m, seconds=round(time.time() - t0, 1))
            print(f"{name:34s} wMAE={m['wmae']:.3f}C MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} bias={m['bias']:+.3f} "
                  f"extreme={m['mae_extreme']:.3f} | 3h: {m.get('wmae_3h', float('nan')):.3f} vs MAV {m.get('wmae_3h_mav', float('nan')):.3f}", flush=True)
        except Exception as e:
            rec.update(status="failed", error=repr(e)); print(name, "FAILED", repr(e)); raise
        finally:
            with REG.open("a") as f:
                f.write(json.dumps(rec, default=float) + "\n")


if __name__ == "__main__":
    main()
