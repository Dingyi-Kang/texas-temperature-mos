"""Paired comparison: python -m src.mos.compare A B [--holdout] [--mav]
(--mav: compare A against the NWS MAV on the 3-hourly valid times of A's predictions)."""
import glob, sys
import pandas as pd
from .metrics import daily_err, block_bootstrap_diff


def load(e, holdout=False):
    fs = [f for f in sorted(glob.glob(f"experiments/runs/*_{e}{'_HOLDOUT' if holdout else ''}/predictions.parquet"))
          if holdout or "HOLDOUT" not in f]
    return pd.read_parquet(fs[-1])


def main():
    h = "--holdout" in sys.argv
    a = sys.argv[1]; A = load(a, h)
    if "--mav18" in sys.argv:
        A = A[A["mav18_t"].notna() & A["y"].notna()]
        da, db, b = daily_err(A, "pred"), daily_err(A, "mav18_t"), "NWS_MAV18Z(3h)"
    elif "--mav" in sys.argv:
        A = A[A["mav_t"].notna() & A["y"].notna()]
        da, db, b = daily_err(A, "pred"), daily_err(A, "mav_t"), "NWS_MAV(3h)"
    else:
        b = sys.argv[2]; B = load(b, h)
        da, db = daily_err(A), daily_err(B)
    d, ci = block_bootstrap_diff(da, db)
    print(f"{a} - {b}: {d:+.3f} C, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], days {len(da)}")


if __name__ == "__main__":
    main()
