# Day-ahead temperature post-processing (MOS) for Texas load cities (Project B)

Status: **complete (holdout opened once, see [MODEL_CARD.md](MODEL_CARD.md))**; public data.
Not affiliated with NOAA/NWS, ERCOT or Vistra. **No public leaderboard exists for this task.**

Corrects the free GFS / ECMWF temperature forecasts used by the ERCOT load project (`~/ML_forecast`)
against ASOS airport observations, with the operational NWS GFS MOS (MAV) as a reference.
Frozen protocol: [PROTOCOL.md](PROTOCOL.md).

## Layout
```
src/mos/download.py     ASOS hourly METARs + NWS MAV bulletins (Iowa Environmental Mesonet)
src/mos/data.py         model table: vintage-safe forecasts, observations known at the issue, MAV
src/mos/features.py     blend, error at the issue, rolling biases, spread
src/mos/models.py       raw, rolling bias, linear MOS, LightGBM residual, average
src/mos/backtest.py     rolling-origin backtest, registry; compare.py paired bootstrap; dailymax.py
```

## Reproduce
```bash
PY=~/ML_forecast/.venv/bin/python
$PY -m src.mos.download && $PY -m src.mos.data
$PY -m pytest -q tests
$PY -m src.mos.backtest R2_raw_blend G4_lgbm_resid_recent_err
$PY -m src.mos.compare G4_lgbm_resid_recent_err --mav18
```
