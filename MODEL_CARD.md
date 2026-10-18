# Model card — Project B: day-ahead temperature post-processing (MOS) for 11 Texas load cities

Written 2026-10-14; not operational meteorology experience. Public data only (Open-Meteo,
Iowa Environmental Mesonet); not affiliated with NOAA/NWS, ERCOT or Vistra. **No public leaderboard
exists for this task**; the NWS MAV comparison is an internal comparison on 11 stations and one
period. Protocol: [PROTOCOL.md](PROTOCOL.md) (amendment B-1); locked choice: `configs/final_B.yaml`.

## Task
Issue 18:00 America/Chicago on T-1 (same as the ERCOT load project); predict the ASOS hourly
temperature at every hour of local day T at HOU, DAL, FTW, SAT, AUS, CRP, MFE, MAF, ABI, TYR, SPS.
Inputs: GFS / ECMWF point forecasts at the load project's city points (vintage-safe), station
observations up to issue - 1 h. Primary metric: load-weighted hourly MAE (deg C).

## Selected model: G4
Global LightGBM (L1 loss) on the residual *observation - mean(GFS, ECMWF)*; features: both models'
temperature, dew point, cloud, radiation, wind, model spread, the error at the issue time,
yesterday's same-hour error, 7- and 30-day station x hour bias, last observation and dew-point
depression, hour, season, station id, lead time. Trained from 2021-04 (expanding, refit quarterly).

## Results (validation = 2024-07..2025-06, holdout = 2025-07..2026-09, opened once)

| Run | Validation wMAE | Holdout wMAE | Holdout RMSE |
|---|---|---|---|
| R0 raw GFS | 1.755 | 1.711 | 2.330 |
| R1 raw ECMWF | 1.566 | — | — |
| R2 raw blend (the load project's input) | 1.425 | 1.320 | 1.899 |
| B1 blend - 30-day station x hour bias | 1.317 | 1.175 | 1.690 |
| L1 linear MOS per station x hour (2021+) | 1.318 | — | — |
| L2 linear MOS (2024+) | 1.290 | 1.140 | 1.636 |
| G1 LightGBM residual | 1.162 | 1.019 | 1.518 |
| G2 = G1 trained 2024+ only | 1.187 | — | — |
| G3 = G1, 3 seeds | 1.158 | — | — |
| **G4 = G1 + recent-error features** | **1.140** | **1.004** | **1.506** |
| G5 = larger G4 | 1.144 | — | — |
| E1 average G4 + L2 | 1.173 | — | — |

Paired block bootstrap, G4 minus other (holdout): G1 -0.014 [-0.019, -0.010]; L2 -0.136
[-0.152, -0.119]; raw blend -0.316 [-0.358, -0.279] C (-24%).

**NWS GFS MOS (MAV), 3-hourly valid times only:**

| | Validation | Holdout |
|---|---|---|
| G4 vs MAV 12Z run | -0.159 C [-0.199, -0.125] | -0.195 C [-0.232, -0.162] |
| G4 vs MAV 18Z run (released ~22Z, before the issue) | -0.112 C [-0.152, -0.077] | -0.147 C [-0.187, -0.109] |
| G4 / MAV 18Z wMAE | — | 1.004 / 1.151 C |

Caveats on the MAV comparison: (1) G4 uses observations to issue - 1 h and two global models,
including ECMWF, which MAV does not; (2) MAV is defined at the station and was not designed for our
weighting; (3) MAV is a fixed multiple-regression product, while G4 is refit on these stations.
Daytime max (13Z-01Z): G4 1.04 C vs MAV 1.37 C on holdout, **but our "observed max" is the max of
hourly reports, which is lower than the true daily max that MAV targets**, so this comparison is
biased in G4's favour and is reported only for completeness.

## Findings
- Raw GFS is warm-biased at these points (+0.6 C); ECMWF is better; the 50/50 blend is better than
  both; a simple 30-day bias removal gets most of the way to MAV.
- The non-linear residual model is the main gain (G1 vs L2: ~0.12 C on holdout); recent-error
  features add a small but consistent 0.014-0.022 C.
- More training data beat "consistent" data: training from 2021 (GFS-only blend) beat 2024+ only.
- All models score better in the holdout year than in validation (the raw blend too), so levels
  across periods are not comparable; rankings are.

## Limits
- City points vs airports: part of the gain is the location offset that any MOS learns.
- Hourly METAR temperature has whole-F resolution (~0.3 C rounding).
- The downstream effect on the load project's forecast has not been measured (would need a dated
  amendment there).
