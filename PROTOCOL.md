# PROTOCOL — Project B: day-ahead temperature forecast post-processing (MOS) for Texas load cities

Frozen 2026-10-14 before any model was fit (only the raw files'
structure was inspected). Changes only as dated amendments at the end. Personal study project on
public data; not affiliated with NOAA/NWS, ERCOT or Vistra.

## Question
Can statistical post-processing of free global NWP forecasts (GFS, ECMWF IFS 0.25 deg via
Open-Meteo, the inputs of the ERCOT load project `~/ML_forecast`) reduce the error of the
day-ahead hourly temperature at the 11 load-project cities, and how does it compare with the
**operational NWS GFS MOS (MAV)**, a professional statistical product?

## Data
- **Observations (truth):** ASOS routine hourly METARs (report_type 3) from the Iowa Environmental
  Mesonet, stations HOU, DAL, FTW, SAT, AUS, CRP, MFE, MAF, ABI, TYR, SPS (nearest major airport to
  each load-project city point). Reports at hh:53 are assigned to valid time hh+1:00 UTC.
  Temperature in deg C (METAR tmpf converted; whole-deg-F resolution, ~0.3 C rounding noise).
- **NWP forecasts:** Open-Meteo previous-runs archive at the *city* points (as used by the load
  project; the city-station offset is part of what MOS corrects). Vintage rule as in the load
  project (reports/weather_semantics.md there): `previous_dayN` = run floor6(v - 24N h), usable
  after a publication delay of 6 h (GFS) / 8 h (ECMWF); day1 used when available at the issue,
  else day2. GFS temperature from 2021-03-25; GFS dew point / cloud / radiation / wind and all
  ECMWF fields from 2024.
- **NWS MAV (baseline only, never a feature):** IEM archive of GFS MOS bulletins; the 12Z run of
  day T-1 (released ~16Z, before the issue) is used: 3-hourly temperature and daytime max /
  nighttime min (`n_x`).

## Task
- **Issue:** 18:00 America/Chicago on day T-1 (same as the load project).
- **Target:** observed temperature at every top-of-hour UTC valid time of local day T at each of
  the 11 stations (23/24/25 hours).
- **Information allowed at the issue:** NWP vintages per the rule above; station observations with
  valid time <= issue - 1 h (METARs are real-time); calendar. Nothing from the future.

## Splits
- Training from 2021-04-01. Models using ECMWF or non-temperature GFS fields train from 2024-01-01
  (or fall back to the GFS-temperature-only model where those are missing).
- **Validation:** 4 quarterly folds, target days 2024-07-01..2025-06-30. Refit at each fold start S
  on targets <= S-1; online updates within a fold only with observations known at each issue.
- **Holdout:** 5 quarterly folds, 2025-07-01..2026-09-30, opened once for a pre-declared set.
  (Caveat: the load project used these months' forecasts and ERA5 weather, and its notes discuss
  a few large forecast misses; the station observations themselves were not used.)

## Metrics
- **Primary:** load-weighted hourly MAE (deg C) over the 11 stations, weights = the load project's
  fixed city weights (zone load shares 2021-2023, split equally within a zone).
- Secondary: unweighted MAE and RMSE per station; bias; MAE on extreme hours (observed >= 38 C or
  <= -5 C); **MAV comparison** on the 3-hourly MAV valid times only (no interpolation of MAV), and
  daytime max / nighttime min MAE against MAV `n_x` (observed max over 13Z-01Z and min over
  01Z-14Z of the corresponding NWS periods).
- Statistics: paired moving-block bootstrap (7-day blocks) on daily weighted absolute errors.

## Models (weakest first)
- R0 raw GFS, R1 raw ECMWF (2024+, GFS where missing), R2 raw mean of both (load project's blend).
- B1 rolling bias correction per station x local hour (mean error of the last 30 known days).
- Linear MOS per station x hour group, LightGBM global with station id, Kalman-filter bias
  updates, combinations — each with a hypothesis logged before it is run.
- External baseline: NWS MAV (reported, not selectable).

## Budget and selection
- 30 validation runs, all logged in experiments/registry.jsonl including failures.
- Selection rule: lowest validation primary metric; if the gain over the next simpler candidate
  has a 95% CI including 0, the simpler model is chosen. Holdout set declared in
  configs/final_B.yaml before opening.

## Honesty rules
- No public leaderboard exists for this task; MAV is an operational product, and beating or not
  beating it on 11 stations over one period is reported as an internal comparison only.
- Not claimed as operational meteorology experience.
- Optional downstream check (does MOS-corrected temperature improve the load project's model?)
  only by a later dated amendment, before running it.

## Amendments

### B-1 (2026-10-14, after R0-R2, B1, L1, G1 validation runs)
- Fairness of the MAV reference: our models use station observations up to issue - 1 h, the 12Z
  MAV only data to ~12Z. Added the **18Z MAV run** (released ~22Z, before the 23Z/00Z issue) as a
  second, stronger reference, compared on its own 3-hourly valid times. MAV stays a reference,
  never a feature or a selection candidate. Rule, splits and models unchanged.
