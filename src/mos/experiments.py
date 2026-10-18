"""Project B experiment registry; hypotheses written before each run."""
from .models import Col, RollingBias, LinearMOS, LGBMResid, Average

EXPERIMENTS = {
    "R0_raw_gfs": dict(model=lambda: Col("g_t"),
                       hypothesis="Raw GFS at the city point is the floor; expected MAE ~1.5-2 C at day-ahead."),
    "R1_raw_ecmwf": dict(model=lambda: Col("e_t", "g_t"),
                         hypothesis="ECMWF IFS 0.25 beats GFS (ECMWF is generally the more skilful global model)."),
    "R2_raw_blend": dict(model=lambda: Col("bl"),
                         hypothesis="Mean of GFS and ECMWF beats both (partly independent errors); this is the load project's input."),
    "B1_rolling_bias30": dict(model=RollingBias,
                              hypothesis="Removing the 30-day station x hour mean error of the blend beats R2: city-point grid forecasts have systematic local biases (e.g. airport vs grid cell, diurnal cycle)."),
}

EXPERIMENTS.update({
    "L1_linear_mos": dict(model=lambda: LinearMOS(), config={"feats": ["bl", "err_last", "ob_prev", "sin", "cos"], "groups": "station x hour"},
                          hypothesis="Ridge per station x local hour on the blend, the error at the issue (ob_last - blend), yesterday's observation and seasonality beats B1: the error at the issue persists into the next day and the bias depends on season."),
    "G1_lgbm_resid": dict(model=lambda: LGBMResid(), config={"feats": LGBMResid.FEATS, "target": "y - blend", "loss": "l1"},
                          hypothesis="Global LightGBM on the blend residual with humidity, cloud, radiation, wind, recent observations and rolling bias beats L1 (non-linear, regime-dependent errors e.g. cloud / cold-air events)."),
})

# registered after G1 (1.162) / L1 (1.318); L1 trains mostly on 2021-23 when the blend = GFS only
EXPERIMENTS.update({
    "L2_linear_mos_2024": dict(model=lambda: LinearMOS(start="2024-01-01"), config={"start": "2024-01-01"},
                               hypothesis="L1 trained only from 2024 (blend = GFS+ECMWF as at prediction time) beats L1: before 2024 the blend is GFS only, so L1's coefficients describe the wrong predictor."),
    "G2_lgbm_resid_2024": dict(model=lambda: LGBMResid(start="2024-01-01"), config={"start": "2024-01-01"},
                               hypothesis="G1 trained only from 2024 beats G1 (same reason as L2); may lose because the first fold has only 6 months of data. Uncertain."),
    "G3_lgbm_resid_3seeds": dict(model=lambda: LGBMResid(seeds=(0, 1, 2)), config={"seeds": 3},
                                 hypothesis="Averaging 3 seeds of G1 (row/column subsampling) gives a small gain (< 0.02 C)."),
})

# registered after G1-G3: G3 (3 seeds) gained only 0.004 C; G2 (2024+) lost
G4_FEATS = LGBMResid.FEATS + ["bias7", "err_prev", "spread", "dep_last"]
EXPERIMENTS.update({
    "G4_lgbm_resid_recent_err": dict(model=lambda: LGBMResid(feats=G4_FEATS), config={"feats": G4_FEATS},
                                     hypothesis="Adding short-memory error features (7-day station x hour bias, yesterday's same-hour error), model disagreement (|GFS-ECMWF|) and dew-point depression at the issue beats G1 by >= 0.02 C: errors persist over a few days (regimes) and moist/dry air shifts the diurnal range."),
})

# registered after G4 (1.140)
EXPERIMENTS.update({
    "G5_lgbm_recent_err_big": dict(model=lambda: LGBMResid(feats=G4_FEATS, params=dict(learning_rate=0.02, n_estimators=3000, num_leaves=127, min_child_samples=100)),
                                   config={"lr": 0.02, "trees": 3000, "leaves": 127, "min_child": 100},
                                   hypothesis="A larger G4 (more, deeper trees, slower learning) gains <= 0.01 C; G1/G4 settings were never tuned. Expected CI to include 0."),
    "E1_avg_G4_L2": dict(model=lambda: Average(LGBMResid(feats=G4_FEATS), LinearMOS(start="2024-01-01")), config={"members": ["G4", "L2"]},
                         hypothesis="Equal average of G4 and L2 (linear MOS) beats G4: different model families. Expected to lose because L2 is 0.15 C worse."),
})
