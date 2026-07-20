"""
Config for the RNN mortality pipeline.

IMPORTANT: the column-name lists below are reconstructed from the R
pipeline's utils_variable_pool.R / 01_multiple_imputation.R, NOT verified
directly against the exported CSVs. data.py checks these against the
actual CSV columns and raises a clear error naming exactly what's
missing/unexpected -- fix these lists first if that check fires.
"""

import os

# ---- EDIT THIS to your project's root folder (the one containing
#      "prediction/", same folder your R scripts treat as the working
#      directory). Everything else below is built from this, so the
#      scripts work regardless of which directory you launch `python`
#      from -- e.g. running from code/RNN pipeline/ still finds the
#      right prediction/ folder up at the project root. ----
PROJECT_ROOT = r"C:/repo/BDSY 2026/EHR-BDSY-Project"   # <<== EDIT

DATA_DIR    = os.path.join(PROJECT_ROOT, "prediction", "imputations_csv")
MODEL_DIR   = os.path.join(PROJECT_ROOT, "prediction", "rnn_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "prediction", "rnn_results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- identifiers / outcome / time ----
ID_COL      = "subject_id"
OUTCOME_COL = "is_dead"
TIME_COL    = "binned_hrs_since_icu"   # values assumed to be bin ENDPOINTS: 4, 8, ..., 48
MAX_SEQ_LEN = 12                        # twelve 4-hour bins across the first 48h

# ---- feature variable is the primary target of the ablation/permutation analysis ----
MAP_COL = "map"

# ---- time-varying (repeated-measurement) features, fed to the RNN at every timestep ----
TIME_VARYING_CONTINUOUS = [
    "map", "heart_rate", "respiratory_rate", "spo2", "gcs", "sbp", "dbp",
    "rass", "temperature_c", "bun", "creatinine", "glucose_lab", "hemoglobin",
    "inr", "platelet", "ptt", "sodium", "wbc", "tidal_volume_obs",
]
TIME_VARYING_BINARY = [
    "nacl3_hypertonic", "vasopressor_baseline", "mechanical_vent_baseline",
]
TIME_VARYING_COLS = TIME_VARYING_CONTINUOUS + TIME_VARYING_BINARY

# ---- time-invariant (patient-level) features, concatenated once at the end ----
STATIC_CONTINUOUS = ["age", "bmi", "charlson_score"]
STATIC_BINARY = ["sex", "hypertension", "afib", "cad", "race_binary", "stroke_binary"]
STATIC_COLS = STATIC_CONTINUOUS + STATIC_BINARY

# ---- explicit string -> 0/1 encodings for any TIME_VARYING/STATIC binary
#      column that turns out to be stored as text labels rather than
#      numeric 0/1 in the exported CSV (R factors get written as their
#      label text by write.csv, not their underlying integer codes).
#      Direction here matches the glmmLasso dummy-variable naming seen
#      earlier ("stroke_binaryischemic"), i.e. ischemic=1. VERIFY this is
#      the convention you actually want before trusting downstream
#      interpretation (it doesn't affect the RNN's raw predictive
#      performance either way, only which direction "1" points).
CATEGORICAL_ENCODINGS = {
    "stroke_binary": {"ischemic": 1, "hemorrhagic": 0},
    # add more here if other columns turn out to be string-valued, e.g.:
    # "sex": {"M": 1, "F": 0},
}

# ---- model hyperparameters (kept small given ~3-4k patients -- see earlier
#      discussion on convergence/overfitting risk at this sample size) ----
HIDDEN_SIZE  = 24
NUM_LAYERS   = 1
DROPOUT      = 0.3
BATCH_SIZE   = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
MAX_EPOCHS    = 200
EARLY_STOP_PATIENCE = 15
GRAD_CLIP_NORM = 1.0
VAL_FRACTION  = 0.15   # carved out of TRAIN patients only, never touches test
SEED = 2025