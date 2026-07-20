# ============================================================
# SHARED CONFIG: candidate variable pool for the LMM stage
# Sourced by 02a, 02b, 03, and 04 so the pool can never drift between
# scripts (previously full_variable_pool was copy-pasted in 02a/02b).
#
# CHANGES from the old full_variable_pool:
#   - all is_missing_* indicators removed (no longer present in cleanData.csv)
#   - "cpp" removed (no longer present in cleanData.csv)
#   - "sbp" and "dbp" added as candidates
#
# sbp/dbp are excluded automatically when response_var == "map", since
# MAP is an arithmetic function of SBP/DBP ( MAP ~ (SBP + 2*DBP)/3 ) and
# would otherwise be a near-perfect collinear predictor of itself.
# ============================================================

full_variable_pool <- c(
  "map", "respiratory_rate", "sbp", "dbp",
  "heart_rate", "spo2", "gcs", "rass",
  "temperature_c", "bun", "creatinine", "glucose_lab", "hemoglobin",
  "inr", "platelet", "ptt", "sodium", "wbc", "tidal_volume_obs",
  "nacl3_hypertonic", "vasopressor_baseline", "mechanical_vent_baseline",
  "bmi", "age", "charlson_score",
  "sex", "hypertension", "afib", "cad", "race_binary", "stroke_binary",
  "binned_hrs_since_icu"
)

# variables that can never appear as an LMM covariate for a given response_var
get_mutual_exclusion_vars <- function(response_var) {
  base <- c("map", "respiratory_rate")   # response var + its "twin" always excluded
  if (response_var == "map") {
    base <- c(base, "sbp", "dbp")        # map is a direct linear function of sbp/dbp
  }
  base
}

get_candidate_predictors <- function(response_var) {
  setdiff(full_variable_pool, get_mutual_exclusion_vars(response_var))
}

# time-invariant (patient-level, constant across all of a patient's rows)
# vs. time-varying (repeated-measurement) split -- used in Step 5 to build
# the patient-level covariate summary table for the mortality logistic model
level2_vars_time_invariant <- c("bmi", "age", "charlson_score", "sex",
                                 "hypertension", "afib", "cad",
                                 "race_binary", "stroke_binary")

# variables excluded specifically from the MORTALITY logistic regression's
# covariate list (still eligible as LMM fixed effects / trajectory covariates
# in Step 4 -- this exclusion applies only to Step 5 onward)
mortality_model_exclude_vars <- c("rass", "binned_hrs_since_icu",
                                  "heart_rate", "spo2", "gcs", "rass",
                                  "temperature_c", "bun", "creatinine", "glucose_lab", "hemoglobin",
                                  "inr", "platelet", "ptt", "sodium", "wbc", "tidal_volume_obs",
                                  "nacl3_hypertonic", "vasopressor_baseline", "mechanical_vent_baseline",
                                  "hypertension", "afib", "cad")
