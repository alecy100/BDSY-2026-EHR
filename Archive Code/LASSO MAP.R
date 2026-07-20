# ============================================================
# LASSO LINEAR MIXED MODEL -- OUTCOME: MAP
# ============================================================
# Run 01_imputation.R first (it saves output/imp.rds and
# output/scale_params.rds). This script loads those saved objects
# rather than re-running imputation, so a crash here doesn't require
# redoing the slow imputation step.
#
# This script only fits the MAP model. See 03_lasso_respiratory_rate.R
# for the respiratory-rate model -- kept separate so each can be run
# (and rerun, if it crashes) independently.
# ============================================================

# ------------------------------------------------------------
# 0. LOAD LIBRARIES
# ------------------------------------------------------------
# install.packages("glmmLasso")   # run once, then comment out
library(mice)
library(dplyr)
library(glmmLasso)

# ------------------------------------------------------------
# 1. LOAD SAVED OBJECTS FROM 01_imputation.R
# ------------------------------------------------------------
imp          <- readRDS("output/imp.rds")
scale_params <- readRDS("output/scale_params.rds")

# ------------------------------------------------------------
# 2. EXTRACT THE m COMPLETED (IMPUTED) DATASETS
# ------------------------------------------------------------
imp_list <- mice::complete(imp, action = "all")   # list of m data frames
m <- length(imp_list)

# ------------------------------------------------------------
# 3. LOG-TRANSFORM RESPIRATORY RATE IN EACH IMPUTED DATASET
# ------------------------------------------------------------
# respiratory_rate isn't the outcome in this script, but it's excluded
# as a covariate either way (Section 4 below). This step just keeps
# the data consistent with the respiratory-rate script.
rr_mean <- scale_params$respiratory_rate_mean
rr_sd   <- scale_params$respiratory_rate_sd

imp_list <- lapply(imp_list, function(d) {
  rr_raw                 <- d$respiratory_rate * rr_sd + rr_mean
  d$log_respiratory_rate <- as.numeric(scale(log(rr_raw)))
  d$respiratory_rate     <- NULL
  d
})

# ------------------------------------------------------------
# 4. CANDIDATE PREDICTOR POOL
# ------------------------------------------------------------
# map and log_respiratory_rate are removed from the pool so that
# neither can ever be a covariate -- including in its own equation,
# and in the other outcome's equation.
#
# Also excluded (as before):
#   - cpp: derived directly from map, so it would leak the map outcome
#   - height_in / weight_lb: collinear with bmi
full_variable_pool <- c(
  "map", "log_respiratory_rate",
  "heart_rate", "spo2", "gcs", "rass",
  "temperature_c", "bun", "creatinine", "glucose_lab", "hemoglobin",
  "inr", "platelet", "ptt", "sodium", "wbc", "tidal_volume_obs",
  "nacl3_hypertonic", "vasopressor_baseline", "mechanical_vent_baseline",
  "bmi", "age", "charlson_score",
  "sex", "hypertension", "afib", "cad", "race_binary", "stroke_binary",
  "binned_hrs_since_icu"
)

mutual_exclusion_vars <- c("map", "log_respiratory_rate")

candidate_predictors <- setdiff(full_variable_pool, mutual_exclusion_vars)

# Random effects: random intercept + random slope on binned_hrs_since_icu,
# nested within stay_id
rnd_structure <- list(stay_id = ~ 1 + binned_hrs_since_icu)

# ------------------------------------------------------------
# 5. HELPER: FIT LASSO LMM ON ONE IMPUTED DATASET
# ------------------------------------------------------------
fit_lasso_lmm <- function(dat, lambda, fix_formula, rnd_structure, response_var) {
  
  dat$stay_id <- as.factor(dat$stay_id)
  
  fit <- glmmLasso(
    fix       = fix_formula,
    rnd       = rnd_structure,
    data      = dat,
    lambda    = lambda,
    family    = gaussian(link = "identity"),
    switch.NR = TRUE,
    final.re  = TRUE
  )
  
  y      <- dat[[response_var]]
  fitted <- fit$fitted.values
  rss    <- sum((y - fitted)^2)
  
  list(coefficients = fit$coefficients, rss = rss)
}

# ------------------------------------------------------------
# 6. OUTCOME: MAP
# ------------------------------------------------------------
response_var <- "map"
lambda       <- 30   # <<== set/tune the MAP-model penalty here

fix_formula <- as.formula(
  paste(response_var, "~", paste(candidate_predictors, collapse = " + "))
)

coef_list <- vector("list", m)
rss_vec   <- numeric(m)

for (i in seq_len(m)) {
  
  result <- tryCatch(
    fit_lasso_lmm(imp_list[[i]], lambda, fix_formula, rnd_structure, response_var),
    error = function(e) {
      warning(sprintf("Imputation %d failed to converge: %s", i, conditionMessage(e)))
      NULL
    }
  )
  
  if (!is.null(result)) {
    coef_list[[i]] <- result$coefficients
    rss_vec[i]     <- result$rss
    cat(sprintf("Imputation %d/%d done -- RSS = %.3f\n", i, m, result$rss))
  } else {
    coef_list[[i]] <- NA
    rss_vec[i]      <- NA
  }
}

# ------------------------------------------------------------
# 7. AVERAGE COEFFICIENTS AND RSS ACROSS IMPUTATIONS
# ------------------------------------------------------------
ok <- !sapply(coef_list, function(x) length(x) == 1 && is.na(x))

if (sum(ok) == 0) {
  stop("glmmLasso failed to converge on every imputed dataset -- try a different lambda.")
}

coef_mat  <- do.call(rbind, coef_list[ok])
avg_coefs <- colMeans(coef_mat)
avg_rss   <- mean(rss_vec[ok])

avg_coefficients_df <- data.frame(
  term            = names(avg_coefs),
  avg_coefficient = as.numeric(avg_coefs),
  row.names       = NULL
)

lasso_results_map <- list(
  response_var                = response_var,
  lambda                      = lambda,
  n_converged                 = sum(ok),
  coefficients_by_imputation  = coef_list,
  rss_by_imputation           = rss_vec,
  avg_coefficients            = avg_coefficients_df,
  avg_rss                     = avg_rss
)

cat("\n=== LASSO linear mixed model for outcome: map -- averaged over",
    sum(ok), "of", m, "imputations (lambda =", lambda, ") ===\n\n")
print(avg_coefficients_df, digits = 4)
cat("\nAverage RSS across imputations:", round(avg_rss, 3), "\n")

# ------------------------------------------------------------
# 8. SAVE RESULTS
# ------------------------------------------------------------
saveRDS(lasso_results_map, file = "output/lasso_results_map.rds")
cat("\nSaved lasso_results_map -> output/lasso_results_map.rds\n")

# ------------------------------------------------------------
# 9. (OPTIONAL) BACK-TRANSFORM A COEFFICIENT TO ORIGINAL UNITS
# ------------------------------------------------------------
# beta_scaled_age   <- avg_coefficients_df %>%
#                        filter(term == "age") %>% pull(avg_coefficient)
# beta_original_age <- beta_scaled_age / scale_params$age_sd