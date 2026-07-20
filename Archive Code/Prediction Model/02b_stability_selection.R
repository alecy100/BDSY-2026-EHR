# ============================================================
# STEP 3b: VARIABLE-SELECTION STABILITY (repeated 10% subsamples)
# ============================================================
# Uses ONLY imputed training dataset 1 (already completed in Step 2 --
# NOT re-imputed here). This is a key difference from the earlier
# version of this script: that version called mice() fresh inside
# every one of the 100 replicates (100 mice() calls + 500 glmmLasso
# fits), which was the main source of the runtime problem. The
# advisor's plan only requires imputing once (Step 2) and then
# repeatedly subsampling rows from that single completed dataset --
# so this version is ~100x cheaper on the imputation side.
#
# lambda is FIXED at the value chosen in 02a_penalty_selection_cv.R.
# Output: selection_frequency per variable across 100 fits.
# Random effects / coefficients from these 100 fits are NOT combined
# or carried forward -- they are used only to pick a stable variable
# subset for Step 4.
# ============================================================

library(dplyr)
library(glmmLasso)
library(doParallel)
library(foreach)

n_reps      <- 100
sample_frac <- 0.10
response_var <- "respiratory_rate"

train_imp1 <- readRDS("output/imputations/imputed_pair_1.rds")$train
lambda     <- readRDS("output/penalty_selection_cv.rds")$selected_lambda
cat(sprintf("Using fixed lambda = %s (from Step 3a)\n", lambda))

full_variable_pool <- c(
  "map", "respiratory_rate",
  "heart_rate", "spo2", "gcs", "rass",
  "temperature_c", "bun", "creatinine", "glucose_lab", "hemoglobin",
  "inr", "platelet", "ptt", "sodium", "wbc", "tidal_volume_obs",
  "nacl3_hypertonic", "vasopressor_baseline", "mechanical_vent_baseline",
  "bmi", "age", "charlson_score",
  "sex", "hypertension", "afib", "cad", "race_binary", "stroke_binary",
  "binned_hrs_since_icu", "is_missing_rass", "is_missing_bun",
  "is_missing_creatinine", "is_missing_glucose_lab", "is_missing_hemoglobin",
  "is_missing_inr", "is_missing_platelet", "is_missing_ptt", "is_missing_sodium",
  "is_missing_wbc", "is_missing_tidal_volume_obs", "is_missing_bmi")
mutual_exclusion_vars <- c("map", "respiratory_rate")
candidate_predictors  <- setdiff(full_variable_pool, mutual_exclusion_vars)
fix_formula   <- as.formula(paste(response_var, "~", paste(candidate_predictors, collapse = " + ")))
rnd_structure <- list(stay_id = ~ 1 + binned_hrs_since_icu)

all_train_ids <- unique(train_imp1$subject_id)

cores <- max(1, parallel::detectCores() - 4)
cl <- makeCluster(cores)
registerDoParallel(cl)

cat(sprintf("Starting %d replicates on %d cores...\n", n_reps, cores))
t0 <- Sys.time()

rep_results <- foreach(r = seq_len(n_reps),
                        .packages = c("dplyr", "glmmLasso"),
                        .options.RNG = 2025) %dopar% {
  n_sample <- round(length(all_train_ids) * sample_frac)
  ids_r <- sample(all_train_ids, n_sample)
  dat_r <- train_imp1 %>% filter(subject_id %in% ids_r)

  tryCatch({
    fit <- glmmLasso(fix = fix_formula, rnd = rnd_structure, data = dat_r,
                      lambda = lambda, family = gaussian(link = "identity"),
                      switch.NR = TRUE, final.re = TRUE)
    list(success = TRUE, coef = fit$coefficients)
  }, error = function(e) list(success = FALSE, error_msg = conditionMessage(e)))
}
stopCluster(cl)

cat(sprintf("Finished in %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "secs")) / 60))

ok <- vapply(rep_results, function(x) x$success, logical(1))
cat(sprintf("%d/%d replicates succeeded\n", sum(ok), n_reps))

coef_mat <- do.call(rbind, lapply(rep_results[ok], function(x) x$coef))
selection_frequency <- colMeans(coef_mat != 0)

stability_df <- data.frame(
  term = names(selection_frequency),
  selection_frequency = as.numeric(selection_frequency)
) %>% arrange(desc(selection_frequency))

print(stability_df, digits = 3)

# ---- choose a reduced variable set: EDIT this threshold as you see fit
selection_threshold <- 0.70
selected_vars <- stability_df$term[stability_df$selection_frequency >= selection_threshold &
                                    stability_df$term != "(Intercept)"]
cat(sprintf("\nVariables selected at >= %.0f%% frequency (%d vars):\n",
            selection_threshold * 100, length(selected_vars)))
print(selected_vars)

dir.create("output", showWarnings = FALSE)
saveRDS(list(stability_df = stability_df, selected_vars = selected_vars,
             lambda_used = lambda, n_reps = n_reps, n_ok = sum(ok)),
        file = "output/variable_selection_stability.rds")
write.csv(stability_df, "output/variable_selection_stability.csv", row.names = FALSE)

cat("\nSaved -> output/variable_selection_stability.rds\n")
