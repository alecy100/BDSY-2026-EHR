# ============================================================
# BOOTSTRAP VARIABLE SELECTION -- LASSO LINEAR MIXED MODEL
# OUTCOME: MAP
#edit
# ============================================================
# For each of n_bootstrap replicates:
#   1. Randomly sample sample_frac (10%) of subjects (subject_id),
#      WITHOUT replacement by default, and keep all of their recorded
#      rows -- this becomes that replicate's dataset.
#   2. Impute that subsampled dataset n_imputations (5) times.
#   3. Fit one glmmLasso model (outcome = map, lambda set below) on
#      each of the 5 imputed datasets.
#   4. Average those 5 fits' coefficients, RSS, and standard errors
#      into one summary row for this bootstrap replicate.
# After all n_bootstrap replicates, those summary rows are averaged
# into final grand-average coefficients, RSS, and standard errors.
#
# NOTE ON SAMPLING: "randomly sample 10% of subjects" is implemented
# as simple random sampling WITHOUT replacement -- a fresh 10% draw
# each replicate, not classic bootstrap-with-replacement at full
# sample size. Set SAMPLE_WITH_REPLACEMENT <- TRUE below if you
# intended the latter instead.
#
# Since subject_id and stay_id are 1:1 in this data, sampling by
# subject_id and keeping all matching rows is equivalent to sampling
# by stay_id.
#
# WARNING: this runs n_bootstrap * n_imputations glmmLasso fits
# (100 * 5 = 500 by default) plus 100 separate mice() calls -- a large
# amount of computation. Test with a small n_bootstrap (e.g. 3-5)
# first to estimate total runtime before committing to the full run.
# ============================================================

# ------------------------------------------------------------
# 0. LOAD LIBRARIES
# ------------------------------------------------------------
library(mice)
library(miceadds)
library(dplyr)
library(glmmLasso)
library(doParallel)
library(foreach)


# ------------------------------------------------------------
# 0. Set Up Parallel
# ------------------------------------------------------------

cores <- detectCores() - 4
cl <- makeCluster(cores)
registerDoParallel(cl)


# ------------------------------------------------------------
# 1. USER-DEFINED PARAMETERS
# ------------------------------------------------------------
lambda                  <- 0.125   # <<== glmmLasso penalty -- set/tune here
n_bootstrap             <- 100    # <<== number of bootstrap replicates
sample_frac             <- 0.10   # <<== fraction of subjects sampled each replicate
n_imputations           <- 5      # <<== imputations run per bootstrap replicate
mice_maxit              <- 10     # <<== mice iterations per imputation run (lower if too slow)
SAMPLE_WITH_REPLACEMENT <- FALSE  # <<== TRUE = classic with-replacement bootstrap instead
response_var            <- "map"

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------
df <- read.csv("output/cleanData.csv")
all_subject_ids <- unique(df$subject_id)

cat(sprintf("Loaded data: %d rows, %d unique subjects.\n", nrow(df), length(all_subject_ids)))

# ------------------------------------------------------------
# 3. VARIABLE GROUPS (same groups used throughout the pipeline)
# ------------------------------------------------------------
level1_vars_continuous <- c("heart_rate", "respiratory_rate", "spo2",
                            "gcs", "cpp", "rass", "temperature_c", "bun",
                            "creatinine", "glucose_lab", "hemoglobin", "inr",
                            "platelet", "ptt", "sodium", "wbc", "tidal_volume_obs")

level1_vars_binary <- c("nacl3_hypertonic", "vasopressor_baseline",
                        "mechanical_vent_baseline")

level2_vars_continuous <- c("bmi", "height_in", "weight_lb", "age", "charlson_score")

level2_vars_categorical <- c("sex", "hypertension", "afib", "cad",
                             "race_binary", "stroke_binary")

redundant_dupes <- c("race", "stroke_type")

all_continuous <- c(level1_vars_continuous, level2_vars_continuous,
                    "binned_hrs_since_icu")

# Candidate predictor pool for the LASSO model. map and
# log_respiratory_rate are excluded so neither can be a covariate for
# the other's equation or its own.
full_variable_pool <- c(
  "map", "log_respiratory_rate",
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

mutual_exclusion_vars <- c("map", "log_respiratory_rate")
candidate_predictors  <- setdiff(full_variable_pool, mutual_exclusion_vars)

rnd_structure <- list(stay_id = ~ 1 + binned_hrs_since_icu)

fix_formula <- as.formula(
  paste(response_var, "~", paste(candidate_predictors, collapse = " + "))
)

# ------------------------------------------------------------
# 4. HELPER: IMPUTE A (SUB)DATASET, RETURN n_imputations COMPLETED SETS
# ------------------------------------------------------------
impute_subset <- function(df_subset, m, maxit, seed) {
  
  never_impute_local <- c("subject_id", "stay_id", "bin",
                          grep("^is_missing_", names(df_subset), value = TRUE))
  
  scale_params_local <- df_subset %>%
    summarise(across(all_of(all_continuous),
                     list(mean = ~mean(.x, na.rm = TRUE),
                          sd   = ~sd(.x, na.rm = TRUE))))
  
  df_scaled_local <- df_subset %>%
    mutate(across(all_of(all_continuous), ~ as.numeric(scale(.x))))
  
  df_scaled_local$stay_id <- as.integer(as.factor(df_scaled_local$stay_id))
  
  ini  <- mice(df_scaled_local, maxit = 0)
  meth <- ini$method
  pred <- ini$predictorMatrix
  
  meth[c(never_impute_local, redundant_dupes)] <- ""
  pred[c(never_impute_local, redundant_dupes), ] <- 0
  pred[, c(never_impute_local[never_impute_local != "stay_id"], redundant_dupes)] <- 0
  
  meth[level1_vars_continuous]  <- "2l.pmm"
  meth[level1_vars_binary]      <- "2l.bin"
  meth[level2_vars_continuous]  <- "2lonly.pmm"
  meth[level2_vars_categorical] <- "2lonly.pmm"
  
  if (all(c("height_in", "weight_lb", "bmi") %in% names(df_scaled_local))) {
    pred[c("height_in", "weight_lb"), "bmi"] <- 0
    pred["bmi", c("height_in", "weight_lb")] <- 0
  }
  if (all(c("map", "cpp") %in% names(df_scaled_local))) {
    pred["map", "cpp"] <- 0
    pred["cpp", "map"] <- 0
  }
  
  pred_auto <- quickpred(df_scaled_local, mincor = 0.25,
                         exclude = c(never_impute_local, redundant_dupes))
  pred[pred == 1 & pred_auto == 0] <- 0
  
  pred[, "stay_id"] <- -2
  pred["stay_id", ] <- 0 
  
  imp_local <- mice(df_scaled_local, method = meth, predictorMatrix = pred,
                    m = m, maxit = maxit, ridge = 1e-2, seed = seed)
  
  imp_list_local <- mice::complete(imp_local, action = "all")
  
  rr_mean <- scale_params_local$respiratory_rate_mean
  rr_sd   <- scale_params_local$respiratory_rate_sd
  
  imp_list_local <- lapply(imp_list_local, function(d) {
    rr_raw                 <- d$respiratory_rate * rr_sd + rr_mean
    d$log_respiratory_rate <- as.numeric(scale(log(rr_raw)))
    d$respiratory_rate     <- NULL
    d$stay_id              <- as.factor(d$stay_id)
    d
  })
  
  imp_list_local
}

# ------------------------------------------------------------
# 5. HELPER: FIT ONE LASSO LMM, RETURN COEFFICIENTS / SE / RSS
# ------------------------------------------------------------
fit_lasso_lmm <- function(dat, lambda, fix_formula, rnd_structure, response_var) {
  
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
  
  # fit$fixerror holds the fixed-effect standard errors (this is what
  # summary.glmmLasso uses internally to build its coefficient table).
  list(coefficients = fit$coefficients, se = fit$fixerror, rss = rss)
}

# ------------------------------------------------------------
# 6. BOOTSTRAP LOOP
# ------------------------------------------------------------
boot_coef_list <- vector("list", n_bootstrap)
boot_se_list   <- vector("list", n_bootstrap)
boot_rss_vec   <- numeric(n_bootstrap)
boot_ok        <- logical(n_bootstrap)

cat(sprintf("\nStarting %d bootstrap replicates (%.0f%% of subjects, %d imputations each, lambda = %s)...\n",
            n_bootstrap, sample_frac * 100, n_imputations, lambda))
boot_start_time <- Sys.time()

# --- 1. Parallel Setup ---
# Set up your cluster backend before running this script
# num_cores <- parallel::detectCores() - 1
# cl <- parallel::makeCluster(num_cores)
# registerDoParallel(cl)

boot_start_time <- Sys.time()
cat(sprintf("Starting %d bootstrap replicates in parallel...\n", n_bootstrap))

# --- 2. Execute Parallel Foreach ---
# `.combine = list` ensures all returned objects are captured in a structured list
boot_raw_results <- foreach(
  b = seq_len(n_bootstrap),
  .packages = c("dplyr", "mice", "miceadds", "glmmLasso"),  # added miceadds + glmmLasso
  .options.RNG = TRUE
) %dopar% {
  
  # --- Step 1: sample subjects, keep all their recorded rows ---
  n_sample <- round(length(all_subject_ids) * sample_frac)
  sampled_subject_ids <- sample(all_subject_ids, size = n_sample,
                                replace = SAMPLE_WITH_REPLACEMENT)
  df_boot <- df %>% filter(subject_id %in% sampled_subject_ids)
  
  # --- Step 2: Impute and Fit ---
  boot_result <- tryCatch({
    
    # Impute this subsample n_imputations times
    imp_list <- impute_subset(df_boot, m = n_imputations, maxit = mice_maxit, seed = b)
    
    # Fit LASSO LMM on each of the n_imputations sets
    coef_mat_this_boot <- NULL
    se_mat_this_boot   <- NULL
    rss_this_boot       <- numeric(n_imputations)
    
    for (j in seq_len(n_imputations)) {
      fit_j <- fit_lasso_lmm(imp_list[[j]], lambda, fix_formula, rnd_structure, response_var)
      coef_mat_this_boot <- rbind(coef_mat_this_boot, fit_j$coefficients)
      se_mat_this_boot   <- rbind(se_mat_this_boot, fit_j$se)
      rss_this_boot[j]    <- fit_j$rss
    }
    
    # Return formatted list on success
    list(
      success  = TRUE,
      avg_coef = colMeans(coef_mat_this_boot),
      avg_se   = colMeans(se_mat_this_boot),
      avg_rss  = mean(rss_this_boot)
    )
    
  }, error = function(e) {
    # If a worker fails, return a marker instead of crashing the full loop
    list(success = FALSE, error_msg = conditionMessage(e))
  })
  
  # The last object evaluated in the block is automatically collected by foreach
  boot_result
}

# --- 3. Post-Processing and Restructuring ---
# Reconstruct your original status vectors and lists from the parallel output
boot_coef_list <- vector("list", n_bootstrap)
boot_se_list   <- vector("list", n_bootstrap)
boot_rss_vec   <- numeric(n_bootstrap)
boot_ok        <- logical(n_bootstrap)

for (b in seq_len(n_bootstrap)) {
  res <- boot_raw_results[[b]]
  
  if (res$success) {
    boot_coef_list[[b]] <- res$avg_coef
    boot_se_list[[b]]   <- res$avg_se
    boot_rss_vec[b]      <- res$avg_rss
    boot_ok[b]           <- TRUE
  } else {
    boot_ok[b]           <- FALSE
    warning(sprintf("Bootstrap replicate %d failed: %s", b, res$error_msg))
  }
}

total_secs <- as.numeric(difftime(Sys.time(), boot_start_time, units = "secs"))
cat(sprintf("\nAll %d bootstrap replicates processed in %.1f minutes (%d succeeded, %d failed).\n",
            n_bootstrap, total_secs / 60, sum(boot_ok), sum(!boot_ok)))

# ------------------------------------------------------------
# 7. AVERAGE ACROSS ALL BOOTSTRAP REPLICATES
# ------------------------------------------------------------
if (sum(boot_ok) == 0) {
  stop("Every bootstrap replicate failed -- check lambda / mice settings before rerunning.")
}

coef_mat_boot <- do.call(rbind, boot_coef_list[boot_ok])
se_mat_boot   <- do.call(rbind, boot_se_list[boot_ok])

final_avg_coef <- colMeans(coef_mat_boot)
final_avg_se   <- colMeans(se_mat_boot, na.rm = TRUE)
final_avg_rss  <- mean(boot_rss_vec[boot_ok])

# Supplementary (not explicitly requested, but a natural byproduct of
# bootstrapping): SD of each coefficient across replicates -- distinct
# from the model-based SE averaged above, this reflects how much each
# estimate itself varies from one resample to the next. Also: the
# fraction of replicates where a term was NOT shrunk to exactly zero,
# the standard stability metric for bootstrapped variable selection.
boot_sd_coef         <- apply(coef_mat_boot, 2, sd)
selection_frequency  <- colMeans(coef_mat_boot != 0)

results_df <- data.frame(
  term                = names(final_avg_coef),
  avg_coefficient     = as.numeric(final_avg_coef),
  avg_standard_error  = as.numeric(final_avg_se),
  bootstrap_sd        = as.numeric(boot_sd_coef),
  selection_frequency = as.numeric(selection_frequency),
  row.names           = NULL
)

cat(sprintf("\n=== Bootstrap LASSO LMM results for outcome '%s' (lambda = %s) ===\n", response_var, lambda))
cat(sprintf("Based on %d successful bootstrap replicates (of %d attempted).\n\n", sum(boot_ok), n_bootstrap))
print(results_df, digits = 4)
cat(sprintf("\nAverage RSS across bootstrap replicates: %.3f\n", final_avg_rss))

# Clean up
stopCluster(cl)

# ------------------------------------------------------------
# 8. SAVE RESULTS
# ------------------------------------------------------------
bootstrap_lasso_results_map <- list(
  response_var      = response_var,
  lambda            = lambda,
  n_bootstrap       = n_bootstrap,
  n_succeeded       = sum(boot_ok),
  sample_frac       = sample_frac,
  n_imputations     = n_imputations,
  coef_by_replicate = boot_coef_list,
  se_by_replicate   = boot_se_list,
  rss_by_replicate  = boot_rss_vec,
  results_summary   = results_df,
  avg_rss           = final_avg_rss
)

saveRDS(bootstrap_lasso_results_map, file = "output/bootstrap_lasso_results_map.rds")
cat("\nSaved bootstrap_lasso_results_map -> output/bootstrap_lasso_results_map.rds\n")