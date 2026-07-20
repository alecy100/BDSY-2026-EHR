# ============================================================
# BOOTSTRAP VARIABLE SELECTION -- LASSO LINEAR MIXED MODEL
# OUTCOME: MAP
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
library(miceadds)  # supplies mice.impute.2l.pmm (2lonly.pmm is actually
# built into base mice itself, so it needs no extra
# package). Every parallel worker needs miceadds
# loaded too -- see .packages in the foreach() call
# below -- not just the main session.
library(dplyr)
library(glmmLasso)
library(doParallel)
library(foreach)

# ------------------------------------------------------------
# 0b. SET UP PARALLEL CLUSTER
# ------------------------------------------------------------
cores <- detectCores() - 6
cat(sprintf("Detected %d cores; using %d for the cluster.\n", detectCores(), cores))
if (cores < 1) stop("Not enough cores available after reserving 6 -- lower the reservation.")

cl <- makeCluster(cores)

# Propagate the master session's library paths to every worker. PSOCK
# workers are fresh R processes and do NOT automatically inherit
# .libPaths() -- if a package like miceadds lives in a user-specific
# library (common without admin rights), the interactive session sees
# it fine but a worker can silently fail to find it: foreach's
# .packages mechanism calls require() on each worker, which just
# returns FALSE on failure rather than erroring, so nothing looks wrong
# until code tries to actually use a function from that package.
clusterCall(cl, function(lp) .libPaths(lp), .libPaths())
registerDoParallel(cl)

# ------------------------------------------------------------
# 1. USER-DEFINED PARAMETERS
# ------------------------------------------------------------
lambda                  <- 0.1   # <<== glmmLasso penalty -- set/tune here
n_bootstrap             <- 100    # <<== number of bootstrap replicates
sample_frac             <- 0.10   # <<== fraction of subjects sampled each replicate
n_imputations           <- 5      # <<== imputations run per bootstrap replicate
mice_maxit              <- 10     # <<== mice iterations per imputation run (lower if too slow)
SAMPLE_WITH_REPLACEMENT <- FALSE  # <<== TRUE = classic with-replacement bootstrap instead
response_var            <- "map"

DEBUG_FIRST_REPLICATE <- TRUE  # <<== if TRUE, replicate 1 is run once, serially,
# errors NOT caught, BEFORE the parallel foreach
# starts. This surfaces a real traceback in
# seconds instead of finding out after all 100
# replicates fail the same way. Set FALSE once
# things are confirmed working.

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
# 6. HELPER: RUN ONE FULL BOOTSTRAP REPLICATE
# ------------------------------------------------------------
run_one_bootstrap_replicate <- function(b) {
  
  # set.seed(b) makes each replicate's subject draw reproducible no
  # matter which worker runs it or in what order -- this replaces the
  # non-functional .options.RNG = TRUE from the previous version (that
  # argument isn't recognized by foreach/doParallel without the doRNG
  # package registered, so it was silently doing nothing).
  set.seed(b)
  n_sample <- round(length(all_subject_ids) * sample_frac)
  sampled_subject_ids <- sample(all_subject_ids, size = n_sample,
                                replace = SAMPLE_WITH_REPLACEMENT)
  df_boot <- df %>% filter(subject_id %in% sampled_subject_ids)
  
  imp_list <- impute_subset(df_boot, m = n_imputations, maxit = mice_maxit, seed = b)
  
  coef_mat_this_boot <- NULL
  se_mat_this_boot   <- NULL
  rss_this_boot       <- numeric(n_imputations)
  
  for (j in seq_len(n_imputations)) {
    fit_j <- fit_lasso_lmm(imp_list[[j]], lambda, fix_formula, rnd_structure, response_var)
    coef_mat_this_boot <- rbind(coef_mat_this_boot, fit_j$coefficients)
    se_mat_this_boot   <- rbind(se_mat_this_boot, fit_j$se)
    rss_this_boot[j]    <- fit_j$rss
  }
  
  list(
    success  = TRUE,
    avg_coef = colMeans(coef_mat_this_boot),
    avg_se   = colMeans(se_mat_this_boot),
    avg_rss  = mean(rss_this_boot)
  )
}

# ------------------------------------------------------------
# 7. OPTIONAL DEBUG PRE-FLIGHT: run replicate 1 serially, uncaught
# ------------------------------------------------------------
# Catches structural problems (missing package methods on the worker,
# formula issues, etc.) with a real traceback, before spending time on
# 100 replicates that would all fail the same way.
if (DEBUG_FIRST_REPLICATE) {
  cat("\n[DEBUG] Running bootstrap replicate 1 serially (errors NOT caught)...\n")
  debug_start <- Sys.time()
  debug_result <- run_one_bootstrap_replicate(1)
  cat(sprintf("[DEBUG] Replicate 1 succeeded serially in %.1fs (avg RSS = %.3f).\n",
              as.numeric(difftime(Sys.time(), debug_start, units = "secs")), debug_result$avg_rss))
}

# ------------------------------------------------------------
# 7b. CLUSTER HEALTH CHECK
# ------------------------------------------------------------
# Confirms every worker can actually load the packages this loop needs
# and see the specific imputation methods before running 100 replicates.
# Checks each package individually and captures the REAL error message
# so a failure here tells you exactly which package/step is broken.
worker_check <- clusterEvalQ(cl, {
  pkg_status <- list()
  for (pkg in c("mice", "miceadds", "dplyr", "glmmLasso")) {
    pkg_status[[pkg]] <- tryCatch({
      suppressWarnings(library(pkg, character.only = TRUE))
      "loaded ok"
    }, error = function(e) paste("ERROR:", conditionMessage(e)))
  }
  methods_found <- tryCatch({
    exists("mice.impute.2l.pmm", mode = "function") &&
      exists("mice.impute.2lonly.pmm", mode = "function")
  }, error = function(e) FALSE)
  list(
    pkg_status        = pkg_status,
    methods_found     = methods_found,
    lib_paths         = .libPaths(),
    r_version         = R.version.string,
    miceadds_installed = "miceadds" %in% rownames(installed.packages())
  )
})

worker_ok <- vapply(worker_check, function(x) isTRUE(x$methods_found), logical(1))

if (!all(worker_ok)) {
  cat("\n[CLUSTER HEALTH CHECK FAILED] Diagnostic from worker 1:\n")
  cat("  R version:          ", worker_check[[1]]$r_version, "\n")
  cat("  miceadds installed?:", worker_check[[1]]$miceadds_installed, "\n")
  cat("  .libPaths():\n")
  print(worker_check[[1]]$lib_paths)
  cat("  Per-package load status:\n")
  print(worker_check[[1]]$pkg_status)
  stopCluster(cl)
  stop(sprintf(
    "%d of %d workers cannot find mice.impute.2l.pmm / mice.impute.2lonly.pmm. See diagnostic above.",
    sum(!worker_ok), cores))
}
cat(sprintf("[CLUSTER HEALTH CHECK] All %d workers can load mice/miceadds/dplyr/glmmLasso. Proceeding.\n", cores))

# ------------------------------------------------------------
# 8. BOOTSTRAP LOOP (PARALLEL)
# ------------------------------------------------------------
cat(sprintf("\nStarting %d bootstrap replicates (%.0f%% of subjects, %d imputations each, lambda = %s) on %d cores...\n",
            n_bootstrap, sample_frac * 100, n_imputations, lambda, cores))
boot_start_time <- Sys.time()

boot_raw_results <- foreach(
  b = seq_len(n_bootstrap),
  .packages = c("dplyr", "mice", "miceadds", "glmmLasso")
  # No .export needed -- foreach auto-detects the globals referenced
  # inside run_one_bootstrap_replicate() (df, candidate_predictors,
  # fix_formula, impute_subset, fit_lasso_lmm, etc.) and ships them to
  # each worker on its own.
) %dopar% {
  tryCatch(
    run_one_bootstrap_replicate(b),
    error = function(e) list(success = FALSE, error_msg = conditionMessage(e))
  )
}

total_secs <- as.numeric(difftime(Sys.time(), boot_start_time, units = "secs"))

# ------------------------------------------------------------
# 9. UNPACK RESULTS FROM THE PARALLEL RUN
# ------------------------------------------------------------
boot_coef_list <- vector("list", n_bootstrap)
boot_se_list   <- vector("list", n_bootstrap)
boot_rss_vec   <- numeric(n_bootstrap)
boot_ok        <- logical(n_bootstrap)

for (b in seq_len(n_bootstrap)) {
  res <- boot_raw_results[[b]]
  
  if (isTRUE(res$success)) {
    boot_coef_list[[b]] <- res$avg_coef
    boot_se_list[[b]]   <- res$avg_se
    boot_rss_vec[b]      <- res$avg_rss
    boot_ok[b]           <- TRUE
  } else {
    boot_ok[b] <- FALSE
    warning(sprintf("Bootstrap replicate %d failed: %s", b, res$error_msg))
  }
}

cat(sprintf("\nAll %d bootstrap replicates processed in %.1f minutes on %d cores (%d succeeded, %d failed).\n",
            n_bootstrap, total_secs / 60, cores, sum(boot_ok), sum(!boot_ok)))

# Cluster no longer needed once the loop is done
stopCluster(cl)

# ------------------------------------------------------------
# 10. AVERAGE ACROSS ALL BOOTSTRAP REPLICATES
# ------------------------------------------------------------
if (sum(boot_ok) == 0) {
  stop("Every bootstrap replicate failed -- check lambda / mice settings before rerunning.")
}

coef_mat_boot <- do.call(rbind, boot_coef_list[boot_ok])
se_mat_boot   <- do.call(rbind, boot_se_list[boot_ok])

final_avg_coef <- colMeans(coef_mat_boot, na.rm = TRUE)
final_avg_se   <- colMeans(se_mat_boot, na.rm = TRUE)
final_avg_rss  <- mean(boot_rss_vec[boot_ok])

# Supplementary (not explicitly requested, but a natural byproduct of
# bootstrapping): SD of each coefficient across replicates -- distinct
# from the model-based SE averaged above, this reflects how much each
# estimate itself varies from one resample to the next. Also: the
# fraction of replicates where a term was NOT shrunk to exactly zero,
# the standard stability metric for bootstrapped variable selection.
boot_sd_coef         <- apply(coef_mat_boot, 2, sd, na.rm = TRUE)
selection_frequency  <- colMeans(coef_mat_boot != 0, na.rm = TRUE)

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

# ------------------------------------------------------------
# 11. SAVE RESULTS
# ------------------------------------------------------------
bootstrap_lasso_results_map <- list(
  response_var      = response_var,
  lambda            = lambda,
  n_bootstrap       = n_bootstrap,
  n_succeeded       = sum(boot_ok),
  sample_frac       = sample_frac,
  n_imputations     = n_imputations,
  n_cores_used      = cores,
  coef_by_replicate = boot_coef_list,
  se_by_replicate   = boot_se_list,
  rss_by_replicate  = boot_rss_vec,
  results_summary   = results_df,
  avg_rss           = final_avg_rss
)

saveRDS(bootstrap_lasso_results_map, file = "output/bootstrap_lasso_results_map.rds")
cat("\nSaved bootstrap_lasso_results_map -> output/bootstrap_lasso_results_map.rds\n")