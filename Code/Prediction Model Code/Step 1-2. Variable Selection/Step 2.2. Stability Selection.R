# ============================================================
# STEP 3b: VARIABLE-SELECTION STABILITY (repeated 10% subsamples, with progress bar)
# ============================================================
# Uses ONLY imputed training dataset 1. lambda is FIXED at the value chosen
# in 02a_penalty_selection_cv.R. Output: selection_frequency per variable
# across 100 fits. Random effects / coefficients from these 100 fits are
# NOT combined or carried forward -- they are used only to pick a stable
# variable subset for Step 4.
#
# CHANGES from the previous version:
#   - full_variable_pool now comes from utils_variable_pool.R (shared with
#     02a/03/04); is_missing_*/cpp removed, sbp/dbp added (response-var-aware)
#   - switched doParallel -> doSNOW so a live progress bar can be shown
#     while the 100 replicate fits run
#   - per-replicate seeding is now explicit (rep_seeds), since the previous
#     .options.RNG = 2025 argument is a doRNG-specific option that has no
#     effect under doParallel/doSNOW unless doRNG is also loaded -- so it
#     was silently not guaranteeing independent, reproducible per-replicate
#     draws before. This version seeds each replicate explicitly.
#   - outputs written to prediction/<response_var>/ instead of output/
# ============================================================

library(dplyr)
library(glmmLasso)
library(doSNOW)
library(foreach)

response_var <- "respiratory_rate"   # <<== EDIT: must match 02a's response_var
n_reps      <- 100
sample_frac <- 0.10

source("utils_variable_pool.R")

out_dir <- sprintf("prediction/%s", response_var)
train_imp1 <- readRDS("prediction/imputations/imputed_pair_1.rds")$train
lambda     <- readRDS(file.path(out_dir, "penalty_selection_cv.rds"))$selected_lambda
cat(sprintf("Using fixed lambda = %s (from Step 3a, response_var = %s)\n", lambda, response_var))

candidate_predictors <- get_candidate_predictors(response_var)
fix_formula   <- as.formula(paste(response_var, "~", paste(candidate_predictors, collapse = " + ")))
rnd_structure <- list(stay_id = ~ 1 + binned_hrs_since_icu)

all_train_ids <- unique(train_imp1$subject_id)

cores <- max(1, parallel::detectCores() - 4)
cl <- makeCluster(cores)
registerDoSNOW(cl)

cat(sprintf("Starting %d replicates on %d cores...\n", n_reps, cores))
t0 <- Sys.time()

pb <- txtProgressBar(min = 0, max = n_reps, style = 3)
progress_fn <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress_fn)

set.seed(2025)
rep_seeds <- sample.int(1e6, n_reps)

rep_results <- foreach(r = seq_len(n_reps),
                        .packages = c("dplyr", "glmmLasso"),
                        .options.snow = opts) %dopar% {
  set.seed(rep_seeds[r])
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
close(pb)
stopCluster(cl)

cat(sprintf("\nFinished in %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "secs")) / 60))

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

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
saveRDS(list(stability_df = stability_df, selected_vars = selected_vars,
             lambda_used = lambda, n_reps = n_reps, n_ok = sum(ok),
             response_var = response_var),
        file = file.path(out_dir, "variable_selection_stability.rds"))
write.csv(stability_df, file.path(out_dir, "variable_selection_stability.csv"), row.names = FALSE)

cat(sprintf("\nSaved -> %s/variable_selection_stability.rds\n", out_dir))
