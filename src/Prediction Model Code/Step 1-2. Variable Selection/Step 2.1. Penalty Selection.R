# ============================================================
# STEP 3a: SELECT THE glmmLasso PENALTY (parallelized, with progress bar)
# ============================================================
# Run once per response_var choice ("respiratory_rate" or "map") -- set
# response_var below and re-run to get the comparison pipeline for the
# other biomarker trajectory. Outputs go to prediction/<response_var>/...
# so the two runs never overwrite each other.
#
# CHANGES from the previous version:
#   - full_variable_pool now lives in utils_variable_pool.R (shared with
#     02b/03/04) instead of being copy-pasted per script
#   - is_missing_* / cpp removed from the pool; sbp/dbp added, excluded
#     automatically when response_var == "map"
#   - switched doParallel -> doSNOW so a live progress bar can be shown
#     while the parallel (lambda x fold) grid runs
#   - outputs written to prediction/<response_var>/ instead of output/
# ============================================================

library(dplyr)
library(glmmLasso)
library(doSNOW)
library(foreach)

set.seed(2025)

response_var  <- "respiratory_rate"   # <<== EDIT: "respiratory_rate" or "map"
lambda_grid   <- c(500, 200, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.25, 0.125, 0.05, 0.01)
n_folds       <- 5
sample_frac   <- 0.10

source("Prediction Model/utils_variable_pool.R")

train_imp1 <- readRDS("prediction/imputations/imputed_pair_1.rds")$train

candidate_predictors <- get_candidate_predictors(response_var)
fix_formula   <- as.formula(paste(response_var, "~", paste(candidate_predictors, collapse = " + ")))
rnd_structure <- list(stay_id = ~ 1 + binned_hrs_since_icu)

cat(sprintf("Response variable: %s\n", response_var))
cat(sprintf("Candidate predictors (%d): %s\n", length(candidate_predictors),
            paste(candidate_predictors, collapse = ", ")))

# ---- draw the one 10% patient sample
all_train_ids <- unique(train_imp1$subject_id)
n_sample <- round(length(all_train_ids) * sample_frac)
cv_sample_ids <- sample(all_train_ids, n_sample)
cv_df <- train_imp1 %>% filter(subject_id %in% cv_sample_ids)

# ---- assign folds at the PATIENT level
fold_assignment <- setNames(sample(rep(1:n_folds, length.out = length(cv_sample_ids))),
                            cv_sample_ids)
cv_df$.fold <- fold_assignment[as.character(cv_df$subject_id)]

# ---- parallel grid with progress bar (doSNOW) --------------------------
cores <- max(1, parallel::detectCores() - 8)   # <<== EDIT: cores to leave free
cl <- makeCluster(cores)
registerDoSNOW(cl)

grid <- expand.grid(lambda = lambda_grid, fold = seq_len(n_folds))
cat(sprintf("Starting %d (lambda x fold) fits on %d cores...\n", nrow(grid), cores))
t0 <- Sys.time()

pb <- txtProgressBar(min = 0, max = nrow(grid), style = 3)
progress_fn <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress_fn)

grid_results <- foreach(i = seq_len(nrow(grid)),
                        .packages = c("dplyr", "glmmLasso"),
                        .combine = rbind,
                        .options.snow = opts) %dopar% {
                          lam <- grid$lambda[i]
                          f   <- grid$fold[i]

                          fit_data  <- cv_df %>% filter(.fold != f)
                          test_data <- cv_df %>% filter(.fold == f)

                          fit <- tryCatch(
                            glmmLasso(fix = fix_formula, rnd = rnd_structure, data = fit_data,
                                      lambda = lam, family = gaussian(link = "identity"),
                                      switch.NR = TRUE, final.re = TRUE),
                            error = function(e) NULL)

                          if (is.null(fit)) {
                            rss <- NA_real_
                          } else {
                            # predict on held-out fold: fixed effects only (new patients ->
                            # no trained random effect yet at this stage), consistent with
                            # evaluating the fixed-effect / penalty choice itself
                            X_test <- model.matrix(delete.response(terms(fix_formula)), data = test_data)
                            coef_names <- names(fit$coefficients)
                            common <- intersect(colnames(X_test), coef_names)
                            pred <- as.numeric(X_test[, common, drop = FALSE] %*% fit$coefficients[common])
                            rss <- sum((test_data[[response_var]] - pred)^2, na.rm = TRUE)
                          }

                          data.frame(lambda = lam, fold = f, rss = rss)
                        }
close(pb)
stopCluster(cl)

cat(sprintf("\nFinished in %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "secs")) / 60))

cv_results <- grid_results %>%
  group_by(lambda) %>%
  summarise(mean_held_out_rss = mean(rss, na.rm = TRUE),
            n_folds_ok = sum(!is.na(rss)),
            .groups = "drop") %>%
  arrange(lambda) %>%
  as.data.frame()

print(cv_results, digits = 4)

best_lambda <- cv_results$lambda[which.min(cv_results$mean_held_out_rss)]
best_rss    <- cv_results$mean_held_out_rss[which.min(cv_results$mean_held_out_rss)]
cat(sprintf("\nSelected lambda (lowest mean HELD-OUT RSS) = %s\n", best_lambda))
cat(sprintf("Corresponding mean held-out RSS = %.4f\n", best_rss))

out_dir <- sprintf("prediction/%s", response_var)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
saveRDS(list(cv_results = cv_results,
             selected_lambda = best_lambda,
             selected_rss = best_rss,
             cv_sample_ids = cv_sample_ids,
             response_var = response_var),
        file = file.path(out_dir, "penalty_selection_cv.rds"))
write.csv(cv_results, file.path(out_dir, "penalty_selection_cv_table.csv"), row.names = FALSE)

cat(sprintf("Saved -> %s/penalty_selection_cv.rds\n", out_dir))
