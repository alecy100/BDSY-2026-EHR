# ============================================================
# STEP 3a: SELECT THE glmmLasso PENALTY (parallelized)
# ============================================================
# Done ONCE, on a single 10% sample of TRAINING patients drawn from
# imputed training dataset 1. Patient-level 5-fold CV: fit on 4 folds,
# evaluate held-out RSS on the 5th, average across folds, pick the
# lambda with lowest average HELD-OUT RSS (not training RSS).
#
# Parallelized over every (lambda, fold) combination -- 14 lambdas x
# 5 folds = 70 independent glmmLasso fits distributed across cores.
# ============================================================

library(dplyr)
library(glmmLasso)
library(doParallel)
library(foreach)

set.seed(2025)

response_var  <- "respiratory_rate"                # <<== EDIT if outcome differs
#lambda_grid   <- c(500, 200, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.25, 0.125, 0.05, 0.01)
lambda_grid   <- c(500, 200, 100, 50, 20, 10, 5, 2)
n_folds       <- 5
sample_frac   <- 0.10

train_imp1 <- readRDS("output/imputations/imputed_pair_1.rds")$train

# ---- same candidate pool / formula setup as your stability-selection script
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

# ---- draw the one 10% patient sample
all_train_ids <- unique(train_imp1$subject_id)
n_sample <- round(length(all_train_ids) * sample_frac)
cv_sample_ids <- sample(all_train_ids, n_sample)
cv_df <- train_imp1 %>% filter(subject_id %in% cv_sample_ids)

# ---- assign folds at the PATIENT level
fold_assignment <- setNames(sample(rep(1:n_folds, length.out = length(cv_sample_ids))),
                            cv_sample_ids)
cv_df$.fold <- fold_assignment[as.character(cv_df$subject_id)]

# ---- parallel grid: one task per (lambda, fold) combination -----------
cores <- max(1, parallel::detectCores() - 8)   # <<== EDIT: cores to leave free
cl <- makeCluster(cores)
registerDoParallel(cl)

grid <- expand.grid(lambda = lambda_grid, fold = seq_len(n_folds))
cat(sprintf("Starting %d (lambda x fold) fits on %d cores...\n", nrow(grid), cores))
t0 <- Sys.time()

grid_results <- foreach(i = seq_len(nrow(grid)),
                        .packages = c("dplyr", "glmmLasso"),
                        .combine = rbind) %dopar% {
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
                            # predict on held-out fold: fixed effects only (new patients -> no
                            # trained random effect available yet at this stage), consistent
                            # with evaluating the fixed-effect / penalty choice itself
                            X_test <- model.matrix(delete.response(terms(fix_formula)), data = test_data)
                            coef_names <- names(fit$coefficients)
                            common <- intersect(colnames(X_test), coef_names)
                            pred <- as.numeric(X_test[, common, drop = FALSE] %*% fit$coefficients[common])
                            rss <- sum((test_data[[response_var]] - pred)^2, na.rm = TRUE)
                          }
                          
                          data.frame(lambda = lam, fold = f, rss = rss)
                        }
stopCluster(cl)

cat(sprintf("Finished in %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "secs")) / 60))

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

dir.create("output", showWarnings = FALSE)
saveRDS(list(cv_results = cv_results,
             selected_lambda = best_lambda,
             selected_rss = best_rss,
             cv_sample_ids = cv_sample_ids),
        file = "output/penalty_selection_cv.rds")
write.csv(cv_results, "output/penalty_selection_cv_table.csv", row.names = FALSE)

cat("Saved -> output/penalty_selection_cv.rds\n")