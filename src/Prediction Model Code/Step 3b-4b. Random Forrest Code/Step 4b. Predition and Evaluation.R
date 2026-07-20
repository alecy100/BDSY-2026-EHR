# ============================================================
# STEP 6/7 (RF VERSION): predict test mortality with each variant's RF
# models, evaluate, and produce ONE comparison table across all 4
# MAP-feature variants (re / mean / sd / mean_sd).
#
# "re" variant: reverse-engineers random intercept/slope for test
#   patients using the map LMM saved by the logistic pipeline's 03
#   (same BLUP logic as that pipeline's 04 -- duplicated here so this
#   RF pipeline stays self-contained rather than sourcing their file).
# "mean"/"sd"/"mean_sd": just recompute the same raw per-patient
#   mean/sd of map directly from the test patients' own observed
#   values -- no LMM, no BLUP estimation needed for these three.
#
# NEW: excluded_subject_ids (patients who died within the first 48h)
# removed from test_imp as well, so the test population matches the
# excluded-adjusted training population from 03_rf_fit_random_forest.R.
# Without this, train and test would reflect two different cohort
# definitions -- the model would be trained on the restricted population
# but evaluated against a test set that still includes early-death
# patients, which is exactly the mismatch this exclusion is meant to fix.
#
# NEW: build_map_summary_stats() sd_map fix mirrors 03 -- sd() of a
# single observation is undefined (NA); treated as 0 rather than
# dropped, with n_obs also carried through for visibility.
# ============================================================

library(lme4)
library(randomForest)
library(dplyr)
library(pROC)
library(PRROC)   # for AUPRC -- install.packages("PRROC") if not already present

n_imputations <- 5
response_var  <- "map"        # <<== EDIT: must match 03_rf's response_var
mortality_var <- "is_dead"
id_var        <- "subject_id"
stay_var      <- "stay_id"
time_var      <- "binned_hrs_since_icu"
variants      <- c("re", "mean", "sd", "mean_sd")

source("Prediction Model/utils_variable_pool.R")
source("Prediction Model/utils_excluded_patients.R")

out_dir    <- sprintf("prediction/%s", response_var)
rf_out_dir <- file.path(out_dir, "rf")

# excluded_subject_ids (patients who died within 48h of ICU admission)
# now lives in utils_excluded_patients.R -- sourced above, shared with
# 03_rf_fit_random_forest.R and the logistic pipeline's 03/04

estimate_new_random_effects <- function(lmm_fit, newdata, id_var, stay_var, time_var, response_var) {
  beta_hat <- fixef(lmm_fit)
  sigma2   <- sigma(lmm_fit)^2
  vc       <- as.data.frame(lme4::VarCorr(lmm_fit))
  G <- matrix(0, 2, 2, dimnames = list(c("(Intercept)", time_var), c("(Intercept)", time_var)))
  G["(Intercept)", "(Intercept)"] <- vc$vcov[vc$grp == stay_var & vc$var1 == "(Intercept)" & is.na(vc$var2)]
  G[time_var, time_var]           <- vc$vcov[vc$grp == stay_var & vc$var1 == time_var & is.na(vc$var2)]
  cov_term <- vc$vcov[vc$grp == stay_var & !is.na(vc$var2)]
  G["(Intercept)", time_var] <- cov_term
  G[time_var, "(Intercept)"] <- cov_term

  fixed_form <- formula(lmm_fit, fixed.only = TRUE)
  X_all <- model.matrix(fixed_form, data = newdata)
  nd <- newdata[rownames(X_all), ]

  results <- lapply(split(seq_len(nrow(nd)), nd[[id_var]]), function(rows) {
    Xi <- X_all[rows, names(beta_hat), drop = FALSE]
    yi <- nd[[response_var]][rows]
    Zi <- cbind(1, nd[[time_var]][rows])
    ri <- yi - as.numeric(Xi %*% beta_hat)
    V  <- Zi %*% G %*% t(Zi) + diag(sigma2, length(rows))
    b_hat <- G %*% t(Zi) %*% solve(V) %*% ri
    c(random_intercept = b_hat[1], random_slope = b_hat[2])
  })

  out <- do.call(rbind, results)
  data.frame(id = names(results), out, row.names = NULL) %>%
    setNames(c(id_var, "random_intercept", "random_slope"))
}

build_patient_covariates <- function(df, covariate_vars) {
  time_invariant_selected <- intersect(covariate_vars, level2_vars_time_invariant)
  time_varying_selected   <- setdiff(covariate_vars, level2_vars_time_invariant)

  invariant_df <- if (length(time_invariant_selected) > 0) {
    df %>% distinct(across(all_of(c(id_var, time_invariant_selected))))
  } else {
    df %>% distinct(across(all_of(id_var)))
  }
  varying_df <- if (length(time_varying_selected) > 0) {
    df %>%
      group_by(across(all_of(id_var))) %>%
      summarise(across(all_of(time_varying_selected), ~mean(.x, na.rm = TRUE)),
                .groups = "drop")
  } else {
    df %>% distinct(across(all_of(id_var)))
  }
  invariant_df %>% inner_join(varying_df, by = id_var)
}

# sd() of a single observation is undefined (NA) -- treated as 0 rather
# than dropped, mirroring the fix in 03_rf_fit_random_forest.R
build_map_summary_stats <- function(df) {
  df %>%
    group_by(across(all_of(id_var))) %>%
    summarise(mean_map = mean(.data[[response_var]], na.rm = TRUE),
              sd_map   = if (n() > 1) sd(.data[[response_var]], na.rm = TRUE) else 0,
              n_obs    = n(),
              .groups = "drop")
}

results_by_variant <- setNames(vector("list", length(variants)), variants)

for (variant in variants) {

  test_predictions    <- vector("list", n_imputations)
  per_imputation_perf  <- vector("list", n_imputations)

  for (k in seq_len(n_imputations)) {
    rf_set   <- readRDS(file.path(rf_out_dir, variant, "fitted_models", sprintf("rf_pair_%d.rds", k)))
    test_imp <- readRDS(sprintf("prediction/imputations/imputed_pair_%d.rds", k))$test

    # ---- apply the SAME subject-level exclusion used on the training side ----
    n_before <- n_distinct(test_imp[[id_var]])
    test_imp <- test_imp %>% filter(!.data[[id_var]] %in% excluded_subject_ids)
    n_after <- n_distinct(test_imp[[id_var]])
    cat(sprintf("[variant=%s | Imputation %d] Excluded %d test patients (%d -> %d unique patients)\n",
                variant, k, n_before - n_after, n_before, n_after))

    covariate_vars <- rf_set$rf_covariate_vars
    test_covars <- build_patient_covariates(test_imp, covariate_vars)
    test_covars[[id_var]] <- as.character(test_covars[[id_var]])

    if (variant == "re") {
      logistic_model_set <- readRDS(file.path(out_dir, "fitted_models", sprintf("model_pair_%d.rds", k)))
      feat_test <- estimate_new_random_effects(logistic_model_set$lmm_fit, test_imp,
                                                id_var, stay_var, time_var, response_var)
    } else {
      feat_test <- build_map_summary_stats(test_imp)
    }
    feat_test[[id_var]] <- as.character(feat_test[[id_var]])

    feat_test <- feat_test %>% inner_join(test_covars, by = id_var)

    pred_prob <- predict(rf_set$rf_fit, newdata = feat_test, type = "prob")[, "1"]

    patient_mortality <- test_imp %>% distinct(.data[[id_var]], .data[[mortality_var]])
    patient_mortality[[id_var]] <- as.character(patient_mortality[[id_var]])

    this_pred <- feat_test %>%
      mutate(pred_prob = pred_prob) %>%
      left_join(patient_mortality, by = id_var) %>%
      rename(observed_mortality = all_of(mortality_var)) %>%
      mutate(imputation = k, variant = variant)

    test_predictions[[k]] <- this_pred

    roc_k   <- roc(this_pred$observed_mortality, this_pred$pred_prob, quiet = TRUE)
    brier_k <- mean((this_pred$pred_prob - this_pred$observed_mortality)^2)
    per_imputation_perf[[k]] <- data.frame(variant = variant, imputation = k,
                                            auc = as.numeric(auc(roc_k)), brier = brier_k)

    cat(sprintf("[variant=%s] Imputation %d: n=%d test patients | AUC=%.3f | Brier=%.4f | (mtry=%d, nodesize=%d, train OOB AUC=%.3f)\n",
                variant, k, nrow(feat_test), as.numeric(auc(roc_k)), brier_k,
                rf_set$tuned_mtry, rf_set$tuned_nodesize, rf_set$tuned_oob_auc))
  }

  all_preds <- bind_rows(test_predictions)

  final_preds <- all_preds %>%
    group_by(.data[[id_var]]) %>%
    summarise(avg_pred_prob = mean(pred_prob), observed_mortality = first(observed_mortality),
               .groups = "drop")

  roc_obj     <- roc(final_preds$observed_mortality, final_preds$avg_pred_prob, quiet = TRUE)
  final_auc   <- as.numeric(auc(roc_obj))
  final_brier <- mean((final_preds$avg_pred_prob - final_preds$observed_mortality)^2)

  base_rate         <- mean(final_preds$observed_mortality)
  brier_base_rate   <- mean((base_rate - final_preds$observed_mortality)^2)
  brier_skill_score <- 1 - (final_brier / brier_base_rate)

  pr <- pr.curve(scores.class0 = final_preds$avg_pred_prob,
                 weights.class0 = final_preds$observed_mortality, curve = FALSE)

  perf_by_imputation <- bind_rows(per_imputation_perf)

  cat(sprintf("\n================= variant=%s: TEST-SET PERFORMANCE SUMMARY =================\n", variant))
  print(perf_by_imputation, digits = 4)
  cat(sprintf("\nAcross-imputation variability -- AUC: mean=%.3f, sd=%.3f | Brier: mean=%.4f, sd=%.4f\n",
              mean(perf_by_imputation$auc), sd(perf_by_imputation$auc),
              mean(perf_by_imputation$brier), sd(perf_by_imputation$brier)))
  cat(sprintf("FINAL (averaged-probability) AUC:   %.3f\n", final_auc))
  cat(sprintf("FINAL (averaged-probability) Brier: %.4f\n", final_brier))
  cat(sprintf("FINAL AUPRC: %.3f (baseline/random = base rate = %.3f)\n", pr$auc.integral, base_rate))
  cat(sprintf("Brier Skill Score (%% improvement over base rate): %.1f%%\n", 100 * brier_skill_score))

  results_by_variant[[variant]] <- list(
    variant = variant, all_preds = all_preds, final_preds = final_preds,
    perf_by_imputation = perf_by_imputation,
    final_auc = final_auc, final_brier = final_brier, auprc = pr$auc.integral,
    base_rate = base_rate, brier_base_rate = brier_base_rate,
    brier_skill_score = brier_skill_score)
}

# ---- one comparison table across all 4 variants ----
comparison_table <- bind_rows(lapply(results_by_variant, function(r) {
  data.frame(variant = r$variant, auc = r$final_auc, brier = r$final_brier,
             auprc = r$auprc, brier_skill_score_pct = 100 * r$brier_skill_score)
}))

cat("\n================= RF VARIANT COMPARISON (re / mean / sd / mean_sd) =================\n")
print(comparison_table, digits = 4)

dir.create(rf_out_dir, showWarnings = FALSE, recursive = TRUE)
saveRDS(list(results_by_variant = results_by_variant, comparison_table = comparison_table,
             excluded_subject_ids = excluded_subject_ids),
        file = file.path(rf_out_dir, "rf_variant_comparison.rds"))
write.csv(comparison_table, file.path(rf_out_dir, "rf_variant_comparison.csv"), row.names = FALSE)

cat(sprintf("\nSaved -> %s/rf_variant_comparison.rds / .csv\n", rf_out_dir))