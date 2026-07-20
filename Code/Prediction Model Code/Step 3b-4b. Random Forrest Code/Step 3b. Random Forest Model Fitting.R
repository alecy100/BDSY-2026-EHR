# ============================================================
# STEP 4/5 (RF VERSION): FIT RANDOM FOREST MORTALITY MODELS across 4
# MAP-feature variants:
#   "re"      -> random_intercept + random_slope (from the map LMM)
#   "mean"    -> mean(map) per patient over the observed window
#   "sd"      -> sd(map) per patient over the observed window
#   "mean_sd" -> both mean(map) and sd(map)
#
# All 4 variants share the SAME demographic covariate set (whatever
# survives mortality_model_exclude_vars in utils_variable_pool.R --
# currently race_binary, age, sex, bmi, stroke_binary). Only the
# MAP-representation columns differ between variants.
#
# DEPENDENCY: the "re" variant reuses the map LMM already fit by the
# logistic pipeline's 03_fit_lmm_and_logistic.R (reads its saved
# fitted_models/model_pair_k.rds for lmm_fit + patient_re) rather than
# refitting the LMM here -- so that script must be run first. "mean",
# "sd", "mean_sd" only need the imputed data (01's output) and do NOT
# depend on the LMM at all.
#
# excluded_subject_ids (patients who died within the first 48h,
# identified by a separate upstream script) removed from train_imp
# before any RF is fit -- shared with the logistic pipeline via
# utils_excluded_patients.R so both pipelines use the exact same
# excluded population. Applied here to train_imp directly (rather than
# only relying on the "re" variant's already-filtered patient_re) so
# all 4 variants -- including "mean"/"sd"/"mean_sd", which read
# train_imp directly and don't go through the logistic pipeline's
# filtered outputs -- use a consistent, identically-sized population.
#
# NEW: HYPERPARAMETER TUNING (mtry x nodesize) + CLASS-IMBALANCE
# HANDLING, per imputation per variant:
#   - sampsize/strata: each tree is grown on a balanced bootstrap
#     sample (both classes downsampled to the size of the minority
#     class, "is_dead"==1), rather than a sample that reflects the
#     ~15-20% event rate at face value. Plain bootstrapping on an
#     imbalanced outcome tends to bias trees toward the majority class
#     (predicting "survives" too often); balancing the sample every
#     tree sees is the standard fix.
#   - mtry x nodesize grid, evaluated via OOB AUC (not OOB error rate --
#     error rate is a poor metric under class imbalance, since a model
#     that always predicts "survives" would still get a low-looking
#     error rate). This uses ONLY the training data's own out-of-bag
#     predictions -- no test-set peeking, no extra CV split needed.
#   - the winning (mtry, nodesize) combination's fit is what gets saved
#     and carried forward into 04/05, exactly as before -- only the
#     search process for picking that model is new.
#
# Outputs -> prediction/<response_var>/rf/<variant>/...
# ============================================================

library(randomForest)
library(dplyr)
library(pROC)

n_imputations <- 5
response_var  <- "map"          # <<== EDIT: must match the logistic pipeline's response_var
mortality_var <- "is_dead"
id_var        <- "subject_id"
variants      <- c("re", "mean", "sd", "mean_sd")

# ---- tuning grid: mtry values are capped per-model to the number of
#      features actually available (randomForest errors if mtry > p)
mtry_candidates     <- c(2, 3, 4)
nodesize_candidates <- c(1, 10, 20)
ntree_tune          <- 500

source("Prediction Model/utils_variable_pool.R")
source("Prediction Model/utils_excluded_patients.R")

out_dir    <- sprintf("prediction/%s", response_var)
rf_out_dir <- file.path(out_dir, "rf")

cat(sprintf("Excluding %d subject IDs (died within 48h) before RF fitting.\n", length(excluded_subject_ids)))

# ---- same covariate derivation as the logistic pipeline's 03, kept
#      self-contained here so this script doesn't depend on 03's globals
sel <- readRDS(file.path(out_dir, "variable_selection_stability.rds"))
sel$selected_vars[sel$selected_vars == "stroke_binaryischemic"] <- "stroke_binary"
rf_covariate_vars <- setdiff(sel$selected_vars, mortality_model_exclude_vars)
cat("Covariates used in every RF variant:\n"); print(rf_covariate_vars)

time_invariant_selected <- intersect(rf_covariate_vars, level2_vars_time_invariant)
time_varying_selected   <- setdiff(rf_covariate_vars, level2_vars_time_invariant)

# ---- patient-level covariate table: time-invariant -> distinct value,
#      time-varying -> MEAN (identical logic to the logistic pipeline)
build_patient_covariates <- function(df) {
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

# ---- raw per-patient mean/sd of the response_var (map), no LMM involved.
#      sd() of a single observation is undefined (NA) -- treated as 0
#      (zero observed within-patient variability) rather than dropped.
build_map_summary_stats <- function(df) {
  df %>%
    group_by(across(all_of(id_var))) %>%
    summarise(mean_map = mean(.data[[response_var]], na.rm = TRUE),
              sd_map   = if (n() > 1) sd(.data[[response_var]], na.rm = TRUE) else 0,
              n_obs    = n(),
              .groups = "drop")
}

get_variant_feature_cols <- function(variant) {
  switch(variant,
         re      = c("random_intercept", "random_slope"),
         mean    = c("mean_map"),
         sd      = c("sd_map"),
         mean_sd = c("mean_map", "sd_map"))
}

# ---- tune mtry x nodesize via OOB AUC, with class-balanced sampling
#      applied at every grid point (not just the winner) so the
#      comparison across grid points is apples-to-apples
tune_rf_oob <- function(rf_formula, data, mortality_var,
                         mtry_candidates, nodesize_candidates, ntree) {
  y <- data[[mortality_var]]
  class_counts <- table(y)
  min_n <- min(class_counts)
  sampsize_balanced <- setNames(rep(min_n, length(class_counts)), names(class_counts))

  p <- length(all.vars(rf_formula)) - 1
  mtry_grid <- unique(pmin(mtry_candidates, p))

  grid <- expand.grid(mtry = mtry_grid, nodesize = nodesize_candidates)

  grid_results <- lapply(seq_len(nrow(grid)), function(i) {
    fit <- randomForest(rf_formula, data = data, ntree = ntree,
                         mtry = grid$mtry[i], nodesize = grid$nodesize[i],
                         sampsize = sampsize_balanced, strata = y,
                         importance = TRUE)
    oob_prob <- fit$votes[, "1"]
    oob_auc  <- as.numeric(auc(roc(y, oob_prob, quiet = TRUE)))
    list(fit = fit, mtry = grid$mtry[i], nodesize = grid$nodesize[i], oob_auc = oob_auc)
  })

  oob_aucs  <- sapply(grid_results, function(r) r$oob_auc)
  best_idx  <- which.max(oob_aucs)

  tuning_table <- data.frame(mtry = grid$mtry, nodesize = grid$nodesize, oob_auc = oob_aucs) %>%
    arrange(desc(oob_auc))

  list(best_fit      = grid_results[[best_idx]]$fit,
       best_mtry      = grid_results[[best_idx]]$mtry,
       best_nodesize  = grid_results[[best_idx]]$nodesize,
       best_oob_auc   = grid_results[[best_idx]]$oob_auc,
       tuning_table   = tuning_table,
       sampsize_used  = sampsize_balanced)
}

for (variant in variants) {
  dir.create(file.path(rf_out_dir, variant, "fitted_models"), showWarnings = FALSE, recursive = TRUE)
}

rf_fits <- setNames(lapply(variants, function(v) vector("list", n_imputations)), variants)

for (k in seq_len(n_imputations)) {

  train_imp <- readRDS(sprintf("prediction/imputations/imputed_pair_%d.rds", k))$train

  # ---- apply subject-level exclusion before any modeling touches this data ----
  n_before <- n_distinct(train_imp[[id_var]])
  train_imp <- train_imp %>% filter(!.data[[id_var]] %in% excluded_subject_ids)
  n_after <- n_distinct(train_imp[[id_var]])
  cat(sprintf("[Imputation %d] Excluded %d patients (%d -> %d unique patients, %d rows remain)\n",
              k, n_before - n_after, n_before, n_after, nrow(train_imp)))

  patient_covars <- build_patient_covariates(train_imp)
  map_stats      <- build_map_summary_stats(train_imp)

  # "re" variant's random effects + outcome come straight from the
  # logistic pipeline's already-fit LMM (no LMM refit here). Its
  # patient_re was already built from an excluded_subject_ids-filtered
  # train_imp inside the logistic 03, so it's already consistent with
  # the filtering applied above -- no double-filtering needed here.
  logistic_model_set <- readRDS(file.path(out_dir, "fitted_models", sprintf("model_pair_%d.rds", k)))
  patient_re <- logistic_model_set$patient_re %>%
    select(all_of(id_var), all_of(mortality_var), random_intercept, random_slope)

  base_table <- patient_covars %>% inner_join(map_stats, by = id_var)

  for (variant in variants) {
    feature_cols <- get_variant_feature_cols(variant)

    feat_df <- if (variant == "re") {
      patient_re %>% inner_join(patient_covars, by = id_var)
    } else {
      base_table %>%
        inner_join(patient_re %>% select(all_of(id_var), all_of(mortality_var)), by = id_var)
    }

    feat_df[[mortality_var]] <- factor(feat_df[[mortality_var]], levels = c(0, 1))

    rf_formula <- as.formula(paste(
      mortality_var, "~", paste(c(feature_cols, rf_covariate_vars), collapse = " + ")))

    tuned <- tune_rf_oob(rf_formula, feat_df, mortality_var,
                         mtry_candidates, nodesize_candidates, ntree_tune)
    rf_fit <- tuned$best_fit

    rf_fits[[variant]][[k]] <- rf_fit

    cat(sprintf("\n================= variant=%s | imputation %d =================\n", variant, k))
    cat(sprintf("Tuning grid (top 3 by OOB AUC):\n"))
    print(head(tuned$tuning_table, 3), digits = 4)
    cat(sprintf("Selected: mtry=%d, nodesize=%d (OOB AUC=%.4f), balanced sampsize=%s\n",
                tuned$best_mtry, tuned$best_nodesize, tuned$best_oob_auc,
                paste(tuned$sampsize_used, collapse = "/")))
    print(rf_fit)
    cat("\n--- Variable importance ---\n")
    print(importance(rf_fit))

    saveRDS(list(rf_fit = rf_fit, feature_cols = feature_cols,
                 rf_covariate_vars = rf_covariate_vars, feat_df = feat_df,
                 excluded_subject_ids = excluded_subject_ids,
                 tuned_mtry = tuned$best_mtry, tuned_nodesize = tuned$best_nodesize,
                 tuned_oob_auc = tuned$best_oob_auc, tuning_table = tuned$tuning_table,
                 sampsize_used = tuned$sampsize_used),
            file = file.path(rf_out_dir, variant, "fitted_models", sprintf("rf_pair_%d.rds", k)))
  }
}

# ---- pooled summary per variant across the 5 imputations ----
# (no Rubin's-rules analog for tree ensembles -- reporting mean +/- sd
#  across imputations is the honest equivalent, same approach used for
#  AUC/Brier variability in the logistic pipeline's Step 7)
pool_rf_importance <- function(fit_list) {
  imp_list <- lapply(seq_along(fit_list), function(k) {
    imp_df <- as.data.frame(importance(fit_list[[k]]))
    imp_df$variable   <- rownames(imp_df)
    imp_df$imputation <- k
    imp_df
  })
  imp_all <- bind_rows(imp_list)
  numeric_cols <- setdiff(names(imp_all), c("variable", "imputation"))
  imp_all %>%
    group_by(variable) %>%
    summarise(across(all_of(numeric_cols), list(mean = ~mean(.x), sd = ~sd(.x)), .names = "{.col}_{.fn}"),
              .groups = "drop") %>%
    arrange(desc(MeanDecreaseAccuracy_mean))
}

pool_rf_oob <- function(fit_list) {
  oob_err <- sapply(fit_list, function(f) f$err.rate[nrow(f$err.rate), "OOB"])
  oob_auc <- sapply(fit_list, function(f) as.numeric(auc(roc(f$y, f$votes[, "1"], quiet = TRUE))))
  data.frame(mean_oob_error = mean(oob_err), sd_oob_error = sd(oob_err),
             mean_oob_auc = mean(oob_auc), sd_oob_auc = sd(oob_auc),
             imputation = seq_along(oob_err), oob_error = oob_err, oob_auc = oob_auc)
}

pool_rf_tuning <- function(fit_dir, variant, n_imputations) {
  tuning_list <- lapply(seq_len(n_imputations), function(k) {
    rf_set <- readRDS(file.path(fit_dir, variant, "fitted_models", sprintf("rf_pair_%d.rds", k)))
    data.frame(imputation = k, mtry = rf_set$tuned_mtry, nodesize = rf_set$tuned_nodesize,
               oob_auc = rf_set$tuned_oob_auc)
  })
  bind_rows(tuning_list)
}

for (variant in variants) {
  cat(sprintf("\n================= POOLED RF SUMMARY: variant=%s =================\n", variant))

  pooled_importance <- pool_rf_importance(rf_fits[[variant]])
  cat("\n--- Averaged variable importance across 5 imputations ---\n")
  print(pooled_importance, digits = 3)

  pooled_oob <- pool_rf_oob(rf_fits[[variant]])
  cat(sprintf("\nAveraged OOB error rate: %.4f (sd=%.4f) | Averaged OOB AUC: %.4f (sd=%.4f)\n",
              pooled_oob$mean_oob_error[1], pooled_oob$sd_oob_error[1],
              pooled_oob$mean_oob_auc[1], pooled_oob$sd_oob_auc[1]))

  pooled_tuning <- pool_rf_tuning(rf_out_dir, variant, n_imputations)
  cat("\n--- Selected hyperparameters per imputation ---\n")
  print(pooled_tuning, digits = 4)

  saveRDS(list(pooled_importance = pooled_importance, pooled_oob = pooled_oob,
               pooled_tuning = pooled_tuning,
               variant = variant, feature_cols = get_variant_feature_cols(variant),
               rf_covariate_vars = rf_covariate_vars,
               excluded_subject_ids = excluded_subject_ids),
          file = file.path(rf_out_dir, variant, "pooled_rf_summary.rds"))
}

cat(sprintf("\nSaved RF fits -> %s/<variant>/fitted_models/\n", rf_out_dir))
cat(sprintf("Saved pooled summaries -> %s/<variant>/pooled_rf_summary.rds\n", rf_out_dir))