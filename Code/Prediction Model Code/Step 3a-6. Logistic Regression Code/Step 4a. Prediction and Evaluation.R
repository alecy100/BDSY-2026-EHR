# ============================================================
# STEP 6: "REVERSE ENGINEER" RANDOM EFFECTS FOR TEST PATIENTS
#         + predict mortality using each imputation's logistic model
#         (random effects + mean-summarized covariates)
# STEP 7: AVERAGE THE 5 PREDICTED PROBABILITIES, EVALUATE, DETAILED SUMMARY
# ============================================================
# CHANGES from the previous version:
#   - test-set covariate summaries (mean for time-varying vars) are built
#     with the SAME logic as 03, using logistic_covariate_vars saved in
#     each model_pair_k.rds, and joined onto new_re before prediction
#   - detailed performance summary now includes PER-IMPUTATION AUC/Brier
#     (and their across-imputation mean/SD) in addition to the final
#     averaged-probability AUC/Brier, printed and saved to rds
#   - outputs written to prediction/<response_var>/ instead of output/
#   - NEW: excluded_subject_ids (inlined below, 337 IDs, matching
#     03_fit_lmm_and_logistic.R and 05_mean_sd_map_mortality_models.R)
#     are dropped from test_imp immediately after each imputation's
#     test set is loaded, before random effects are reverse-engineered
#     or mortality is predicted/evaluated for anyone.
# ============================================================

library(lme4)
library(dplyr)
library(pROC)

n_imputations <- 5
response_var  <- "map"   # <<== EDIT: must match 02a/02b/03's response_var
mortality_var <- "is_dead"
id_var        <- "subject_id"
stay_var      <- "stay_id"
time_var      <- "binned_hrs_since_icu"

source("utils_variable_pool.R")

# EXCLUDED SUBJECT IDs: inlined directly (337 IDs), copied to match the
# excluded_subject_ids vector embedded in 03_fit_lmm_and_logistic.R and
# 05_mean_sd_map_mortality_models.R exactly. This is a hand-kept copy,
# not a shared/sourced file -- if the list changes in 03, it needs to be
# updated here (and in 05) too.
excluded_subject_ids <- c(
  10326273, 10442543, 10547640, 11481318, 11802193, 12130963, 12852254, 14368383, 14815027, 15822558, 16004662,
  16433374, 16508905, 17471925, 17528095, 17726962, 18202796, 18585906, 18891052, 19181803, 19463304, 19472788,
  19523707, 19751955, 19848900, 10111471, 10226399, 10584941, 11272778, 11707036, 11921090, 12259105, 12725295,
  13852466, 14112332, 14288514, 14490698, 14628472, 15057220, 15364928, 15612258, 15882381, 16126644, 16259761,
  16275019, 17959982, 18819549, 18966710, 19340375, 19418926, 10075506, 10675361, 11885267, 12429606, 13007347,
  13973605, 14223179, 14247206, 14975201, 15111957, 15429606, 15558105, 15625079, 16540359, 17886662, 17982005,
  18914886, 19069982, 19593271, 19787497, 10253355, 10257895, 10478935, 10535413, 10602808, 12360636, 12625272,
  12713322, 12905072, 13343032, 13871417, 15123048, 15442759, 15585673, 15768571, 16326458, 16514011, 16727737,
  17536748, 18310386, 18635171, 18941300, 19122895, 19231248, 19278876, 19403388, 19412083, 10153420, 10179938,
  11061972, 11114467, 11854457, 12646229, 13101078, 14089575, 14663112, 15281579, 15373990, 15962265, 16034243,
  16137601, 16525967, 10144089, 10720228, 11560497, 12053011, 13582491, 13684309, 13720005, 13762552, 14233915,
  14520615, 15600053, 15907529, 17418890, 17947722, 19352227, 19402063, 10106165, 10112880, 11549544, 11722172,
  11732244, 11913938, 12265009, 12458851, 12702538, 13720987, 13723356, 14369272, 15048581, 15226441, 17293846,
  17327419, 18250712, 18473223, 18793846, 19287751, 19307423, 19470403, 10049642, 10270064, 10825313, 11004183,
  11018658, 12221723, 12233558, 12309099, 13603188, 14114656, 14725974, 14988847, 15311288, 15552515, 15854999,
  15909915, 16081944, 16687584, 16778395, 18172078, 18248681, 18554370, 18951588, 19904685, 10245374, 10262574,
  10382177, 11027472, 11350044, 11569817, 13033181, 13613546, 13783774, 13902673, 14284282, 14421640, 14838267,
  15113915, 16056209, 16248948, 16393152, 17226685, 17975678, 19440297, 10111101, 10593481, 11121125, 11622426,
  12173825, 13090296, 13198822, 13348568, 14514630, 14633465, 15125253, 15149974, 16139162, 16155414, 17553130,
  17635990, 17737643, 18290572, 10504539, 10780769, 11436324, 11457273, 11669818, 11731531, 12441528, 12686478,
  12784758, 13522532, 14795899, 17355673, 17357560, 17782556, 18341422, 19102039, 19885636, 10810168, 11235409,
  11289321, 11625891, 11722148, 12735239, 13042472, 13877230, 13916620, 15696612, 16093032, 16325086, 16528283,
  16940596, 16956980, 17261054, 17958466, 18119847, 18669115, 18684087, 19783267, 10034171, 10806809, 11264564,
  11667980, 12091602, 12811067, 13190878, 13263975, 13947218, 13992867, 14415776, 14539739, 14961555, 15947104,
  16122037, 16377213, 16613429, 16691656, 16739392, 17997171, 18475468, 19356128, 19645794, 11149148, 11788497,
  12074215, 12168724, 12453379, 12546031, 13470745, 14546693, 14664560, 15519399, 15642021, 16148902, 17210785,
  17509752, 18324626, 18383938, 19496979, 19904446, 10030566, 10246238, 10622130, 10804631, 11223490, 11614016,
  12151776, 12827336, 13122131, 13313150, 13315557, 13333286, 13676372, 14247396, 14385971, 14515889, 14888649,
  15148777, 15616506, 15693180, 16047924, 16342851, 16389404, 16615529, 16926353, 18477796, 18546142, 18749464,
  18891541, 19528390, 19980800, 11449507, 11635373, 11743987, 12957096, 13771243, 13855662, 15290913, 15510752,
  15898276, 16021029, 16025717, 17880221, 18115673, 18434318, 19558328
)

cat(sprintf("Excluding %d subject IDs before test-set evaluation.\n", length(excluded_subject_ids)))

out_dir <- sprintf("prediction/%s", response_var)

estimate_new_random_effects <- function(lmm_fit, newdata, id_var, stay_var, time_var, response_var) {
  beta_hat <- fixef(lmm_fit)
  sigma2   <- sigma(lmm_fit)^2
  vc       <- as.data.frame(VarCorr(lmm_fit))
  G <- matrix(0, 2, 2, dimnames = list(c("(Intercept)", time_var), c("(Intercept)", time_var)))
  G["(Intercept)", "(Intercept)"] <- vc$vcov[vc$grp == stay_var & vc$var1 == "(Intercept)" & is.na(vc$var2)]
  G[time_var, time_var]           <- vc$vcov[vc$grp == stay_var & vc$var1 == time_var & is.na(vc$var2)]
  cov_term <- vc$vcov[vc$grp == stay_var & !is.na(vc$var2)]
  G["(Intercept)", time_var] <- cov_term
  G[time_var, "(Intercept)"] <- cov_term
  
  fixed_form <- formula(lmm_fit, fixed.only = TRUE)
  X_all <- model.matrix(fixed_form, data = newdata)
  # keep only rows model.matrix kept (in case of NAs) -- match by ROW NAME
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

# same summarization logic as 03: time-invariant -> distinct value,
# time-varying -> MEAN. level2_vars_time_invariant comes from utils_variable_pool.R
build_patient_covariates <- function(df, logistic_covariate_vars) {
  time_invariant_selected <- intersect(logistic_covariate_vars, level2_vars_time_invariant)
  time_varying_selected   <- setdiff(logistic_covariate_vars, level2_vars_time_invariant)
  
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

test_predictions    <- vector("list", n_imputations)
per_imputation_perf  <- vector("list", n_imputations)

for (k in seq_len(n_imputations)) {
  model_set <- readRDS(file.path(out_dir, "fitted_models", sprintf("model_pair_%d.rds", k)))
  test_imp  <- readRDS(sprintf("prediction/imputations/imputed_pair_%d.rds", k))$test
  
  # ---- drop excluded subjects BEFORE random effects / predictions are computed ----
  n_before <- length(unique(test_imp[[id_var]]))
  test_imp <- test_imp %>% filter(!(.data[[id_var]] %in% excluded_subject_ids))
  n_after <- length(unique(test_imp[[id_var]]))
  cat(sprintf("Imputation %d: excluded %d patients (%d -> %d test patients)\n",
              k, n_before - n_after, n_before, n_after))
  
  # patients must have their observed pre-prediction-window measurements only --
  # confirm test_imp already reflects the intended prediction-time window
  # before this point (i.e. no post-outcome / future data leaking in).
  new_re <- estimate_new_random_effects(model_set$lmm_fit, test_imp,
                                        id_var, stay_var, time_var, response_var)
  new_re[[id_var]] <- as.character(new_re[[id_var]])   # <-- add this line
  
  test_covars <- build_patient_covariates(test_imp, model_set$logistic_covariate_vars)
  test_covars[[id_var]] <- as.character(test_covars[[id_var]])   # <-- and this one
  
  new_re <- new_re %>% inner_join(test_covars, by = id_var)
  
  pred_prob <- predict(model_set$logistic_fit, newdata = new_re, type = "response")
  
  patient_mortality <- test_imp %>% distinct(.data[[id_var]], .data[[mortality_var]])
  
  new_re[[id_var]]            <- as.character(new_re[[id_var]])
  patient_mortality[[id_var]] <- as.character(patient_mortality[[id_var]])
  
  this_pred <- new_re %>%
    mutate(pred_prob = pred_prob) %>%
    left_join(patient_mortality, by = id_var) %>%
    rename(observed_mortality = all_of(mortality_var)) %>%
    mutate(imputation = k)
  
  test_predictions[[k]] <- this_pred
  
  roc_k   <- roc(this_pred$observed_mortality, this_pred$pred_prob, quiet = TRUE)
  brier_k <- mean((this_pred$pred_prob - this_pred$observed_mortality)^2)
  per_imputation_perf[[k]] <- data.frame(imputation = k, auc = as.numeric(auc(roc_k)), brier = brier_k)
  
  cat(sprintf("Imputation %d: n=%d test patients | AUC=%.3f | Brier=%.4f\n",
              k, nrow(new_re), as.numeric(auc(roc_k)), brier_k))
}

# ---- Step 7: average the 5 probabilities per patient (final reported metric) ----
all_preds <- bind_rows(test_predictions)

final_preds <- all_preds %>%
  group_by(.data[[id_var]]) %>%
  summarise(avg_pred_prob = mean(pred_prob),
            observed_mortality = first(observed_mortality),
            .groups = "drop")

roc_obj     <- roc(final_preds$observed_mortality, final_preds$avg_pred_prob, quiet = TRUE)
final_auc   <- as.numeric(auc(roc_obj))
final_brier <- mean((final_preds$avg_pred_prob - final_preds$observed_mortality)^2)

base_rate         <- mean(final_preds$observed_mortality)
brier_base_rate   <- mean((base_rate - final_preds$observed_mortality)^2)
brier_skill_score <- 1 - (final_brier / brier_base_rate)

perf_by_imputation <- bind_rows(per_imputation_perf)

# ---- detailed printed summary ----
cat("\n================= TEST-SET PERFORMANCE SUMMARY =================\n")
cat(sprintf("Response variable used for trajectory: %s\n\n", response_var))
cat("Per-imputation performance:\n")
print(perf_by_imputation, digits = 4)
cat(sprintf("\nAcross-imputation variability -- AUC: mean=%.3f, sd=%.3f | Brier: mean=%.4f, sd=%.4f\n",
            mean(perf_by_imputation$auc), sd(perf_by_imputation$auc),
            mean(perf_by_imputation$brier), sd(perf_by_imputation$brier)))
cat(sprintf("\nFINAL (averaged-probability) test-set AUC:   %.3f\n", final_auc))
cat(sprintf("FINAL (averaged-probability) test-set Brier: %.4f\n", final_brier))
cat(sprintf("Observed test-set mortality (base) rate: %.4f\n", base_rate))
cat(sprintf("Base-rate (null model) Brier score:       %.4f\n", brier_base_rate))
cat(sprintf("Brier Skill Score (%% improvement over base rate): %.1f%%\n", 100 * brier_skill_score))

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
saveRDS(list(all_preds = all_preds, final_preds = final_preds,
             perf_by_imputation = perf_by_imputation,
             final_auc = final_auc, final_brier = final_brier,
             base_rate = base_rate, brier_base_rate = brier_base_rate,
             brier_skill_score = brier_skill_score,
             response_var = response_var),
        file = file.path(out_dir, "test_predictions_final.rds"))
write.csv(final_preds, file.path(out_dir, "test_predictions_final.csv"), row.names = FALSE)

cat(sprintf("\nSaved -> %s/test_predictions_final.rds / .csv\n", out_dir))
