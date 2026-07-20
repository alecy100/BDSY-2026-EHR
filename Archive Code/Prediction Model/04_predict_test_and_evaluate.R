# ============================================================
# STEP 6: "REVERSE ENGINEER" RANDOM EFFECTS FOR TEST PATIENTS
#         + predict mortality from each imputation's logistic model
# STEP 7: AVERAGE THE 5 PREDICTED PROBABILITIES, EVALUATE
# ============================================================
# There is no built-in function for this (lme4::ranef only returns
# effects for clusters that were IN the original fit; predict(...,
# allow.new.levels=TRUE) gives population-average predictions with
# random effect = 0, which is not what we want).
#
# Instead we use the standard closed-form empirical-Bayes / BLUP
# formula, treating the fixed effects (beta_hat), residual variance
# (sigma^2) and random-effect covariance matrix (G) from the TRAINING
# fit as known/fixed, and solving for each test patient's own random
# effect using their own observed biomarker trajectory:
#
#     b_hat_i = G Z_i' (Z_i G Z_i' + sigma^2 I)^(-1) (y_i - X_i beta_hat)
#
# where, for patient i: y_i are their observed outcome values (map),
# X_i is their fixed-effect design matrix, Z_i is their random-effect
# design matrix (intercept + time), and (y_i - X_i beta_hat) are their
# "residuals" under the population-average (fixed-effects-only) model.
# This is exactly the formula lme4 uses internally for in-sample BLUPs;
# here we just apply it out-of-sample.
# ============================================================

library(lme4)
library(dplyr)
library(pROC)

n_imputations <- 5
response_var  <- "respiratory_rate"
mortality_var <- "is_dead"
id_var        <- "subject_id"
stay_var      <- "stay_id"
time_var      <- "binned_hrs_since_icu"

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
  
  # fixed-effects-only design matrix, built from the same terms as the model
  fixed_form <- formula(lmm_fit, fixed.only = TRUE)
  X_all <- model.matrix(fixed_form, data = newdata)
  # keep only rows model.matrix kept (in case of NAs) -- match by ROW NAME,
  # not position: rownames(X_all) are the original labels carried over from
  # newdata (not a clean 1:nrow sequence), so converting them to integers
  # and using them as positional indices silently pulls the wrong rows.
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

test_predictions <- vector("list", n_imputations)

for (k in seq_len(n_imputations)) {
  model_set <- readRDS(sprintf("output/fitted_models/model_pair_%d.rds", k))
  test_imp  <- readRDS(sprintf("output/imputations/imputed_pair_%d.rds", k))$test
  
  # patients must have their observed pre-prediction-window measurements only --
  # confirm test_imp already reflects the intended prediction-time window
  # before this point (i.e. no post-outcome / future data leaking in).
  new_re <- estimate_new_random_effects(model_set$lmm_fit, test_imp,
                                        id_var, stay_var, time_var, response_var)
  
  pred_prob <- predict(model_set$logistic_fit,
                       newdata = new_re, type = "response")
  
  patient_mortality <- test_imp %>% distinct(.data[[id_var]], .data[[mortality_var]])
  
  # new_re's id column comes from names(results) inside estimate_new_random_effects,
  # and list names are always character -- even though subject_id started as
  # integer. Coerce both sides to character before joining so the types match
  # regardless of subject_id's original class.
  new_re[[id_var]]            <- as.character(new_re[[id_var]])
  patient_mortality[[id_var]] <- as.character(patient_mortality[[id_var]])
  
  test_predictions[[k]] <- new_re %>%
    mutate(pred_prob = pred_prob) %>%
    left_join(patient_mortality, by = id_var) %>%
    rename(observed_mortality = all_of(mortality_var)) %>%
    mutate(imputation = k)
  
  cat(sprintf("Imputation %d: predicted mortality probability for %d test patients\n",
              k, nrow(new_re)))
}

# ---- Step 7: average the 5 probabilities per patient ----
all_preds <- bind_rows(test_predictions)

final_preds <- all_preds %>%
  group_by(.data[[id_var]]) %>%
  summarise(avg_pred_prob = mean(pred_prob),
            observed_mortality = first(observed_mortality),
            .groups = "drop")

# ---- performance ----
roc_obj <- roc(final_preds$observed_mortality, final_preds$avg_pred_prob, quiet = TRUE)
cat(sprintf("\nTest-set AUC: %.3f\n", auc(roc_obj)))

brier <- mean((final_preds$avg_pred_prob - final_preds$observed_mortality)^2)
cat(sprintf("Test-set Brier score: %.4f\n", brier))

dir.create("output", showWarnings = FALSE)
saveRDS(list(all_preds = all_preds, final_preds = final_preds,
             auc = as.numeric(auc(roc_obj)), brier = brier),
        file = "output/test_predictions_final.rds")
write.csv(final_preds, "output/test_predictions_final.csv", row.names = FALSE)

cat("\nSaved -> output/test_predictions_final.rds / .csv\n")

# ---- base-rate (null model) comparison ----
base_rate <- mean(final_preds$observed_mortality)
brier_base_rate <- mean((base_rate - final_preds$observed_mortality)^2)
brier_skill_score <- 1 - (brier / brier_base_rate)

cat(sprintf("Observed test-set mortality (base) rate: %.4f\n", base_rate))
cat(sprintf("Base-rate (null model) Brier score:       %.4f\n", brier_base_rate))
cat(sprintf("Brier Skill Score (%% improvement over base rate): %.1f%%\n", 100 * brier_skill_score))
