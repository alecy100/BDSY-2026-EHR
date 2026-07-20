# ============================================================
# STEP 4: FIT REDUCED ORDINARY LINEAR MIXED-EFFECTS MODEL
#         (same fixed effects / time var / random structure,
#          separately in each of the 5 imputed TRAINING datasets)
# STEP 5: FIT MORTALITY LOGISTIC REGRESSION using each model's
#         patient-level random intercepts/slopes as predictors
# ============================================================
# Because the variable set is already reduced (Step 3), lmer() here
# should be fast -- no glmmLasso needed for this step.
# ============================================================

library(lme4)
library(dplyr)

n_imputations <- 5
response_var  <- "respiratory_rate"
mortality_var <- "is_dead"     # <<== EDIT to your outcome column
id_var        <- "subject_id"
stay_var      <- "stay_id"

sel <- readRDS("output/variable_selection_stability.rds")
sel$selected_vars[sel$selected_vars == "stroke_binaryischemic"] <- "stroke_binary"
selected_vars <- sel$selected_vars
cat("Reduced fixed-effect set:\n"); print(selected_vars)

fix_formula <- as.formula(
  paste(response_var, "~", paste(selected_vars, collapse = " + "),
        "+ (1 + binned_hrs_since_icu |", stay_var, ")"))

dir.create("output/fitted_models", showWarnings = FALSE, recursive = TRUE)

for (k in seq_len(n_imputations)) {

  train_imp <- readRDS(sprintf("output/imputations/imputed_pair_%d.rds", k))$train

  # ---- Step 4: ordinary LMM ----
  lmm_fit <- lmer(fix_formula, data = train_imp, REML = TRUE,
                   control = lmerControl(optimizer = "bobyqa"))

  # ---- patient-level random intercept/slope table ----
  re <- ranef(lmm_fit)[[stay_var]]
  re_df <- data.frame(stay_id_factor = rownames(re),
                       random_intercept = re[, "(Intercept)"],
                       random_slope     = re[, "binned_hrs_since_icu"])

  # map stay_id_factor back to subject_id / mortality (patient level, 1 row/patient)
  patient_map <- train_imp %>%
    distinct(across(all_of(c(id_var, mortality_var)))) %>%
    mutate(stay_id_factor = as.character(train_imp[[stay_var]][match(.data[[id_var]], train_imp[[id_var]])]))
  # safer: build the id<->stay_id map directly
  patient_map <- train_imp %>%
    distinct(.data[[id_var]], .data[[stay_var]], .data[[mortality_var]]) %>%
    mutate(stay_id_factor = as.character(.data[[stay_var]]))

  patient_re <- patient_map %>%
    inner_join(re_df, by = "stay_id_factor")

  # ---- Step 5: logistic mortality model on the random effects ----
  logistic_formula <- as.formula(paste(mortality_var, "~ random_intercept + random_slope"))
  logistic_fit <- glm(logistic_formula, data = patient_re, family = binomial())

  cat(sprintf("\n[Imputation %d] LMM converged: %s | logistic AIC: %.1f\n",
              k, !is.null(lmm_fit@optinfo$conv$lme4$messages) == FALSE, AIC(logistic_fit)))

  saveRDS(list(lmm_fit = lmm_fit, logistic_fit = logistic_fit,
               patient_re = patient_re, fix_formula = fix_formula,
               selected_vars = selected_vars),
          file = sprintf("output/fitted_models/model_pair_%d.rds", k))
}

cat("\nSaved 5 (lmm_fit, logistic_fit, patient_re) sets -> output/fitted_models/\n")
