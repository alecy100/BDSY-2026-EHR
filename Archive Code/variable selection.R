df <- read.csv("output/cleanData.csv")

# ============================================================
# 0. INSTALL PACKAGES (run once, comment out after)
# ============================================================
install.packages(c("mice", "miceadds", "micemd", "mitml",
                   "lme4", "caret", "dplyr", "glmmLasso"))

# ============================================================
# 1. LOAD LIBRARIES
# ============================================================
library(mice)
library(miceadds)
library(micemd)
library(mitml)
library(lme4)
library(caret)
library(dplyr)
library(glmmLasso)

# ============================================================
# 2. DIAGNOSTICS -- run these BEFORE imputation to find
#    collinearity / singularity / cluster-size problems
# ============================================================

# 2a. Zero / near-zero variance columns
nzv <- nearZeroVar(df, saveMetrics = TRUE)
print(nzv[nzv$nzv == TRUE | nzv$zeroVar == TRUE, ])

# 2b. Pairwise correlations among numeric predictors (>0.9 flagged)
num_vars <- df %>% select(where(is.numeric)) %>% select(-subject_id, -stay_id) %>% names()
cor_mat <- cor(df[num_vars], use = "pairwise.complete.obs")
high_cor <- which(abs(cor_mat) > 0.9 & abs(cor_mat) < 1, arr.ind = TRUE)
high_cor_df <- data.frame(
  var1 = rownames(cor_mat)[high_cor[, 1]],
  var2 = colnames(cor_mat)[high_cor[, 2]],
  corr = cor_mat[high_cor]
) %>% distinct(corr, .keep_all = TRUE) %>% arrange(desc(abs(corr)))
print(high_cor_df)

# 2c. Rank deficiency check (exact linear dependence)
X <- model.matrix(~ ., data = df[num_vars])
cat("Matrix rank:", qr(X)$rank, " / Number of columns:", ncol(X), "\n")

# 2d. Cluster sizes -- singleton or tiny clusters can break 2-level methods
cluster_size_table <- table(table(df$stay_id))
print(cluster_size_table)

# ============================================================
# 3. DEFINE VARIABLE GROUPS
# ============================================================

never_impute <- c("subject_id", "stay_id", "bin",
                  grep("^is_missing_", names(df), value = TRUE))
#"map"
level1_vars_continuous <- c("heart_rate", "respiratory_rate", "spo2",
                            "gcs", "cpp", "rass", "temperature_c", "bun",
                            "creatinine", "glucose_lab", "hemoglobin", "inr",
                            "platelet", "ptt", "sodium", "wbc", "tidal_volume_obs")

#Should we take out GCS and respiratory rate? 

level1_vars_binary <- c("nacl3_hypertonic", "vasopressor_baseline",
                        "mechanical_vent_baseline")

level2_vars_continuous <- c("bmi", "height_in", "weight_lb", "age", "charlson_score")

level2_vars_categorical <- c("sex", "hypertension", "afib", "cad",
                             "race_binary", "stroke_binary")

# NOTE: if race/stroke_type are redundant recodes of race_binary/stroke_binary,
# drop them from analysis & imputation entirely rather than carrying both
redundant_dupes <- c("race", "stroke_type")

all_continuous <- c(level1_vars_continuous, level2_vars_continuous,
                    "binned_hrs_since_icu")

# ============================================================
# 4. SCALE CONTINUOUS VARIABLES (store params to back-transform later)
# ============================================================

scale_params <- df %>%
  summarise(across(all_of(all_continuous),
                   list(mean = ~mean(.x, na.rm = TRUE),
                        sd   = ~sd(.x, na.rm = TRUE))))

df_scaled <- df %>%
  mutate(across(all_of(all_continuous), ~ as.numeric(scale(.x))))

# cluster ID must be numeric/integer with no missing
df_scaled$stay_id <- as.integer(as.factor(df_scaled$stay_id))

# ============================================================
# 5. BUILD METHOD VECTOR AND PREDICTOR MATRIX
# ============================================================

ini  <- mice(df_scaled, maxit = 0)
meth <- ini$method
pred <- ini$predictorMatrix

# 5a. never impute / never use as predictor: IDs, bin index, missingness flags, dupes
meth[c(never_impute, redundant_dupes)] <- ""
pred[c(never_impute, redundant_dupes), ] <- 0
pred[, c(never_impute[never_impute != "stay_id"], redundant_dupes)] <- 0

# 5b. assign 2-level imputation methods
meth[level1_vars_continuous] <- "2l.pmm"
meth[level1_vars_binary]     <- "2l.bin"
meth[level2_vars_continuous] <- "2lonly.pmm"
meth[level2_vars_categorical] <- "2lonly.pmm"

# 5c. break known deterministic / near-collinear pairs
if (all(c("height_in", "weight_lb", "bmi") %in% names(df_scaled))) {
  pred[c("height_in", "weight_lb"), "bmi"] <- 0
  pred["bmi", c("height_in", "weight_lb")] <- 0
}
if (all(c("sbp", "dbp", "map") %in% names(df_scaled))) {
  pred[c("sbp", "dbp"), "map"] <- 0
  pred["map", c("sbp", "dbp")] <- 0
}
if (all(c("map", "cpp") %in% names(df_scaled))) {
  pred["map", "cpp"] <- 0
  pred["cpp", "map"] <- 0
}

# 5d. auto-thin remaining weak/redundant predictors based on correlation
pred_auto <- quickpred(df_scaled, mincor = 0.25,
                       exclude = c(never_impute, redundant_dupes))
# merge: keep manual overrides above, but use quickpred thinning elsewhere
pred[pred == 1 & pred_auto == 0] <- 0

# 5e. cluster variable coding (REQUIRED for 2l methods)
pred[, "stay_id"] <- -2
pred["stay_id", ] <- 0

# ============================================================
# 6. RUN MULTIPLE IMPUTATION
# ============================================================

imp <- mice(df_scaled, method = meth, predictorMatrix = pred,
            m = 20, maxit = 10, ridge = 1e-2, seed = 1)

# check convergence -- should look like white noise, not trending
plot(imp)

# ============================================================
# 7. FIT MIXED MODEL ON EACH IMPUTED DATASET AND POOL
# ============================================================

fits <- with(imp, lmer(map ~ stroke_binary + age + bmi + charlson_score + gcs +
                         binned_hrs_since_icu + (binned_hrs_since_icu | stay_id)))




pooled <- testEstimates(as.mitml.result(fits), var.comp = TRUE)
summary(pooled)

# ============================================================
# 8. BACK-TRANSFORM A COEFFICIENT TO ORIGINAL UNITS (example: age)
# ============================================================

beta_scaled_age <- pooled$estimates["age", "Estimate"]
beta_original_age <- beta_scaled_age / scale_params$age_sd
beta_original_age


# ============================================================
# ============================================================
# LASSO VARIABLE SELECTION FOR LINEAR MIXED-EFFECTS MODELS
# TWO OUTCOMES: MAP and (log) RESPIRATORY RATE
# ============================================================
# ============================================================
# Uses the `imp` (mids object) and `scale_params` created above.
#
# Two separate sets of LASSO linear mixed models are fit -- one with
# map as the outcome, one with (log-transformed) respiratory_rate as
# the outcome. Each model uses a random intercept + random slope for
# binned_hrs_since_icu, grouped by stay_id -- the LASSO-penalized
# analogue of  (binned_hrs_since_icu | stay_id)  in lme4 syntax.
#
# MAP and respiratory rate are mutually excluded as covariates: MAP
# never appears as a predictor in the respiratory-rate equations, and
# respiratory rate never appears as a predictor in the MAP equations
# (each is naturally excluded from its own equation as well).
#
# map is used on its native scale (it was deliberately left out of the
# standardization step above -- the "#\"map\"" line in Section 3).
# lambda is left as a user-defined value per outcome (Section 13) so
# either model can be re-tuned independently, e.g. via cross-validation.
# ============================================================

# ------------------------------------------------------------
# 9. EXTRACT THE m COMPLETED (IMPUTED) DATASETS
# ------------------------------------------------------------
imp_list <- mice::complete(imp, action = "all")   # list of m data frames
m <- length(imp_list)

# ------------------------------------------------------------
# 10. LOG-TRANSFORM RESPIRATORY RATE IN EACH IMPUTED DATASET
# ------------------------------------------------------------
# respiratory_rate was z-scored (mean/sd stored in scale_params) before
# imputation, so log() can't be applied to it directly (values can be
# negative on the standardized scale). We back-transform to the native
# scale first, log it, then re-standardize the logged variable so it's
# on a comparable scale to the other (already-standardized) predictors.
# This log_respiratory_rate variable is used below as its own outcome
# (Section 13) as well as being excluded as a covariate everywhere else.

rr_mean <- scale_params$respiratory_rate_mean
rr_sd   <- scale_params$respiratory_rate_sd

imp_list <- lapply(imp_list, function(d) {
  rr_raw                 <- d$respiratory_rate * rr_sd + rr_mean
  d$log_respiratory_rate <- as.numeric(scale(log(rr_raw)))
  d$respiratory_rate     <- NULL
  d
})

# ------------------------------------------------------------
# 11. SHARED CANDIDATE PREDICTOR POOL
# ------------------------------------------------------------
# One predictor set is used for BOTH outcome models. map and
# log_respiratory_rate are removed from this pool so that neither can
# ever be a covariate -- including in its own equation, and in the
# other outcome's equation.
#
# Also excluded (as before):
#   - cpp: derived directly from map, so it would leak the map outcome
#   - height_in / weight_lb: collinear with bmi (Section 5c)
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

candidate_predictors <- setdiff(full_variable_pool, mutual_exclusion_vars)

# Random effects: random intercept + random slope on binned_hrs_since_icu,
# nested within stay_id -- used for both outcome models
rnd_structure <- list(stay_id = ~ 1 + binned_hrs_since_icu)

# ------------------------------------------------------------
# 12. HELPER: FIT LASSO LMM ON ONE IMPUTED DATASET
# ------------------------------------------------------------
fit_lasso_lmm <- function(dat, lambda, fix_formula, rnd_structure, response_var) {
  
  dat$stay_id <- as.factor(dat$stay_id)
  
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
  
  list(coefficients = fit$coefficients, rss = rss)
}

# ------------------------------------------------------------
# 13. OUTCOME CONFIGURATIONS (lambda is user-defined per outcome)
# ------------------------------------------------------------
outcomes_config <- list(
  map = list(
    response_var = "map",
    lambda       = 0.01    # <<== set/tune the MAP-model penalty here
  ),
  respiratory_rate = list(
    response_var = "log_respiratory_rate",
    lambda       = 0.01    # <<== set/tune the respiratory-rate-model penalty here
  )
)

# ------------------------------------------------------------
# 14. RUN THE LASSO LMM FOR EACH OUTCOME, ACROSS EVERY IMPUTED DATASET
# ------------------------------------------------------------
lasso_results <- list()   # top-level storage: one entry per outcome

for (outcome_name in names(outcomes_config)) {
  
  cfg          <- outcomes_config[[outcome_name]]
  response_var <- cfg$response_var
  lambda       <- cfg$lambda
  
  fix_formula <- as.formula(
    paste(response_var, "~", paste(candidate_predictors, collapse = " + "))
  )
  
  coef_list <- vector("list", m)
  rss_vec   <- numeric(m)
  
  for (i in seq_len(m)) {
    
    result <- tryCatch(
      fit_lasso_lmm(imp_list[[i]], lambda, fix_formula, rnd_structure, response_var),
      error = function(e) {
        warning(sprintf("[%s] Imputation %d failed to converge: %s",
                        outcome_name, i, conditionMessage(e)))
        NULL
      }
    )
    
    if (!is.null(result)) {
      coef_list[[i]] <- result$coefficients
      rss_vec[i]     <- result$rss
      cat(sprintf("[%s] Imputation %d/%d done -- RSS = %.3f\n",
                  outcome_name, i, m, result$rss))
    } else {
      coef_list[[i]] <- NA
      rss_vec[i]      <- NA
    }
  }
  
  ok <- !sapply(coef_list, function(x) length(x) == 1 && is.na(x))
  
  if (sum(ok) == 0) {
    warning(sprintf(
      "[%s] glmmLasso failed to converge on every imputed dataset -- try a different lambda.",
      outcome_name
    ))
    next
  }
  
  coef_mat  <- do.call(rbind, coef_list[ok])
  avg_coefs <- colMeans(coef_mat)
  avg_rss   <- mean(rss_vec[ok])
  
  avg_coefficients_df <- data.frame(
    term            = names(avg_coefs),
    avg_coefficient = as.numeric(avg_coefs),
    row.names       = NULL
  )
  
  # ---- storage for this outcome ----
  lasso_results[[outcome_name]] <- list(
    response_var                = response_var,
    lambda                      = lambda,
    n_converged                 = sum(ok),
    coefficients_by_imputation  = coef_list,
    rss_by_imputation           = rss_vec,
    avg_coefficients            = avg_coefficients_df,
    avg_rss                     = avg_rss
  )
  
  cat("\n=== LASSO linear mixed model for outcome:", outcome_name,
      "-- averaged over", sum(ok), "of", m,
      "imputations (lambda =", lambda, ") ===\n\n")
  print(avg_coefficients_df, digits = 4)
  cat("\nAverage RSS across imputations:", round(avg_rss, 3), "\n\n")
}

# Access results, e.g.:
#   lasso_results$map$avg_coefficients
#   lasso_results$map$avg_rss
#   lasso_results$respiratory_rate$avg_coefficients
#   lasso_results$respiratory_rate$avg_rss

# ------------------------------------------------------------
# 15. (OPTIONAL) BACK-TRANSFORM A COEFFICIENT TO ORIGINAL UNITS
# ------------------------------------------------------------
# All predictors above except the 0/1 binary indicators were
# standardized before imputation (mean/sd stored in scale_params), so
# the averaged LASSO coefficients are "per 1 SD" effects. To convert
# one back to its native units -- e.g. age, in the MAP model, to
# "per 1 year":
#
# beta_scaled_age   <- lasso_results$map$avg_coefficients %>%
#                        filter(term == "age") %>% pull(avg_coefficient)
# beta_original_age <- beta_scaled_age / scale_params$age_sd