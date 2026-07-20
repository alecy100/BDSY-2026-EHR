# ============================================================
# STEP 2: MULTIPLE IMPUTATION (train + test together, test rows
#         imputed but excluded from estimating imputation models)
# ============================================================
# Uses mice's `ignore=` argument: rows with ignore==TRUE (test rows)
# are imputed, but do NOT contribute to fitting the imputation models.
# This is the standard, correct way to avoid test-set leakage into
# imputation while still producing completed test data.
#
# Produces 5 (train, test) pairs: output/imputed_pair_1.rds ... _5.rds
#
# Reuses the variable-group / method setup from your existing bootstrap
# script (level1_vars_continuous, etc.) -- EDIT these to match your
# actual cleanData.csv columns if they differ.
# ============================================================

library(mice)
library(miceadds)
library(dplyr)

set.seed(2025)

n_imputations <- 5     # advisor: 5 (not 10)
mice_maxit    <- 10
id_var        <- "subject_id"
stay_var      <- "stay_id"
mortality_var <- "is_dead"   # <<== EDIT to your outcome column name

split <- readRDS("output/train_test_split.rds")

# ---- combine, tag which rows are test (to be ignored during estimation)
train_df <- split$train_df %>% mutate(.is_test = FALSE)
test_df  <- split$test_df  %>% mutate(.is_test = TRUE)
full_df  <- bind_rows(train_df, test_df)
ignore_vec <- full_df$.is_test
full_df$.is_test <- NULL

# ---- variable groups (copy from your cleaning pipeline / doc2 script)
level1_vars_continuous <- c("heart_rate", "respiratory_rate", "spo2",
                             "gcs", "cpp", "rass", "temperature_c", "bun",
                             "creatinine", "glucose_lab", "hemoglobin", "inr",
                             "platelet", "ptt", "sodium", "wbc", "tidal_volume_obs")
level1_vars_binary <- c("nacl3_hypertonic", "vasopressor_baseline",
                         "mechanical_vent_baseline")
level2_vars_continuous  <- c("bmi", "height_in", "weight_lb", "age", "charlson_score")
level2_vars_categorical <- c("sex", "hypertension", "afib", "cad",
                              "race_binary", "stroke_binary")
redundant_dupes <- c("race", "stroke_type")
all_continuous  <- c(level1_vars_continuous, level2_vars_continuous, "binned_hrs_since_icu")

never_impute <- c(id_var, stay_var, "bin", mortality_var,
                   grep("^is_missing_", names(full_df), value = TRUE))

# ---- scale continuous vars using TRAINING rows' mean/sd only (no test leakage)
scale_stats <- full_df %>% filter(!ignore_vec) %>%
  summarise(across(all_of(all_continuous),
                    list(mean = ~mean(.x, na.rm = TRUE), sd = ~sd(.x, na.rm = TRUE))))

df_scaled <- full_df
for (v in all_continuous) {
  m <- scale_stats[[paste0(v, "_mean")]]
  s <- scale_stats[[paste0(v, "_sd")]]
  df_scaled[[v]] <- (df_scaled[[v]] - m) / s
}
df_scaled[[stay_var]] <- as.integer(as.factor(df_scaled[[stay_var]]))

ini  <- mice(df_scaled, maxit = 0)
meth <- ini$method
pred <- ini$predictorMatrix

meth[c(never_impute, redundant_dupes)]      <- ""
pred[c(never_impute, redundant_dupes), ]    <- 0
pred[, c(never_impute[never_impute != stay_var], redundant_dupes)] <- 0
# advisor requirement: never use (test) mortality to impute predictors
pred[, mortality_var] <- 0

meth[level1_vars_continuous]  <- "2l.pmm"
meth[level1_vars_binary]      <- "2l.bin"
meth[level2_vars_continuous]  <- "2lonly.pmm"
meth[level2_vars_categorical] <- "2lonly.pmm"

pred_auto <- quickpred(df_scaled, mincor = 0.25, exclude = c(never_impute, redundant_dupes))
pred[pred == 1 & pred_auto == 0] <- 0
pred[, stay_var]   <- -2
pred[stay_var, ]   <- 0

cat(sprintf("Running mice on %d rows (%d train / %d test) with ignore=...\n",
            nrow(df_scaled), sum(!ignore_vec), sum(ignore_vec)))

imp <- mice(df_scaled, method = meth, predictorMatrix = pred,
            m = n_imputations, maxit = mice_maxit, ridge = 1e-2,
            ignore = ignore_vec, seed = 2025)

imp_list <- mice::complete(imp, action = "all")

dir.create("output/imputations", showWarnings = FALSE, recursive = TRUE)

for (k in seq_len(n_imputations)) {
  d <- imp_list[[k]]
  d[[stay_var]] <- as.factor(d[[stay_var]])
  d$.row_is_test <- ignore_vec
  
  train_imp <- d[!d$.row_is_test, ]
  test_imp  <- d[ d$.row_is_test, ]
  train_imp$.row_is_test <- NULL
  test_imp$.row_is_test  <- NULL
  
  saveRDS(list(train = train_imp, test = test_imp),
          file = sprintf("output/imputations/imputed_pair_%d.rds", k))
  cat(sprintf("Saved imputation %d: %d train rows, %d test rows\n",
              k, nrow(train_imp), nrow(test_imp)))
}