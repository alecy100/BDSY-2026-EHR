# ============================================================
# EXPORT IMPUTED DATA TO CSV (for the Python RNN pipeline)
# ============================================================
# Reads each imputed_pair_k.rds (train + test data frames) produced by
# 01_multiple_imputation.R and writes them out as plain CSVs, since
# Python doesn't natively read .rds. One CSV per (imputation, split):
#   prediction/imputations_csv/imputed_pair_1_train.csv
#   prediction/imputations_csv/imputed_pair_1_test.csv
#   ... through imputation 5
#
# APPLIES THE SAME excluded_subject_ids FILTER used everywhere else in
# the project (patients who died within the first 48h of ICU admission
# -- see utils_excluded_patients.R). This is on by default (APPLY_EXCLUSION
# below) so the RNN's data matches what the logistic/RF pipelines are
# using; set it to FALSE only if you deliberately want the unfiltered
# cohort for some other comparison.
#
# NOTE: this writes ALL columns present in the imputed data (not a
# reduced covariate set) -- matches the plan to feed the RNN the full
# originally-identified variable set rather than the glmmLasso-reduced
# subset used by the logistic/RF mortality models.
# ============================================================

library(dplyr)

n_imputations  <- 5
APPLY_EXCLUSION <- TRUE   # <<== set FALSE to export the unfiltered cohort instead

source("Code/Prediction Model Code/Utils/utils_excluded_patients.R")

out_dir <- "prediction/imputations_csv"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

id_var <- "subject_id"

for (k in seq_len(n_imputations)) {
  imp <- readRDS(sprintf("prediction/imputations/imputed_pair_%d.rds", k))
  train_df <- imp$train
  test_df  <- imp$test

  if (APPLY_EXCLUSION) {
    n_train_before <- n_distinct(train_df[[id_var]])
    n_test_before  <- n_distinct(test_df[[id_var]])

    train_df <- train_df %>% filter(!.data[[id_var]] %in% excluded_subject_ids)
    test_df  <- test_df  %>% filter(!.data[[id_var]] %in% excluded_subject_ids)

    cat(sprintf("[Imputation %d] Train: %d -> %d patients | Test: %d -> %d patients (excluded_subject_ids applied)\n",
                k, n_train_before, n_distinct(train_df[[id_var]]),
                n_test_before, n_distinct(test_df[[id_var]])))
  } else {
    cat(sprintf("[Imputation %d] Train: %d patients | Test: %d patients (NO exclusion applied)\n",
                k, n_distinct(train_df[[id_var]]), n_distinct(test_df[[id_var]])))
  }

  train_path <- file.path(out_dir, sprintf("imputed_pair_%d_train.csv", k))
  test_path  <- file.path(out_dir, sprintf("imputed_pair_%d_test.csv", k))

  write.csv(train_df, train_path, row.names = FALSE)
  write.csv(test_df,  test_path,  row.names = FALSE)

  cat(sprintf("  Saved -> %s (%d rows), %s (%d rows)\n",
              train_path, nrow(train_df), test_path, nrow(test_df)))
}

cat(sprintf("\nAll done. CSVs written to %s/\n", out_dir))
