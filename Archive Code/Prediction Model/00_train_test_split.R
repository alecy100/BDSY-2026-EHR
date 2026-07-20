# ============================================================
# STEP 1: PATIENT-LEVEL 80/20 TRAIN/TEST SPLIT (stratified by mortality)
# ============================================================
# Splits at the PATIENT level (subject_id), keeping all repeated rows
# for a patient together in whichever set they're assigned to.
# Stratifies on the outcome so train/test mortality rates match.
#
# EDIT: set `mortality_var` and `id_var` to match your cleanData.csv
# ============================================================

library(dplyr)

set.seed(2025)  # fix for reproducibility

id_var         <- "subject_id"
mortality_var  <- "is_dead"    # <<== EDIT: your patient-level 0/1 outcome column
train_frac     <- 0.70

df <- read.csv("output/stroke_clean_with_mortality.csv")

# One row per patient with their outcome (assumes outcome is constant
# within subject_id -- check this assumption on your real data)
patient_lookup <- df %>%
  distinct(across(all_of(c(id_var, mortality_var))))

stopifnot(nrow(patient_lookup) == length(unique(patient_lookup[[id_var]])))

# Stratified split: sample train_frac within each mortality stratum
train_ids <- patient_lookup %>%
  group_by(across(all_of(mortality_var))) %>%
  slice_sample(prop = train_frac) %>%
  ungroup() %>%
  pull(all_of(id_var))

test_ids <- setdiff(patient_lookup[[id_var]], train_ids)

train_df <- df %>% filter(.data[[id_var]] %in% train_ids)
test_df  <- df %>% filter(.data[[id_var]] %in% test_ids)

cat(sprintf("Patients: %d train / %d test (%.1f%% train)\n",
            length(train_ids), length(test_ids),
            100 * length(train_ids) / (length(train_ids) + length(test_ids))))
cat(sprintf("Mortality rate -- train: %.3f, test: %.3f, overall: %.3f\n",
            mean(patient_lookup[[mortality_var]][patient_lookup[[id_var]] %in% train_ids]),
            mean(patient_lookup[[mortality_var]][patient_lookup[[id_var]] %in% test_ids]),
            mean(patient_lookup[[mortality_var]])))
cat(sprintf("Rows -- train: %d, test: %d\n", nrow(train_df), nrow(test_df)))

dir.create("output", showWarnings = FALSE)
saveRDS(list(train_ids = train_ids, test_ids = test_ids,
             train_df = train_df, test_df = test_df),
        file = "output/train_test_split.rds")

write.csv(train_df, "output/train_raw.csv", row.names = FALSE)
write.csv(test_df,  "output/test_raw.csv",  row.names = FALSE)

cat("\nSaved -> output/train_test_split.rds, output/train_raw.csv, output/test_raw.csv\n")
cat("IMPORTANT: from this point on, never touch test_raw.csv until Step 6 --\n")
cat("no variable selection, no penalty tuning, no model fitting should see it.\n")
