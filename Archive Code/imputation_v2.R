df <- read.csv("output/cleanData.csv")

# ============================================================
# 0. INSTALL PACKAGES (run once, comment out after)
# ============================================================
install.packages(c("mice", "miceadds", "micemd", "mitml",
                   "lme4", "caret", "dplyr"))

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
# 9. SAVE OBJECTS NEEDED BY THE DOWNSTREAM LASSO SCRIPTS
# ============================================================
# Saving here means the (slow) imputation step never has to be re-run
# just because a downstream glmmLasso script crashes.

saveRDS(imp, file = "output/imp.rds")
saveRDS(scale_params, file = "output/scale_params.rds")

cat("Saved imp -> output/imp.rds\n")
cat("Saved scale_params -> output/scale_params.rds\n")