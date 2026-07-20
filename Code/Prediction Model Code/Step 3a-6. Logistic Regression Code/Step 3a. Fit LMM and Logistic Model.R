# ============================================================
# STEP 4: FIT REDUCED ORDINARY LMM (fixed effects + random intercept/slope)
# STEP 5: FIT MORTALITY LOGISTIC REGRESSION using each imputation's random
#         effects PLUS patient-level covariate summaries (mean for
#         time-varying vars, single value for time-invariant vars)
# ============================================================
# CHANGES from the previous version:
#   - the mortality logistic model now includes the selected covariates
#     directly (not just random_intercept/random_slope) -- time-varying
#     covariates are summarized per patient using the MEAN over their
#     observed window; "rass" is excluded from this covariate list per
#     project decision (still usable as an LMM fixed effect upstream)
#   - full_variable_pool / exclusion logic now come from
#     utils_variable_pool.R (shared with 02a/02b/04)
#   - detailed model summaries are printed AND saved to rds for each of
#     the 5 imputations, plus a pooled (Rubin's rules) summary across all 5
#   - outputs written to prediction/<response_var>/ instead of output/
#   - removed a duplicate/overwritten patient_map computation that was
#     left over in the original script
#   - NEW: excluded_subject_ids removed from train_imp before any model
#     is fit; propagates downstream to 04/05/06 since they consume this
#     script's saved outputs rather than reloading raw imputed data
# ============================================================

library(lme4)
library(dplyr)


n_imputations <- 5
response_var  <- "map"   # <<== EDIT: must match 02a/02b's response_var
mortality_var <- "is_dead"            # <<== EDIT to your outcome column
id_var        <- "subject_id"
stay_var      <- "stay_id"

source("Prediction Model/utils_variable_pool.R")
source("Prediction Model/utils_pooling.R")

out_dir <- sprintf("prediction/%s", response_var)

# ============================================================
# EXCLUDE SPECIFIC SUBJECT IDs FROM ALL DOWNSTREAM MODELING
# Applied here, before any model is fit, so 04/05/06 never see these
# patients (they consume this script's saved outputs, not raw imputed data).
# ============================================================
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

cat(sprintf("Excluding %d subject IDs before model fitting.\n", length(excluded_subject_ids)))

sel <- readRDS(file.path(out_dir, "variable_selection_stability.rds"))
sel$selected_vars[sel$selected_vars == "stroke_binaryischemic"] <- "stroke_binary"
selected_vars <- sel$selected_vars
cat("Reduced fixed-effect set (LMM):\n"); print(selected_vars)

# covariates going into the MORTALITY logistic model: same reduced set,
# minus anything on the exclusion list (rass excluded per project decision)
logistic_covariate_vars <- setdiff(selected_vars, mortality_model_exclude_vars)
cat("Covariates added directly to the mortality logistic model:\n")
print(logistic_covariate_vars)

time_invariant_selected <- intersect(logistic_covariate_vars, level2_vars_time_invariant)
time_varying_selected   <- setdiff(logistic_covariate_vars, level2_vars_time_invariant)
cat("  time-invariant (single value/patient):", paste(time_invariant_selected, collapse = ", "), "\n")
cat("  time-varying (summarized as MEAN/patient):", paste(time_varying_selected, collapse = ", "), "\n")

fix_formula <- as.formula(
  paste(response_var, "~", paste(selected_vars, collapse = " + "),
        "+ (1 + binned_hrs_since_icu |", stay_var, ")"))

# ---- build one patient-level covariate summary table from a given data
#      frame: time-invariant vars -> distinct value; time-varying -> MEAN
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

dir.create(file.path(out_dir, "fitted_models"), showWarnings = FALSE, recursive = TRUE)

lmm_fits      <- vector("list", n_imputations)
logistic_fits <- vector("list", n_imputations)

for (k in seq_len(n_imputations)) {
  
  train_imp <- readRDS(sprintf("prediction/imputations/imputed_pair_%d.rds", k))$train
  
  # ---- apply subject-level exclusion before any modeling touches this data ----
  n_before <- n_distinct(train_imp[[id_var]])
  train_imp <- train_imp %>% filter(!.data[[id_var]] %in% excluded_subject_ids)
  n_after <- n_distinct(train_imp[[id_var]])
  cat(sprintf("[Imputation %d] Excluded %d patients (%d -> %d unique patients, %d rows remain)\n",
              k, n_before - n_after, n_before, n_after, nrow(train_imp)))
  
  # ---- Step 4: ordinary LMM ----
  lmm_fit <- lmer(fix_formula, data = train_imp, REML = TRUE,
                  control = lmerControl(optimizer = "bobyqa"))
  
  re <- ranef(lmm_fit)[[stay_var]]
  re_df <- data.frame(stay_id_factor = rownames(re),
                      random_intercept = re[, "(Intercept)"],
                      random_slope     = re[, "binned_hrs_since_icu"])
  
  # patient <-> stay_id map (built directly; the original script computed
  # this twice, once via a buggy version that got overwritten -- removed here)
  patient_map <- train_imp %>%
    distinct(.data[[id_var]], .data[[stay_var]], .data[[mortality_var]]) %>%
    mutate(stay_id_factor = as.character(.data[[stay_var]]))
  
  patient_re <- patient_map %>%
    inner_join(re_df, by = "stay_id_factor")
  
  # ---- covariate summary table (mean for time-varying, distinct for static) ----
  patient_covars <- build_patient_covariates(train_imp)
  patient_re <- patient_re %>% inner_join(patient_covars, by = id_var)
  
  # ---- Step 5: logistic mortality model on random effects + covariates ----
  logistic_formula <- as.formula(paste(
    mortality_var, "~ random_intercept + random_slope",
    if (length(logistic_covariate_vars) > 0)
      paste("+", paste(logistic_covariate_vars, collapse = " + ")) else ""
  ))
  logistic_fit <- glm(logistic_formula, data = patient_re, family = binomial())
  
  lmm_fits[[k]]      <- lmm_fit
  logistic_fits[[k]] <- logistic_fit
  
  # ---- detailed per-imputation summaries: print + save ----
  cat(sprintf("\n================= Imputation %d (response_var = %s) =================\n",
              k, response_var))
  cat("\n--- LMM summary ---\n")
  print(summary(lmm_fit))
  cat("\n--- Mortality logistic regression summary ---\n")
  print(summary(logistic_fit))
  cat(sprintf("\nLMM converged: %s | Logistic AIC: %.1f\n",
              length(lmm_fit@optinfo$conv$lme4$messages) == 0, AIC(logistic_fit)))
  
  saveRDS(list(lmm_fit = lmm_fit, logistic_fit = logistic_fit,
               lmm_summary = summary(lmm_fit), logistic_summary = summary(logistic_fit),
               patient_re = patient_re, fix_formula = fix_formula,
               logistic_formula = logistic_formula,
               selected_vars = selected_vars,
               logistic_covariate_vars = logistic_covariate_vars,
               time_invariant_selected = time_invariant_selected,
               time_varying_selected = time_varying_selected,
               excluded_subject_ids = excluded_subject_ids),
          file = file.path(out_dir, "fitted_models", sprintf("model_pair_%d.rds", k)))
}

# ---- pooled summary across the 5 imputations (Rubin's rules) ----
cat("\n================= POOLED SUMMARY ACROSS 5 IMPUTATIONS =================\n")

cat("\n--- Pooled LMM fixed effects ---\n")
pooled_lmm <- pool_lmer_fixef_list(lmm_fits)
print(pooled_lmm, digits = 4)

cat("\n--- Pooled mortality logistic regression coefficients ---\n")
pooled_logistic <- pool_glm_list(logistic_fits)
print(pooled_logistic, digits = 4)

saveRDS(list(pooled_lmm = pooled_lmm, pooled_logistic = pooled_logistic,
             response_var = response_var, n_imputations = n_imputations,
             selected_vars = selected_vars,
             logistic_covariate_vars = logistic_covariate_vars,
             excluded_subject_ids = excluded_subject_ids),
        file = file.path(out_dir, "pooled_model_summary.rds"))

cat(sprintf("\nSaved %d (lmm_fit, logistic_fit, summaries) sets -> %s/fitted_models/\n",
            n_imputations, out_dir))
cat(sprintf("Saved pooled summary -> %s/pooled_model_summary.rds\n", out_dir))
