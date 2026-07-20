# ============================================================
# STEP 8: FIT MEAN-MAP AND SD-MAP MORTALITY LOGISTIC REGRESSIONS
# ============================================================
# Two additional mortality models, fit the same way as the existing
# random-effects model in 03/04 -- once per imputed (train, test) pair,
# then pooled across the 5 imputations with Rubin's rules
# (utils_pooling.R). Both use the same demographic/clinical covariates
# and differ only in which patient-level MAP summary is included:
#
#   Model A ("mean_map"): race_binary + age + sex + bmi + stroke_binary
#                          + each patient's MEAN MAP over their observed window
#   Model B ("sd_map"):   race_binary + age + sex + bmi + stroke_binary
#                          + each patient's SD of MAP over their observed window
#
# ASSUMPTIONS (flagging since the prompt left these implicit):
#   - "race (binary)" -> race_binary
#   - "stroke type"   -> stroke_binary. The pipeline already uses
#     stroke_binary (ischemic/hemorrhagic) everywhere else as THE
#     stroke-type covariate (see utils_variable_pool.R); the raw
#     multi-level "stroke_type"/"race" columns were deliberately
#     excluded upstream as redundant_dupes and are not re-introduced here.
#   - age/bmi/map are used AS THEY APPEAR in the imputed datasets, i.e.
#     standardized (z-scored on TRAIN-only mean/sd, see
#     01_multiple_imputation.R) -- consistent with how these variables
#     are used everywhere else in this pipeline (e.g. the LMM in 03).
#     Odds ratios for age/bmi/mean_map/sd_map are therefore "per 1 SD"
#     effects, not per-raw-unit effects.
#   - MAP mean/sd are computed from the response_var = "map" imputed
#     pairs (prediction/imputations/imputed_pair_k.rds) -- the same
#     five imputations already used for the existing random-effects
#     mortality model, so all three models are evaluated on the same
#     imputed data.
#   - Outcome = is_dead (mortality_var), at the patient level.
#
# EXCLUDED SUBJECT IDs: inlined directly below (337 IDs), copied to
# match the excluded_subject_ids vector embedded in the current
# 03_fit_lmm_and_logistic.R exactly. This is a hand-kept copy, not a
# shared/sourced file -- if 03's list changes again, this vector needs
# to be updated here too. Applied to both train and test right after
# each imputed pair is loaded, before any patient-level table is built
# or model is fit. 06_auc_ci_and_or_plots.R needs no separate copy: it
# only reads outputs already produced by 03/05, which will already
# reflect this exclusion.
#
# NEW: also computes/saves imputation-averaged TRAINING predictions
# (final_mean_train, final_sd_train), not just test (final_mean,
# final_sd). These exist purely so 11_confusion_matrices.R can pick a
# classification threshold (e.g. Youden's J) from TRAINING data instead
# of the test set itself, then apply that fixed threshold to the test
# predictions -- picking a threshold from the same test data you then
# evaluate on would bias the reported sensitivity/specificity.
# ============================================================

library(dplyr)

n_imputations <- 5
id_var        <- "subject_id"
mortality_var <- "is_dead"     # <<== EDIT to your outcome column if different

demo_vars <- c("race_binary", "age", "sex", "bmi", "stroke_binary")

source("utils_pooling.R")

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

out_dir <- "prediction/mean_sd_map_models"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- build one row per patient: demographics (distinct) + MAP mean/sd ----
build_patient_table <- function(df, id_var, demo_vars, mortality_var) {
  demo_df <- df %>% distinct(across(all_of(c(id_var, demo_vars, mortality_var))))
  map_summary <- df %>%
    group_by(across(all_of(id_var))) %>%
    summarise(mean_map = mean(map, na.rm = TRUE),
              sd_map   = sd(map, na.rm = TRUE),
              .groups = "drop")
  demo_df %>% inner_join(map_summary, by = id_var)
}

form_mean <- as.formula(paste(mortality_var, "~", paste(c(demo_vars, "mean_map"), collapse = " + ")))
form_sd   <- as.formula(paste(mortality_var, "~", paste(c(demo_vars, "sd_map"),   collapse = " + ")))

fit_mean     <- vector("list", n_imputations)
fit_sd       <- vector("list", n_imputations)
train_tables <- vector("list", n_imputations)
test_tables  <- vector("list", n_imputations)

for (k in seq_len(n_imputations)) {
  pair <- readRDS(sprintf("prediction/imputations/imputed_pair_%d.rds", k))
  
  # ---- drop excluded subjects BEFORE building patient tables / fitting ----
  train_raw <- pair$train %>% filter(!(.data[[id_var]] %in% excluded_subject_ids))
  test_raw  <- pair$test  %>% filter(!(.data[[id_var]] %in% excluded_subject_ids))
  cat(sprintf("Imputation %d: excluded %d train patients, %d test patients\n",
              k,
              length(unique(pair$train[[id_var]])) - length(unique(train_raw[[id_var]])),
              length(unique(pair$test[[id_var]]))  - length(unique(test_raw[[id_var]]))))
  
  train_pt <- build_patient_table(train_raw, id_var, demo_vars, mortality_var)
  test_pt  <- build_patient_table(test_raw,  id_var, demo_vars, mortality_var)
  
  # a patient with only 1 observed MAP row has sd_map == NA -- can't
  # inform a within-patient spread from a single measurement, so those
  # patients are dropped from the SD model (train and test) only
  fit_mean[[k]] <- glm(form_mean, data = train_pt, family = binomial())
  fit_sd[[k]]   <- glm(form_sd,   data = train_pt %>% filter(!is.na(sd_map)), family = binomial())
  
  train_tables[[k]] <- train_pt %>% mutate(imputation = k)
  test_tables[[k]]  <- test_pt  %>% mutate(imputation = k)
  
  cat(sprintf("Imputation %d: %d train patients (mean model), %d train patients (sd model)\n",
              k, nrow(train_pt), sum(!is.na(train_pt$sd_map))))
}

# ---- pooled coefficient tables (Rubin's rules) ----
pooled_mean <- pool_glm_list(fit_mean)
pooled_sd   <- pool_glm_list(fit_sd)

cat("\n--- Pooled mortality logistic regression: MEAN MAP model ---\n")
print(pooled_mean, digits = 4)
cat("\n--- Pooled mortality logistic regression: SD MAP model ---\n")
print(pooled_sd, digits = 4)

# ---- test-set prediction, averaged across the 5 imputations per patient ----
# (same convention as 04_predict_test_and_evaluate.R's Step 7)
predict_and_average <- function(fit_list, test_tables, mortality_var, drop_na_var = NULL) {
  preds <- lapply(seq_along(fit_list), function(k) {
    d <- test_tables[[k]]
    if (!is.null(drop_na_var)) d <- d %>% filter(!is.na(.data[[drop_na_var]]))
    d %>% mutate(pred_prob = predict(fit_list[[k]], newdata = d, type = "response"))
  })
  bind_rows(preds) %>%
    group_by(across(all_of(id_var))) %>%
    summarise(avg_pred_prob = mean(pred_prob),
              observed_mortality = first(.data[[mortality_var]]),
              .groups = "drop")
}

final_mean <- predict_and_average(fit_mean, test_tables, mortality_var)
final_sd   <- predict_and_average(fit_sd,   test_tables, mortality_var, drop_na_var = "sd_map")

# ---- TRAINING-set predictions, averaged across the 5 imputations per
# patient, using the same predict_and_average() logic. These are for
# choosing a classification threshold (e.g. Youden's J) WITHOUT looking
# at the test set -- 11_confusion_matrices.R uses these, not final_mean/
# final_sd, to pick each model's threshold, then applies that fixed
# threshold to the test-set predictions above.
final_mean_train <- predict_and_average(fit_mean, train_tables, mortality_var)
final_sd_train   <- predict_and_average(fit_sd,   train_tables, mortality_var, drop_na_var = "sd_map")

saveRDS(list(fit_mean = fit_mean, fit_sd = fit_sd,
             pooled_mean = pooled_mean, pooled_sd = pooled_sd,
             final_mean = final_mean, final_sd = final_sd,
             final_mean_train = final_mean_train, final_sd_train = final_sd_train,
             demo_vars = demo_vars, mortality_var = mortality_var,
             excluded_subject_ids = excluded_subject_ids),
        file = file.path(out_dir, "mean_sd_map_models.rds"))

cat(sprintf("\nSaved -> %s/mean_sd_map_models.rds\n", out_dir))
