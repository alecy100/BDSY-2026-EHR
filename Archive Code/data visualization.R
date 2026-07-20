ds <- read.csv("output/stroke_longitudinal.csv")

## =============================================================
## Exploratory Data Analysis: ICU Stroke Longitudinal Dataset (ds)
## =============================================================
## Assumes `ds` is already loaded in your R environment with columns:
## subject_id, stay_id, stroke_type, stroke_binary, hours_since_icu_intime,
## map, gcs, temperature_c, bun, creatinine, inr, platelets, sodium, wbc,
## lactate, pf_ratio, age, sex, race, bmi, height_in, weight_lb,
## hypertension, afib, ckd, diabetes, cancer, copd, heart_failure, cad,
## charlson_score, reperfusion_therapy, vasopressor_baseline,
## mechanical_vent_baseline, code_status_dnr

## ---- 0. Packages ---------------------------------------------------------
pkgs <- c("tidyverse", "naniar", "GGally", "patchwork", "scales")
invisible(lapply(pkgs, function(p) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
}))

library(tidyverse)
library(naniar)
library(GGally)
library(patchwork)
library(scales)

theme_set(theme_minimal(base_size = 12))

## ---- 1. Basic structure ---------------------------------------------------
glimpse(ds)

n_subjects <- n_distinct(ds$subject_id)
n_stays    <- n_distinct(ds$stay_id)
cat("Subjects:", n_subjects, " | Stays:", n_stays,
    " | Rows:", nrow(ds), "\n")

# Observations per subject (repeated-measures structure)
obs_per_subj <- ds %>%
  count(subject_id, name = "n_obs")

p_obs_per_subj <- ggplot(obs_per_subj, aes(x = n_obs)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "white") +
  labs(title = "Number of Observations per Subject",
       x = "Observations per subject_id", y = "Count of subjects")
print(p_obs_per_subj)

# Follow-up duration per stay
followup <- ds %>%
  group_by(stay_id) %>%
  summarise(max_hours = max(hours_since_icu_intime, na.rm = TRUE))

p_followup <- ggplot(followup, aes(x = max_hours)) +
  geom_histogram(bins = 40, fill = "darkorange", color = "white") +
  labs(title = "ICU Follow-up Duration per Stay",
       x = "Hours since ICU admission (max observed)", y = "Count of stays")
print(p_followup)

## ---- 2. Missingness overview ----------------------------------------------
p_missing <- gg_miss_var(ds, show_pct = TRUE) +
  labs(title = "Percent Missing by Variable")
print(p_missing)

p_missing_upset <- tryCatch(
  gg_miss_upset(ds, nsets = 8),
  error = function(e) NULL
)
if (!is.null(p_missing_upset)) print(p_missing_upset)

## ---- 3. Demographics -------------------------------------------------------
# One row per subject for demographic summaries (avoid double-counting repeats)
demo <- ds %>%
  distinct(subject_id, .keep_all = TRUE)

p_age <- ggplot(demo, aes(x = age)) +
  geom_histogram(bins = 30, fill = "seagreen", color = "white") +
  labs(title = "Age Distribution", x = "Age (years)", y = "Count")

p_sex <- ggplot(demo, aes(x = sex, fill = sex)) +
  geom_bar() +
  labs(title = "Sex Distribution", x = NULL, y = "Count") +
  theme(legend.position = "none")

p_race <- ggplot(demo, aes(y = fct_infreq(race))) +
  geom_bar(fill = "slateblue") +
  labs(title = "Race Distribution", x = "Count", y = NULL)

p_bmi <- ggplot(demo, aes(x = bmi)) +
  geom_histogram(fill = "tomato", color = "white") +
  labs(title = "BMI Distribution", x = "BMI", y = "Count")

(p_age + p_sex) / (p_race + p_bmi)

# Height / weight relationship
p_ht_wt <- ggplot(demo, aes(x = height_in, y = weight_lb)) +
  geom_point(alpha = 0.5, color = "darkcyan") +
  geom_smooth(method = "lm", se = FALSE, color = "black") +
  labs(title = "Height vs Weight", x = "Height (in)", y = "Weight (lb)")
print(p_ht_wt)

## ---- 4. Stroke type / outcome distribution --------------------------------
p_stroke_type <- ggplot(demo, aes(x = fct_infreq(stroke_type), fill = stroke_type)) +
  geom_bar() +
  labs(title = "Stroke Type Distribution", x = NULL, y = "Count") +
  theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1))

p_stroke_binary <- ggplot(demo, aes(x = factor(stroke_binary), fill = factor(stroke_binary))) +
  geom_bar() +
  labs(title = "Stroke (Binary) Distribution", x = "stroke_binary", y = "Count") +
  theme(legend.position = "none")

p_stroke_type + p_stroke_binary

## ---- 5. Comorbidity prevalence --------------------------------------------
comorbidities <- c("hypertension", "afib", "ckd", "diabetes", "cancer",
                   "copd", "heart_failure", "cad", "code_status_dnr",
                   "reperfusion_therapy", "vasopressor_baseline",
                   "mechanical_vent_baseline")

comorb_long <- demo %>%
  select(subject_id, all_of(comorbidities)) %>%
  pivot_longer(-subject_id, names_to = "condition", values_to = "present") %>%
  mutate(present = as.logical(present)) %>%
  group_by(condition) %>%
  summarise(pct = mean(present, na.rm = TRUE) * 100) %>%
  arrange(pct)

p_comorb <- ggplot(comorb_long, aes(x = pct, y = fct_reorder(condition, pct))) +
  geom_col(fill = "firebrick") +
  labs(title = "Prevalence of Comorbidities / Baseline Flags",
       x = "% of subjects", y = NULL) +
  scale_x_continuous(labels = label_percent(scale = 1))
print(p_comorb)

# Charlson score distribution
p_charlson <- ggplot(demo, aes(x = charlson_score)) +
  geom_histogram(binwidth = 1, fill = "purple4", color = "white") +
  labs(title = "Charlson Comorbidity Score Distribution",
       x = "Charlson score", y = "Count of subjects")
print(p_charlson)

## ---- 6. Vital sign trajectories over time (binned trend, scalable) --------
vitals <- c("map", "gcs", "temperature_c")

vitals_long <- ds %>%
  select(subject_id, hours_since_icu_intime, stroke_type, all_of(vitals)) %>%
  pivot_longer(all_of(vitals), names_to = "vital", values_to = "value")

# Bin hours into 6-hour windows to summarize instead of smoothing raw points
bin_width <- 6

vitals_binned <- vitals_long %>%
  filter(!is.na(value)) %>%
  mutate(time_bin = floor(hours_since_icu_intime / bin_width) * bin_width) %>%
  group_by(vital, time_bin) %>%
  summarise(mean_val = mean(value, na.rm = TRUE),
            se_val = sd(value, na.rm = TRUE) / sqrt(n()),
            n_obs = n(),
            .groups = "drop")

p_vitals_traj <- ggplot(vitals_binned, aes(x = time_bin, y = mean_val)) +
  geom_ribbon(aes(ymin = mean_val - se_val, ymax = mean_val + se_val),
              alpha = 0.2, fill = "steelblue") +
  geom_line(color = "steelblue", linewidth = 0.8) +
  facet_wrap(~ vital, scales = "free_y") +
  labs(title = "Vital Sign Trajectories Over Time (binned mean \u00b1 SE)",
       x = paste0("Hours since ICU admission (", bin_width, "-hour bins)"),
       y = "Value")
print(p_vitals_traj)

# Same, stratified by stroke_type
vitals_binned_stroke <- vitals_long %>%
  filter(!is.na(value)) %>%
  mutate(time_bin = floor(hours_since_icu_intime / bin_width) * bin_width) %>%
  group_by(vital, stroke_type, time_bin) %>%
  summarise(mean_val = mean(value, na.rm = TRUE),
            se_val = sd(value, na.rm = TRUE) / sqrt(n()),
            n_obs = n(),
            .groups = "drop")

p_vitals_traj_stroke <- ggplot(vitals_binned_stroke,
                               aes(x = time_bin, y = mean_val, color = stroke_type,
                                   fill = stroke_type)) +
  geom_ribbon(aes(ymin = mean_val - se_val, ymax = mean_val + se_val),
              alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~ vital, scales = "free_y") +
  labs(title = "Vital Sign Trends Over Time by Stroke Type (binned mean \u00b1 SE)",
       x = paste0("Hours since ICU admission (", bin_width, "-hour bins)"),
       y = "Value", color = "Stroke type", fill = "Stroke type")
print(p_vitals_traj_stroke)


## ---- 7. Lab value trajectories over time (binned trend, scalable) --------
labs_vars <- c("bun", "creatinine", "inr", "platelets", "sodium",
               "wbc", "lactate", "pf_ratio")

labs_long <- ds %>%
  select(subject_id, hours_since_icu_intime, stroke_type, all_of(labs_vars)) %>%
  pivot_longer(all_of(labs_vars), names_to = "lab", values_to = "value")

bin_width <- 6

labs_binned <- labs_long %>%
  filter(!is.na(value)) %>%
  mutate(time_bin = floor(hours_since_icu_intime / bin_width) * bin_width) %>%
  group_by(lab, time_bin) %>%
  summarise(mean_val = mean(value, na.rm = TRUE),
            se_val = sd(value, na.rm = TRUE) / sqrt(n()),
            n_obs = n(),
            .groups = "drop")

p_labs_traj <- ggplot(labs_binned, aes(x = time_bin, y = mean_val)) +
  geom_ribbon(aes(ymin = mean_val - se_val, ymax = mean_val + se_val),
              alpha = 0.2, fill = "darkorange") +
  geom_line(color = "darkorange", linewidth = 0.8) +
  facet_wrap(~ lab, scales = "free_y") +
  labs(title = "Lab Value Trajectories Over Time (binned mean \u00b1 SE)",
       x = paste0("Hours since ICU admission (", bin_width, "-hour bins)"),
       y = "Value")
print(p_labs_traj)

## ---- 8. Distributions of continuous clinical variables --------------------
p_labs_dist <- ggplot(labs_long, aes(x = value)) +
  geom_histogram(bins = 30, fill = "cadetblue", color = "white") +
  facet_wrap(~ lab, scales = "free") +
  labs(title = "Distributions of Lab Values (all observations)",
       x = NULL, y = "Count")
print(p_labs_dist)

## ---- 9. Correlation among continuous variables ----------------------------
cont_vars <- ds %>%
  select(map, gcs, temperature_c, bun, creatinine, inr, platelets,
         sodium, wbc, lactate, pf_ratio, age, bmi, charlson_score)

p_corr <- ggcorr(cont_vars, method = c("pairwise", "pearson"),
                 label = TRUE, label_size = 3, hjust = 0.8,
                 layout.exp = 2) +
  labs(title = "Correlation Matrix: Continuous Clinical Variables")
print(p_corr)

## ---- 10. Outcome comparisons by stroke type/binary ------------------------
key_clinical <- c("map", "gcs", "lactate", "pf_ratio", "charlson_score", "age")

compare_long <- ds %>%
  select(subject_id, stroke_binary, all_of(key_clinical)) %>%
  distinct(subject_id, .keep_all = TRUE) %>%
  pivot_longer(all_of(key_clinical), names_to = "variable", values_to = "value")

p_compare <- ggplot(compare_long, aes(x = factor(stroke_binary), y = value,
                                      fill = factor(stroke_binary))) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ variable, scales = "free_y") +
  labs(title = "Key Clinical Variables by Stroke Status (subject-level baseline)",
       x = "stroke_binary", y = NULL, fill = "stroke_binary")
print(p_compare)

## ---- 11. Reperfusion therapy & baseline supports vs stroke type -----------
p_reperf <- ds %>%
  distinct(subject_id, .keep_all = TRUE) %>%
  count(stroke_type, reperfusion_therapy) %>%
  ggplot(aes(x = stroke_type, y = n, fill = factor(reperfusion_therapy))) +
  geom_col(position = "fill") +
  labs(title = "Reperfusion Therapy by Stroke Type",
       x = NULL, y = "Proportion", fill = "Reperfusion\ntherapy") +
  scale_y_continuous(labels = percent) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
print(p_reperf)

## =============================================================
## End of script. Each plot object (p_*) can be saved individually, e.g.:
## ggsave("vitals_trajectory.png", p_vitals_traj, width = 10, height = 6, dpi = 300)
## =============================================================

library(lme4)
library(ggplot2)

#LOOK AT MAP: MAYBE THIS IS WHAT HAS TOO MANY MEASUREMENTS AND NEEDS TO BE GROUPED DOWN

#gcs, map, temperature
lab_var <- "platelets"   # change this to whichever lab you want to plot

ggplot(ds, aes(x = hours_since_icu_intime, y = .data[[lab_var]],
               group = subject_id)) +
  geom_line(alpha = 0.3, color = "steelblue") +
  labs(title = paste("Individual Subject Trajectories:", lab_var),
       x = "Hours since ICU admission",
       y = lab_var) +
  theme_minimal()

fit <- lmer(lactate ~ hours_since_icu_intime+sodium+
              (hours_since_icu_intime|subject_id),data = ds, REML=TRUE)

#MISSING VARIABLES --> IF WE CAN CONDENSE THESE, WE CAN HAVE A MUCH LONGER TIMELINE
na_counts <- ds %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "n_missing") %>%
  arrange(desc(n_missing))

print(na_counts, n = Inf)

library(tidyverse)

#MAX FOR EACH ONE
## ---- For each subject, count distinct non-missing values per variable ----
distinct_counts <- ds %>%
  group_by(subject_id) %>%
  summarise(across(everything(), ~ n_distinct(.x[!is.na(.x)])), .groups = "drop")

## ---- Classify each variable: does it vary within any subject? ----
var_classification <- distinct_counts %>%
  select(-subject_id) %>%
  summarise(across(everything(), ~ max(.x, na.rm = TRUE))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "max_distinct_values") %>%
  mutate(type = if_else(max_distinct_values > 1,
                        "time-varying (repeated measurements)",
                        "static (1 value per subject)")) %>%
  arrange(type, desc(max_distinct_values))

print(var_classification, n = Inf)

#MISSING DATA / HOW MANY PEOPLE ACTUALLY HAVE REPEATED MEASUREMENTS
## THIS IS SUPER IMPORTANT
# MIGHT HAVE TO REDUCE DATASET TO PEOPLE WITH MULTIPLE MEASUREMENTS? IDK 

library(tidyverse)

## ---- Number of distinct non-missing values per subject, for each lab/vital ----
lab_vital_vars <- c("map", "gcs", "temperature_c", "bun", "creatinine",
                    "inr", "platelets", "sodium", "wbc", "lactate", "pf_ratio")

measurement_counts <- ds %>%
  group_by(subject_id) %>%
  summarise(across(all_of(lab_vital_vars),
                   ~ n_distinct(.x[!is.na(.x)]),
                   .names = "n_{.col}"),
            .groups = "drop")

## ---- Categorize each subject as 0 / 1 / 2+ measurements, per variable ----
measurement_categories <- measurement_counts %>%
  pivot_longer(-subject_id, names_to = "variable", values_to = "n_distinct_values") %>%
  mutate(variable = str_remove(variable, "^n_"),
         category = case_when(
           n_distinct_values == 0 ~ "0 (never measured)",
           n_distinct_values == 1 ~ "1 (single measurement)",
           n_distinct_values >= 2 ~ "2+ (repeated measurements)"
         ))

## ---- Summary table: patient counts (and %) per category, per variable ----
summary_table <- measurement_categories %>%
  count(variable, category) %>%
  group_by(variable) %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  ungroup() %>%
  arrange(variable, category)

print(summary_table, n = Inf)


##JUST TWO LACTATE MEASUREMENTS
#ONLY 1000 with two +  lactate measurements in first 24hr
lactate_multi_subjects <- ds %>%
  group_by(subject_id) %>%
  summarise(n_distinct_lactate = n_distinct(lactate[!is.na(lactate)]), .groups = "drop") %>%
  filter(n_distinct_lactate >= 2) %>%
  pull(subject_id)

## ---- Find row numbers (in ds) belonging to those subjects ----
lactate_multi_rownums <- which(ds$subject_id %in% lactate_multi_subjects)

## ---- Create ds_lactate: all rows/timepoints for those subjects ----
ds_lactate <- ds[lactate_multi_rownums, ]

## ---- Quick checks ----
length(lactate_multi_subjects)      # number of qualifying subjects
length(lactate_multi_rownums)       # number of rows pulled into ds_lactate
n_distinct(ds_lactate$subject_id)   # should match length(lactate_multi_subjects)
nrow(ds_lactate)                    # should match length(lactate_multi_rownums)

##LACTATE ONLY DS
library(tidyverse)

## ---- Identify subjects with 2+ distinct non-missing lactate values ----
lactate_multi_subjects <- ds %>%
  group_by(subject_id) %>%
  summarise(n_distinct_lactate = n_distinct(lactate[!is.na(lactate)]), .groups = "drop") %>%
  filter(n_distinct_lactate >= 2) %>%
  pull(subject_id)

## ---- Find row numbers (in ds) belonging to those subjects ----
lactate_multi_rownums <- which(ds$subject_id %in% lactate_multi_subjects)

## ---- Create ds_lactate: all rows/timepoints for those subjects ----
ds_lactate <- ds[lactate_multi_rownums, ]

## ---- Quick checks ----
length(lactate_multi_subjects)      # number of qualifying subjects
length(lactate_multi_rownums)       # number of rows pulled into ds_lactate
n_distinct(ds_lactate$subject_id)   # should match length(lactate_multi_subjects)
nrow(ds_lactate)                    # should match length(lactate_multi_rownums)


##BINNED DATASET

library(tidyverse)

bin_width <- 4

## Variables that should be averaged if multiple values fall in the same bin
numeric_vars <- c("map", "gcs", "temperature_c", "bun", "creatinine", "inr",
                  "platelets", "sodium", "wbc", "lactate", "pf_ratio")

## Variables that are static per subject — just carry the first non-missing value
static_vars <- c("stroke_type", "stroke_binary", "age", "sex", "race", "bmi",
                 "height_in", "weight_lb", "hypertension", "afib", "ckd",
                 "diabetes", "cancer", "copd", "heart_failure", "cad",
                 "charlson_score", "reperfusion_therapy", "vasopressor_baseline",
                 "mechanical_vent_baseline", "code_status_dnr")

ds_binned <- ds %>%
  mutate(time_bin = floor(hours_since_icu_intime / bin_width) * bin_width) %>%
  group_by(subject_id, stay_id, time_bin) %>%
  summarise(
    across(all_of(numeric_vars), ~ mean(.x, na.rm = TRUE)),
    across(all_of(static_vars), ~ first(na.omit(.x))),
    .groups = "drop"
  ) %>%
  mutate(across(all_of(numeric_vars), ~ na_if(., NaN))) %>%   # mean() of all-NA returns NaN
  arrange(subject_id, time_bin)

## Quick check
glimpse(ds_binned)
nrow(ds)          # original row count
nrow(ds_binned)   # binned row count (should be smaller)

model1 <- lmer(gcs ~ time_bin + stroke_type + age + sex + race + bmi +
  charlson_score + reperfusion_therapy + vasopressor_baseline +
  mechanical_vent_baseline + code_status_dnr +
  (1 | subject_id), ds_binned)


model2 <- lmer(gcs ~ time_bin + stroke_type + age + sex + race+
                 charlson_score + reperfusion_therapy + vasopressor_baseline +
                 mechanical_vent_baseline + code_status_dnr +
                 (time_bin| subject_id), ds_binned)

model3 <- lmer(gcs ~ time_bin + stroke_binary + age + sex + race+
                 charlson_score + reperfusion_therapy + vasopressor_baseline +
                 mechanical_vent_baseline + code_status_dnr +
                 (time_bin| subject_id), ds_binned)

#anova(model2, model3) --> offers the comparison

##BINNED DATASET VISUALIZATIONS
library(tidyverse)
library(patchwork)
library(scales)

theme_set(theme_minimal(base_size = 12))

## ---- 1. How many bins per subject? ----
bins_per_subject <- ds_binned %>%
  count(subject_id, name = "n_bins")

p_bins_per_subj <- ggplot(bins_per_subject, aes(x = n_bins)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "white") +
  labs(title = "Number of 4-Hour Bins per Subject",
       x = "Number of bins", y = "Count of subjects")
print(p_bins_per_subj)

## ---- 2. Missingness in the binned dataset ----
naniar::gg_miss_var(ds_binned, show_pct = TRUE) +
  labs(title = "Percent Missing by Variable (Binned Data)")

## ---- 3. Spaghetti plots of biomarkers over binned time ----
biomarkers <- c("map", "gcs", "temperature_c", "bun", "creatinine",
                "inr", "platelets", "sodium", "wbc", "lactate", "pf_ratio")

biomarkers_long <- ds_binned %>%
  select(subject_id, time_bin, stroke_type, all_of(biomarkers)) %>%
  pivot_longer(all_of(biomarkers), names_to = "biomarker", values_to = "value")

p_spaghetti <- ggplot(biomarkers_long,
                      aes(x = time_bin, y = value, group = subject_id)) +
  geom_line(alpha = 0.15, color = "steelblue") +
  stat_summary(aes(group = 1), fun = mean, geom = "line",
               color = "black", linewidth = 1) +
  facet_wrap(~ biomarker, scales = "free_y") +
  labs(title = "Biomarker Trajectories Over Time (4-Hour Bins)",
       x = "Hours since ICU admission (binned)", y = "Value")
print(p_spaghetti)

## ---- 4. Same, split by stroke_type (mean trend only, cleaner to read) ----
p_by_stroke <- ggplot(biomarkers_long,
                      aes(x = time_bin, y = value, color = stroke_type)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1) +
  stat_summary(fun.data = mean_se, geom = "ribbon",
               aes(fill = stroke_type), alpha = 0.15, color = NA) +
  facet_wrap(~ biomarker, scales = "free_y") +
  labs(title = "Mean Biomarker Trends by Stroke Type (4-Hour Bins)",
       x = "Hours since ICU admission (binned)", y = "Value",
       color = "Stroke type", fill = "Stroke type")
print(p_by_stroke)

## ---- 5. Distribution of each biomarker (binned values) ----
p_dist <- ggplot(biomarkers_long, aes(x = value)) +
  geom_histogram(bins = 30, fill = "darkorange", color = "white") +
  facet_wrap(~ biomarker, scales = "free") +
  labs(title = "Distributions of Binned Biomarker Values",
       x = NULL, y = "Count")
print(p_dist)

## ---- 6. Boxplots of biomarkers by stroke_binary, faceted ----
biomarkers_long_binary <- ds_binned %>%
  select(subject_id, time_bin, stroke_binary, all_of(biomarkers)) %>%
  pivot_longer(all_of(biomarkers), names_to = "biomarker", values_to = "value")

p_box_stroke <- ggplot(biomarkers_long_binary,
                       aes(x = factor(stroke_binary), y = value,
                           fill = factor(stroke_binary))) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ biomarker, scales = "free_y") +
  labs(title = "Biomarker Distributions by Stroke Status (all bins)",
       x = "stroke_binary", y = NULL, fill = "stroke_binary")
print(p_box_stroke)

## ---- 7. Correlation among biomarkers (binned) ----
GGally::ggcorr(ds_binned %>% select(all_of(biomarkers)),
               method = c("pairwise", "pearson"),
               label = TRUE, label_size = 3) +
  labs(title = "Correlation Among Biomarkers (Binned Data)")

## ---- 8. Heatmap: data availability by subject and time bin ----
availability <- ds_binned %>%
  select(subject_id, time_bin, all_of(biomarkers)) %>%
  pivot_longer(all_of(biomarkers), names_to = "biomarker", values_to = "value") %>%
  group_by(subject_id, time_bin) %>%
  summarise(n_available = sum(!is.na(value)), .groups = "drop")

p_availability <- ggplot(availability, aes(x = time_bin, y = fct_rev(factor(subject_id)),
                                           fill = n_available)) +
  geom_tile() +
  scale_fill_viridis_c(name = "# biomarkers\navailable") +
  labs(title = "Biomarker Data Availability by Subject and Time Bin",
       x = "Hours since ICU admission (binned)", y = "Subject") +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
print(p_availability)

## Is lactate missingness related to severity? (a rough check)
ds_binned %>%
  mutate(lactate_missing = is.na(lactate)) %>%
  group_by(lactate_missing) %>%
  summarise(mean_map = mean(map, na.rm = TRUE),
            mean_gcs = mean(gcs, na.rm = TRUE),
            n = n())

#IS LACTATE NOT MISSING AT RANDOM (AS IN ONLY MEASURED IN SICKER PATIENTS)

library(tidyverse)
library(patchwork)
library(scales)

theme_set(theme_minimal(base_size = 12))

## ---- Create the missingness indicator ----
ds_binned <- ds_binned %>%
  mutate(lactate_missing = if_else(is.na(lactate), "Missing", "Measured"))

## ---- 1. Continuous biomarkers/vitals compared across missingness groups ----
compare_vars <- c("map", "gcs", "temperature_c", "bun", "creatinine",
                  "inr", "platelets", "sodium", "wbc", "pf_ratio",
                  "age", "bmi", "charlson_score")

compare_long <- ds_binned %>%
  select(lactate_missing, all_of(compare_vars)) %>%
  pivot_longer(-lactate_missing, names_to = "variable", values_to = "value")

p_compare_box <- ggplot(compare_long, aes(x = lactate_missing, y = value,
                                          fill = lactate_missing)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ variable, scales = "free_y") +
  labs(title = "Patient Characteristics by Lactate Measurement Status",
       x = NULL, y = NULL, fill = NULL) +
  theme(legend.position = "top")
print(p_compare_box)

##NOTICE THE DIFFERENCE IN GCS --> NOT MISSING AT RANDOM!

library(tidyverse)
library(patchwork)
library(scales)

theme_set(theme_minimal(base_size = 12))

## ---- Create the missingness indicator ----
ds_binned <- ds_binned %>%
  mutate(lactate_missing = if_else(is.na(lactate), "Missing", "Measured"))

## ---- 1. Continuous biomarkers/vitals compared across missingness groups ----
compare_vars <- c("map", "gcs", "temperature_c", "bun", "creatinine",
                  "inr", "platelets", "sodium", "wbc", "pf_ratio",
                  "age", "bmi", "charlson_score")

compare_long <- ds_binned %>%
  select(lactate_missing, all_of(compare_vars)) %>%
  pivot_longer(-lactate_missing, names_to = "variable", values_to = "value")

p_compare_box <- ggplot(compare_long, aes(x = lactate_missing, y = value,
                                          fill = lactate_missing)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ variable, scales = "free_y") +
  labs(title = "Patient Characteristics by Lactate Measurement Status",
       x = NULL, y = NULL, fill = NULL) +
  theme(legend.position = "top")
print(p_compare_box)

## ---- 2. Same comparison, as density/violin plots (shows distribution shape) ----
p_compare_violin <- ggplot(compare_long, aes(x = lactate_missing, y = value,
                                             fill = lactate_missing)) +
  geom_violin(alpha = 0.6) +
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +
  facet_wrap(~ variable, scales = "free_y") +
  labs(title = "Distribution Shape by Lactate Measurement Status",
       x = NULL, y = NULL, fill = NULL) +
  theme(legend.position = "top")
print(p_compare_violin)

## ---- 3. Categorical/comorbidity variables: % present by missingness group ----
categorical_vars <- c("stroke_type", "hypertension", "afib", "ckd", "diabetes",
                      "cancer", "copd", "heart_failure", "cad",
                      "reperfusion_therapy", "vasopressor_baseline",
                      "mechanical_vent_baseline", "code_status_dnr")

categorical_long <- ds_binned %>%
  select(lactate_missing, all_of(categorical_vars)) %>%
  pivot_longer(-lactate_missing, names_to = "variable", values_to = "value",
               values_transform = as.character)

p_categorical <- categorical_long %>%
  count(lactate_missing, variable, value) %>%
  group_by(lactate_missing, variable) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ungroup() %>%
  filter(value %in% c("1", "TRUE", "Yes") | variable == "stroke_type") %>%
  ggplot(aes(x = variable, y = pct, fill = lactate_missing)) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(title = "Comorbidities / Care Intensity by Lactate Measurement Status",
       x = NULL, y = "% of group", fill = NULL) +
  theme(legend.position = "top")
print(p_categorical)

## ---- 4. Stroke type composition by missingness group (proportion) ----
p_stroke_type <- ds_binned %>%
  count(lactate_missing, stroke_type) %>%
  group_by(lactate_missing) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(x = lactate_missing, y = pct, fill = stroke_type)) +
  geom_col(position = "fill") +
  scale_y_continuous(labels = percent) +
  labs(title = "Stroke Type Composition by Lactate Measurement Status",
       x = NULL, y = "Proportion", fill = "Stroke type")
print(p_stroke_type)

##THIS ONE IS KEY

## ---- 5. Missingness rate over time (does it change across the ICU stay?) ----
p_missing_over_time <- ds_binned %>%
  group_by(time_bin) %>%
  summarise(pct_missing = mean(is.na(lactate)) * 100, n = n()) %>%
  ggplot(aes(x = time_bin, y = pct_missing)) +
  geom_line(color = "firebrick", linewidth = 1) +
  geom_point(aes(size = n), color = "firebrick", alpha = 0.6) +
  labs(title = "Lactate Missingness Rate Over Time",
       x = "Hours since ICU admission (binned)", y = "% Missing",
       size = "N (bin)")
print(p_missing_over_time)



p_missing_over_time <- ds_binned %>%
  group_by(time_bin) %>%
  summarise(pct_missing = mean(is.na(gcs)) * 100, n = n()) %>%
  ggplot(aes(x = time_bin, y = pct_missing)) +
  geom_line(color = "firebrick", linewidth = 1) +
  geom_point(aes(size = n), color = "firebrick", alpha = 0.6) +
  labs(title = "GCS Missingness Rate Over Time",
       x = "Hours since ICU admission (binned)", y = "% Missing",
       size = "N (bin)")
print(p_missing_over_time)

## ---- 6. Missingness rate by subject-level severity (charlson score bins) ----
p_missing_by_severity <- ds_binned %>%
  mutate(charlson_group = cut(charlson_score, breaks = c(-Inf, 2, 4, 6, Inf),
                              labels = c("0-2", "3-4", "5-6", "7+"))) %>%
  group_by(charlson_group) %>%
  summarise(pct_missing = mean(is.na(lactate)) * 100, n = n()) %>%
  filter(!is.na(charlson_group)) %>%
  ggplot(aes(x = charlson_group, y = pct_missing)) +
  geom_col(fill = "steelblue") +
  labs(title = "Lactate Missingness by Charlson Comorbidity Score Group",
       x = "Charlson score group", y = "% Missing")
print(p_missing_by_severity)


#PROJECT IDEA: DO GCS? 

library(tidyverse)
library(patchwork)
library(scales)

theme_set(theme_minimal(base_size = 12))

## ---- 1. Overall GCS distribution ----
p_gcs_dist <- ggplot(ds_binned, aes(x = gcs)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "white") +
  labs(title = "Distribution of GCS Scores",
       x = "GCS (3-15)", y = "Count of observations")
print(p_gcs_dist)

## ---- 2. GCS severity categories ----
p_gcs_severity <- ds_binned %>%
  mutate(gcs_category = case_when(
    gcs >= 13 ~ "Mild (13-15)",
    gcs >= 9  ~ "Moderate (9-12)",
    gcs >= 3  ~ "Severe (3-8)",
    TRUE ~ NA_character_
  ),
  gcs_category = factor(gcs_category,
                        levels = c("Mild (13-15)", "Moderate (9-12)", "Severe (3-8)"))) %>%
  filter(!is.na(gcs_category)) %>%
  ggplot(aes(x = gcs_category, fill = gcs_category)) +
  geom_bar() +
  labs(title = "GCS Severity Category Distribution",
       x = NULL, y = "Count of observations") +
  theme(legend.position = "none")
print(p_gcs_severity)

## ---- 3. GCS trajectory over time (binned) — individual + mean trend ----
p_gcs_traj <- ggplot(ds_binned, aes(x = time_bin, y = gcs, group = subject_id)) +
  geom_line(alpha = 0.15, color = "steelblue") +
  stat_summary(aes(group = 1), fun = mean, geom = "line",
               color = "black", linewidth = 1) +
  labs(title = "GCS Trajectories Over Time (4-Hour Bins)",
       x = "Hours since ICU admission (binned)", y = "GCS")
print(p_gcs_traj)

## ---- 4. GCS trend by stroke_type, with SE ribbon ----
p_gcs_by_stroke <- ggplot(ds_binned %>% filter(!is.na(gcs)),
                          aes(x = time_bin, y = gcs, color = stroke_type, fill = stroke_type)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1) +
  stat_summary(fun.data = mean_se, geom = "ribbon", alpha = 0.15, color = NA) +
  labs(title = "Mean GCS Trend by Stroke Type Over Time",
       x = "Hours since ICU admission (binned)", y = "GCS",
       color = "Stroke type", fill = "Stroke type")
print(p_gcs_by_stroke)

## ---- 5. GCS distribution by stroke_binary ----
p_gcs_stroke_binary <- ggplot(ds_binned %>% filter(!is.na(gcs)),
                              aes(x = factor(stroke_binary), y = gcs,
                                  fill = factor(stroke_binary))) +
  geom_violin(alpha = 0.6) +
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +
  labs(title = "GCS Distribution by Stroke Status",
       x = "stroke_binary", y = "GCS", fill = "stroke_binary") +
  theme(legend.position = "none")
print(p_gcs_stroke_binary)

## ---- 6. GCS by mechanical ventilation status (low GCS often drives intubation) ----
p_gcs_vent <- ggplot(ds_binned %>% filter(!is.na(gcs)),
                     aes(x = factor(mechanical_vent_baseline), y = gcs,
                         fill = factor(mechanical_vent_baseline))) +
  geom_boxplot(outlier.alpha = 0.3) +
  labs(title = "GCS by Baseline Mechanical Ventilation Status",
       x = "mechanical_vent_baseline", y = "GCS", fill = NULL) +
  theme(legend.position = "none")
print(p_gcs_vent)

## ---- 7. GCS vs age ----
p_gcs_age <- ggplot(ds_binned %>% filter(!is.na(gcs)), aes(x = age, y = gcs)) +
  geom_jitter(alpha = 0.2, color = "darkcyan", height = 0.2) +
  geom_smooth(method = "loess", color = "black") +
  labs(title = "GCS vs Age", x = "Age", y = "GCS")
print(p_gcs_age)

## ---- 8. GCS vs Charlson comorbidity score ----
p_gcs_charlson <- ggplot(ds_binned %>% filter(!is.na(gcs)), aes(x = charlson_score, y = gcs)) +
  geom_jitter(alpha = 0.2, color = "firebrick", height = 0.2, width = 0.2) +
  geom_smooth(method = "loess", color = "black") +
  labs(title = "GCS vs Charlson Comorbidity Score", x = "Charlson score", y = "GCS")
print(p_gcs_charlson)

## ---- 9. GCS vs other biomarkers (correlation check) ----
gcs_corr_vars <- c("gcs", "map", "lactate", "temperature_c", "wbc", "pf_ratio")

GGally::ggcorr(ds_binned %>% select(all_of(gcs_corr_vars)),
               method = c("pairwise", "pearson"),
               label = TRUE, label_size = 3) +
  labs(title = "GCS Correlation with Other Biomarkers")

## ---- 10. GCS missingness over time ----
p_gcs_missing_time <- ds_binned %>%
  group_by(time_bin) %>%
  summarise(pct_missing = mean(is.na(gcs)) * 100, n = n()) %>%
  ggplot(aes(x = time_bin, y = pct_missing)) +
  geom_line(color = "firebrick", linewidth = 1) +
  geom_point(aes(size = n), color = "firebrick", alpha = 0.6) +
  labs(title = "GCS Missingness Rate Over Time",
       x = "Hours since ICU admission (binned)", y = "% Missing", size = "N (bin)")
print(p_gcs_missing_time)
