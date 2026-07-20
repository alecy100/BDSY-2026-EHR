# ============================================================
# STEP 9: 95% CI FOR AUC (all 3 mortality models) + ODDS-RATIO PLOTS
# ============================================================
#   Model A = "Mean MAP"        (05_mean_sd_map_mortality_models.R)
#   Model B = "SD MAP"          (05_mean_sd_map_mortality_models.R)
#   Model C = "Random effects"  (the existing model from 03_fit_lmm_and_logistic.R /
#             04_predict_test_and_evaluate.R, response_var = "map")
#
# AUC + 95% CI:
#   pROC::ci.auc() (DeLong method) applied to each model's TEST-SET
#   predicted probability, averaged across the 5 imputations per patient
#   -- the same "final" (avg_pred_prob, observed_mortality) pairing that
#   04 already reports the point AUC for; this just adds the CI around it.
#
# Odds ratios:
#   exp(pooled Rubin's-rules coefficient), with
#   95% CI = exp(estimate +/- qt(0.975, df) * pooled_SE), using the
#   pooled df from Barnard & Rubin already computed by
#   utils_pooling.R::pool_rubin() (used for both the mean/sd MAP models
#   and the existing random-effects model's pooled_logistic table).
# ============================================================

library(dplyr)
library(pROC)
library(ggplot2)

response_var <- "map"   # <<== must match the response_var used to fit Model C

meansd  <- readRDS("prediction/mean_sd_map_models/mean_sd_map_models.rds")
modelC_preds  <- readRDS(file.path(sprintf("prediction/%s", response_var), "test_predictions_final.rds"))
modelC_pooled <- readRDS(file.path(sprintf("prediction/%s", response_var), "pooled_model_summary.rds"))$pooled_logistic

out_dir <- "prediction/model_comparison"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------------------------------------
# 1. AUC + 95% CI table
# ------------------------------------------------------------
get_auc_ci <- function(observed, predicted) {
  roc_obj <- roc(observed, predicted, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  ci_obj  <- ci.auc(roc_obj, method = "delong")
  data.frame(auc = auc_val, ci_lower = as.numeric(ci_obj[1]), ci_upper = as.numeric(ci_obj[3]))
}

auc_table <- bind_rows(
  get_auc_ci(meansd$final_mean$observed_mortality, meansd$final_mean$avg_pred_prob) %>%
    mutate(model = "Mean MAP"),
  get_auc_ci(meansd$final_sd$observed_mortality, meansd$final_sd$avg_pred_prob) %>%
    mutate(model = "SD MAP"),
  get_auc_ci(modelC_preds$final_preds$observed_mortality, modelC_preds$final_preds$avg_pred_prob) %>%
    mutate(model = "Random effects")
) %>% select(model, auc, ci_lower, ci_upper)

cat("\n================= TEST-SET AUC (95% CI, DeLong) =================\n")
print(auc_table, digits = 4)
write.csv(auc_table, file.path(out_dir, "auc_ci_table.csv"), row.names = FALSE)

# ------------------------------------------------------------
# 2. odds-ratio tables (exponentiated pooled coefficients + 95% CI)
# ------------------------------------------------------------
or_table_from_pooled <- function(pooled_df, model_name) {
  pooled_df %>%
    filter(term != "(Intercept)") %>%
    mutate(or       = exp(estimate),
           ci_lower = exp(estimate - qt(0.975, df) * std.error),
           ci_upper = exp(estimate + qt(0.975, df) * std.error),
           model    = model_name)
}

or_mean <- or_table_from_pooled(meansd$pooled_mean, "Mean MAP")
or_sd   <- or_table_from_pooled(meansd$pooled_sd,   "SD MAP")
or_C    <- or_table_from_pooled(modelC_pooled,      "Random effects")

or_all <- bind_rows(or_mean, or_sd, or_C) %>%
  select(model, term, estimate, std.error, df, or, ci_lower, ci_upper, p.value)

cat("\n================= ODDS RATIOS (95% CI) =================\n")
print(or_all, digits = 3)
write.csv(or_all, file.path(out_dir, "odds_ratio_table.csv"), row.names = FALSE)

# ------------------------------------------------------------
# 3. odds-ratio forest plots (one per model)
# ------------------------------------------------------------
plot_or <- function(df, title) {
  df <- df %>% arrange(or)
  df$term <- factor(df$term, levels = df$term)
  ggplot(df, aes(x = or, y = term)) +
    geom_point(size = 2) +
    geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper), height = 0.2) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "grey40") +
    scale_x_log10() +
    labs(x = "Odds ratio (95% CI, log scale)", y = NULL, title = title) +
    theme_minimal(base_size = 12)
}

p_mean <- plot_or(or_mean, "Mortality model: Mean MAP")
p_sd   <- plot_or(or_sd,   "Mortality model: SD of MAP")
p_C    <- plot_or(or_C,    "Mortality model: Random effects (LMM trajectory)")

ggsave(file.path(out_dir, "or_plot_mean_map.png"),       p_mean, width = 7, height = 5, dpi = 300)
ggsave(file.path(out_dir, "or_plot_sd_map.png"),         p_sd,   width = 7, height = 5, dpi = 300)
ggsave(file.path(out_dir, "or_plot_random_effects.png"), p_C,    width = 7, height = 5, dpi = 300)

# ------------------------------------------------------------
# 4. ONE combined odds-ratio plot, all 3 models on the same figure
# ------------------------------------------------------------
# or_all is already long-format (one row per model x term), so terms
# shared across models (e.g. race_binary, age, sex, bmi,
# stroke_binaryischemic) naturally get 3 dodged points/CIs on the same
# row, while terms unique to one model (mean_map; sd_map;
# random_intercept/random_slope) naturally get just 1 point on their
# own row -- nothing needs to be hardcoded to tell shared from unique,
# it falls out of how many models have that term in or_all.
#
# position_dodge2(preserve = "single") is used instead of plain
# position_dodge() so a term's dot/error-bar stays a consistent width
# and is centered correctly whether it has 1, 2, or 3 models present at
# that row (plain position_dodge divides the row width by however many
# groups ARE present, which would make single-model rows look
# different/misaligned from 3-model rows).

# term order: alphabetical within "shared across all 3 models" first,
# then alphabetical within "unique to 1-2 models", so the figure reads
# top-to-bottom as demographic/clinical covariates -> model-specific terms
term_model_counts <- or_all %>% count(term, name = "n_models")
term_order <- term_model_counts %>%
  arrange(desc(n_models), term) %>%
  pull(term)

# display labels for the plot axis -- raw term name -> readable label.
# anything not in this map falls back to its raw term name unchanged
# (guards against a term-name mismatch silently dropping off the plot).
term_labels <- c(
  age                    = "Age",
  bmi                    = "BMI",
  charlson_score         = "Charlson score",
  race_binary            = "Race(non-white)",
  sex                    = "Sex(male)",
  stroke_binaryischemic  = "Stroke type(ischemic)",
  mean_map               = "Mean MAP",
  random_intercept       = "Random intercept of MAP",
  random_slope           = "Random slope of MAP",
  sd_map                 = "SD MAP"
)

label_order <- ifelse(term_order %in% names(term_labels), term_labels[term_order], term_order)

or_all_plot <- or_all %>%
  mutate(term_label = ifelse(term %in% names(term_labels), term_labels[term], term),
         term_label = factor(term_label, levels = rev(label_order)))

# same red/green/blue role assignment as before, now with these 3 hex values
model_colors <- c(
  "SD MAP"          = "#E140E3",
  "Random effects"  = "#4BBE41",
  "Mean MAP"        = "#3F54BF"
)

# NOTE: base_family = "Georgia" requires the "Georgia" font to be
# registered/available to the graphics device you're using. On Windows
# this usually just works (Georgia ships with the OS). On Mac/Linux, or
# when saving with ggsave() through a non-Cairo device, you may need to
# register it first, e.g.:
#   library(extrafont); font_import(); loadfonts()
# If Georgia isn't found, ggplot will silently fall back to a default
# sans-serif font rather than erroring.
#
# geom_linerange draws the CI as a plain line with no end caps, paired
# with geom_point() for the center dot. Both share the same dodge_pos so
# they stay aligned with each other across the 3 models.
dodge_pos <- position_dodge2(width = 0.6, preserve = "single")

# switched from geom_errorbar() to geom_linerange() -- same CI line,
# no end caps/brackets.
p_combined <- ggplot(or_all_plot, aes(x = term_label, y = or, color = model)) +
  geom_linerange(aes(ymin = ci_lower, ymax = ci_upper),
                 position = dodge_pos, linewidth = 1.6) +
  geom_point(position = dodge_pos, size = 4.8) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
  scale_y_log10() +
  scale_color_manual(values = model_colors) +
  coord_flip() +
  labs(x = NULL, y = "Odds ratio (95% CI, log scale)", color = "Model",
       title = "Odds ratios across all 3 mortality models") +
  theme_minimal(base_size = 24, base_family = "Georgia") +
  theme(legend.position = "top")

ggsave(file.path(out_dir, "or_plot_all_models_combined.png"),
       p_combined, width = 17, height = 12, dpi = 300)

cat(sprintf("\nSaved -> %s/{auc_ci_table.csv, odds_ratio_table.csv,\n", out_dir))
cat("          or_plot_mean_map.png, or_plot_sd_map.png, or_plot_random_effects.png,\n")
cat("          or_plot_all_models_combined.png}\n")
