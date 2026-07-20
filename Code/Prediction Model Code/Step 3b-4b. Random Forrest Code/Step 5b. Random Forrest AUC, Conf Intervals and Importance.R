# ============================================================
# STEP 5 (RF VERSION): 95% CI FOR AUC (all 4 RF variants) + VARIABLE
# IMPORTANCE PLOTS
#
#   Variant "re"      = random_intercept + random_slope
#   Variant "mean"    = mean(map)
#   Variant "sd"      = sd(map)
#   Variant "mean_sd" = mean(map) + sd(map)
#   (all from 03_rf_fit_random_forest.R / 04_rf_predict_test_and_evaluate.R,
#    response_var = "map")
#
# AUC + 95% CI:
#   pROC::ci.auc() (DeLong method) applied to each variant's TEST-SET
#   averaged predicted probability -- the same (avg_pred_prob,
#   observed_mortality) pairing 04_rf already reports the point AUC for.
#   This is a direct port of the logistic pipeline's Step 9 (their
#   numbering, not this pipeline's) -- ci.auc() only needs predicted
#   probabilities + observed outcomes, so it works identically
#   regardless of what produced the probabilities.
#
# NO ODDS RATIOS FOR RF:
#   Random forest has no coefficients to exponentiate -- it's an
#   ensemble of trees, not a logit equation, so "odds ratio" isn't a
#   defined quantity here. The closest analog is variable importance
#   (pooled across the 5 imputations, mean only -- no SD/error bars, see
#   prior discussion on why RF importance has no meaningful CI), plotted
#   as bar charts rather than dot+errorbar. IMPORTANT DIFFERENCE FROM THE
#   OR PLOTS: importance measures how much a variable helped prediction,
#   not which direction it pushes risk -- so there is no "no effect"
#   reference line at 1 / no log scale here; the dashed line marks 0
#   (no contribution) instead.
#
# STYLE UPDATE (matching the logistic pipeline's updated combined OR
# plot): readable variable labels (with safe fallback to the raw name
# if unmapped, so nothing silently disappears), a shared color palette
# keyed to the SAME model names across both pipelines (so "Random
# effects"/"Mean MAP"/"SD MAP" are the same color in the RF figures as
# in the logistic figures), Georgia font, larger base size, higher dpi.
# NOTE: the underlying geom is still a bar chart (geom_col), not
# dot+linerange -- that was an earlier, deliberate change (RF importance
# has no CI to draw), so only the color/font/label/resolution styling is
# ported here, not the point+linerange geometry itself.
# ============================================================

library(dplyr)
library(pROC)
library(ggplot2)

response_var   <- "map"   # <<== must match the response_var used to fit the RF models
variants       <- c("re", "mean", "sd")   # <<== "mean_sd" temporarily excluded for consistency with the logistic pipeline (re/mean/sd only)
variant_labels <- c(re = "Random effects", mean = "Mean MAP", sd = "SD MAP", mean_sd = "Mean + SD MAP")

rf_out_dir <- sprintf("prediction/%s/rf", response_var)
out_dir    <- "prediction/model_comparison/rf"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

rf_comparison <- readRDS(file.path(rf_out_dir, "rf_variant_comparison.rds"))$results_by_variant

# ------------------------------------------------------------
# 1. AUC + 95% CI table (all 4 RF variants)
# ------------------------------------------------------------
get_auc_ci <- function(observed, predicted) {
  roc_obj <- roc(observed, predicted, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  ci_obj  <- ci.auc(roc_obj, method = "delong")
  data.frame(auc = auc_val, ci_lower = as.numeric(ci_obj[1]), ci_upper = as.numeric(ci_obj[3]))
}

auc_table <- bind_rows(lapply(variants, function(v) {
  fp <- rf_comparison[[v]]$final_preds
  get_auc_ci(fp$observed_mortality, fp$avg_pred_prob) %>%
    mutate(model = variant_labels[[v]], variant = v)
})) %>% select(model, variant, auc, ci_lower, ci_upper)

cat("\n================= RF TEST-SET AUC (95% CI, DeLong) =================\n")
print(auc_table, digits = 4)
write.csv(auc_table, file.path(out_dir, "rf_auc_ci_table.csv"), row.names = FALSE)

# ------------------------------------------------------------
# 1b. Confusion matrices (all RF variants), at a fixed probability
#     threshold applied to the averaged-across-imputations predicted
#     probability (the same avg_pred_prob used for the AUC table above).
#
# NOTE: 0.5 is a NAIVE default threshold -- with a ~20% mortality base
# rate, a model can have real ranking ability (decent AUC) while still
# looking poor at 0.5 specifically, since 0.5 was never calibrated to
# this outcome's prevalence. Treat this as one diagnostic view, not the
# final word on the model's usefulness; a different operating threshold
# (e.g. chosen via Youden's index, or a clinically-set sensitivity
# target) may tell a different story. THRESHOLDS is easy to edit below --
# fill in each variant's value from rf_oob_threshold_selection.R's
# output (mean_threshold per variant) once you've run it; defaults to
# 0.5 for all variants until then.
#
# STYLE UPDATE: matches the logistic pipeline's updated Step 11
# (confusion matrices) -- ppv/npv added alongside accuracy/sensitivity/
# specificity; plot styling (theme_minimal base_size=14, no gridlines,
# bold count labels, the DCE6F1->1F4E79 blue gradient, y-axis reversed
# via scale_y_discrete(limits=rev), "Observed/Predicted mortality" axis
# titles, raw 0/1 labels rather than "Survived"/"Died" text) now matches
# theirs exactly, rather than the Georgia/base_size=24 styling used for
# the OR/importance plots elsewhere in this pipeline -- this confusion
# matrix style is deliberately its own, separate convention.
# ------------------------------------------------------------
# <<== EDIT: replace each value with rf_oob_threshold_selection.R's
# mean_threshold for that variant (from oob_threshold_selection.csv)
THRESHOLDS <- c(re = 0.5, mean = 0.5, sd = 0.5, mean_sd = 0.5)

build_confusion <- function(observed, predicted_prob, threshold, model_name) {
  predicted_class <- factor(ifelse(predicted_prob >= threshold, 1, 0), levels = c(0, 1))
  observed_f      <- factor(observed, levels = c(0, 1))

  cm <- table(Predicted = predicted_class, Observed = observed_f)

  tn <- cm["0", "0"]; fp <- cm["1", "0"]
  fn <- cm["0", "1"]; tp <- cm["1", "1"]

  metrics <- data.frame(
    model       = model_name,
    n           = length(observed),
    threshold   = threshold,
    tn = tn, fp = fp, fn = fn, tp = tp,
    accuracy    = (tp + tn) / (tp + tn + fp + fn),
    sensitivity = tp / (tp + fn),   # a.k.a. recall / true positive rate
    specificity = tn / (tn + fp),
    ppv         = tp / (tp + fp),   # precision
    npv         = tn / (tn + fn)
  )

  list(cm = cm, metrics = metrics)
}

confusion_results <- lapply(variants, function(v) {
  fp <- rf_comparison[[v]]$final_preds
  build_confusion(fp$observed_mortality, fp$avg_pred_prob, THRESHOLDS[[v]], variant_labels[[v]])
})
names(confusion_results) <- variants

for (v in variants) {
  cat(sprintf("\n================= CONFUSION MATRIX: %s (threshold = %.4f) =================\n",
              variant_labels[[v]], THRESHOLDS[[v]]))
  print(confusion_results[[v]]$cm)
}

confusion_table <- bind_rows(lapply(confusion_results, function(r) r$metrics)) %>%
  mutate(variant = variants) %>%
  select(model, variant, n, threshold, tn, fp, fn, tp, accuracy, sensitivity, specificity, ppv, npv)

cat("\n================= CLASSIFICATION METRICS (all RF variants) =================\n")
print(confusion_table, digits = 3)
write.csv(confusion_table, file.path(out_dir, "rf_confusion_matrix_table.csv"), row.names = FALSE)

plot_confusion <- function(cm, title) {
  cm_df <- as.data.frame(cm) %>%  # columns: Predicted, Observed, Freq
    mutate(Predicted = factor(Predicted, levels = c("0", "1"), labels = c("Survived", "Died")),
           Observed  = factor(Observed,  levels = c("0", "1"), labels = c("Survived", "Died")))

  ggplot(cm_df, aes(x = Observed, y = Predicted, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq, color = Freq > max(Freq) / 2), fontface = "bold", size = 6, family = "Georgia") +
    scale_color_manual(values = c("TRUE" = "white", "FALSE" = "black"), guide = "none") +
    scale_fill_gradient(low = "#DCE6F1", high = "#1F4E79") +
    labs(x = "Observed mortality", y = "Predicted mortality",
         title = title, fill = "Count") +
    theme_minimal(base_size = 14, base_family = "Georgia") +
    theme(panel.grid = element_blank(),
          plot.title = element_text(size = 11))
}

for (v in variants) {
  p_cm <- plot_confusion(confusion_results[[v]]$cm,
                          sprintf("Confusion matrix: Random forest, %s", variant_labels[[v]]))
  ggsave(file.path(out_dir, sprintf("confusion_matrix_%s.png", v)), p_cm, width = 6, height = 5, dpi = 300)
}

# ------------------------------------------------------------
# 2. load pooled importance for all variants ONCE (used by both the
#    per-variant plots below and the combined plot in section 4)
# ------------------------------------------------------------
pooled_importance_list <- lapply(variants, function(v) {
  readRDS(file.path(rf_out_dir, v, "pooled_rf_summary.rds"))$pooled_importance %>%
    select(variable, MeanDecreaseAccuracy_mean, MeanDecreaseAccuracy_sd) %>%
    mutate(model = variant_labels[[v]], variant = v)
})
names(pooled_importance_list) <- variants

importance_all <- bind_rows(pooled_importance_list)

write.csv(importance_all, file.path(out_dir, "rf_importance_table.csv"), row.names = FALSE)

# ------------------------------------------------------------
# display labels for the plot axis -- raw variable name -> readable
# label. anything not in this map falls back to its raw variable name
# unchanged (guards against a name mismatch silently dropping off the
# plot) -- same approach as the logistic pipeline's term_labels.
# ------------------------------------------------------------
variable_labels <- c(
  age               = "Age",
  bmi               = "BMI",
  charlson_score    = "Charlson score",
  race_binary       = "Race (non-white)",
  sex               = "Sex (male)",
  stroke_binary     = "Stroke type (ischemic)",
  mean_map          = "Mean MAP",
  random_intercept  = "Random intercept of MAP",
  random_slope      = "Random slope of MAP",
  sd_map            = "SD MAP"
)

relabel <- function(x) ifelse(x %in% names(variable_labels), variable_labels[x], x)

# same color-per-model-name scheme as the logistic pipeline's plots, so
# "Random effects"/"Mean MAP"/"SD MAP" are visually identical across
# both sets of figures
model_colors <- c(
  "SD MAP"         = "#E140E3",
  "Random effects" = "#4BBE41",
  "Mean MAP"       = "#3F54BF",
  "Mean + SD MAP"  = "#E39B31"
)

# NOTE: base_family = "Georgia" requires the "Georgia" font to be
# registered/available to the graphics device you're using. On Windows
# this usually just works (Georgia ships with the OS). On Mac/Linux, or
# when saving with ggsave() through a non-Cairo device, you may need to
# register it first, e.g.:
#   library(extrafont); font_import(); loadfonts()
# If Georgia isn't found, ggplot will silently fall back to a default
# sans-serif font rather than erroring.

# ------------------------------------------------------------
# 3. variable importance BAR plots (mean across 5 imputations only --
#    no SD shown, since importance isn't a parameter estimate with a
#    meaningful CI; see prior discussion). One per variant -- the RF
#    analog of the odds-ratio forest plot.
# ------------------------------------------------------------
plot_importance <- function(df, title, bar_color) {
  df <- df %>% arrange(MeanDecreaseAccuracy_mean)
  df$variable_label <- factor(relabel(df$variable), levels = relabel(df$variable))
  ggplot(df, aes(x = MeanDecreaseAccuracy_mean, y = variable_label)) +
    geom_col(fill = bar_color, width = 0.7) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
    scale_y_discrete(expand = expansion(mult = 0.15)) +
    labs(x = "Mean decrease in accuracy (mean across 5 imputations)",
         y = NULL, title = title) +
    theme_minimal(base_size = 24, base_family = "Georgia")
}

for (v in variants) {
  model_name <- variant_labels[[v]]
  p <- plot_importance(pooled_importance_list[[v]],
                        sprintf("RF variable importance: %s", model_name),
                        bar_color = model_colors[[model_name]])
  n_rows <- nrow(pooled_importance_list[[v]])
  ggsave(file.path(out_dir, sprintf("importance_plot_%s.png", v)), p,
         width = 10, height = max(6, 0.8 * n_rows), dpi = 300)
}

# ------------------------------------------------------------
# 4. ONE combined importance BAR plot, all RF variants on the same
#    figure (dodged bars -- SD dropped, as importance has no meaningful CI)
# ------------------------------------------------------------
# importance_all is already long-format (one row per variant x
# variable), so variables shared across variants (the demographic
# covariates, present in all 3) naturally get 3 dodged bars on the same
# row, while variables unique to fewer variants (random_intercept/
# random_slope only in "re"; mean_map in "mean"; sd_map in "sd")
# naturally get fewer bars on their own row -- nothing hardcoded, it
# falls out of how many variants' tables contain that variable.
#
# position_dodge2(preserve = "single") is used instead of plain
# position_dodge() so a variable's bar stays a consistent width and is
# centered correctly whether it has 1, 2, or 3 variants present at that
# row (plain position_dodge divides the row width by however many
# groups ARE present, which would make rarer variables look
# different/misaligned from universally-shared ones).

# variable order: alphabetical within "shared across most variants"
# first, then alphabetical within "unique to fewer variants", so the
# figure reads top-to-bottom as demographic/clinical covariates ->
# MAP-representation-specific terms
variable_model_counts <- importance_all %>% count(variable, name = "n_variants")
variable_order <- variable_model_counts %>%
  arrange(desc(n_variants), variable) %>%
  pull(variable)
label_order <- relabel(variable_order)

importance_all_plot <- importance_all %>%
  mutate(variable_label = factor(relabel(variable), levels = rev(label_order)))

dodge_pos <- position_dodge2(width = 0.6, preserve = "single")

p_combined <- ggplot(importance_all_plot, aes(x = variable_label, y = MeanDecreaseAccuracy_mean, fill = model)) +
  geom_col(position = dodge_pos, width = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  scale_x_discrete(expand = expansion(mult = 0.1)) +
  scale_fill_manual(values = model_colors) +
  coord_flip() +
  labs(x = NULL, y = "Mean decrease in accuracy (mean across 5 imputations)",
       fill = "RF variant",
       title = "Variable importance across RF variants") +
  theme_minimal(base_size = 24, base_family = "Georgia") +
  theme(legend.position = "top")

n_variables <- n_distinct(importance_all_plot$variable_label)
ggsave(file.path(out_dir, "importance_plot_all_variants_combined.png"),
       p_combined, width = 17, height = max(10, 1.1 * n_variables), dpi = 300)

cat(sprintf("\nSaved -> %s/rf_auc_ci_table.csv, rf_confusion_matrix_table.csv, rf_importance_table.csv,\n", out_dir))
cat("          confusion_matrix_{re,mean,sd}.png,\n")
cat("          importance_plot_{re,mean,sd}.png,\n")
cat("          importance_plot_all_variants_combined.png\n")