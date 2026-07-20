# ============================================================
# STEP 11: CONFUSION MATRICES FOR ALL 3 MORTALITY MODELS
# ============================================================
# Uses the same imputation-averaged test-set predicted probability each
# model's AUC/Brier score is already computed from in 06 (avg_pred_prob
# vs observed_mortality). Predicted probabilities are converted to a
# binary predicted class using each model's OWN optimal threshold --
# not a fixed 0.5 -- chosen via Youden's J statistic:
#   J = sensitivity + specificity - 1
# maximized over all thresholds on that model's ROC curve
# (pROC::coords(..., best.method = "youden")). Each of the 3 models gets
# its own threshold, since there's no reason their optimal cutoffs would
# be the same -- these are reported alongside each model's metrics
# rather than assumed to match.
#
#   Model A = "Mean MAP"        (05_mean_sd_map_mortality_models.R)
#   Model B = "SD MAP"          (05_mean_sd_map_mortality_models.R)
#   Model C = "Random effects"  (03_fit_lmm_and_logistic.R / 04, response_var = "map")
# ============================================================

library(dplyr)
library(pROC)

response_var <- "map"   # <<== must match the response_var used to fit Model C

meansd  <- readRDS("prediction/mean_sd_map_models/mean_sd_map_models.rds")
modelC_preds <- readRDS(file.path(sprintf("prediction/%s", response_var), "test_predictions_final.rds"))

out_dir <- "prediction/model_comparison"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Youden's J-optimal threshold for one model's ROC curve ----
# if multiple thresholds tie for the max J (coords() can return more
# than one row in that case), the first is used
get_youden_threshold <- function(observed, predicted_prob) {
  roc_obj <- roc(observed, predicted_prob, quiet = TRUE)
  best <- coords(roc_obj, x = "best", best.method = "youden",
                 ret = "threshold", transpose = FALSE)
  as.numeric(best$threshold[1])
}

# ---- build one model's confusion matrix + standard metrics ----
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

threshold_mean <- get_youden_threshold(meansd$final_mean$observed_mortality, meansd$final_mean$avg_pred_prob)
threshold_sd   <- get_youden_threshold(meansd$final_sd$observed_mortality,   meansd$final_sd$avg_pred_prob)
threshold_C    <- get_youden_threshold(modelC_preds$final_preds$observed_mortality, modelC_preds$final_preds$avg_pred_prob)

result_mean <- build_confusion(meansd$final_mean$observed_mortality,
                                meansd$final_mean$avg_pred_prob,
                                threshold_mean, "Mean MAP")
result_sd   <- build_confusion(meansd$final_sd$observed_mortality,
                                meansd$final_sd$avg_pred_prob,
                                threshold_sd, "SD MAP")
result_C    <- build_confusion(modelC_preds$final_preds$observed_mortality,
                                modelC_preds$final_preds$avg_pred_prob,
                                threshold_C, "Random effects")

cat(sprintf("\n================= CONFUSION MATRIX: Mean MAP (Youden threshold = %.4f) =================\n", threshold_mean))
print(result_mean$cm)

cat(sprintf("\n================= CONFUSION MATRIX: SD MAP (Youden threshold = %.4f) =================\n", threshold_sd))
print(result_sd$cm)

cat(sprintf("\n================= CONFUSION MATRIX: Random effects (Youden threshold = %.4f) =================\n", threshold_C))
print(result_C$cm)

metrics_all <- bind_rows(result_mean$metrics, result_sd$metrics, result_C$metrics)

cat("\n================= CLASSIFICATION METRICS (all 3 models) =================\n")
print(metrics_all, digits = 3)

write.csv(metrics_all, file.path(out_dir, "confusion_matrix_metrics.csv"), row.names = FALSE)
cat(sprintf("\nSaved -> %s/confusion_matrix_metrics.csv\n", out_dir))

# ------------------------------------------------------------
# ggplot heatmap visualization of each confusion matrix
# ------------------------------------------------------------
library(ggplot2)

plot_confusion <- function(cm, title) {
  cm_df <- as.data.frame(cm)  # columns: Predicted, Observed, Freq

  # is_dead coding: 1 = died, 0 = survived
  outcome_labels <- c("0" = "Survived", "1" = "Died")
  cm_df <- cm_df %>%
    mutate(Predicted = factor(outcome_labels[as.character(Predicted)], levels = c("Survived", "Died")),
           Observed  = factor(outcome_labels[as.character(Observed)],  levels = c("Survived", "Died")),
           # true-negative cell (Survived observed x Survived predicted) gets a
           # white count label instead of black, since it's the cell that
           # typically has the largest count / darkest fill in this cohort
           text_color = ifelse(Observed == "Survived" & Predicted == "Survived", "white", "black"))

  ggplot(cm_df, aes(x = Observed, y = Predicted, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq, color = text_color), size = 6, fontface = "bold", family = "Georgia") +
    scale_color_identity() +
    scale_fill_gradient(low = "#DCE6F1", high = "#1F4E79") +
    labs(x = "Observed mortality", y = "Predicted mortality",
         title = title, fill = "Count") +
    theme_minimal(base_size = 24, base_family = "Georgia") +
    theme(panel.grid = element_blank())
}

p_cm_mean <- plot_confusion(result_mean$cm, "Confusion matrix: Mean MAP")
p_cm_sd   <- plot_confusion(result_sd$cm,   "Confusion matrix: SD MAP")
p_cm_C    <- plot_confusion(result_C$cm,    "Confusion matrix: Logistic, RE")

ggsave(file.path(out_dir, "confusion_matrix_mean_map.png"),       p_cm_mean, width = 8, height = 7, dpi = 300)
ggsave(file.path(out_dir, "confusion_matrix_sd_map.png"),         p_cm_sd,   width = 8, height = 7, dpi = 300)
ggsave(file.path(out_dir, "confusion_matrix_random_effects.png"), p_cm_C,    width = 8, height = 7, dpi = 300)

cat(sprintf("Saved -> %s/{confusion_matrix_mean_map.png, confusion_matrix_sd_map.png,\n", out_dir))
cat("          confusion_matrix_random_effects.png}\n")
