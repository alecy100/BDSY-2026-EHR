ds <- read.csv("output/stroke_longitudinal_v3.csv")

library(tidyverse)

# ============================================================
# STROKE DATASET — BINNED
# Prerequisite: ds is already in the environment.
#   It is the wide-format data frame produced by stroke_dataset_v3.R.
#   If loading from disk, run first:
#     ds <- data.table::fread("output/stroke_longitudinal_v3.csv")
#
# Bins the 48-hour ICU window into 12 equal 4-hour intervals:
#   [0,4), [4,8), [8,12), ..., [44,48]   <- last bin closed on right to capture hour 48
#
# Each row in the output is one patient-bin combination.
# Numeric columns:   mean of all measurements in the bin (NA if no measurements)
# Character columns: first value (these are static — same across all rows per patient)
# ============================================================

library(dplyr)
library(data.table)

# safe_mean returns NA (not NaN) when a bin contains no measurements for a variable
safe_mean <- function(x) {
  res <- mean(x, na.rm = TRUE)
  if (is.nan(res)) NA_real_ else res
}

bin_breaks <- seq(0, 48, by = 4)
bin_labels <- paste0(head(bin_breaks, -1), "-", tail(bin_breaks, -1), "h")
# e.g. "0-4h", "4-8h", ..., "44-48h"

ds_binned <- ds |>
  filter(!is.na(hours_since_icu_intime)) |>
  mutate(
    bin = cut(
      hours_since_icu_intime,
      breaks         = bin_breaks,
      labels         = bin_labels,
      right          = FALSE,        # left-closed: [0,4), [4,8), ..., [44,48]
      include.lowest = TRUE          # includes hour 48 in the last bin
    )
  ) |>
  filter(!is.na(bin)) |>
  select(-hours_since_icu_intime) |>   # bin replaces the raw time column
  group_by(subject_id, stay_id, bin) |>
  summarise(
    across(where(is.numeric),  safe_mean),
    across(where(is.character), first),
    .groups = "drop"
  ) |>
  mutate(
    binned_hrs_since_icu = as.numeric(sub(".*-(\\d+)h", "\\1", as.character(bin)))
  ) |>
  select(subject_id, stay_id, stroke_type, stroke_binary, bin, binned_hrs_since_icu,
         everything()) |>
  arrange(subject_id, bin)

cat("Rows:", nrow(ds_binned), "| Columns:", ncol(ds_binned), "\n\n")
cat("Row count per bin:\n")
print(table(ds_binned$bin))

dir.create("output", showWarnings = FALSE)
fwrite(ds_binned, "output/stroke_binned.csv")
cat("\nSaved -> output/stroke_binned.csv\n")

