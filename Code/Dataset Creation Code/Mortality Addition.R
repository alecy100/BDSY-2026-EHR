# ============================================================
# STROKE 30-DAY MORTALITY
# Prerequisite: ds is already in the environment — the cleaned
#   wide-format dataset with a subject_id column.
#   If loading from disk, run first:
#     ds <- data.table::fread("output/stroke_binned.csv")
#     # or: ds <- data.table::fread("output/stroke_longitudinal_v3.csv")
#
# For each subject_id, is_dead = 1 if the patient died within
# 30 days of their ICU admission intime, 0 otherwise.
# The flag is joined onto every row of ds (repeats per subject).
# ============================================================

ds <- read.csv("output/cleanData.csv")

ids_to_remove <- c(14390259, 14477097, 16899163, 17124711, 10558762, 14252222, 10707442)

ds <- ds %>% filter(!subject_id %in% ids_to_remove)


library(DBI)
library(duckdb)
library(dplyr)
library(dbplyr)
library(data.table)

# ─────────────────────────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────────────────────────
mimic_dir <- "C:/Users/ajy20/Downloads/3.1/3.1"
hosp <- file.path(mimic_dir, "hosp")
icu  <- file.path(mimic_dir, "icu")

con <- dbConnect(duckdb::duckdb())

reg <- function(name, path) {
  path <- gsub("\\\\", "/", path)
  invisible(dbExecute(con, sprintf(
    "CREATE OR REPLACE VIEW %s AS SELECT * FROM read_csv_auto('%s')", name, path)))
}

reg("patients",   file.path(hosp, "patients.csv.gz"))
reg("admissions", file.path(hosp, "admissions.csv.gz"))
reg("icustays",   file.path(icu,  "icustays.csv.gz"))

# ─────────────────────────────────────────────────────────────
# 2. ICU INTIME PER SUBJECT
# Reproduces the cohort's anchor point: the first ICU stay on
# each patient's first hospital admission.
# ─────────────────────────────────────────────────────────────
subject_ids <- unique(ds$subject_id)

icu_intimes <- tbl(con, "admissions") |>
  filter(subject_id %in% !!subject_ids) |>
  group_by(subject_id) |>
  slice_min(order_by = admittime, n = 1, with_ties = FALSE) |>   # first admission
  ungroup() |>
  select(subject_id, hadm_id) |>
  inner_join(tbl(con, "icustays"), by = c("subject_id", "hadm_id")) |>
  group_by(subject_id) |>
  slice_min(order_by = intime, n = 1, with_ties = FALSE) |>       # first ICU stay
  ungroup() |>
  select(subject_id, icu_intime = intime) |>
  collect()

# ─────────────────────────────────────────────────────────────
# 3. DATE OF DEATH
# patients.dod: date of death from death records (NA if unknown/alive).
# Captures post-discharge deaths — more complete than admissions.deathtime.
# ─────────────────────────────────────────────────────────────
dod_df <- tbl(con, "patients") |>
  filter(subject_id %in% !!subject_ids) |>
  select(subject_id, dod) |>
  collect()

# ─────────────────────────────────────────────────────────────
# 4. 30-DAY MORTALITY FLAG
# is_dead = 1 if dod is known AND falls within [0, 30] days of
# ICU intime. days_to_death >= 0 guards against any data anomaly
# where dod predates ICU admission.
# ─────────────────────────────────────────────────────────────
mortality_df <- icu_intimes |>
  left_join(dod_df, by = "subject_id") |>
  mutate(
    icu_intime    = as.POSIXct(icu_intime),
    dod           = as.Date(dod),
    days_to_death = as.numeric(difftime(dod, as.Date(icu_intime), units = "days")),
    is_dead       = if_else(!is.na(dod) & days_to_death >= 0 & days_to_death <= 30, 1L, 0L)
  ) |>
  select(subject_id, is_dead)

cat("30-day mortality:\n")
print(table(mortality_df$is_dead, dnn = "is_dead"))
cat(sprintf("Mortality rate: %.1f%%\n\n", mean(mortality_df$is_dead) * 100))

# ─────────────────────────────────────────────────────────────
# 5. JOIN ONTO EXISTING DATASET
# left_join propagates is_dead to every row sharing a subject_id.
# ─────────────────────────────────────────────────────────────
ds_with_mortality <- ds |>
  left_join(mortality_df, by = "subject_id")

cat("Rows:", nrow(ds_with_mortality), "| Columns:", ncol(ds_with_mortality), "\n")

# ─────────────────────────────────────────────────────────────
# 6. WRITE OUTPUT
# ─────────────────────────────────────────────────────────────
dir.create("output", showWarnings = FALSE)
fwrite(ds_with_mortality, "output/stroke_clean_with_mortality.csv")
cat("Saved -> output/stroke_clean_with_mortality.csv\n")

dbDisconnect(con, shutdown = TRUE)

df2 <- read.csv("output/stroke_clean_with_mortality.csv")

library(dplyr)

# 1. Get unique subject ids
unique_ids <- df2 %>%
  distinct(subject_id)

# 2. For each unique subject id, pull is_dead from the original dataset
#    (taking the first non-missing value per subject, in case is_dead
#    is repeated identically across rows for that subject)
subject_status <- df2 %>%
  group_by(subject_id) %>%
  summarise(is_dead = first(na.omit(is_dead)), .groups = "drop")

# 3. Proportion of patients who died
# assumes is_dead is coded as 1/0 or TRUE/FALSE
prop_died <- mean(subject_status$is_dead == 1, na.rm = TRUE)
# or if is_dead is logical: mean(subject_status$is_dead, na.rm = TRUE)

prop_died