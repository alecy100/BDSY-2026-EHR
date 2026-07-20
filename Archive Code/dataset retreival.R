# ============================================================
# STROKE LONGITUDINAL DATASET BUILDER
# Produces a long-format CSV with static and time-varying
# covariates for stroke patients first admitted to the ICU.
# Standalone — no dependency on other scripts.
#
# Output columns:
#   subject_id, stay_id, stroke_type, stroke_binary,
#   hours_since_icu_intime (NA = static variable),
#   variable, value
# ============================================================

library(DBI)
library(duckdb)
library(dplyr)
library(dbplyr)
library(stringr)
library(tidyr)
library(data.table)

# ─────────────────────────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────────────────────────
mimic_dir <- "C:/electronic health reccords/3.1"
hosp <- file.path(mimic_dir, "hosp")
icu  <- file.path(mimic_dir, "icu")

con <- dbConnect(duckdb::duckdb())

reg <- function(name, path) {
  path <- gsub("\\\\", "/", path)
  invisible(dbExecute(con, sprintf(
    "CREATE OR REPLACE VIEW %s AS SELECT * FROM read_csv_auto('%s')", name, path)))
}

reg("patients",        file.path(hosp, "patients.csv.gz"))
reg("admissions",      file.path(hosp, "admissions.csv.gz"))
reg("icustays",        file.path(icu,  "icustays.csv.gz"))
reg("diagnoses_icd",   file.path(hosp, "diagnoses_icd.csv.gz"))
reg("omr",             file.path(hosp, "omr.csv.gz"))
reg("labevents",       file.path(hosp, "labevents.csv.gz"))
reg("chartevents",     file.path(icu,  "chartevents.csv.gz"))
reg("procedures_icd",  file.path(hosp, "procedures_icd.csv.gz"))
reg("inputevents",     file.path(icu,  "inputevents.csv.gz"))

# ─────────────────────────────────────────────────────────────
# 2. STROKE COHORT
#    Only patients whose very first hospital admission was an
#    ICU admission AND who carry a stroke diagnosis.
#
#    stroke_type  : ischemic | hemorrhagic | SAH
#    stroke_binary: ischemic | hemorrhagic  (SAH grouped into hemorrhagic)
# ─────────────────────────────────────────────────────────────

# Each patient's first hospital admission
first_hosp_tbl <- tbl(con, "admissions") |>
  group_by(subject_id) |>
  slice_min(order_by = admittime, n = 1, with_ties = FALSE) |>
  ungroup()

# Require first admission to include an ICU stay; keep adults only
cohort_tbl <- first_hosp_tbl |>
  inner_join(tbl(con, "icustays"), by = c("subject_id", "hadm_id")) |>
  left_join(tbl(con, "patients"), by = "subject_id") |>
  mutate(admit_year = sql("year(admittime)"),
         age = anchor_age + (admit_year - anchor_year)) |>
  filter(age >= 18) |>
  transmute(
    subject_id, hadm_id, stay_id, gender, age, race,
    admittime, dischtime,
    icu_intime  = intime,
    icu_outtime = outtime,
    hospital_expire_flag, dod
  ) |>
  group_by(subject_id) |>
  slice_min(order_by = icu_intime, n = 1, with_ties = FALSE) |>
  ungroup() |>
  compute(name = "cohort", temporary = FALSE)

# ICD code reference:
#   Ischemic:    ICD-10 I63.*   ICD-9 433_1 / 434_1 / 436
#   SAH:         ICD-10 I60.*   ICD-9 430
#   Hemorrhagic: ICD-10 I61.*, I62.*   ICD-9 431, 432.*
stroke_raw <- dbGetQuery(con, "
  SELECT
    d.subject_id, d.hadm_id, d.icd_code, d.icd_version,
    CASE
      WHEN d.icd_version = 10 AND d.icd_code LIKE 'I63%'  THEN 'ischemic'
      WHEN d.icd_version = 9  AND d.icd_code LIKE '433_1' THEN 'ischemic'
      WHEN d.icd_version = 9  AND d.icd_code LIKE '434_1' THEN 'ischemic'
      WHEN d.icd_version = 9  AND d.icd_code = '436'      THEN 'ischemic'
      WHEN d.icd_version = 10 AND d.icd_code LIKE 'I60%'  THEN 'SAH'
      WHEN d.icd_version = 9  AND d.icd_code = '430'      THEN 'SAH'
      ELSE 'hemorrhagic'
    END AS stroke_type_detail
  FROM diagnoses_icd d
  INNER JOIN cohort c ON d.subject_id = c.subject_id AND d.hadm_id = c.hadm_id
  WHERE
    (d.icd_version = 10 AND d.icd_code LIKE 'I63%')
    OR (d.icd_version = 9  AND (d.icd_code LIKE '433_1' OR d.icd_code LIKE '434_1' OR d.icd_code = '436'))
    OR (d.icd_version = 10 AND (d.icd_code LIKE 'I60%' OR d.icd_code LIKE 'I61%' OR d.icd_code LIKE 'I62%'))
    OR (d.icd_version = 9  AND (d.icd_code = '430' OR d.icd_code = '431' OR d.icd_code LIKE '432%'))
")

# One stroke_type per patient: ischemic > SAH > hemorrhagic when multiple codes present
stroke_types <- stroke_raw |>
  mutate(type_priority = case_when(
    stroke_type_detail == "ischemic"    ~ 1L,
    stroke_type_detail == "SAH"         ~ 2L,
    stroke_type_detail == "hemorrhagic" ~ 3L
  )) |>
  group_by(subject_id, hadm_id) |>
  slice_min(order_by = type_priority, n = 1, with_ties = FALSE) |>
  ungroup() |>
  mutate(stroke_binary = if_else(stroke_type_detail == "ischemic", "ischemic", "hemorrhagic")) |>
  select(subject_id, hadm_id, stroke_type = stroke_type_detail, stroke_binary)

cohort_df <- collect(tbl(con, "cohort")) |>
  inner_join(stroke_types, by = c("subject_id", "hadm_id"))

cat("Stroke patients (first ICU admission):", nrow(cohort_df), "\n")

if (nrow(cohort_df) == 0) {
  dbDisconnect(con, shutdown = TRUE)
  stop("No stroke patients found — no output written.")
}

# ─────────────────────────────────────────────────────────────
# 3. STATIC COVARIATES
# ─────────────────────────────────────────────────────────────

# ── 3a. Height / weight / BMI from OMR ──
# Take the single measurement closest to (and preferring before) ICU admission.
body_raw <- tbl(con, "omr") |>
  filter(subject_id %in% !!cohort_df$subject_id) |>
  collect() |>
  filter(str_detect(result_name, regex("weight|height|bmi", ignore_case = TRUE))) |>
  mutate(
    variable  = case_when(
      str_detect(result_name, regex("bmi",    TRUE)) ~ "bmi",
      str_detect(result_name, regex("weight", TRUE)) ~ "weight_lb",
      str_detect(result_name, regex("height", TRUE)) ~ "height_in"
    ),
    value_num = suppressWarnings(as.numeric(result_value))
  ) |>
  filter(!is.na(value_num)) |>
  left_join(cohort_df |> select(subject_id, stay_id, icu_intime), by = "subject_id") |>
  mutate(gap_days = as.numeric(difftime(chartdate, as.Date(icu_intime), units = "days"))) |>
  group_by(subject_id, stay_id, variable) |>
  arrange(gap_days > 0, abs(gap_days), .by_group = TRUE) |>   # on/before first, then nearest
  slice(1) |>
  ungroup() |>
  transmute(subject_id, stay_id, variable, value = as.character(value_num),
            hours_since_icu_intime = NA_real_)

# ── 3b. Comorbidities from admission diagnoses ──
# Coded at discharge; includes all conditions active during the stay.
# Components for simplified Charlson score: diabetes(1), cancer(2), COPD(1), HF(1), CAD(1).
comorbidities_df <- dbGetQuery(con, "
  SELECT
    d.subject_id,
    MAX(CASE
      WHEN (d.icd_version=10 AND d.icd_code LIKE 'I10%')
        OR (d.icd_version=9  AND d.icd_code LIKE '401%') THEN 1 ELSE 0 END) AS hypertension,
    MAX(CASE
      WHEN (d.icd_version=10 AND d.icd_code LIKE 'I48%')
        OR (d.icd_version=9  AND d.icd_code = '42731')   THEN 1 ELSE 0 END) AS afib,
    MAX(CASE
      WHEN (d.icd_version=10 AND d.icd_code LIKE 'N18%')
        OR (d.icd_version=9  AND d.icd_code LIKE '585%') THEN 1 ELSE 0 END) AS ckd,
    MAX(CASE
      WHEN (d.icd_version=10 AND (d.icd_code LIKE 'E10%' OR d.icd_code LIKE 'E11%' OR d.icd_code LIKE 'E13%'))
        OR (d.icd_version=9  AND d.icd_code LIKE '250%') THEN 1 ELSE 0 END) AS diabetes,
    MAX(CASE
      WHEN (d.icd_version=10 AND d.icd_code LIKE 'C%' AND d.icd_code NOT LIKE 'C44%')
        OR (d.icd_version=9  AND TRY_CAST(SUBSTRING(d.icd_code,1,3) AS INTEGER) BETWEEN 140 AND 208)
      THEN 1 ELSE 0 END) AS cancer,
    MAX(CASE
      WHEN (d.icd_version=10 AND d.icd_code LIKE 'J44%')
        OR (d.icd_version=9  AND (d.icd_code LIKE '496%' OR d.icd_code LIKE '491%'))
      THEN 1 ELSE 0 END) AS copd,
    MAX(CASE
      WHEN (d.icd_version=10 AND d.icd_code LIKE 'I50%')
        OR (d.icd_version=9  AND d.icd_code LIKE '428%') THEN 1 ELSE 0 END) AS heart_failure,
    MAX(CASE
      WHEN (d.icd_version=10 AND (d.icd_code LIKE 'I21%' OR d.icd_code LIKE 'I22%' OR d.icd_code LIKE 'I25%'))
        OR (d.icd_version=9  AND d.icd_code LIKE '414%') THEN 1 ELSE 0 END) AS cad
  FROM diagnoses_icd d
  INNER JOIN cohort c ON d.subject_id = c.subject_id AND d.hadm_id = c.hadm_id
  GROUP BY d.subject_id
") |>
  mutate(charlson_score = diabetes*1 + cancer*2 + copd*1 + heart_failure*1 + cad*1)

# ── 3c. Reperfusion therapy (tPA or mechanical thrombectomy) ──
#  ICD-9  9910  = infusion of thrombolytic agent
#  ICD-10 3E0*GC = alteplase administration (various routes)
#  ICD-10 03C*3ZZ = extirpation (thrombectomy) of intracranial/carotid arteries
reperfusion_df <- dbGetQuery(con, "
  SELECT DISTINCT p.subject_id, 1 AS reperfusion_therapy
  FROM procedures_icd p
  INNER JOIN cohort c ON p.subject_id = c.subject_id AND p.hadm_id = c.hadm_id
  WHERE (p.icd_version=9  AND p.icd_code = '9910')
     OR (p.icd_version=10 AND p.icd_code LIKE '3E0%GC')
     OR (p.icd_version=10 AND p.icd_code LIKE '03C%3ZZ')
")

# ── 3d. Vasopressor use (any vasopressor in first 24h of ICU) ──
#  Itemids: norepinephrine=221906, epinephrine=221289, dopamine=221662,
#           phenylephrine=221749, vasopressin=222315
vaso_df <- dbGetQuery(con, "
  SELECT DISTINCT ie.subject_id, 1 AS vasopressor_baseline
  FROM inputevents ie
  INNER JOIN cohort c ON ie.subject_id = c.subject_id AND ie.stay_id = c.stay_id
  WHERE ie.itemid IN (221906, 221289, 221662, 221749, 222315)
    AND ie.amount > 0
    AND ie.starttime >= c.icu_intime
    AND ie.starttime <= c.icu_intime + INTERVAL 24 HOURS
")

# ── 3e. Mechanical ventilation (any vent mode charted in first 24h) ──
vent_df <- dbGetQuery(con, "
  SELECT DISTINCT ce.subject_id, 1 AS mechanical_vent_baseline
  FROM chartevents ce
  INNER JOIN cohort c ON ce.subject_id = c.subject_id AND ce.stay_id = c.stay_id
  WHERE ce.itemid = 223849
    AND ce.value NOT IN ('None', '')
    AND ce.value IS NOT NULL
    AND ce.charttime >= c.icu_intime
    AND ce.charttime <= c.icu_intime + INTERVAL 24 HOURS
")

# ── 3f. Code status / DNR-DNI (first charted value in first 24h) ──
code_df <- dbGetQuery(con, "
  SELECT ce.subject_id, ce.value AS code_status_value
  FROM chartevents ce
  INNER JOIN cohort c ON ce.subject_id = c.subject_id AND ce.stay_id = c.stay_id
  WHERE ce.itemid = 223758
    AND ce.charttime >= c.icu_intime
    AND ce.charttime <= c.icu_intime + INTERVAL 24 HOURS
  QUALIFY ROW_NUMBER() OVER (PARTITION BY ce.subject_id ORDER BY ce.charttime) = 1
") |>
  mutate(code_status_dnr = if_else(
    str_detect(code_status_value, regex("DNR|DNI|Comfort", ignore_case = TRUE)), 1L, 0L))

# ─────────────────────────────────────────────────────────────
# 4. LONGITUDINAL MEASUREMENTS (first 24h, anchored to ICU admission = hour 0)
# ─────────────────────────────────────────────────────────────

# ── 4a. Labs ──
# itemids:  BUN=51006, lactate=50813, INR=51237, creatinine=50912, WBC=51301,
#           platelets=51265, PaO2=50821, FiO2=50816, sodium=50983
lab_item_map <- c(
  "51006" = "bun",    "50813" = "lactate",    "51237" = "inr",
  "50912" = "creatinine", "51301" = "wbc",    "51265" = "platelets",
  "50821" = "pao2",   "50816" = "fio2_lab",   "50983" = "sodium"
)

labs_raw <- dbGetQuery(con, sprintf("
  SELECT l.subject_id, l.hadm_id, l.itemid, l.charttime, l.valuenum
  FROM labevents l
  INNER JOIN cohort c ON l.subject_id = c.subject_id AND l.hadm_id = c.hadm_id
  WHERE l.itemid IN (%s)
    AND l.valuenum IS NOT NULL AND l.valuenum > 0
    AND l.charttime >= c.icu_intime
    AND l.charttime <= c.icu_intime + INTERVAL 24 HOURS
", paste(names(lab_item_map), collapse = ",")))

labs_timed <- labs_raw |>
  left_join(cohort_df |> select(subject_id, hadm_id, stay_id, icu_intime),
            by = c("subject_id", "hadm_id")) |>
  mutate(
    variable               = lab_item_map[as.character(itemid)],
    hours_since_icu_intime = as.numeric(difftime(charttime, icu_intime, units = "hours"))
  ) |>
  filter(hours_since_icu_intime >= 0, hours_since_icu_intime <= 24)

# P/F ratio: match each PaO2 to the nearest FiO2 within ±1 hour
# FiO2 in labevents (50816) is stored as percentage (21–100)
pao2_df <- labs_timed |>
  filter(variable == "pao2") |>
  select(subject_id, stay_id, charttime_pao2 = charttime,
         hours_since_icu_intime, pao2 = valuenum)

fio2_df <- labs_timed |>
  filter(variable == "fio2_lab") |>
  select(subject_id, stay_id, charttime_fio2 = charttime, fio2 = valuenum)

pf_long <- pao2_df |>
  left_join(fio2_df, by = c("subject_id", "stay_id")) |>
  mutate(tdiff = abs(as.numeric(difftime(charttime_fio2, charttime_pao2, units = "hours")))) |>
  filter(tdiff <= 1) |>
  group_by(subject_id, stay_id, charttime_pao2) |>
  slice_min(order_by = tdiff, n = 1, with_ties = FALSE) |>
  ungroup() |>
  mutate(
    fio2_frac = if_else(fio2 > 1, fio2 / 100, fio2),     # normalize percent -> fraction
    value     = as.character(round(pao2 / fio2_frac, 1)),
    variable  = "pf_ratio"
  ) |>
  filter(as.numeric(value) > 0, as.numeric(value) < 700) |>
  select(subject_id, stay_id, hours_since_icu_intime, variable, value)

# Regular labs (exclude raw pao2 / fio2 intermediates from final output)
labs_long <- labs_timed |>
  filter(!variable %in% c("pao2", "fio2_lab")) |>
  transmute(subject_id, stay_id, hours_since_icu_intime,
            variable, value = as.character(valuenum))

# ── 4b. Vitals from chartevents ──
# GCS eye=220739, verbal=223900, motor=223901
# MAP invasive=220052, non-invasive=220181
# Temp °C=223762, Temp °F=223761
vital_items <- c(220739, 223900, 223901, 220052, 220181, 223762, 223761)

vitals_raw <- dbGetQuery(con, sprintf("
  SELECT ce.subject_id, ce.stay_id, ce.itemid, ce.charttime, ce.valuenum
  FROM chartevents ce
  INNER JOIN cohort c ON ce.subject_id = c.subject_id AND ce.stay_id = c.stay_id
  WHERE ce.itemid IN (%s)
    AND ce.valuenum IS NOT NULL
    AND (ce.warning IS NULL OR ce.warning = 0)
    AND ce.charttime >= c.icu_intime
    AND ce.charttime <= c.icu_intime + INTERVAL 24 HOURS
", paste(vital_items, collapse = ",")))

vitals_timed <- vitals_raw |>
  left_join(cohort_df |> select(subject_id, stay_id, icu_intime), by = c("subject_id", "stay_id")) |>
  mutate(hours_since_icu_intime = as.numeric(difftime(charttime, icu_intime, units = "hours"))) |>
  filter(hours_since_icu_intime >= 0, hours_since_icu_intime <= 24)

# GCS: sum all three components charted at the same charttime
gcs_long <- vitals_timed |>
  filter(itemid %in% c(220739, 223900, 223901)) |>
  group_by(subject_id, stay_id, charttime, hours_since_icu_intime) |>
  summarise(n_comp = n_distinct(itemid), gcs = sum(valuenum, na.rm = TRUE), .groups = "drop") |>
  filter(n_comp == 3, gcs >= 3, gcs <= 15) |>        # complete assessments, valid range
  transmute(subject_id, stay_id, hours_since_icu_intime,
            variable = "gcs", value = as.character(gcs))

# MAP: prefer arterial (220052); fall back to non-invasive (220181)
map_long <- vitals_timed |>
  filter(itemid %in% c(220052, 220181)) |>
  mutate(src = if_else(itemid == 220052L, 1L, 2L)) |>
  group_by(subject_id, stay_id, charttime, hours_since_icu_intime) |>
  slice_min(order_by = src, n = 1, with_ties = FALSE) |>
  ungroup() |>
  filter(valuenum > 20, valuenum < 200) |>
  transmute(subject_id, stay_id, hours_since_icu_intime,
            variable = "map", value = as.character(valuenum))

# Temperature: convert °F -> °C; prefer direct Celsius (223762)
temp_long <- vitals_timed |>
  filter(itemid %in% c(223762, 223761)) |>
  mutate(temp_c = if_else(itemid == 223762L, valuenum, (valuenum - 32) * 5 / 9)) |>
  filter(temp_c > 25, temp_c < 45) |>
  group_by(subject_id, stay_id, charttime, hours_since_icu_intime) |>
  slice_min(order_by = itemid, n = 1, with_ties = FALSE) |>   # prefer 223762
  ungroup() |>
  transmute(subject_id, stay_id, hours_since_icu_intime,
            variable = "temperature_c", value = as.character(round(temp_c, 2)))

# ─────────────────────────────────────────────────────────────
# 5. ASSEMBLE WIDE-FORMAT DATAFRAME
#    Longitudinal measurements define the rows (one per time point).
#    Static covariates are joined on and repeat at every time point.
# ─────────────────────────────────────────────────────────────

all_patients <- cohort_df |> select(subject_id, stay_id)

# ── 5a. Build one wide static row per patient ──

static_demo <- cohort_df |>
  mutate(sex = if_else(gender == "M", 1L, 0L)) |>
  select(subject_id, stay_id, age, sex, race)

static_body <- body_raw |>
  select(subject_id, stay_id, variable, value) |>
  pivot_wider(names_from = variable, values_from = value) |>
  type.convert(as.is = TRUE)

static_comorbid <- comorbidities_df |>
  left_join(all_patients, by = "subject_id")

static_treatment <- all_patients |>
  left_join(reperfusion_df, by = "subject_id") |>
  left_join(vaso_df,        by = "subject_id") |>
  left_join(vent_df,        by = "subject_id") |>
  left_join(code_df |> select(subject_id, code_status_dnr), by = "subject_id") |>
  mutate(
    reperfusion_therapy      = replace_na(reperfusion_therapy,      0L),
    vasopressor_baseline     = replace_na(vasopressor_baseline,     0L),
    mechanical_vent_baseline = replace_na(mechanical_vent_baseline, 0L),
    code_status_dnr          = replace_na(code_status_dnr,          0L)
  )

stroke_meta <- cohort_df |> select(subject_id, stroke_type, stroke_binary)

static_wide <- static_demo |>
  left_join(static_body,      by = c("subject_id", "stay_id")) |>
  left_join(static_comorbid,  by = c("subject_id", "stay_id")) |>
  left_join(static_treatment, by = c("subject_id", "stay_id")) |>
  left_join(stroke_meta,      by = "subject_id")

# ── 5b. Pivot longitudinal measurements to wide (one row per time point) ──

longitudinal_wide <- bind_rows(labs_long, pf_long, gcs_long, map_long, temp_long) |>
  group_by(subject_id, stay_id, hours_since_icu_intime, variable) |>
  slice(1) |>   # drop any exact duplicate (subject, stay, time, variable) entries
  ungroup() |>
  pivot_wider(names_from = variable, values_from = value) |>
  type.convert(as.is = TRUE)

# ── 5c. Join static onto longitudinal: values repeat at every time point ──

dataset_wide <- longitudinal_wide |>
  left_join(static_wide, by = c("subject_id", "stay_id")) |>
  select(subject_id, stay_id, stroke_type, stroke_binary,
         hours_since_icu_intime, everything()) |>
  arrange(subject_id, hours_since_icu_intime)

cat("\nRows:", nrow(dataset_wide), "| Columns:", ncol(dataset_wide), "\n")
cat("Covariate columns:\n")
print(sort(setdiff(names(dataset_wide),
                   c("subject_id", "stay_id", "stroke_type", "stroke_binary",
                     "hours_since_icu_intime"))))

# ─────────────────────────────────────────────────────────────
# 6. WRITE OUTPUT
# ─────────────────────────────────────────────────────────────
dir.create("output", showWarnings = FALSE)
data.table::fwrite(dataset_wide, "output/stroke_longitudinal.csv")
cat("\nSaved -> output/stroke_longitudinal.csv\n")

dbDisconnect(con, shutdown = TRUE)
