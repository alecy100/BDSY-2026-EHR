# ============================================================
# STROKE DATASET V3  —  union of stroke_dataset.R + stroke_dataset_v2.R
# Stroke patients (age >= 18) whose first hospital admission was to the ICU.
# Time window: first 48 hours of ICU admission.
#
# Static covariates (repeated at every row):
#   age, sex, race
#   height_in, weight_lb, bmi                        [v1]
#   hypertension, afib, ckd, diabetes, cancer,
#     copd, heart_failure, cad                        [v1 individual flags]
#   charlson_score (comprehensive 13-component)       [v2]
#   reperfusion_therapy (tPA or thrombectomy,
#     from procedures_icd)                            [v1]
#   alteplase (tPA from inputevents + procedures_icd) [v2]
#   mannitol, nacl3_hypertonic                        [v2]
#   vasopressor_baseline, mechanical_vent_baseline,
#     code_status_dnr                                 [v1]
#
# Longitudinal columns (one row per measurement time, 0–48 h):
#   bun, lactate, inr, creatinine, sodium,            [v1 labs]
#   pf_ratio (PaO2 / [FiO2/100])                      [v1 computed]
#   glucose_lab, hemoglobin, wbc, platelet, ptt        [v2 labs]
#   gcs, map, temperature_c                            [v1 chartevents]
#   sbp, dbp, heart_rate, spo2, glucose_chart,         [v2 chartevents]
#   icp, cpp, rass, tidal_volume_obs,
#   respiratory_rate                                   [v2 chartevents]
#   strength_left_arm, strength_left_leg,
#   strength_right_arm, strength_right_leg             [v2 chartevents]
#
# NOTE — verify limb-strength itemids against d_items before relying on results:
#   dbGetQuery(con, "SELECT itemid, label, param_type
#                    FROM d_items WHERE lower(label) LIKE '%strength%'")
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

reg("patients",       file.path(hosp, "patients.csv.gz"))
reg("admissions",     file.path(hosp, "admissions.csv.gz"))
reg("icustays",       file.path(icu,  "icustays.csv.gz"))
reg("diagnoses_icd",  file.path(hosp, "diagnoses_icd.csv.gz"))
reg("omr",            file.path(hosp, "omr.csv.gz"))
reg("labevents",      file.path(hosp, "labevents.csv.gz"))
reg("chartevents",    file.path(icu,  "chartevents.csv.gz"))
reg("d_items",        file.path(icu,  "d_items.csv.gz"))
reg("procedures_icd", file.path(hosp, "procedures_icd.csv.gz"))
reg("inputevents",    file.path(icu,  "inputevents.csv.gz"))

# ─────────────────────────────────────────────────────────────
# 2. STROKE COHORT
# ─────────────────────────────────────────────────────────────

first_hosp_tbl <- tbl(con, "admissions") |>
  group_by(subject_id) |>
  slice_min(order_by = admittime, n = 1, with_ties = FALSE) |>
  ungroup()

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

# Stroke type: ischemic > SAH > hemorrhagic when multiple codes present
stroke_raw <- dbGetQuery(con, "
  SELECT d.subject_id, d.hadm_id,
    CASE
      WHEN d.icd_version=10 AND d.icd_code LIKE 'I63%'  THEN 'ischemic'
      WHEN d.icd_version=9  AND d.icd_code LIKE '433_1' THEN 'ischemic'
      WHEN d.icd_version=9  AND d.icd_code LIKE '434_1' THEN 'ischemic'
      WHEN d.icd_version=9  AND d.icd_code = '436'      THEN 'ischemic'
      WHEN d.icd_version=10 AND d.icd_code LIKE 'I60%'  THEN 'SAH'
      WHEN d.icd_version=9  AND d.icd_code = '430'      THEN 'SAH'
      ELSE 'hemorrhagic'
    END AS stroke_detail
  FROM diagnoses_icd d
  INNER JOIN cohort c ON d.subject_id=c.subject_id AND d.hadm_id=c.hadm_id
  WHERE
    (d.icd_version=10 AND d.icd_code LIKE 'I63%')
    OR (d.icd_version=9  AND (d.icd_code LIKE '433_1' OR d.icd_code LIKE '434_1' OR d.icd_code='436'))
    OR (d.icd_version=10 AND (d.icd_code LIKE 'I60%' OR d.icd_code LIKE 'I61%' OR d.icd_code LIKE 'I62%'))
    OR (d.icd_version=9  AND (d.icd_code='430' OR d.icd_code='431' OR d.icd_code LIKE '432%'))
")

stroke_types <- stroke_raw |>
  mutate(priority = case_when(stroke_detail=="ischemic" ~ 1L,
                              stroke_detail=="SAH"      ~ 2L,
                              TRUE                      ~ 3L)) |>
  group_by(subject_id, hadm_id) |>
  slice_min(order_by = priority, n = 1, with_ties = FALSE) |>
  ungroup() |>
  mutate(stroke_binary = if_else(stroke_detail == "ischemic", "ischemic", "hemorrhagic")) |>
  select(subject_id, hadm_id, stroke_type = stroke_detail, stroke_binary)

cohort_df <- collect(tbl(con, "cohort")) |>
  inner_join(stroke_types, by = c("subject_id", "hadm_id"))

cat("Stroke patients (first ICU admission, age >= 18):", nrow(cohort_df), "\n")

if (nrow(cohort_df) == 0) {
  dbDisconnect(con, shutdown = TRUE)
  stop("No stroke patients found — no output written.")
}

all_patients <- cohort_df |> select(subject_id, stay_id)

# ─────────────────────────────────────────────────────────────
# 3. STATIC COVARIATES
# ─────────────────────────────────────────────────────────────

# ── 3a. Demographics ──
static_demo <- cohort_df |>
  mutate(sex = if_else(gender == "M", 1L, 0L)) |>
  select(subject_id, stay_id, age, sex, race)

# ── 3b. Height / weight / BMI from OMR [v1] ──
# Closest measurement to ICU admission (before preferred, then nearest after).
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
  arrange(gap_days > 0, abs(gap_days), .by_group = TRUE) |>
  slice(1) |>
  ungroup()

static_body <- body_raw |>
  select(subject_id, stay_id, variable, value = value_num) |>
  pivot_wider(names_from = variable, values_from = value) |>
  type.convert(as.is = TRUE)

# ── 3c. Comorbidity flags + comprehensive Charlson score ──
#
# Individual flags (v1): hypertension, afib, ckd, diabetes, cancer, copd, heart_failure, cad
# Charlson components (v2, 13-component; cerebrovascular disease excluded — constant here):
#   MI(1), CHF(1), PVD(1), COPD(1), Dementia(1), Mild liver disease(1),
#   Diabetes uncomplicated(1), Hemiplegia(2), Diabetes complicated(2),
#   CKD moderate-severe(2), Non-skin malignancy(2),
#   Moderate-severe liver disease(3), Metastatic tumor(6)
comorbid_df <- dbGetQuery(con, "
  SELECT d.subject_id,
    -- Individual comorbidity flags
    MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'I10%')
                 OR (d.icd_version=9  AND d.icd_code LIKE '401%')
             THEN 1 ELSE 0 END) AS hypertension,
    MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'I48%')
                 OR (d.icd_version=9  AND d.icd_code = '42731')
             THEN 1 ELSE 0 END) AS afib,
    MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'N18%')
                 OR (d.icd_version=9  AND d.icd_code LIKE '585%')
             THEN 1 ELSE 0 END) AS ckd,
    MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'E10%' OR d.icd_code LIKE 'E11%' OR d.icd_code LIKE 'E13%'))
                 OR (d.icd_version=9  AND d.icd_code LIKE '250%')
             THEN 1 ELSE 0 END) AS diabetes,
    MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'C%' AND d.icd_code NOT LIKE 'C44%')
                 OR (d.icd_version=9  AND TRY_CAST(SUBSTRING(d.icd_code,1,3) AS INTEGER) BETWEEN 140 AND 208)
             THEN 1 ELSE 0 END) AS cancer,
    MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'J44%')
                 OR (d.icd_version=9  AND (d.icd_code LIKE '496%' OR d.icd_code LIKE '491%'))
             THEN 1 ELSE 0 END) AS copd,
    MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'I50%')
                 OR (d.icd_version=9  AND d.icd_code LIKE '428%')
             THEN 1 ELSE 0 END) AS heart_failure,
    MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'I21%' OR d.icd_code LIKE 'I22%' OR d.icd_code LIKE 'I25%'))
                 OR (d.icd_version=9  AND d.icd_code LIKE '414%')
             THEN 1 ELSE 0 END) AS cad,
    -- Charlson score (13-component comprehensive)
    MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'I21%' OR d.icd_code LIKE 'I22%'))
                 OR (d.icd_version=9  AND d.icd_code LIKE '410%') THEN 1 ELSE 0 END)
    + MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'I50%')
                   OR (d.icd_version=9  AND d.icd_code LIKE '428%') THEN 1 ELSE 0 END)
    + MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'I70%' OR d.icd_code LIKE 'I73%'))
                   OR (d.icd_version=9  AND (d.icd_code LIKE '440%' OR d.icd_code LIKE '443%')) THEN 1 ELSE 0 END)
    + MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'J44%')
                   OR (d.icd_version=9  AND (d.icd_code LIKE '496%' OR d.icd_code LIKE '491%')) THEN 1 ELSE 0 END)
    + MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'F00%' OR d.icd_code LIKE 'F01%'
                                        OR d.icd_code LIKE 'F02%' OR d.icd_code LIKE 'F03%'
                                        OR d.icd_code LIKE 'G30%'))
                   OR (d.icd_version=9  AND d.icd_code LIKE '290%') THEN 1 ELSE 0 END)
    + MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'K70%' OR d.icd_code LIKE 'K73%'))
                   OR (d.icd_version=9  AND d.icd_code LIKE '571%') THEN 1 ELSE 0 END)
    + MAX(CASE WHEN (d.icd_version=10 AND d.icd_code IN ('E100','E109','E110','E119','E130','E139'))
                   OR (d.icd_version=9  AND d.icd_code IN ('2500','2501','2502','2503')) THEN 1 ELSE 0 END)
    + 2 * MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'G81%' OR d.icd_code LIKE 'G82%'))
                      OR (d.icd_version=9  AND (d.icd_code LIKE '342%' OR d.icd_code LIKE '344%')) THEN 1 ELSE 0 END)
    + 2 * MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'E112%' OR d.icd_code LIKE 'E113%'
                                            OR d.icd_code LIKE 'E114%' OR d.icd_code LIKE 'E115%'
                                            OR d.icd_code LIKE 'E116%'))
                      OR (d.icd_version=9  AND d.icd_code LIKE '250%'
                          AND d.icd_code NOT IN ('2500','2501','2502','2503')) THEN 1 ELSE 0 END)
    + 2 * MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'N18%'
                         AND d.icd_code NOT IN ('N180','N181','N182'))
                      OR (d.icd_version=9  AND (d.icd_code LIKE '5853%' OR d.icd_code LIKE '5854%'
                                            OR d.icd_code LIKE '5855%')) THEN 1 ELSE 0 END)
    + 2 * MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'C%'
                         AND d.icd_code NOT LIKE 'C44%' AND d.icd_code NOT LIKE 'C77%'
                         AND d.icd_code NOT LIKE 'C78%' AND d.icd_code NOT LIKE 'C79%'
                         AND d.icd_code NOT LIKE 'C80%')
                      OR (d.icd_version=9
                          AND TRY_CAST(SUBSTRING(d.icd_code,1,3) AS INTEGER) BETWEEN 140 AND 195)
                   THEN 1 ELSE 0 END)
    + 3 * MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'K704%' OR d.icd_code LIKE 'K720%'
                                            OR d.icd_code LIKE 'K729%'))
                      OR (d.icd_version=9  AND (d.icd_code LIKE '5722%' OR d.icd_code LIKE '5723%'))
                   THEN 1 ELSE 0 END)
    + 6 * MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'C77%' OR d.icd_code LIKE 'C78%'
                                            OR d.icd_code LIKE 'C79%' OR d.icd_code LIKE 'C80%'))
                      OR (d.icd_version=9
                          AND TRY_CAST(SUBSTRING(d.icd_code,1,3) AS INTEGER) BETWEEN 196 AND 199)
                   THEN 1 ELSE 0 END)
    AS charlson_score
  FROM diagnoses_icd d
  INNER JOIN cohort c ON d.subject_id=c.subject_id AND d.hadm_id=c.hadm_id
  GROUP BY d.subject_id
")

# ── 3d. Treatment flags (presence within first 48h of ICU) ──

# reperfusion_therapy: tPA or mechanical thrombectomy (from procedures_icd) [v1]
reperfusion_df <- dbGetQuery(con, "
  SELECT DISTINCT p.subject_id, 1 AS reperfusion_therapy
  FROM procedures_icd p
  INNER JOIN cohort c ON p.subject_id=c.subject_id AND p.hadm_id=c.hadm_id
  WHERE (p.icd_version=9  AND p.icd_code = '9910')
     OR (p.icd_version=10 AND p.icd_code LIKE '3E0%GC')
     OR (p.icd_version=10 AND p.icd_code LIKE '03C%3ZZ')
")

# alteplase: tPA specifically (inputevents + procedures_icd) [v2]
alteplase_df <- dbGetQuery(con, "
  SELECT DISTINCT subject_id, 1 AS alteplase FROM (
    SELECT ie.subject_id
    FROM inputevents ie
    INNER JOIN cohort c ON ie.subject_id=c.subject_id AND ie.stay_id=c.stay_id
    WHERE ie.itemid = 221243 AND ie.amount > 0
      AND ie.starttime >= c.icu_intime
      AND ie.starttime <= c.icu_intime + INTERVAL 48 HOURS
    UNION ALL
    SELECT p.subject_id
    FROM procedures_icd p
    INNER JOIN cohort c ON p.subject_id=c.subject_id AND p.hadm_id=c.hadm_id
    WHERE (p.icd_version=9  AND p.icd_code = '9910')
       OR (p.icd_version=10 AND p.icd_code LIKE '3E0%GC')
  ) t
")

# Mannitol [v2]: itemid 220970
mannitol_df <- dbGetQuery(con, "
  SELECT DISTINCT ie.subject_id, 1 AS mannitol
  FROM inputevents ie
  INNER JOIN cohort c ON ie.subject_id=c.subject_id AND ie.stay_id=c.stay_id
  WHERE ie.itemid = 220970 AND ie.amount > 0
    AND ie.starttime >= c.icu_intime
    AND ie.starttime <= c.icu_intime + INTERVAL 48 HOURS
")

# NaCl 3% / hypertonic saline [v2]: itemids 220954, 225158
nacl3_df <- dbGetQuery(con, "
  SELECT DISTINCT ie.subject_id, 1 AS nacl3_hypertonic
  FROM inputevents ie
  INNER JOIN cohort c ON ie.subject_id=c.subject_id AND ie.stay_id=c.stay_id
  WHERE ie.itemid IN (220954, 225158) AND ie.amount > 0
    AND ie.starttime >= c.icu_intime
    AND ie.starttime <= c.icu_intime + INTERVAL 48 HOURS
")

# Vasopressors [v1]: NE=221906, Epi=221289, Dopamine=221662, PE=221749, VP=222315
vaso_df <- dbGetQuery(con, "
  SELECT DISTINCT ie.subject_id, 1 AS vasopressor_baseline
  FROM inputevents ie
  INNER JOIN cohort c ON ie.subject_id=c.subject_id AND ie.stay_id=c.stay_id
  WHERE ie.itemid IN (221906, 221289, 221662, 221749, 222315)
    AND ie.amount > 0
    AND ie.starttime >= c.icu_intime
    AND ie.starttime <= c.icu_intime + INTERVAL 48 HOURS
")

# Mechanical ventilation [v1]: ventilator mode charted (itemid 223849)
vent_df <- dbGetQuery(con, "
  SELECT DISTINCT ce.subject_id, 1 AS mechanical_vent_baseline
  FROM chartevents ce
  INNER JOIN cohort c ON ce.subject_id=c.subject_id AND ce.stay_id=c.stay_id
  WHERE ce.itemid = 223849
    AND ce.value IS NOT NULL AND ce.value NOT IN ('None', '')
    AND ce.charttime >= c.icu_intime
    AND ce.charttime <= c.icu_intime + INTERVAL 48 HOURS
")

# Code status / DNR-DNI [v1]: first charted value (itemid 223758)
code_df <- dbGetQuery(con, "
  SELECT ce.subject_id, ce.value AS code_status_value
  FROM chartevents ce
  INNER JOIN cohort c ON ce.subject_id=c.subject_id AND ce.stay_id=c.stay_id
  WHERE ce.itemid = 223758
    AND ce.charttime >= c.icu_intime
    AND ce.charttime <= c.icu_intime + INTERVAL 48 HOURS
  QUALIFY ROW_NUMBER() OVER (PARTITION BY ce.subject_id ORDER BY ce.charttime) = 1
") |>
  mutate(code_status_dnr = if_else(
    str_detect(code_status_value, regex("DNR|DNI|Comfort", ignore_case = TRUE)), 1L, 0L))

# Merge all static into one wide row per patient
static_wide <- static_demo |>
  left_join(static_body,    by = c("subject_id", "stay_id")) |>
  left_join(comorbid_df,    by = "subject_id") |>
  left_join(reperfusion_df, by = "subject_id") |>
  left_join(alteplase_df,   by = "subject_id") |>
  left_join(mannitol_df,    by = "subject_id") |>
  left_join(nacl3_df,       by = "subject_id") |>
  left_join(vaso_df,        by = "subject_id") |>
  left_join(vent_df,        by = "subject_id") |>
  left_join(code_df |> select(subject_id, code_status_dnr), by = "subject_id") |>
  left_join(cohort_df |> select(subject_id, stroke_type, stroke_binary), by = "subject_id") |>
  mutate(
    reperfusion_therapy      = replace_na(reperfusion_therapy,      0L),
    alteplase                = replace_na(alteplase,                0L),
    mannitol                 = replace_na(mannitol,                 0L),
    nacl3_hypertonic         = replace_na(nacl3_hypertonic,         0L),
    vasopressor_baseline     = replace_na(vasopressor_baseline,     0L),
    mechanical_vent_baseline = replace_na(mechanical_vent_baseline, 0L),
    code_status_dnr          = replace_na(code_status_dnr,          0L)
  )

# ─────────────────────────────────────────────────────────────
# 4. LONGITUDINAL MEASUREMENTS (first 48h of ICU stay)
# ─────────────────────────────────────────────────────────────

# ── 4a. Labs ──
# v1: BUN=51006, Lactate=50813, INR=51237, Creatinine=50912,
#     PaO2=50821, FiO2=50816 (for P/F ratio), Sodium=50983
# v2: Glucose=50931, Hemoglobin=51222, WBC=51301, Platelet=51265, PTT=51275
# Both: WBC=51301, Platelet=51265 (same itemid — de-duplicated automatically)
lab_item_map <- c(
  "51006" = "bun",
  "50813" = "lactate",
  "51237" = "inr",
  "50912" = "creatinine",
  "50983" = "sodium",
  "50821" = "pao2",        # intermediate: used to compute pf_ratio
  "50816" = "fio2_lab",    # intermediate: used to compute pf_ratio
  "50931" = "glucose_lab",
  "51222" = "hemoglobin",
  "51301" = "wbc",
  "51265" = "platelet",
  "51275" = "ptt"
)

labs_raw <- dbGetQuery(con, sprintf("
  SELECT l.subject_id, l.hadm_id, l.itemid, l.charttime, l.valuenum
  FROM labevents l
  INNER JOIN cohort c ON l.subject_id=c.subject_id AND l.hadm_id=c.hadm_id
  WHERE l.itemid IN (%s)
    AND l.valuenum IS NOT NULL AND l.valuenum > 0
    AND l.charttime >= c.icu_intime
    AND l.charttime <= c.icu_intime + INTERVAL 48 HOURS
", paste(names(lab_item_map), collapse = ",")))

labs_timed <- labs_raw |>
  left_join(cohort_df |> select(subject_id, hadm_id, stay_id, icu_intime),
            by = c("subject_id", "hadm_id")) |>
  mutate(
    variable               = lab_item_map[as.character(itemid)],
    hours_since_icu_intime = as.numeric(difftime(charttime, icu_intime, units = "hours"))
  ) |>
  filter(hours_since_icu_intime >= 0, hours_since_icu_intime <= 48)

# P/F ratio [v1]: match each PaO2 to the nearest FiO2 within ±1 hour
# FiO2 in labevents (50816) is stored as a percentage (21–100)
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
    fio2_frac = if_else(fio2 > 1, fio2 / 100, fio2),
    value     = as.character(round(pao2 / fio2_frac, 1)),
    variable  = "pf_ratio"
  ) |>
  filter(as.numeric(value) > 0, as.numeric(value) < 700) |>
  select(subject_id, stay_id, hours_since_icu_intime, variable, value)

labs_long <- labs_timed |>
  filter(!variable %in% c("pao2", "fio2_lab")) |>
  transmute(subject_id, stay_id, hours_since_icu_intime, variable,
            value = as.character(valuenum))

# ── 4b. Chartevents ──
#
# v1: GCS (3 components), MAP, Temperature
# v2: SBP, DBP, Heart rate, SpO2, Glucose (bedside), ICP, CPP, RASS,
#     Tidal volume, Respiratory rate, Limb strength (x4)
#
# NOTE: Limb-strength itemids 224664–224667 are best estimates.
#   Verify with: dbGetQuery(con, "SELECT itemid, label, param_type FROM d_items
#                WHERE lower(label) LIKE '%strength%'")

chart_item_map <- c(
  # GCS components [v1]
  "220739" = "gcs_eye",    "223900" = "gcs_verbal",   "223901" = "gcs_motor",
  # MAP [v1]
  "220052" = "map_art",    "220181" = "map_ni",
  # SBP / DBP [v2]
  "220050" = "sbp_art",    "220179" = "sbp_ni",
  "220051" = "dbp_art",    "220180" = "dbp_ni",
  # Temperature [v1]
  "223762" = "temp_c",     "223761" = "temp_f",
  # Other vitals [v2]
  "220045" = "heart_rate",
  "220277" = "spo2",
  "220621" = "glucose_chart",
  # Neurological [v2]
  "220765" = "icp",
  "220058" = "cpp",
  "228096" = "rass",
  # Limb strength [v2] — verify itemids before use
  "224664" = "strength_left_arm",
  "224665" = "strength_left_leg",
  "224666" = "strength_right_arm",
  "224667" = "strength_right_leg",
  # Respiratory [v2]
  "220210" = "respiratory_rate",
  "224685" = "tidal_volume_obs"
)

vitals_raw <- dbGetQuery(con, sprintf("
  SELECT ce.subject_id, ce.stay_id, ce.itemid, ce.charttime, ce.valuenum, ce.value
  FROM chartevents ce
  INNER JOIN cohort c ON ce.subject_id=c.subject_id AND ce.stay_id=c.stay_id
  WHERE ce.itemid IN (%s)
    AND (ce.warning IS NULL OR ce.warning = 0)
    AND ce.charttime >= c.icu_intime
    AND ce.charttime <= c.icu_intime + INTERVAL 48 HOURS
", paste(names(chart_item_map), collapse = ",")))

vitals_timed <- vitals_raw |>
  left_join(cohort_df |> select(subject_id, stay_id, icu_intime),
            by = c("subject_id", "stay_id")) |>
  mutate(
    variable               = chart_item_map[as.character(itemid)],
    hours_since_icu_intime = as.numeric(difftime(charttime, icu_intime, units = "hours"))
  ) |>
  filter(hours_since_icu_intime >= 0, hours_since_icu_intime <= 48)

# GCS: sum 3 components at the same charttime [v1]
gcs_long <- vitals_timed |>
  filter(variable %in% c("gcs_eye", "gcs_verbal", "gcs_motor"), !is.na(valuenum)) |>
  group_by(subject_id, stay_id, charttime, hours_since_icu_intime) |>
  summarise(n_comp = n_distinct(variable), gcs = sum(valuenum), .groups = "drop") |>
  filter(n_comp == 3, gcs >= 3, gcs <= 15) |>
  transmute(subject_id, stay_id, hours_since_icu_intime,
            variable = "gcs", value = as.character(gcs))

# MAP: prefer arterial; fall back to non-invasive [v1]
map_long <- vitals_timed |>
  filter(variable %in% c("map_art", "map_ni"), !is.na(valuenum),
         valuenum > 20, valuenum < 200) |>
  mutate(src = if_else(variable == "map_art", 1L, 2L)) |>
  group_by(subject_id, stay_id, charttime, hours_since_icu_intime) |>
  slice_min(order_by = src, n = 1, with_ties = FALSE) |>
  ungroup() |>
  transmute(subject_id, stay_id, hours_since_icu_intime,
            variable = "map", value = as.character(valuenum))

# SBP: prefer arterial [v2]
sbp_long <- vitals_timed |>
  filter(variable %in% c("sbp_art", "sbp_ni"), !is.na(valuenum),
         valuenum > 40, valuenum < 300) |>
  mutate(src = if_else(variable == "sbp_art", 1L, 2L)) |>
  group_by(subject_id, stay_id, charttime, hours_since_icu_intime) |>
  slice_min(order_by = src, n = 1, with_ties = FALSE) |>
  ungroup() |>
  transmute(subject_id, stay_id, hours_since_icu_intime,
            variable = "sbp", value = as.character(valuenum))

# DBP: prefer arterial [v2]
dbp_long <- vitals_timed |>
  filter(variable %in% c("dbp_art", "dbp_ni"), !is.na(valuenum),
         valuenum > 10, valuenum < 200) |>
  mutate(src = if_else(variable == "dbp_art", 1L, 2L)) |>
  group_by(subject_id, stay_id, charttime, hours_since_icu_intime) |>
  slice_min(order_by = src, n = 1, with_ties = FALSE) |>
  ungroup() |>
  transmute(subject_id, stay_id, hours_since_icu_intime,
            variable = "dbp", value = as.character(valuenum))

# Temperature: unify °C and °F; prefer direct Celsius [v1]
temp_long <- vitals_timed |>
  filter(variable %in% c("temp_c", "temp_f"), !is.na(valuenum)) |>
  mutate(temp_celsius = if_else(variable == "temp_c", valuenum, (valuenum - 32) * 5 / 9)) |>
  filter(temp_celsius > 25, temp_celsius < 45) |>
  group_by(subject_id, stay_id, charttime, hours_since_icu_intime) |>
  slice_min(order_by = if_else(variable == "temp_c", 1L, 2L), n = 1, with_ties = FALSE) |>
  ungroup() |>
  transmute(subject_id, stay_id, hours_since_icu_intime,
            variable = "temperature_c", value = as.character(round(temp_celsius, 2)))

# Limb strength: valuenum if numeric; fall back to text value [v2]
strength_long <- vitals_timed |>
  filter(str_starts(variable, "strength_")) |>
  mutate(val = coalesce(as.character(valuenum), value)) |>
  filter(!is.na(val)) |>
  transmute(subject_id, stay_id, hours_since_icu_intime, variable, value = val)

# All remaining numeric chartevents variables (direct valuenum)
other_chart_vars <- c("heart_rate", "spo2", "glucose_chart", "icp", "cpp",
                      "rass", "respiratory_rate", "tidal_volume_obs")

other_long <- vitals_timed |>
  filter(variable %in% other_chart_vars, !is.na(valuenum)) |>
  transmute(subject_id, stay_id, hours_since_icu_intime, variable,
            value = as.character(valuenum))

# ─────────────────────────────────────────────────────────────
# 5. ASSEMBLE WIDE-FORMAT DATAFRAME
#    Longitudinal measurements define the rows.
#    Static covariates are joined on and repeat at every time point.
# ─────────────────────────────────────────────────────────────

longitudinal_wide <- bind_rows(
  labs_long, pf_long,
  gcs_long, map_long, sbp_long, dbp_long, temp_long,
  strength_long, other_long
) |>
  group_by(subject_id, stay_id, hours_since_icu_intime, variable) |>
  slice(1) |>   # drop any exact duplicate (subject, stay, time, variable) entries
  ungroup() |>
  pivot_wider(names_from = variable, values_from = value) |>
  type.convert(as.is = TRUE)

dataset_wide <- longitudinal_wide |>
  left_join(static_wide, by = c("subject_id", "stay_id")) |>
  select(subject_id, stay_id, stroke_type, stroke_binary,
         hours_since_icu_intime, everything()) |>
  arrange(subject_id, hours_since_icu_intime)

cat("\nRows:", nrow(dataset_wide), "| Columns:", ncol(dataset_wide), "\n")
cat("Column names:\n")
print(names(dataset_wide))

# ─────────────────────────────────────────────────────────────
# 6. WRITE OUTPUT
# ─────────────────────────────────────────────────────────────
dir.create("output", showWarnings = FALSE)
data.table::fwrite(dataset_wide, "output/stroke_longitudinal_v3.csv")
cat("\nSaved -> output/stroke_longitudinal_v3.csv\n")

dbDisconnect(con, shutdown = TRUE)
