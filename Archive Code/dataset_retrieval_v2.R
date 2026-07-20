# ============================================================
# STROKE DATASET V2
# Stroke patients (age >= 18) first admitted to the ICU.
# Time window: first 48 hours of ICU admission.
#
# Static covariates (repeated at every row):
#   age, sex, race, hypertension, afib, charlson_score,
#   alteplase, mannitol, nacl3_hypertonic
#
# Longitudinal columns (one row per measurement time):
#   gcs, map, sbp, dbp, temperature_c, heart_rate, spo2,
#   glucose_lab, glucose_chart*, hemoglobin, wbc, platelet, ptt,
#   icp, cpp, rass, tidal_volume_obs, respiratory_rate,
#   strength_left_arm, strength_left_leg,
#   strength_right_arm, strength_right_leg
#
# * glucose_lab (chemistry panel) and glucose_chart (bedside glucometer)
#   are two sources of the same measurement kept as separate columns.
#   Combine downstream with: coalesce(glucose_lab, glucose_chart)
#
# IMPORTANT — Verify uncertain itemids before using in analysis:
#   Run: dbGetQuery(con, "SELECT itemid, label, param_type FROM d_items
#                  WHERE lower(label) LIKE '%strength%'")
#   Adjust the strength_* itemids in chart_item_map if needed.
#   Strength values are stored as text (e.g. "5/5") in some MIMIC versions;
#   the code falls back to the text value if valuenum is NULL.
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
reg("procedures_icd", file.path(hosp, "procedures_icd.csv.gz"))
reg("labevents",      file.path(hosp, "labevents.csv.gz"))
reg("chartevents",    file.path(icu,  "chartevents.csv.gz"))
reg("d_items",        file.path(icu,  "d_items.csv.gz"))
reg("inputevents",    file.path(icu,  "inputevents.csv.gz"))

# ─────────────────────────────────────────────────────────────
# 2. STROKE COHORT
#    First hospital admission must be an ICU admission; age >= 18;
#    stroke diagnosis on that admission.
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
    icu_outtime = outtime
  ) |>
  group_by(subject_id) |>
  slice_min(order_by = icu_intime, n = 1, with_ties = FALSE) |>
  ungroup() |>
  compute(name = "cohort", temporary = FALSE)

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
  INNER JOIN cohort c ON d.subject_id = c.subject_id AND d.hadm_id = c.hadm_id
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
  select(subject_id, hadm_id, stroke_binary)

cohort_df <- collect(tbl(con, "cohort")) |>
  inner_join(stroke_types, by = c("subject_id", "hadm_id"))

cat("Stroke patients (first ICU admission, age >= 18):", nrow(cohort_df), "\n")

if (nrow(cohort_df) == 0) {
  dbDisconnect(con, shutdown = TRUE)
  stop("No stroke patients found — no output written.")
}

all_patients <- cohort_df |> select(subject_id, stay_id)

# ─────────────────────────────────────────────────────────────
# 3. STATIC COVARIATES (one wide row per patient)
# ─────────────────────────────────────────────────────────────

# ── 3a. Demographics ──
static_demo <- cohort_df |>
  mutate(sex = if_else(gender == "M", 1L, 0L)) |>
  select(subject_id, stay_id, age, sex, race)

# ── 3b. Hypertension, AFib, and Charlson Comorbidity Index ──
#
# Charlson components included (cerebrovascular disease excluded — constant for this cohort):
#   MI(1), CHF(1), PVD(1), COPD(1), Dementia(1), Mild liver disease(1),
#   Diabetes uncomplicated(1), Hemiplegia(2), Diabetes complicated(2),
#   CKD moderate-severe(2), Any non-skin malignancy(2),
#   Moderate-severe liver disease(3), Metastatic tumor(6)
comorbid_df <- dbGetQuery(con, "
  SELECT d.subject_id,
    MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'I10%')
                 OR (d.icd_version=9 AND d.icd_code LIKE '401%')
             THEN 1 ELSE 0 END) AS hypertension,
    MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'I48%')
                 OR (d.icd_version=9 AND d.icd_code='42731')
             THEN 1 ELSE 0 END) AS afib,
    -- Charlson score
    MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'I21%' OR d.icd_code LIKE 'I22%'))
                 OR (d.icd_version=9 AND d.icd_code LIKE '410%') THEN 1 ELSE 0 END)
    + MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'I50%')
                   OR (d.icd_version=9 AND d.icd_code LIKE '428%') THEN 1 ELSE 0 END)
    + MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'I70%' OR d.icd_code LIKE 'I73%'))
                   OR (d.icd_version=9 AND (d.icd_code LIKE '440%' OR d.icd_code LIKE '443%')) THEN 1 ELSE 0 END)
    + MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'J44%')
                   OR (d.icd_version=9 AND (d.icd_code LIKE '496%' OR d.icd_code LIKE '491%')) THEN 1 ELSE 0 END)
    + MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'F00%' OR d.icd_code LIKE 'F01%'
                                        OR d.icd_code LIKE 'F02%' OR d.icd_code LIKE 'F03%'
                                        OR d.icd_code LIKE 'G30%'))
                   OR (d.icd_version=9 AND d.icd_code LIKE '290%') THEN 1 ELSE 0 END)
    + MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'K70%' OR d.icd_code LIKE 'K73%'))
                   OR (d.icd_version=9 AND d.icd_code LIKE '571%') THEN 1 ELSE 0 END)
    + MAX(CASE WHEN (d.icd_version=10 AND d.icd_code IN ('E100','E109','E110','E119','E130','E139'))
                   OR (d.icd_version=9 AND d.icd_code IN ('2500','2501','2502','2503')) THEN 1 ELSE 0 END)
    + 2 * MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'G81%' OR d.icd_code LIKE 'G82%'))
                      OR (d.icd_version=9 AND (d.icd_code LIKE '342%' OR d.icd_code LIKE '344%')) THEN 1 ELSE 0 END)
    + 2 * MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'E112%' OR d.icd_code LIKE 'E113%'
                                            OR d.icd_code LIKE 'E114%' OR d.icd_code LIKE 'E115%'
                                            OR d.icd_code LIKE 'E116%'))
                      OR (d.icd_version=9 AND d.icd_code LIKE '250%'
                          AND d.icd_code NOT IN ('2500','2501','2502','2503')) THEN 1 ELSE 0 END)
    + 2 * MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'N18%'
                         AND d.icd_code NOT IN ('N180','N181','N182'))
                      OR (d.icd_version=9 AND (d.icd_code LIKE '5853%' OR d.icd_code LIKE '5854%'
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
                      OR (d.icd_version=9 AND (d.icd_code LIKE '5722%' OR d.icd_code LIKE '5723%'))
                   THEN 1 ELSE 0 END)
    + 6 * MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'C77%' OR d.icd_code LIKE 'C78%'
                                            OR d.icd_code LIKE 'C79%' OR d.icd_code LIKE 'C80%'))
                      OR (d.icd_version=9
                          AND TRY_CAST(SUBSTRING(d.icd_code,1,3) AS INTEGER) BETWEEN 196 AND 199)
                   THEN 1 ELSE 0 END)
    AS charlson_score
  FROM diagnoses_icd d
  INNER JOIN cohort c ON d.subject_id = c.subject_id AND d.hadm_id = c.hadm_id
  GROUP BY d.subject_id
")

# ── 3c. Medication flags: given at any point during the first 48h of ICU ──
#
# Alteplase (tPA): inputevents itemid 221243 + procedures_icd fallback
#   (tPA is often given before ICU admission; procedures_icd captures those cases)
# Mannitol:        inputevents itemid 220970
# NaCl 3%:         inputevents itemids 220954 (Sodium Chloride 3%) and 225158 (NaCl 3%)
#
# To verify itemids: dbGetQuery(con, "SELECT itemid, label FROM d_items
#   WHERE lower(label) LIKE '%alteplase%' OR lower(label) LIKE '%mannitol%'
#      OR lower(label) LIKE '%sodium chloride 3%' OR lower(label) LIKE '%nacl 3%'")

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
    WHERE (p.icd_version=9 AND p.icd_code='9910')
       OR (p.icd_version=10 AND p.icd_code LIKE '3E0%GC')
  ) t
")

mannitol_df <- dbGetQuery(con, "
  SELECT DISTINCT ie.subject_id, 1 AS mannitol
  FROM inputevents ie
  INNER JOIN cohort c ON ie.subject_id=c.subject_id AND ie.stay_id=c.stay_id
  WHERE ie.itemid = 220970 AND ie.amount > 0
    AND ie.starttime >= c.icu_intime
    AND ie.starttime <= c.icu_intime + INTERVAL 48 HOURS
")

nacl3_df <- dbGetQuery(con, "
  SELECT DISTINCT ie.subject_id, 1 AS nacl3_hypertonic
  FROM inputevents ie
  INNER JOIN cohort c ON ie.subject_id=c.subject_id AND ie.stay_id=c.stay_id
  WHERE ie.itemid IN (220954, 225158) AND ie.amount > 0
    AND ie.starttime >= c.icu_intime
    AND ie.starttime <= c.icu_intime + INTERVAL 48 HOURS
")

# Combine all static into one wide row per patient (default 0 for absent medications)
static_wide <- static_demo |>
  left_join(comorbid_df,  by = "subject_id") |>
  left_join(alteplase_df, by = "subject_id") |>
  left_join(mannitol_df,  by = "subject_id") |>
  left_join(nacl3_df,     by = "subject_id") |>
  left_join(cohort_df |> select(subject_id, stroke_binary), by = "subject_id") |>
  mutate(
    alteplase        = replace_na(alteplase,        0L),
    mannitol         = replace_na(mannitol,         0L),
    nacl3_hypertonic = replace_na(nacl3_hypertonic, 0L)
  )

# ─────────────────────────────────────────────────────────────
# 4. LONGITUDINAL MEASUREMENTS (first 48h of ICU stay)
# ─────────────────────────────────────────────────────────────

# ── 4a. Labs ──
# 50931 = Glucose (chemistry panel)   51222 = Hemoglobin
# 51301 = WBC                          51265 = Platelet Count
# 51275 = PTT
lab_item_map <- c(
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

labs_long <- labs_raw |>
  left_join(cohort_df |> select(subject_id, hadm_id, stay_id, icu_intime),
            by = c("subject_id", "hadm_id")) |>
  mutate(
    variable               = lab_item_map[as.character(itemid)],
    hours_since_icu_intime = as.numeric(difftime(charttime, icu_intime, units = "hours"))
  ) |>
  filter(hours_since_icu_intime >= 0, hours_since_icu_intime <= 48) |>
  transmute(subject_id, stay_id, hours_since_icu_intime, variable, value = as.character(valuenum))

# ── 4b. Chartevents ──
#
# GCS:        eye=220739  verbal=223900  motor=223901  (summed to single gcs column)
# MAP:        arterial=220052  non-invasive=220181     (arterial preferred)
# SBP:        arterial=220050  non-invasive=220179
# DBP:        arterial=220051  non-invasive=220180
# Heart rate: 220045
# Temp °C:    223762   Temp °F: 223761               (converted to °C, °C preferred)
# SpO2:       220277
# Glucose (bedside glucometer): 220621
# ICP:        220765
# CPP (cerebral perfusion pressure): 220058
# RASS:       228096
# Tidal volume (observed): 224685
# Respiratory rate: 220210
# Limb strength — VERIFY these itemids against d_items before relying on results:
#   Left arm: 224664  Left leg: 224665  Right arm: 224666  Right leg: 224667

chart_item_map <- c(
  "220739" = "gcs_eye",   "223900" = "gcs_verbal",  "223901" = "gcs_motor",
  "220052" = "map_art",   "220181" = "map_ni",
  "220050" = "sbp_art",   "220179" = "sbp_ni",
  "220051" = "dbp_art",   "220180" = "dbp_ni",
  "220045" = "heart_rate",
  "223762" = "temp_c",    "223761" = "temp_f",
  "220277" = "spo2",
  "220621" = "glucose_chart",
  "220765" = "icp",
  "220058" = "cpp",
  "228096" = "rass",
  "220210" = "respiratory_rate",
  "224685" = "tidal_volume_obs",
  "224664" = "strength_left_arm",
  "224665" = "strength_left_leg",
  "224666" = "strength_right_arm",
  "224667" = "strength_right_leg"
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
  left_join(cohort_df |> select(subject_id, stay_id, icu_intime), by = c("subject_id", "stay_id")) |>
  mutate(
    variable               = chart_item_map[as.character(itemid)],
    hours_since_icu_intime = as.numeric(difftime(charttime, icu_intime, units = "hours"))
  ) |>
  filter(hours_since_icu_intime >= 0, hours_since_icu_intime <= 48)

# GCS: sum all three components charted at the same charttime
gcs_long <- vitals_timed |>
  filter(variable %in% c("gcs_eye", "gcs_verbal", "gcs_motor"), !is.na(valuenum)) |>
  group_by(subject_id, stay_id, charttime, hours_since_icu_intime) |>
  summarise(n_comp = n_distinct(variable), gcs = sum(valuenum), .groups = "drop") |>
  filter(n_comp == 3, gcs >= 3, gcs <= 15) |>
  transmute(subject_id, stay_id, hours_since_icu_intime, variable = "gcs", value = as.character(gcs))

# MAP, SBP, DBP: arterial preferred; fallback to non-invasive
bp_source_rank <- function(var_prefix) {
  function(df) {
    df |>
      filter(str_starts(variable, var_prefix)) |>
      filter(!is.na(valuenum)) |>
      mutate(src = if_else(str_ends(variable, "_art"), 1L, 2L)) |>
      group_by(subject_id, stay_id, charttime, hours_since_icu_intime) |>
      slice_min(order_by = src, n = 1, with_ties = FALSE) |>
      ungroup() |>
      transmute(subject_id, stay_id, hours_since_icu_intime,
                variable = var_prefix, value = as.character(valuenum))
  }
}

map_long <- bp_source_rank("map")(vitals_timed |> filter(valuenum > 20, valuenum < 200))
sbp_long <- bp_source_rank("sbp")(vitals_timed |> filter(valuenum > 40, valuenum < 300))
dbp_long <- bp_source_rank("dbp")(vitals_timed |> filter(valuenum > 10, valuenum < 200))

# Temperature: convert °F -> °C; prefer direct Celsius measurement
temp_long <- vitals_timed |>
  filter(variable %in% c("temp_c", "temp_f"), !is.na(valuenum)) |>
  mutate(temp_celsius = if_else(variable == "temp_c", valuenum, (valuenum - 32) * 5 / 9)) |>
  filter(temp_celsius > 25, temp_celsius < 45) |>
  group_by(subject_id, stay_id, charttime, hours_since_icu_intime) |>
  slice_min(order_by = if_else(variable == "temp_c", 1L, 2L), n = 1, with_ties = FALSE) |>
  ungroup() |>
  transmute(subject_id, stay_id, hours_since_icu_intime,
            variable = "temperature_c", value = as.character(round(temp_celsius, 2)))

# Limb strength: use valuenum when available; fall back to text value
# (some MIMIC versions store strength as text, e.g. "5/5" or "Paralyzed")
strength_long <- vitals_timed |>
  filter(str_starts(variable, "strength_")) |>
  mutate(val = coalesce(as.character(valuenum), value)) |>
  filter(!is.na(val)) |>
  transmute(subject_id, stay_id, hours_since_icu_intime, variable, value = val)

# All remaining numeric chartevents variables
other_vars <- c("heart_rate", "spo2", "glucose_chart", "icp", "cpp",
                "rass", "respiratory_rate", "tidal_volume_obs")

other_long <- vitals_timed |>
  filter(variable %in% other_vars, !is.na(valuenum)) |>
  transmute(subject_id, stay_id, hours_since_icu_intime, variable,
            value = as.character(valuenum))

# ─────────────────────────────────────────────────────────────
# 5. ASSEMBLE WIDE-FORMAT DATAFRAME
#    Longitudinal data defines the rows; static values are joined
#    on and repeat at every time point.
# ─────────────────────────────────────────────────────────────

longitudinal_wide <- bind_rows(
  labs_long, gcs_long, map_long, sbp_long, dbp_long,
  temp_long, strength_long, other_long
) |>
  group_by(subject_id, stay_id, hours_since_icu_intime, variable) |>
  slice(1) |>   # drop exact duplicate (subject, stay, time, variable) entries
  ungroup() |>
  pivot_wider(names_from = variable, values_from = value) |>
  type.convert(as.is = TRUE)

dataset_wide <- longitudinal_wide |>
  left_join(static_wide, by = c("subject_id", "stay_id")) |>
  select(subject_id, stay_id, stroke_binary, hours_since_icu_intime, everything()) |>
  arrange(subject_id, hours_since_icu_intime)

cat("\nRows:", nrow(dataset_wide), "| Columns:", ncol(dataset_wide), "\n")
cat("Column names:\n")
print(names(dataset_wide))

# ─────────────────────────────────────────────────────────────
# 6. WRITE OUTPUT
# ─────────────────────────────────────────────────────────────
dir.create("output", showWarnings = FALSE)
data.table::fwrite(dataset_wide, "output/stroke_longitudinal_v2.csv")
cat("\nSaved -> output/stroke_longitudinal_v2.csv\n")

dbDisconnect(con, shutdown = TRUE)
