# ============================================================
# MIMIC-IV Stroke ICU Cohort Extraction
# Architecture: DuckDB reading local gzipped CSV files
#
# Follows the cohort-spine pattern from the MIMIC-IV example notebook.
# All heavy filtering/joining runs inside DuckDB (lazy); only the final
# small result tables are pulled into R via collect().
#
# Outputs (written to output/):
#   analysis_long.csv  – one row per measurement per patient
#   analysis_wide.csv  – one row per ICU stay, summary statistics
# ============================================================

# Install once if needed (uncomment):
# install.packages(c("DBI","duckdb","dplyr","dbplyr","data.table","stringr","tidyr","lubridate","purrr"))

library(DBI)
library(duckdb)
library(dplyr)
library(dbplyr)
library(stringr)
library(tidyr)
library(lubridate)
library(purrr)
library(data.table)

# ============================================================
# 0. CONFIGURE  ← change these lines to match your setup
# ============================================================
mimic_dir       <- "C:/electronic health reccords/3.1"
max_hours       <- 168L   # observation window: first 7 days (168 h) after ICU admission
hosp <- file.path(mimic_dir, "hosp")
icu  <- file.path(mimic_dir, "icu")

# ============================================================
# 1. OPEN DuckDB AND REGISTER VIEWS
# ============================================================
con <- dbConnect(duckdb::duckdb())

# reg() creates a lazy DuckDB view over one .csv.gz file.
# Nothing is read into memory here; DuckDB only touches the file
# when a query actually needs it.
reg <- function(name, path) {
  path <- gsub("\\\\", "/", path)   # forward slashes work on all OS
  invisible(dbExecute(con, sprintf(
    "CREATE OR REPLACE VIEW %s AS SELECT * FROM read_csv_auto('%s')",
    name, path)))
}

reg("patients",         file.path(hosp, "patients.csv.gz"))
reg("admissions",       file.path(hosp, "admissions.csv.gz"))
reg("diagnoses_icd",    file.path(hosp, "diagnoses_icd.csv.gz"))
reg("d_icd_diagnoses",  file.path(hosp, "d_icd_diagnoses.csv.gz"))
reg("d_icd_procedures", file.path(hosp, "d_icd_procedures.csv.gz"))
reg("procedures_icd",   file.path(hosp, "procedures_icd.csv.gz"))
reg("prescriptions",    file.path(hosp, "prescriptions.csv.gz"))
reg("d_labitems",       file.path(hosp, "d_labitems.csv.gz"))
reg("labevents",        file.path(hosp, "labevents.csv.gz"))
reg("omr",              file.path(hosp, "omr.csv.gz"))
reg("icustays",         file.path(icu,  "icustays.csv.gz"))
reg("d_items",          file.path(icu,  "d_items.csv.gz"))
reg("chartevents",      file.path(icu,  "chartevents.csv.gz"))
reg("procedureevents",  file.path(icu,  "procedureevents.csv.gz"))

# ============================================================
# 2. STROKE ICD-CODE LOOKUP (printed for your verification)
# ============================================================
# ICD-9 codes covered:
#   430  Subarachnoid hemorrhage
#   431  Intracerebral hemorrhage
#   433x1 / 434x1  Precerebral / cerebral occlusion WITH infarction
#   436  Acute, ill-defined cerebrovascular disease
# ICD-10 codes covered:
#   I60–I62  Haemorrhagic strokes
#   I63      Cerebral infarction (ischaemic)
#   I64      Stroke not specified as haemorrhage or infarction

stroke_icd_lookup <- dbGetQuery(con, "
  SELECT
    icd_code, icd_version, long_title,
    CASE
      WHEN icd_code LIKE 'I60%' THEN 'Subarachnoid Hemorrhage'
      WHEN icd_code LIKE 'I61%' THEN 'Intracerebral Hemorrhage'
      WHEN icd_code LIKE 'I62%' THEN 'Other Intracranial Hemorrhage'
      WHEN icd_code LIKE 'I63%' THEN 'Cerebral Infarction (Ischemic)'
      WHEN icd_code  =   'I64'  THEN 'Stroke, Unspecified'
      WHEN icd_code  =   '430'  THEN 'Subarachnoid Hemorrhage'
      WHEN icd_code  =   '431'  THEN 'Intracerebral Hemorrhage'
      WHEN icd_code LIKE '433%' AND RIGHT(icd_code,1) = '1'
                                THEN 'Precerebral Occlusion with Infarction'
      WHEN icd_code LIKE '434%' AND RIGHT(icd_code,1) = '1'
                                THEN 'Cerebral Arterial Occlusion with Infarction'
      WHEN icd_code  =   '436'  THEN 'Acute Stroke (Ill-defined)'
    END AS stroke_type,
    CASE
      WHEN icd_code LIKE 'I60%' OR icd_code LIKE 'I61%' OR icd_code LIKE 'I62%'
        OR icd_code IN ('430','431')                              THEN 'hemorrhagic'
      WHEN icd_code LIKE 'I63%'
        OR (icd_code LIKE '433%' AND RIGHT(icd_code,1) = '1')
        OR (icd_code LIKE '434%' AND RIGHT(icd_code,1) = '1')
        OR icd_code = '436'                                       THEN 'ischemic'
      ELSE NULL
    END AS ischemic_hemorrhagic
  FROM d_icd_diagnoses
  WHERE
       icd_code LIKE 'I60%' OR icd_code LIKE 'I61%' OR icd_code LIKE 'I62%'
    OR icd_code LIKE 'I63%' OR icd_code = 'I64'
    OR icd_code IN ('430','431','436')
    OR (icd_code LIKE '433%' AND RIGHT(icd_code,1) = '1')
    OR (icd_code LIKE '434%' AND RIGHT(icd_code,1) = '1')
  ORDER BY icd_version, icd_code
")

message("\n=== STROKE ICD CODES CONFIRMED IN d_icd_diagnoses ===")
print(stroke_icd_lookup)

# Register as a DuckDB in-memory table so we can join against it
dbWriteTable(con, "stroke_icd_lookup", stroke_icd_lookup, overwrite = TRUE)

# ============================================================
# 3. COHORT SPINE
#    One row per patient — their first ICU stay that falls
#    within a stroke-coded hospital admission.  Adults (≥18) only.
# ============================================================
message("\nBuilding cohort spine...")

cohort <- tbl(con, "icustays") |>
  # --- join stroke diagnoses (inner join keeps only stroke admissions) ---
  inner_join(tbl(con, "diagnoses_icd"),   by = c("subject_id", "hadm_id")) |>
  inner_join(tbl(con, "stroke_icd_lookup"), by = c("icd_code", "icd_version")) |>
  # Keep only the most-primary stroke diagnosis per admission (lowest seq_num)
  group_by(subject_id, hadm_id, stay_id) |>
  window_order(seq_num) |>
  mutate(dx_rank = row_number()) |>
  filter(dx_rank == 1L) |>
  select(-dx_rank) |>
  ungroup() |>
  # --- attach admission and patient info ---
  inner_join(tbl(con, "admissions"), by = c("subject_id", "hadm_id")) |>
  inner_join(tbl(con, "patients"),   by  = "subject_id") |>
  # --- compute age at admission (MIMIC-IV recommended formula) ---
  mutate(
    admit_year = sql("YEAR(admittime)"),
    age        = anchor_age + (admit_year - anchor_year)
  ) |>
  filter(age >= 18) |>
  transmute(
    subject_id, hadm_id, stay_id,
    # ICD diagnosis info
    icd_code, icd_version, stroke_type, ischemic_hemorrhagic,
    # Demographics
    gender,
    age,
    race          = race,       # column is `race` in MIMIC-IV v2.x (was `ethnicity` in v1.x)
    # Timestamps
    admittime, dischtime,
    icu_intime    = intime,
    icu_outtime   = outtime,
    icu_los_days  = los,
    # ICU details
    first_careunit,
    # Outcomes
    hospital_expire_flag,
    dod
  ) |>
  # --- Keep FIRST stroke ICU stay per patient ---
  group_by(subject_id) |>
  window_order(icu_intime) |>
  mutate(stay_rank = row_number()) |>
  filter(stay_rank == 1L) |>
  select(-stay_rank) |>
  ungroup() |>
  collect()

message(sprintf("  Cohort: %d patients, %d unique ICU stays",
                n_distinct(cohort$subject_id), nrow(cohort)))
message(sprintf("  Stroke subtype breakdown:\n%s",
                paste(capture.output(print(table(cohort$ischemic_hemorrhagic))), collapse = "\n")))

# Register cohort as an in-memory DuckDB table for downstream joins
dbWriteTable(con, "cohort", cohort, overwrite = TRUE)

# ============================================================
# 3a. BASELINE BODY SIZE (from omr — same pattern as example notebook)
# ============================================================
baseline_body <- tbl(con, "omr") |>
  semi_join(tbl(con, "cohort"), by = "subject_id") |>
  collect() |>
  filter(str_detect(result_name, regex("weight|height|bmi", ignore_case = TRUE))) |>
  mutate(
    measure = case_when(
      str_detect(result_name, regex("bmi",    TRUE)) ~ "bmi",
      str_detect(result_name, regex("weight", TRUE)) ~ "weight_lb",
      str_detect(result_name, regex("height", TRUE)) ~ "height_in"
    ),
    value = suppressWarnings(as.numeric(result_value))
  ) |>
  filter(!is.na(value)) |>
  left_join(cohort |> select(subject_id, stay_id, icu_intime), by = "subject_id") |>
  mutate(gap_days = as.numeric(difftime(chartdate, as.Date(icu_intime), units = "days"))) |>
  group_by(subject_id, stay_id, measure) |>
  arrange(gap_days > 0, abs(gap_days), .by_group = TRUE) |>  # prefer pre-admission, closest
  slice(1) |>
  ungroup() |>
  select(subject_id, stay_id, measure, value) |>
  pivot_wider(names_from = measure, values_from = value)

cohort <- cohort |>
  left_join(
    baseline_body |> select(subject_id, stay_id, any_of(c("weight_lb","height_in","bmi"))),
    by = c("subject_id","stay_id"))

# ============================================================
# 3b. BASELINE MEDICATIONS (anticoagulants / antiplatelets)
#     Flag: was this drug class ever prescribed during the admission?
# ============================================================
anticoag_rx <- tbl(con, "prescriptions") |>
  semi_join(tbl(con, "cohort"), by = c("subject_id","hadm_id")) |>
  filter(
    sql("lower(drug) LIKE '%warfarin%'")    |
      sql("lower(drug) LIKE '%heparin%'")     |
      sql("lower(drug) LIKE '%enoxaparin%'")  |
      sql("lower(drug) LIKE '%apixaban%'")    |
      sql("lower(drug) LIKE '%rivaroxaban%'") |
      sql("lower(drug) LIKE '%dabigatran%'")  |
      sql("lower(drug) LIKE '%aspirin%'")     |
      sql("lower(drug) LIKE '%clopidogrel%'") |
      sql("lower(drug) LIKE '%ticagrelor%'")  |
      sql("lower(drug) LIKE '%prasugrel%'")
  ) |>
  mutate(drug_class = case_when(
    sql("lower(drug) LIKE '%warfarin%'")                                       ~ "rx_warfarin",
    sql("lower(drug) LIKE '%heparin%'") | sql("lower(drug) LIKE '%enoxaparin%'") ~ "rx_heparin",
    sql("lower(drug) LIKE '%apixaban%'") | sql("lower(drug) LIKE '%rivaroxaban%'") |
      sql("lower(drug) LIKE '%dabigatran%'")                                   ~ "rx_doac",
    sql("lower(drug) LIKE '%aspirin%'")                                        ~ "rx_aspirin",
    sql("lower(drug) LIKE '%clopidogrel%'") | sql("lower(drug) LIKE '%ticagrelor%'") |
      sql("lower(drug) LIKE '%prasugrel%'")                                    ~ "rx_p2y12",
    TRUE                                                                       ~ "rx_other"
  )) |>
  distinct(subject_id, hadm_id, drug_class) |>
  collect() |>
  pivot_wider(names_from = drug_class, values_from = drug_class,
              values_fn = \(x) 1L, values_fill = 0L) |>
  select(-any_of("rx_other"))

cohort <- cohort |>
  left_join(anticoag_rx, by = c("subject_id","hadm_id")) |>
  mutate(across(starts_with("rx_"), \(x) replace_na(x, 0L)))

# ============================================================
# 3c. REPERFUSION THERAPY (tPA / thrombectomy) from ICD procedures
#     These are major outcome confounders for stroke — do not omit.
# ============================================================
reperfusion <- tbl(con, "procedures_icd") |>
  semi_join(tbl(con, "cohort"), by = c("subject_id","hadm_id")) |>
  inner_join(tbl(con, "d_icd_procedures"), by = c("icd_code","icd_version")) |>
  filter(
    sql("lower(long_title) LIKE '%thrombolytic%'")           |
      sql("lower(long_title) LIKE '%tissue plasminogen%'")     |
      sql("lower(long_title) LIKE '%alteplase%'")              |
      sql("lower(long_title) LIKE '%thrombectomy%'")           |
      sql("lower(long_title) LIKE '%mechanical embolectomy%'") |
      sql("lower(long_title) LIKE '%clot retrieval%'")
  ) |>
  mutate(proc_type = case_when(
    sql("lower(long_title) LIKE '%thrombectomy%'") |
      sql("lower(long_title) LIKE '%mechanical embolectomy%'") |
      sql("lower(long_title) LIKE '%clot retrieval%'")   ~ "proc_thrombectomy",
    TRUE                                                  ~ "proc_tpa"
  )) |>
  distinct(subject_id, hadm_id, proc_type) |>
  collect() |>
  pivot_wider(names_from = proc_type, values_from = proc_type,
              values_fn = \(x) 1L, values_fill = 0L)

cohort <- cohort |>
  left_join(reperfusion, by = c("subject_id","hadm_id")) |>
  mutate(across(any_of(c("proc_tpa","proc_thrombectomy")), \(x) replace_na(x, 0L)))

message(sprintf("  Cohort spine finalised: %d rows × %d columns",
                nrow(cohort), ncol(cohort)))

# ============================================================
# 4. BIOMARKER ITEMID DEFINITIONS
#    For each biomarker: look up itemid(s) in the dictionary tables,
#    confirm labels below, then use itemid %in% c(...) in queries.
# ============================================================

# --- 4a. Labs (source: labevents, join on hadm_id) ---
#
# itemid choices: prefer Blood Chemistry (fluid = 'Blood', category = 'Chemistry')
# for the standard clinical panel. ABG itemids included where relevant.
#
lab_biomarkers <- list(
  glucose         = c(50931L, 50809L),   # Chemistry + ABG glucose
  lactate         = c(50813L, 52442L),   # ABG lactate (52442 absent in demo; present in full dataset)
  creatinine      = 50912L,              # Chemistry
  bun             = 51006L,              # Blood Urea Nitrogen, Chemistry
  sodium          = c(50983L, 50824L),   # Chemistry + Whole Blood
  potassium       = c(50971L, 50822L),   # Chemistry + Whole Blood
  chloride        = c(50902L, 50806L),   # Chemistry + Whole Blood
  bicarbonate     = c(50882L, 50803L),   # Chemistry + Calculated Bicarb (ABG)
  anion_gap       = c(50868L, 52500L),   # Chemistry (52500 absent in demo; present in full dataset)
  wbc             = 51301L,              # Hematology (White Blood Cells)
  hemoglobin      = c(51222L, 50811L),   # Hematology + ABG
  hematocrit      = c(51221L, 50810L),   # Hematology + ABG calculated
  platelet        = 51265L,              # Hematology (Platelet Count)
  inr             = 51237L,              # Hematology INR(PT)
  ptt             = 51275L,              # Hematology PTT
  bilirubin_total = 50885L,              # Chemistry
  alt             = 50861L,              # ALT (Alanine Aminotransferase)
  ast             = 50878L,              # AST (Aspartate Aminotransferase)
  ph_abg          = 50820L,              # Blood Gas pH
  pco2            = 50818L,              # ABG pCO2
  po2             = 50821L               # ABG pO2
)

message("\n=== LAB ITEMIDS — confirm these labels match your expectation ===")
lab_dict <- dbGetQuery(con, sprintf(
  "SELECT itemid, label, fluid, category FROM d_labitems WHERE itemid IN (%s) ORDER BY itemid",
  paste(unlist(lab_biomarkers), collapse = ",")))
print(lab_dict)

# --- 4b. Vitals (source: chartevents, join on stay_id) ---
#
# BP: we capture both arterial line (ABP) and cuff (NIBP); they are merged later.
# GCS: captured as three components; GCS total is computed from them.
# Temperature: both Celsius and Fahrenheit captured; unified to Celsius later.
# FiO2 (223835): only present for mechanically ventilated patients — many NAs expected.
# CPP (227066): only present for patients with invasive ICP monitoring.
#
vital_biomarkers <- list(
  heart_rate      = 220045L,
  sbp_nibp        = 220179L,   # Non-invasive systolic
  dbp_nibp        = 220180L,   # Non-invasive diastolic
  map_nibp        = 220181L,   # Non-invasive mean
  sbp_abp         = 220050L,   # Arterial line systolic
  dbp_abp         = 220051L,   # Arterial line diastolic
  map_abp         = 220052L,   # Arterial line mean
  resp_rate       = 220210L,
  spo2            = 220277L,   # Pulse oximetry
  temp_c          = 223762L,
  temp_f          = 223761L,
  gcs_eye         = 220739L,
  gcs_verbal      = 223900L,
  gcs_motor       = 223901L,
  fio2            = 223835L,   # Inspired O2 Fraction (vented patients only)
  peep            = 220339L,   # PEEP set
  cpp             = 227066L    # Cerebral Perfusion Pressure (ICP monitoring only)
)

message("\n=== VITAL ITEMIDS — confirm these labels match your expectation ===")
vital_dict <- dbGetQuery(con, sprintf(
  "SELECT itemid, label, category FROM d_items WHERE itemid IN (%s) ORDER BY itemid",
  paste(unlist(vital_biomarkers), collapse = ",")))
print(vital_dict)

# ============================================================
# 5. LAB EXTRACTION
#    For each biomarker: filter labevents, inner-join to cohort
#    on (subject_id, hadm_id), restrict to ICU window.
#    All biomarkers unioned into one long table in R.
# ============================================================
message("\nExtracting labs...")

extract_labs <- function(biomarker_list) {
  imap_dfr(biomarker_list, function(itemids, var_name) {
    message("  ", var_name)
    tbl(con, "labevents") |>
      filter(itemid %in% !!itemids) |>
      inner_join(tbl(con, "cohort"),
                 by = c("subject_id","hadm_id")) |>
      filter(!is.na(valuenum),
             valuenum > 0,              # drop non-positive values (implausible)
             charttime >= icu_intime,
             charttime <= icu_outtime,
             sql(sprintf("DATEDIFF('hour', icu_intime, charttime) <= %d", max_hours))) |>
      transmute(
        subject_id, hadm_id, stay_id,
        charttime,
        icu_intime,
        variable_name = !!var_name,
        value         = valuenum
      ) |>
      collect()
  })
}

labs_long <- extract_labs(lab_biomarkers) |>
  mutate(hours_since_icu_intime =
           as.numeric(difftime(charttime, icu_intime, units = "hours"))) |>
  arrange(subject_id, stay_id, variable_name, charttime)

message(sprintf("  Labs: %d rows across %d biomarkers",
                nrow(labs_long), n_distinct(labs_long$variable_name)))

# ============================================================
# 6. VITAL EXTRACTION
#    Source: chartevents, join on stay_id.
#    warning == 1 rows are excluded (clinician-flagged as erroneous).
# ============================================================
message("\nExtracting vitals...")

extract_vitals <- function(biomarker_list) {
  imap_dfr(biomarker_list, function(itemids, var_name) {
    message("  ", var_name)
    tbl(con, "chartevents") |>
      filter(itemid %in% !!itemids,
             !is.na(valuenum),
             warning != 1 | is.na(warning)) |>   # exclude flagged errors
      inner_join(tbl(con, "cohort"), by = c("subject_id", "hadm_id", "stay_id")) |>
      filter(charttime >= icu_intime,
             charttime <= icu_outtime,
             sql(sprintf("DATEDIFF('hour', icu_intime, charttime) <= %d", max_hours))) |>
      transmute(
        subject_id, hadm_id, stay_id,
        charttime,
        icu_intime,
        variable_name = !!var_name,
        value         = valuenum
      ) |>
      collect()
  })
}

vitals_long <- extract_vitals(vital_biomarkers) |>
  mutate(hours_since_icu_intime =
           as.numeric(difftime(charttime, icu_intime, units = "hours"))) |>
  arrange(subject_id, stay_id, variable_name, charttime)

# ============================================================
# 6a. DERIVED VITALS
# ============================================================

# GCS total = eye + verbal + motor, aligned to 30-minute time bins.
# Only emit a total when all three components are present in the same bin.
gcs_total <- vitals_long |>
  filter(variable_name %in% c("gcs_eye","gcs_verbal","gcs_motor")) |>
  mutate(time_bin = floor(hours_since_icu_intime * 2) / 2) |>   # 30-min bins
  group_by(subject_id, hadm_id, stay_id, icu_intime, time_bin) |>
  summarise(
    n_components           = n_distinct(variable_name),
    gcs_total              = sum(value),
    hours_since_icu_intime = mean(hours_since_icu_intime),
    charttime              = min(charttime),
    .groups = "drop"
  ) |>
  filter(n_components == 3L) |>
  transmute(subject_id, hadm_id, stay_id, charttime, icu_intime,
            variable_name = "gcs_total",
            value = gcs_total,
            hours_since_icu_intime)

# Blood pressure: prefer arterial line (ABP) over non-invasive cuff (NIBP).
# Produces unified map / sbp / dbp series.
unify_bp <- function(var_art, var_cuff, out_name) {
  vitals_long |>
    filter(variable_name %in% c(var_art, var_cuff)) |>
    group_by(subject_id, stay_id, charttime) |>
    slice_min(order_by = if_else(variable_name == var_art, 0L, 1L),
              with_ties = FALSE) |>
    ungroup() |>
    mutate(variable_name = out_name)
}

map_unified <- unify_bp("map_abp", "map_nibp", "map")
sbp_unified <- unify_bp("sbp_abp", "sbp_nibp", "sbp")
dbp_unified <- unify_bp("dbp_abp", "dbp_nibp", "dbp")

# Temperature: unify to Celsius (convert Fahrenheit where necessary).
# De-duplicate FIRST (preferring Celsius when both are charted at the same time),
# THEN convert and rename — order matters because the preference logic uses the
# original variable_name before it gets overwritten.
temp_unified <- vitals_long |>
  filter(variable_name %in% c("temp_c","temp_f")) |>
  group_by(subject_id, stay_id, charttime) |>
  slice_min(order_by = if_else(variable_name == "temp_c", 0L, 1L),
            with_ties = FALSE) |>   # prefer Celsius; both convert to same value anyway
  ungroup() |>
  mutate(
    value         = if_else(variable_name == "temp_f", (value - 32) * 5/9, value),
    variable_name = "temperature_c"
  )

# Drop the raw component series; keep derived ones
vitals_clean <- vitals_long |>
  filter(!variable_name %in% c("map_abp","map_nibp",
                               "sbp_abp","sbp_nibp",
                               "dbp_abp","dbp_nibp",
                               "temp_c","temp_f")) |>
  bind_rows(gcs_total, map_unified, sbp_unified, dbp_unified, temp_unified)

message(sprintf("  Vitals: %d rows across %d biomarkers",
                nrow(vitals_clean), n_distinct(vitals_clean$variable_name)))

# ============================================================
# 7. ANALYSIS_LONG
#    Union labs + vitals; repeat spine covariates on every row.
# ============================================================
message("\nBuilding analysis_long...")

all_measurements <- bind_rows(labs_long, vitals_clean) |>
  select(subject_id, hadm_id, stay_id,
         variable_name, hours_since_icu_intime, value, charttime) |>
  arrange(subject_id, stay_id, variable_name, hours_since_icu_intime)

# Spine columns to repeat on every measurement row
spine_for_join <- cohort |>
  select(subject_id, hadm_id, stay_id,
         stroke_type, ischemic_hemorrhagic,
         gender, age, race,
         icu_los_days, first_careunit, hospital_expire_flag, dod,
         any_of(c("weight_lb","height_in","bmi")),
         starts_with("rx_"),
         any_of(c("proc_tpa","proc_thrombectomy")))

analysis_long <- all_measurements |>
  left_join(spine_for_join, by = c("subject_id","hadm_id","stay_id"))

message(sprintf("  analysis_long: %d rows | %d biomarkers | %d patients",
                nrow(analysis_long),
                n_distinct(analysis_long$variable_name),
                n_distinct(analysis_long$subject_id)))

# ============================================================
# 8. ANALYSIS_WIDE
#    One row per ICU stay.
#    Per biomarker: n_obs, first, last, min, max, mean, slope.
#
#    NOTE on slope: linear slope is a rough approximation for
#    bounded/ordinal measures such as GCS. Flag this in models.
# ============================================================
message("\nBuilding analysis_wide...")

safe_slope <- function(t, y) {
  # Fit a linear trend (y ~ t); return NA when fewer than 2 observations.
  ok <- !is.na(t) & !is.na(y)
  if (sum(ok) < 2L) return(NA_real_)
  tryCatch(unname(coef(lm(y[ok] ~ t[ok]))[2L]),
           error = \(e) NA_real_)
}

wide_summary <- analysis_long |>
  group_by(subject_id, hadm_id, stay_id, variable_name) |>
  summarise(
    n_obs     = sum(!is.na(value)),
    val_first = value[which.min(hours_since_icu_intime)],
    val_last  = value[which.max(hours_since_icu_intime)],
    val_min   = min(value,  na.rm = TRUE),
    val_max   = max(value,  na.rm = TRUE),
    val_mean  = mean(value, na.rm = TRUE),
    val_slope = safe_slope(hours_since_icu_intime, value),
    .groups   = "drop"
  ) |>
  pivot_wider(
    names_from  = variable_name,
    values_from = c(n_obs, val_first, val_last, val_min, val_max, val_mean, val_slope),
    names_glue  = "{variable_name}_{.value}"
  )

analysis_wide <- spine_for_join |>
  left_join(wide_summary, by = c("subject_id","hadm_id","stay_id"))

message(sprintf("  analysis_wide: %d rows × %d columns",
                nrow(analysis_wide), ncol(analysis_wide)))

# ============================================================
# 9. SAVE OUTPUTS
# ============================================================
dir.create("output", showWarnings = FALSE)
data.table::fwrite(analysis_long, "output/analysis_long.csv")
data.table::fwrite(analysis_wide, "output/analysis_wide.csv")
message("\nSaved:\n  output/analysis_long.csv\n  output/analysis_wide.csv")

# ============================================================
# 10. DISCONNECT
# ============================================================
dbDisconnect(con, shutdown = TRUE)
message("Done. DuckDB disconnected.")