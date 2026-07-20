# Mortality prediction pipeline — script guide

This implements the workflow your advisor described, as five ordered R scripts.
Run them in order; each reads the previous step's saved `.rds`/`.csv` from `output/`.

| Script | Advisor's step | What it does |
|---|---|---|
| `00_train_test_split.R` | Step 1 | 80/20 **patient-level** split, stratified by mortality |
| `01_multiple_imputation.R` | Step 2 | `mice(..., ignore=)` on combined train+test → 5 (train, test) pairs |
| `02a_penalty_selection_cv.R` | Step 3a | 5-fold patient-level CV on one 10% training sample, picks lambda by **held-out** RSS |
| `02b_stability_selection.R` | Step 3b | 100 reps of glmmLasso on 10% training subsamples (fixed lambda), records selection frequency |
| `03_fit_lmm_and_logistic.R` | Steps 4–5 | ordinary `lmer` on reduced variables (5 imputations), saves random effects, fits 5 logistic mortality models |
| `04_predict_test_and_evaluate.R` | Steps 6–7 | BLUP formula to estimate test patients' random effects, predicts mortality, averages across imputations, reports AUC/Brier |

## Important fix versus your original bootstrap script
Your original script called `mice()` fresh inside every one of the 100 bootstrap
replicates (100 `mice()` calls + 500 `glmmLasso` fits) — that's almost certainly
what was making it computationally infeasible. The advisor's plan only calls for
imputing **once** (Step 2, on the full training set) and then repeatedly
subsampling rows from that single completed dataset for stability selection
(Step 3b). `02b_stability_selection.R` does this — no imputation inside the loop
at all, so it should be dramatically faster than the version you had.

## Things you must edit before running
- **`mortality_var`** — set to your actual patient-level outcome column name (I used `"mortality"` as a placeholder).
- **`response_var`** (`"map"`), **`time_var`** (`"binned_hrs_since_icu"`), and the variable groups (`level1_vars_continuous`, etc.) — copied from your existing script; adjust if your `cleanData.csv` differs.
- **`lambda_grid`** in `02a` — the range of penalties to search; widen/narrow based on what you saw in your earlier exploration.
- **`selection_threshold`** in `02b` — the cutoff selection frequency (e.g. 70%) used to decide which variables make it into Step 4's reduced model.

## Note on the uploaded file
The MIMIC-IV demo zip you attached is the raw ~100-patient PhysioNet demo
(`icu/`, `hosp/` tables) — it isn't your cleaned `output/cleanData.csv` and
doesn't match the 5,567-patient / 59,641×55 dataset you described, so I
couldn't run this pipeline end-to-end against it here (this sandbox also
doesn't have R/CRAN access). These scripts are written to match the structure
your own `document 2` script already assumes; you'll want to run them yourself
against your real `cleanData.csv`, ideally starting with a small `n_bootstrap`
(as your script's own comment already suggests) to sanity-check runtime before
committing to the full 100 reps.

## Honest limitations to report (per your advisor's email)
1. Penalty selection and stability selection use only **imputed training set #1** — variability across imputations in which variables get selected is not captured.
2. A reduced ordinary LMM (not glmmLasso) is fit per imputation in Step 4 — this is a computational compromise, not the ideal "glmmLasso on the full data" analysis.
3. The penalty is chosen once (Step 3a) and held fixed for all 100 stability-selection reps (Step 3b).
4. The BLUP formula in Step 6 treats the training-fit's fixed effects and variance components as known/fixed when solving for test patients' random effects — standard empirical-Bayes practice, but it does not propagate uncertainty in those training estimates into the test-patient random-effect estimates themselves.
