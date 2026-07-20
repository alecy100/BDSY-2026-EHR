"""
Loads one imputation's train/test CSV (long format: multiple rows per
patient) and reshapes it into fixed-length, per-patient tensors:
  X_time    : (n_patients, MAX_SEQ_LEN, n_time_features) -- zero-padded
  lengths   : (n_patients,) -- number of REAL (non-padded) timesteps
  X_static  : (n_patients, n_static_features)
  y         : (n_patients,)
  subject_ids

ASSUMES each patient's observed bins are a contiguous prefix (bin 1..n),
not scattered gaps -- see config.py docstring / earlier discussion. A
warning prints (not an error) if any patient violates this.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from config import (
    ID_COL, OUTCOME_COL, TIME_COL, MAX_SEQ_LEN,
    TIME_VARYING_COLS, TIME_VARYING_CONTINUOUS,
    STATIC_COLS, STATIC_CONTINUOUS,
    CATEGORICAL_ENCODINGS,
)


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts any TIME_VARYING/STATIC column that's stored as text (e.g. an
    R factor written out by write.csv as its label rather than its code)
    into numeric 0/1, using config.CATEGORICAL_ENCODINGS. Any object-dtype
    column NOT in that dict gets a fallback factorize() encoding with a
    loud warning -- so an unexpected string column surfaces immediately
    with a printed mapping to check/copy into config.py, rather than
    crashing deep inside numpy with no indication of which column or what
    values it actually contained.
    """
    df = df.copy()
    for col in TIME_VARYING_COLS + STATIC_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            if col in CATEGORICAL_ENCODINGS:
                mapping = CATEGORICAL_ENCODINGS[col]
                unmapped = set(df[col].unique()) - set(mapping.keys())
                if unmapped:
                    raise ValueError(
                        f"Column '{col}' has values not covered by "
                        f"config.CATEGORICAL_ENCODINGS['{col}']: {unmapped}. "
                        f"Add them to the mapping in config.py."
                    )
                df[col] = df[col].map(mapping)
            else:
                codes, uniques = pd.factorize(df[col])
                df[col] = codes
                print(f"WARNING: column '{col}' was text-valued with no entry in "
                      f"CATEGORICAL_ENCODINGS -- auto-encoded as {dict(zip(uniques, range(len(uniques))))}. "
                      f"Add an explicit mapping to config.py if this direction is wrong.")
    return df


def _check_columns(df):
    expected = set([ID_COL, OUTCOME_COL, TIME_COL] + TIME_VARYING_COLS + STATIC_COLS)
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns not found in CSV: {sorted(missing)}. "
            f"Edit config.py's column lists to match the actual export, "
            f"or check that export_imputations_to_csv.R ran against the "
            f"right imputed data."
        )


def _flatten_valid(X_time, lengths, col_idx):
    """Stack only the REAL (non-padded) timesteps across all patients, for
    fitting a scaler without letting padding zeros pollute the fit."""
    rows = [X_time[i, : lengths[i], :][:, col_idx] for i in range(len(lengths)) if lengths[i] > 0]
    return np.concatenate(rows, axis=0)


def load_imputation_csv(imputation: int, split: str, data_dir: str) -> pd.DataFrame:
    path = f"{data_dir}/imputed_pair_{imputation}_{split}.csv"
    df = pd.read_csv(path)
    _check_columns(df)
    return df


def build_bin_mapping(df: pd.DataFrame) -> dict:
    """
    TIME_COL is z-scored (standardized) by the R imputation pipeline, not
    raw hours -- so instead of assuming specific values (e.g. 4,8,...,48),
    we rank the observed unique values by sorted order and use that rank
    (0..MAX_SEQ_LEN-1) as the bin index. This works regardless of the
    scaling constants, since scaling is a monotonic (order-preserving)
    transform. Build this ONCE per file and reuse it across every
    build_patient_tensors() call on that same file's train/val/test
    subsets, so a given raw value always maps to the same bin index.
    """
    unique_vals = np.sort(df[TIME_COL].unique())
    if len(unique_vals) != MAX_SEQ_LEN:
        print(f"WARNING: expected {MAX_SEQ_LEN} unique {TIME_COL} values, "
              f"found {len(unique_vals)}: {unique_vals}. Check TIME_COL / MAX_SEQ_LEN.")
    return {v: i for i, v in enumerate(unique_vals)}


def build_patient_tensors(df: pd.DataFrame, bin_mapping: dict,
                           time_scaler=None, static_scaler=None, fit_scalers=False):
    df = df.copy()
    if len(df) == 0:
        raise ValueError(
            "build_patient_tensors received an EMPTY data frame (0 rows). "
            "This means the train/val patient-ID split produced no matching "
            f"rows -- check that {ID_COL}'s dtype matches between the CSV "
            "and the split IDs (e.g. int64 vs object/string)."
        )

    df["_bin_idx"] = df[TIME_COL].map(bin_mapping)
    n_before = len(df)
    df = df[df["_bin_idx"].notna()]
    df["_bin_idx"] = df["_bin_idx"].astype(int)
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)}/{n_before} rows whose {TIME_COL} value "
              f"wasn't in bin_mapping (unexpected -- investigate if this is more than a handful).")

    if len(df) == 0:
        raise ValueError(
            f"ALL rows were dropped -- none of this data frame's {TIME_COL} "
            f"values matched bin_mapping's keys. bin_mapping was likely built "
            f"from a different file/subset with different (scaled) values."
        )

    df = _encode_categoricals(df)

    patients = df[ID_COL].unique()
    n = len(patients)
    n_time_feat = len(TIME_VARYING_COLS)
    n_static_feat = len(STATIC_COLS)

    X_time = np.zeros((n, MAX_SEQ_LEN, n_time_feat), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    X_static = np.zeros((n, n_static_feat), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    gap_violations = 0

    grouped = df.sort_values("_bin_idx").groupby(ID_COL)
    for i, pid in enumerate(patients):
        g = grouped.get_group(pid)
        bin_idx = g["_bin_idx"].values
        n_obs = len(g)

        if not np.array_equal(np.sort(bin_idx), np.arange(n_obs)):
            gap_violations += 1

        X_time[i, :n_obs, :] = g[TIME_VARYING_COLS].values
        lengths[i] = n_obs
        X_static[i, :] = g[STATIC_COLS].iloc[0].values
        y[i] = g[OUTCOME_COL].iloc[0]

    if gap_violations > 0:
        print(f"WARNING: {gap_violations}/{n} patients have non-contiguous observed "
              f"bins (gaps, not just trailing truncation). Padding still runs, but "
              f"double-check whether this needs different handling.")

    cont_time_idx = [TIME_VARYING_COLS.index(c) for c in TIME_VARYING_CONTINUOUS]
    cont_static_idx = [STATIC_COLS.index(c) for c in STATIC_CONTINUOUS]

    if fit_scalers:
        time_scaler = StandardScaler().fit(_flatten_valid(X_time, lengths, cont_time_idx))
        static_scaler = StandardScaler().fit(X_static[:, cont_static_idx])

    if time_scaler is not None:
        for i in range(n):
            L = lengths[i]
            if L > 0:
                # NOTE: X_time[i, :L, :][:, idx] = ... would silently no-op here
                # (fancy indexing on the second step returns a copy in numpy) --
                # X_time[i, :L, idx] = ... (single combined index) is what
                # actually writes back to the original array. But combining an
                # integer index (i) with a slice (:L) and a fancy index (idx)
                # puts the fancy-indexed axis FIRST in the result, i.e. the
                # write target has shape (len(idx), L), not (L, len(idx)) --
                # the .T below matches transform()'s (L, len(idx)) output to that.
                X_time[i, :L, cont_time_idx] = time_scaler.transform(X_time[i, :L, :][:, cont_time_idx]).T
    if static_scaler is not None:
        X_static[:, cont_static_idx] = static_scaler.transform(X_static[:, cont_static_idx])

    return {
        "X_time": X_time, "lengths": lengths, "X_static": X_static,
        "y": y, "subject_ids": patients,
        "time_scaler": time_scaler, "static_scaler": static_scaler,
    }


def drop_time_varying_feature(tensors: dict, feature_name: str) -> dict:
    """Zero out one time-varying feature's channel (used for MAP ablation).
    Keeps the array shape identical so the same model architecture can be
    reused for a fair A/B comparison; zeroing (post-scaling, so it's
    exactly the scaled feature's mean) is a defensible neutral value."""
    idx = TIME_VARYING_COLS.index(feature_name)
    t2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tensors.items()}
    t2["X_time"][:, :, idx] = 0.0
    return t2


def permute_time_varying_feature(tensors: dict, feature_name: str, rng: np.random.Generator) -> dict:
    """Shuffle one time-varying feature's ENTIRE trajectory across patients
    (not per-timestep) -- preserves that feature's own within-patient
    trajectory shape while breaking its association with the outcome and
    every other feature. Used for permutation importance."""
    idx = TIME_VARYING_COLS.index(feature_name)
    t2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tensors.items()}
    perm = rng.permutation(len(t2["y"]))
    t2["X_time"][:, :, idx] = tensors["X_time"][perm, :, idx]
    # lengths/y/X_static deliberately stay with the ORIGINAL patient --
    # only this one feature's trajectory gets swapped in from a random
    # other patient, which is what isolates that feature's contribution.
    return t2


class PatientSequenceDataset(Dataset):
    def __init__(self, tensors: dict):
        self.X_time = torch.from_numpy(tensors["X_time"])
        self.lengths = torch.from_numpy(tensors["lengths"])
        self.X_static = torch.from_numpy(tensors["X_static"])
        self.y = torch.from_numpy(tensors["y"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_time[idx], self.lengths[idx], self.X_static[idx], self.y[idx]