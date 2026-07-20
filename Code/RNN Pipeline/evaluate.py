"""
Evaluates trained RNN model(s) on the test set.

Usage:
    python evaluate.py                          # main model, all 5 imputations, pooled
    python evaluate.py --tag _no_map             # evaluate an ablation run instead
    python evaluate.py --permutation-importance  # main model + MAP permutation importance
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

import config as cfg
from data import PatientSequenceDataset, build_bin_mapping, build_patient_tensors, load_imputation_csv, permute_time_varying_feature
from model import MortalityGRU
from train import get_device


def load_model(imputation: int, n_time_features: int, n_static_features: int, tag=""):
    device = get_device()
    model = MortalityGRU(n_time_features, n_static_features,
                          hidden_size=cfg.HIDDEN_SIZE, num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT).to(device)
    model.load_state_dict(torch.load(f"{cfg.MODEL_DIR}/rnn_imputation_{imputation}{tag}.pt", map_location=device))
    model.eval()
    with open(f"{cfg.MODEL_DIR}/scalers_imputation_{imputation}{tag}.pkl", "rb") as f:
        scalers = pickle.load(f)
    return model, scalers, device


def predict_tensors(model, tensors, device, batch_size=256):
    ds = PatientSequenceDataset(tensors)
    probs = []
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            end = min(start + batch_size, len(ds))
            x_time = ds.X_time[start:end].to(device)
            lengths = ds.lengths[start:end].to(device)
            x_static = ds.X_static[start:end].to(device)
            logits = model(x_time, lengths, x_static)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def evaluate_imputation(imputation: int, tag="", drop_features=None, permute_feature=None, rng=None):
    df_test = load_imputation_csv(imputation, "test", cfg.DATA_DIR)
    bin_mapping = build_bin_mapping(df_test)

    # build once (unscaled) just to read feature-count shapes needed to
    # construct the model before loading its saved weights + scalers
    probe = build_patient_tensors(df_test, bin_mapping, fit_scalers=False)
    n_time_feat = probe["X_time"].shape[2]
    n_static_feat = probe["X_static"].shape[1]

    model, scalers, device = load_model(imputation, n_time_feat, n_static_feat, tag=tag)
    test_tensors = build_patient_tensors(
        df_test, bin_mapping, time_scaler=scalers["time_scaler"], static_scaler=scalers["static_scaler"],
    )

    if drop_features:
        from data import drop_time_varying_feature
        for feat in drop_features:
            test_tensors = drop_time_varying_feature(test_tensors, feat)

    if permute_feature is not None:
        test_tensors = permute_time_varying_feature(test_tensors, permute_feature, rng)

    probs = predict_tensors(model, test_tensors, device)
    return pd.DataFrame({
        cfg.ID_COL: test_tensors["subject_ids"],
        "y_true": test_tensors["y"],
        "y_prob": probs,
        "imputation": imputation,
    })


def pooled_metrics(preds_by_imputation: list, label="", save_tag=None):
    """
    preds_by_imputation: list of per-imputation prediction DataFrames (each
    tagged with an "imputation" column by evaluate_imputation).

    Computes BOTH:
      - per-imputation metrics (AUC/Brier/AUPRC computed separately within
        each imputation's own predictions)
      - the "final" pooled metrics: average each patient's predicted
        probability ACROSS imputations first, then compute one AUC/Brier/
        AUPRC on that averaged probability -- same averaging-before-scoring
        approach used throughout the R pipeline (denoises before the
        nonlinear ranking step, so it's expected to differ slightly from
        the mean of the per-imputation numbers, not a bug if it does).

    If save_tag is given (e.g. "" for the main model, "_no_map" for an
    ablation run), writes 3 CSVs to cfg.RESULTS_DIR:
      metrics_per_imputation{save_tag}.csv
      metrics_final{save_tag}.csv
      predictions_final{save_tag}.csv
    """
    per_imp_rows = []
    for imp_df in preds_by_imputation:
        k = imp_df["imputation"].iloc[0]
        per_imp_rows.append({
            "imputation": k,
            "n": len(imp_df),
            "auc": roc_auc_score(imp_df["y_true"], imp_df["y_prob"]),
            "brier": brier_score_loss(imp_df["y_true"], imp_df["y_prob"]),
            "auprc": average_precision_score(imp_df["y_true"], imp_df["y_prob"]),
        })
    per_imputation_df = pd.DataFrame(per_imp_rows)

    all_preds = pd.concat(preds_by_imputation, ignore_index=True)
    final = all_preds.groupby(cfg.ID_COL).agg(avg_prob=("y_prob", "mean"), y_true=("y_true", "first")).reset_index()

    auc = roc_auc_score(final["y_true"], final["avg_prob"])
    brier = brier_score_loss(final["y_true"], final["avg_prob"])
    auprc = average_precision_score(final["y_true"], final["avg_prob"])
    base_rate = final["y_true"].mean()

    print(f"\n================= RNN TEST-SET PERFORMANCE {label} =================")
    print("Per-imputation:")
    print(per_imputation_df.to_string(index=False))
    if len(per_imputation_df) > 1:
        print(f"\nAcross-imputation variability -- AUC: mean={per_imputation_df['auc'].mean():.4f}, "
              f"sd={per_imputation_df['auc'].std():.4f} | Brier: mean={per_imputation_df['brier'].mean():.4f}, "
              f"sd={per_imputation_df['brier'].std():.4f}")
    print(f"\nn patients (pooled) = {len(final)} | base rate = {base_rate:.4f}")
    print(f"FINAL (averaged-probability) AUC   = {auc:.4f}")
    print(f"FINAL (averaged-probability) Brier = {brier:.4f}")
    print(f"FINAL AUPRC = {auprc:.4f} (baseline/random = base rate = {base_rate:.4f})")

    result = {"auc": auc, "brier": brier, "auprc": auprc, "base_rate": base_rate,
              "final_preds": final, "per_imputation": per_imputation_df}

    if save_tag is not None:
        per_imputation_df.to_csv(f"{cfg.RESULTS_DIR}/metrics_per_imputation{save_tag}.csv", index=False)
        final.to_csv(f"{cfg.RESULTS_DIR}/predictions_final{save_tag}.csv", index=False)
        pd.DataFrame([{"auc": auc, "brier": brier, "auprc": auprc,
                       "base_rate": base_rate, "n_patients": len(final)}]
                     ).to_csv(f"{cfg.RESULTS_DIR}/metrics_final{save_tag}.csv", index=False)
        print(f"\nSaved -> {cfg.RESULTS_DIR}/metrics_per_imputation{save_tag}.csv, "
              f"metrics_final{save_tag}.csv, predictions_final{save_tag}.csv")

    return result


def run_permutation_importance(feature_name, n_repeats=20, tag=""):
    """
    Shuffles `feature_name`'s trajectory across test patients (see
    data.permute_time_varying_feature), re-predicts, and reports the AUC
    drop relative to the unpermuted model -- repeated n_repeats times to
    get a distribution of the drop (mean +/- SD) rather than one noisy
    draw. Uses imputation 1's model only (fast prototyping choice from the
    original plan); extend to all 5 + average if needed later.
    """
    baseline_preds = [evaluate_imputation(1, tag=tag)]
    baseline = pooled_metrics(baseline_preds, label="(baseline, no permutation)")
    baseline_auc = baseline["auc"]

    rng = np.random.default_rng(cfg.SEED)
    rows = []
    for r in range(n_repeats):
        permuted_preds = [evaluate_imputation(1, tag=tag, permute_feature=feature_name, rng=rng)]
        permuted = pooled_metrics(permuted_preds, label=f"(permuted '{feature_name}', repeat {r+1}/{n_repeats})")
        rows.append({"repeat": r, "permuted_auc": permuted["auc"], "auc_drop": baseline_auc - permuted["auc"]})

    drops_df = pd.DataFrame(rows)
    print(f"\n================= PERMUTATION IMPORTANCE: {feature_name} =================")
    print(f"Baseline AUC = {baseline_auc:.4f}")
    print(f"Mean AUC drop when '{feature_name}' is permuted: {drops_df['auc_drop'].mean():.4f} "
          f"(SD={drops_df['auc_drop'].std():.4f}, over {n_repeats} repeats)")

    out_path = f"{cfg.RESULTS_DIR}/permutation_importance_{feature_name}{tag}.csv"
    drops_df.insert(0, "baseline_auc", baseline_auc)
    drops_df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return drops_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--permutation-importance", action="store_true")
    parser.add_argument("--feature", type=str, default=cfg.MAP_COL)
    parser.add_argument("--n-repeats", type=int, default=20)
    args = parser.parse_args()

    if args.permutation_importance:
        run_permutation_importance(args.feature, n_repeats=args.n_repeats, tag=args.tag)
    else:
        preds = [evaluate_imputation(k, tag=args.tag) for k in range(1, 6)]
        pooled_metrics(preds, label=f"(tag='{args.tag}')", save_tag=args.tag)