"""
Trains the GRU mortality model for one (or all) imputation(s).

Usage:
    python train.py --imputation 1
    python train.py --all              # loops imputations 1..5

Validation is carved out of TRAIN patients only (patient-level, stratified
by outcome) -- the held-out test set is never touched here. Scalers are
fit on the training split only and reused (not refit) on validation/test.
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import trange

import config as cfg
from data import PatientSequenceDataset, build_bin_mapping, build_patient_tensors, load_imputation_csv
from model import MortalityGRU

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_imputation(imputation: int, drop_features=None, tag=""):
    """
    drop_features: optional list of TIME_VARYING_COLS names to zero out
    BEFORE training (for ablation, e.g. drop_features=["map"]). tag is
    just a filename suffix so ablation runs don't overwrite the main model.
    """
    set_seed(cfg.SEED)
    device = get_device()

    df_train_full = load_imputation_csv(imputation, "train", cfg.DATA_DIR)

    patient_outcomes = df_train_full.groupby(cfg.ID_COL)[cfg.OUTCOME_COL].first()
    train_ids, val_ids = train_test_split(
        patient_outcomes.index.values, test_size=cfg.VAL_FRACTION,
        stratify=patient_outcomes.values, random_state=cfg.SEED,
    )

    df_tr = df_train_full[df_train_full[cfg.ID_COL].isin(train_ids)]
    df_val = df_train_full[df_train_full[cfg.ID_COL].isin(val_ids)]

    bin_mapping = build_bin_mapping(df_train_full)
    train_tensors = build_patient_tensors(df_tr, bin_mapping, fit_scalers=True)
    val_tensors = build_patient_tensors(
        df_val, bin_mapping, time_scaler=train_tensors["time_scaler"], static_scaler=train_tensors["static_scaler"],
    )

    if drop_features:
        from data import drop_time_varying_feature
        for feat in drop_features:
            train_tensors = drop_time_varying_feature(train_tensors, feat)
            val_tensors = drop_time_varying_feature(val_tensors, feat)

    train_ds = PatientSequenceDataset(train_tensors)
    val_ds = PatientSequenceDataset(val_tensors)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    n_pos = train_tensors["y"].sum()
    n_neg = len(train_tensors["y"]) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    print(f"[Imputation {imputation}{tag}] train n={len(train_tensors['y'])} "
          f"(pos={int(n_pos)}, neg={int(n_neg)}) | val n={len(val_tensors['y'])} | pos_weight={pos_weight.item():.3f}")

    model = MortalityGRU(
        n_time_features=train_tensors["X_time"].shape[2],
        n_static_features=train_tensors["X_static"].shape[1],
        hidden_size=cfg.HIDDEN_SIZE, num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = -np.inf
    best_state = None
    patience_counter = 0

    pbar = trange(1, cfg.MAX_EPOCHS + 1, desc=f"Imputation {imputation}{tag}")
    for epoch in pbar:
        model.train()
        for x_time, lengths, x_static, y in train_loader:
            x_time, lengths, x_static, y = (t.to(device) for t in (x_time, lengths, x_static, y))
            optimizer.zero_grad()
            logits = model(x_time, lengths, x_static)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            optimizer.step()

        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for x_time, lengths, x_static, y in val_loader:
                x_time, lengths, x_static = (t.to(device) for t in (x_time, lengths, x_static))
                logits = model(x_time, lengths, x_static)
                val_probs.append(torch.sigmoid(logits).cpu().numpy())
                val_true.append(y.numpy())
        val_probs = np.concatenate(val_probs)
        val_true = np.concatenate(val_true)
        val_auc = roc_auc_score(val_true, val_probs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        pbar.set_postfix(val_auc=f"{val_auc:.4f}", best=f"{best_val_auc:.4f}",
                          patience=f"{patience_counter}/{cfg.EARLY_STOP_PATIENCE}")

        if patience_counter >= cfg.EARLY_STOP_PATIENCE:
            pbar.close()
            print(f"  early stopping at epoch {epoch} (best val AUC = {best_val_auc:.4f})")
            break

    model.load_state_dict(best_state)

    model_path = f"{cfg.MODEL_DIR}/rnn_imputation_{imputation}{tag}.pt"
    scaler_path = f"{cfg.MODEL_DIR}/scalers_imputation_{imputation}{tag}.pkl"
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump({"time_scaler": train_tensors["time_scaler"],
                     "static_scaler": train_tensors["static_scaler"]}, f)

    print(f"[Imputation {imputation}{tag}] done. best val AUC = {best_val_auc:.4f}. "
          f"Saved -> {model_path}, {scaler_path}")
    return best_val_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imputation", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="train imputations 1..5")
    parser.add_argument("--drop-feature", action="append", default=None,
                         help="zero out this time-varying feature before training (repeatable), for ablation")
    parser.add_argument("--tag", type=str, default="", help="filename suffix, e.g. '_no_map'")
    args = parser.parse_args()

    imputations = range(1, 6) if args.all else [args.imputation]
    for k in imputations:
        train_one_imputation(k, drop_features=args.drop_feature, tag=args.tag)