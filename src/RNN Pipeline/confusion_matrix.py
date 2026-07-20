"""
Computes the same confusion-matrix stats (TN/FP/FN/TP, accuracy,
sensitivity, specificity, ppv, npv) as the R pipeline's confusion matrix
scripts, plus a matching heatmap -- run this AFTER evaluate.py, since it
reads evaluate.py's saved predictions_final{tag}.csv rather than
re-predicting anything itself.

Convention (matches the whole project): is_dead / observed mortality is
1 = died, 0 = survived. Predicted class = 1 (died) if avg_prob >= threshold.

Usage:
    python confusion_matrix.py                    # main model (no tag)
    python confusion_matrix.py --tag _no_map       # an ablation run
    python confusion_matrix.py --threshold 0.3     # different cutoff
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import config as cfg

# same blue gradient used in the R confusion matrix plots
CMAP = LinearSegmentedColormap.from_list("dcef1_to_1f4e79", ["#DCE6F1", "#1F4E79"])


def compute_confusion(df: pd.DataFrame, threshold: float) -> dict:
    predicted = (df["avg_prob"] >= threshold).astype(int)
    observed = df["y_true"].astype(int)

    tn = int(((predicted == 0) & (observed == 0)).sum())
    fp = int(((predicted == 1) & (observed == 0)).sum())
    fn = int(((predicted == 0) & (observed == 1)).sum())
    tp = int(((predicted == 1) & (observed == 1)).sum())

    n = tn + fp + fn + tp
    return {
        "n": n, "threshold": threshold,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tp + tn) / n,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        "npv": tn / (tn + fn) if (tn + fn) > 0 else float("nan"),
    }


def plot_confusion(metrics: dict, title: str, out_path: str):
    # rows = Predicted, in order [Died, Survived] so Died plots at the TOP
    # (row 0 in imshow) and Survived at the BOTTOM (row 1) -- matching the
    # R plot's TN-bottom-left / TP-top-right layout.
    # cols = Observed, in order [Survived, Died] so Survived is LEFT, Died is RIGHT.
    cm = np.array([
        [metrics["fp"], metrics["tp"]],   # Predicted = Died:     [Observed=Survived -> FP, Observed=Died -> TP]
        [metrics["tn"], metrics["fn"]],   # Predicted = Survived: [Observed=Survived -> TN, Observed=Died -> FN]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap=CMAP)

    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > vmax / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    color=color, fontsize=16, fontweight="bold", fontfamily="Georgia")

    ax.set_xticks([0, 1]); ax.set_xticklabels(["Survived", "Died"], fontfamily="Georgia", fontsize=12)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Died", "Survived"], fontfamily="Georgia", fontsize=12)
    ax.set_xlabel("Observed mortality", fontfamily="Georgia", fontsize=13)
    ax.set_ylabel("Predicted mortality", fontfamily="Georgia", fontsize=13)
    ax.set_title(title, fontfamily="Georgia", fontsize=14)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    preds_path = f"{cfg.RESULTS_DIR}/predictions_final{args.tag}.csv"
    df = pd.read_csv(preds_path)

    metrics = compute_confusion(df, args.threshold)

    print(f"\n================= CONFUSION MATRIX (tag='{args.tag}', threshold={args.threshold:.2f}) =================")
    print(f"n={metrics['n']} | TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}")
    print(f"accuracy={metrics['accuracy']:.3f} | sensitivity={metrics['sensitivity']:.3f} | "
          f"specificity={metrics['specificity']:.3f} | ppv={metrics['ppv']:.3f} | npv={metrics['npv']:.3f}")

    metrics_path = f"{cfg.RESULTS_DIR}/confusion_matrix_metrics{args.tag}.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Saved -> {metrics_path}")

    plot_path = f"{cfg.RESULTS_DIR}/confusion_matrix{args.tag}.png"
    title = "Confusion matrix: RNN" + (f" ({args.tag.lstrip('_')})" if args.tag else "")
    plot_confusion(metrics, title, plot_path)
    print(f"Saved -> {plot_path}")