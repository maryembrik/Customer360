"""Metrics and diagnostic plots: are the models actually good, and good for what?"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,
                             roc_auc_score, roc_curve)


# ---------- metrics ----------

def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[np.argmax(f1[:-1])])


def lift_at(y, p, top=0.1) -> float:
    n = max(int(len(p) * top), 1)
    idx = np.argsort(-p)[:n]
    return float(np.mean(np.asarray(y)[idx]) / np.mean(y))


def classification_metrics(y, p, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "f1": f1_score(y, pred), "threshold": threshold,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "lift_top10": lift_at(y, p, 0.1), "positive_rate": float(np.mean(y))}


def bootstrap_ci(metric, y, p, n=500, seed=0) -> tuple[float, float]:
    """95% percentile interval of a metric over households resampled with replacement."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            vals.append(metric(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def regression_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    top = p >= np.quantile(p, 0.9)
    return {"mae": mean_absolute_error(y, p), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": r2_score(y, p), "spearman": float(pd.Series(p).corr(pd.Series(y), method="spearman")),
            "total_bias": float(p.sum() / y.sum() - 1), "top_decile_bias": float(p[top].sum() / y[top].sum() - 1),
            "min_pred": float(p.min())}


# ---------- plots ----------

def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def classification_plots(y, probs: dict[str, np.ndarray], best: str, threshold: float, title: str, out: Path) -> Path:
    y = np.asarray(y)
    p = probs[best]
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))

    for name, pr in probs.items():
        fpr, tpr, _ = roc_curve(y, pr)
        ax[0, 0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, pr):.3f})")
        prec, rec, _ = precision_recall_curve(y, pr)
        ax[0, 1].plot(rec, prec, label=f"{name} (AP {average_precision_score(y, pr):.3f})")
    ax[0, 0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0, 0].set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate")
    ax[0, 1].axhline(y.mean(), color="k", ls="--", lw=1, label=f"random ({y.mean():.3f})")
    ax[0, 1].set(title="Precision-recall curve (test)", xlabel="Recall", ylabel="Precision")
    ax[0, 0].legend(fontsize=8); ax[0, 1].legend(fontsize=8)

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax[0, 2].plot(mean_pred, frac_pos, "o-", label=best)
    lim = max(mean_pred.max(), frac_pos.max()) * 1.05
    ax[0, 2].plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
    ax[0, 2].set(title="Calibration (test, deciles)", xlabel="Mean predicted probability", ylabel="Observed rate")
    ax[0, 2].legend(fontsize=8)

    deciles = pd.qcut(pd.Series(p).rank(method="first", ascending=False), 10, labels=range(1, 11))
    rate = pd.Series(y).groupby(deciles, observed=True).mean()
    ax[1, 0].bar(rate.index.astype(int), rate.values / y.mean())
    ax[1, 0].axhline(1, color="k", ls="--", lw=1)
    ax[1, 0].set(title="Lift by decile (1 = highest scores)", xlabel="Decile", ylabel="Lift vs average")

    order = np.argsort(-p)
    gains = np.cumsum(y[order]) / y.sum()
    ax[1, 1].plot(np.arange(1, len(y) + 1) / len(y), gains, label=best)
    ax[1, 1].plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax[1, 1].set(title="Cumulative gains", xlabel="Share of households contacted", ylabel="Share of positives found")
    ax[1, 1].legend(fontsize=8)

    cm = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])
    ax[1, 2].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[1, 2].text(j, i, f"{v:,}", ha="center", va="center", fontsize=13)
    ax[1, 2].set(title=f"Confusion matrix (threshold {threshold:.2f})", xticks=[0, 1], yticks=[0, 1],
                 xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])

    fig.suptitle(f"{title} — best model: {best}", fontsize=14)
    return _save(fig, out)


def regression_plots(y, preds: dict[str, np.ndarray], best: str, title: str, out: Path) -> Path:
    y = np.asarray(y)
    p = preds[best]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

    ax[0].scatter(y + 1, np.maximum(p, 0) + 1, s=5, alpha=0.3)
    lim = [1, max(y.max(), p.max()) * 1.2]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xscale="log", yscale="log", title="Predicted vs actual (log scale, +1)",
              xlabel="Actual 90-day spend", ylabel="Predicted")

    resid = y - p
    ax[1].scatter(p, resid, s=5, alpha=0.3)
    ax[1].axhline(0, color="k", ls="--", lw=1)
    ax[1].set(title="Residuals vs predicted", xlabel="Predicted", ylabel="Actual - predicted")

    for name, pr in preds.items():
        dec = pd.qcut(pd.Series(pr).rank(method="first"), 10, labels=range(1, 11))
        ax[2].plot(range(1, 11), pd.Series(y).groupby(dec, observed=True).mean().values, "o-", label=f"actual | {name}")
    dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=range(1, 11))
    ax[2].plot(range(1, 11), pd.Series(p).groupby(dec, observed=True).mean().values, "k--", label=f"predicted ({best})")
    ax[2].set(title="Mean actual spend by predicted decile", xlabel="Predicted decile (10 = highest)", ylabel="$")
    ax[2].legend(fontsize=7)

    fig.suptitle(f"{title} — best model: {best}", fontsize=14)
    return _save(fig, out)
