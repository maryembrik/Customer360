"""Phase 5: sentiment of Amazon grocery reviews (Negative / Neutral / Positive from stars, A08).

Split by product (no product appears in two splits), so the model cannot score well by memorising
product names. Candidates are compared on validation by macro-F1 (each class counts equally, so the
model cannot win by predicting "Positive" for everything); the winner is refit on train + validation
and reported once on test, with 95% bootstrap intervals. Logged to MLflow.

    python -m src.nlp.sentiment
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, average_precision_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline

from src.config import ROOT
from src.models.evaluation import bootstrap_ci

warnings.filterwarnings("ignore", category=UserWarning)

SAMPLE = ROOT / "data" / "sampled" / "amazon_grocery_reviews_sample.parquet"
META = ROOT / "data" / "sampled" / "amazon_grocery_reviews_sample_meta.json"
FIGURES = ROOT / "reports" / "figures"
ARTIFACTS = ROOT / "artifacts"
CLASSES = ["negative", "neutral", "positive"]
SEED = 42

# Keyword themes for the business read-out. Deliberately simple and transparent: a review "mentions" a
# theme if any of its words appear. Delivery / packaging is the logistics-relevant theme.
THEMES = {
    "taste_flavor": r"\b(taste[sd]?|tasty|flavou?rs?|bland|delicious|sweet|bitter|salty)\b",
    "price_value": r"\b(price[sd]?|expensive|cheap(er)?|overpriced|value|worth|cost)\b",
    "delivery_shipping": r"\b(deliver(y|ed)?|ship(ping|ped)?|arriv(e|ed|al)|late|package arrived|courier)\b",
    "packaging_damage": r"\b(packag(e|ing|ed)|box|damaged?|broken|crushed|leak(ed|ing|y)?|dented|seal(ed)?)\b",
    "freshness_expiry": r"\b(stale|expir(ed|ation|y)|fresh(ness)?|moldy|mould|rancid|best by|old)\b",
    "quality_ingredients": r"\b(quality|ingredients?|organic|natural|artificial|sugar)\b",
}


def label(rating: int) -> str:
    return "negative" if rating <= 2 else "neutral" if rating == 3 else "positive"


# Auto-generated titles that state the star rating ("Five Stars", "Two Stars"): 10% of reviews, 99.9% equal to the
# label. Keeping them lets the model read the answer instead of the text (A23), so they are removed from the input.
STAR_TITLE = r"^\s*(?:one|two|three|four|five|[1-5])\s+stars?\s*$"


def make_doc(title: pd.Series, text: pd.Series) -> pd.Series:
    title = title.fillna("").where(~title.fillna("").str.lower().str.match(STAR_TITLE), "")
    return (title + " . " + text.fillna("")).str.replace(r"<br\s*/?>", " ", regex=True)


def load() -> pd.DataFrame:
    df = pd.read_parquet(SAMPLE)
    df["star_title"] = df.title.fillna("").str.lower().str.match(STAR_TITLE)
    df["doc"] = make_doc(df.title, df.text)
    df["label"] = df.rating.map(label)
    before = len(df)
    # Drop only copy-pastes (same user, same text). Identical short texts from different people ("Good . Good",
    # 34 times) are genuine reviews; dropping them would remove short positives and bias the label mix.
    df = df.drop_duplicates(["user_id", "doc"]).reset_index(drop=True)
    df.attrs["duplicates_dropped"] = before - len(df)
    df["n_words"] = df.doc.str.split().str.len()
    return df


def split(df: pd.DataFrame):
    """Development (~85%, used for 5-fold cross-validation) and test (~15%, used once).
    Stratified by label, grouped by product (parent_asin): a product never sits on both sides."""
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    dev, test = next(outer.split(df, df.label, df.parent_asin))
    dev, test = df.iloc[dev], df.iloc[test]
    assert not set(dev.parent_asin) & set(test.parent_asin)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                 .split(dev, dev.label, dev.parent_asin))
    return dev, test, folds


def word_char_logreg(C):
    """Word 1-2-grams + character 2-5-grams (catches misspellings, 'sooo good', negation fragments).
    Plain sklearn objects only, so the saved model loads anywhere scikit-learn is installed."""
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300_000)
    return make_pipeline(FeatureUnion([("word", word_tfidf()), ("char", char)]),
                         LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


def word_tfidf(**kw):
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, strip_accents="unicode", **kw)


def candidates(C_word=4.0, C_wc=4.0) -> dict:
    return {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "complement_nb": make_pipeline(word_tfidf(), ComplementNB(alpha=0.3)),
        "tfidf_logreg": make_pipeline(word_tfidf(), LogisticRegression(C=C_word, max_iter=3000, class_weight="balanced")),
        "tfidf_word_char_logreg": word_char_logreg(C_wc),
    }


def metrics(y, pred) -> dict:
    return {"macro_f1": f1_score(y, pred, average="macro"), "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            **{f"f1_{c}": f for c, f in zip(CLASSES, f1_score(y, pred, average=None, labels=CLASSES))}}


def cv_scores(dev, folds, make_model) -> list[float]:
    out = []
    for tr, va in folds:
        m = make_model().fit(dev.doc.iloc[tr], dev.label.iloc[tr])
        out.append(f1_score(dev.label.iloc[va], m.predict(dev.doc.iloc[va]), average="macro"))
    return out


def tune_C(dev, folds, name, grid):
    """Inverse regularisation strength by mean 5-fold macro-F1 (one validation split was too noisy:
    ~260 neutral reviews, grid differences of 0.01 inside the noise)."""
    scores = {C: float(np.mean(cv_scores(dev, folds, lambda: candidates(C, C)[name]))) for C in grid}
    return max(scores, key=scores.get), scores


# ---------- plots ----------

def data_plot(df, dev, test, meta, out):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    # Proof the sample is spread over the whole file, not its first rows. A uniform sample of REVIEWS is uniform
    # in bytes only if reviews are equally long everywhere, so the expectation is adjusted for review length.
    dec = pd.cut(df.byte_offset / meta["file_bytes"], 10, labels=range(1, 11))
    approx_bytes = df.title.fillna("").str.len() + df.text.fillna("").str.len() + 260   # + fixed JSON fields
    g = pd.DataFrame({"n": df.groupby(dec, observed=False).size(),
                      "inv": 1 / approx_bytes.groupby(dec, observed=False).mean()})
    ax[0].bar(g.index.astype(int), g.n, label="sampled reviews")
    ax[0].plot(g.index.astype(int), g.inv / g.inv.sum() * len(df), "ko--", label="expected (length-adjusted)")
    ax[0].set(title="Where sampled reviews sit in the 6 GB file", xlabel="File tenth (1 = start, 10 = end)",
              ylabel="Reviews")
    ax[0].legend(fontsize=8, loc="lower right")
    split_mix = pd.DataFrame({n: s.label.value_counts(normalize=True) for n, s in
                              [("all", df), ("development (5-fold CV)", dev), ("test", test)]}).loc[CLASSES]
    split_mix.T.plot.bar(stacked=True, ax=ax[1], rot=0)
    ax[1].set(title="Label mix per split (stratified, grouped by product)", ylabel="Share")
    ax[1].legend(fontsize=8, loc="lower right")
    for c in CLASSES:
        ax[2].hist(np.clip(df.n_words[df.label == c], 0, 300), bins=50, alpha=0.5, density=True, label=c)
    ax[2].set(title="Review length by label (words, clipped at 300)", xlabel="Words")
    ax[2].legend()
    return _save(fig, out)


def diagnostics_plot(y, proba, classes, pred, best, frame, out):
    y = np.asarray(y)
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    cm = confusion_matrix(y, pred, labels=CLASSES)
    for a, m, fmt, t in [(ax[0, 0], cm, "{:,}", "Confusion matrix (counts)"),
                         (ax[0, 1], cm / cm.sum(1, keepdims=True), "{:.0%}", "Confusion matrix (row = true class, recall)")]:
        a.imshow(m, cmap="Blues")
        for (i, j), v in np.ndenumerate(m):
            a.text(j, i, fmt.format(v), ha="center", va="center", fontsize=12)
        a.set(title=t, xticks=range(3), yticks=range(3), xticklabels=[f"pred {c}" for c in CLASSES],
              yticklabels=[f"true {c}" for c in CLASSES])

    for k, c in enumerate(classes):
        prec, rec, _ = precision_recall_curve(y == c, proba[:, k])
        ax[0, 2].plot(rec, prec, label=f"{c} (AP {average_precision_score(y == c, proba[:, k]):.2f}, base {np.mean(y == c):.2f})")
    ax[0, 2].set(title="Precision-recall, one class vs rest", xlabel="Recall", ylabel="Precision")
    ax[0, 2].legend(fontsize=8)

    neg = list(classes).index("negative")
    fp, mp = calibration_curve(y == "negative", proba[:, neg], n_bins=10, strategy="quantile")
    ax[1, 0].plot(mp, fp, "o-")
    ax[1, 0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1, 0].set(title="Calibration of P(negative)", xlabel="Mean predicted", ylabel="Observed share negative")

    lengths = pd.Series(np.asarray(frame["n_words"]))
    bins = pd.qcut(lengths.rank(method="first"), 5, labels=["shortest", "short", "medium", "long", "longest"])
    acc = pd.Series(np.asarray(pred) == y).groupby(bins.values, observed=True).mean()
    f1s = pd.Series({b: f1_score(y[bins.values == b], np.asarray(pred)[bins.values == b], average="macro")
                     for b in bins.cat.categories})
    pd.DataFrame({"accuracy": acc, "macro-F1": f1s}).plot.bar(ax=ax[1, 1], rot=0)
    ax[1, 1].set(title="Quality by review length (quintiles)", ylim=(0, 1))

    stars = pd.Series(np.asarray(frame["rating"]))
    pd.crosstab(stars, np.asarray(pred), normalize="index")[CLASSES].plot.bar(stacked=True, ax=ax[1, 2], rot=0)
    ax[1, 2].set(title="Predicted label by star rating", xlabel="Stars", ylabel="Share")
    ax[1, 2].legend(fontsize=8)

    fig.suptitle(f"Review sentiment (test, products unseen in training) — best model: {best}", fontsize=14)
    return _save(fig, out)


def words_plot(model, out):
    """Strongest word features per class from the word-level logistic regression."""
    vec, clf = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
    vocab = np.array(vec.get_feature_names_out())
    fig, ax = plt.subplots(1, 3, figsize=(17, 6))
    for k, c in enumerate(clf.classes_):
        top = np.argsort(-clf.coef_[k])[:15]
        ax[k].barh(vocab[top][::-1], clf.coef_[k][top][::-1])
        ax[k].set(title=f"Words pushing towards '{c}'", xlabel="Coefficient")
    return _save(fig, out)


def themes_plot(themes, out):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    themes[["share_of_reviews"]].plot.barh(ax=ax[0], legend=False)
    ax[0].set(title="Share of reviews mentioning each theme", xlabel="Share")
    lo = themes.negative_rate - themes.negative_rate_ci.str[0]
    hi = themes.negative_rate_ci.str[1] - themes.negative_rate
    ax[1].barh(themes.index, themes.negative_rate, xerr=[lo, hi], color="tab:red", capsize=3, label="1-2 star share (95% CI)")
    ax[1].axvline(themes.attrs["overall_negative_rate"], color="k", ls="--", lw=1, label="all reviews")
    ax[1].set(title="Negative rate (1-2 stars) when a theme is mentioned, 30K reviews", xlabel="Share negative")
    ax[1].legend()
    return _save(fig, out)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def theme_table(df, test, test_pred) -> pd.DataFrame:
    """Themes on all 30K reviews with the star labels (observed, not modelled), plus the model's negative rate on
    the held-out test set as a cross-check. class_weight='balanced' over-predicts negatives (~8%), so business
    numbers use the stars; the model is for new text that has no rating."""
    rows = {}
    neg = df.label == "negative"
    low, low_t = df.doc.str.lower(), test.doc.str.lower()
    neg_pred = pd.Series(test_pred, index=test.index) == "negative"
    for name, pattern in THEMES.items():
        hit, hit_t = low.str.contains(pattern, regex=True), low_t.str.contains(pattern, regex=True)
        rows[name] = {"reviews": int(hit.sum()), "share_of_reviews": hit.mean(), "negative_rate": neg[hit].mean(),
                      "negative_rate_ci": bootstrap_ci(lambda y, _: y.mean(), neg[hit].values.astype(int),
                                                       neg[hit].values),
                      "share_of_all_negatives": (hit & neg).sum() / neg.sum(),
                      "negative_rate_model_test": neg_pred[hit_t].mean()}
    t = pd.DataFrame(rows).T.infer_objects().sort_values("negative_rate")
    t.attrs["overall_negative_rate"] = float(neg.mean())
    return t


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("customer360-sentiment")
    meta = json.loads(META.read_text())
    df = load()
    dev, test, folds = split(df)
    print(f"{len(df):,} reviews ({df.attrs['duplicates_dropped']} duplicates dropped, "
          f"{int(df.star_title.sum()):,} star-count titles removed) | dev {len(dev):,} (5 folds) test {len(test):,}")
    print(df.label.value_counts(normalize=True).round(3).to_dict())
    data_fig = data_plot(df, dev, test, meta, FIGURES / "sentiment_data.png")

    best_C, grids = {}, {}
    for name, grid in [("tfidf_logreg", [0.5, 1, 2, 4, 8, 16]), ("tfidf_word_char_logreg", [0.5, 1, 2, 4])]:
        best_C[name], grids[name] = tune_C(dev, folds, name, grid)
        print(f"  {name}: CV macro-F1 by C {({k: round(v, 3) for k, v in grids[name].items()})} -> {best_C[name]}")

    cv = {}
    for name in candidates():
        s = cv_scores(dev, folds, lambda: candidates(best_C["tfidf_logreg"], best_C["tfidf_word_char_logreg"])[name])
        cv[name] = {"mean_macro_f1": float(np.mean(s)), "std_macro_f1": float(np.std(s)),
                    "per_fold": [round(x, 3) for x in s], "folds_won": 0}
        with mlflow.start_run(run_name=f"{name}-cv"):
            mlflow.log_params({"model": name, "stage": "cv", "C": best_C.get(name, "n/a")})
            for step, x in enumerate(s):
                mlflow.log_metric("fold_macro_f1", x, step=step)
            mlflow.log_metric("mean_macro_f1", float(np.mean(s)))
    for i in range(len(folds)):
        cv[max(cv, key=lambda n: cv[n]["per_fold"][i])]["folds_won"] += 1
    best = max(cv, key=lambda n: cv[n]["mean_macro_f1"])
    print(pd.DataFrame(cv).T.to_string())
    print(f"selected by 5-fold CV: {best}")
    valid_scores = cv

    trval = dev
    test_scores, test_pred = {}, {}
    for name in candidates():
        m = candidates(C_word=best_C["tfidf_logreg"], C_wc=best_C["tfidf_word_char_logreg"])[name]
        m.fit(trval.doc, trval.label)
        test_pred[name] = m.predict(test.doc)
        test_scores[name] = metrics(test.label, test_pred[name])
        if name == best:
            best_model = m
        if name == "tfidf_logreg":
            word_model = m
    y = test.label.values
    for name in test_scores:
        test_scores[name]["macro_f1_ci"] = bootstrap_ci(lambda a, b: f1_score(a, b, average="macro"), y, test_pred[name])
    print(pd.DataFrame(test_scores).T.to_string())
    print(classification_report(y, test_pred[best], digits=3))

    proba = best_model.predict_proba(test.doc)
    diag_fig = diagnostics_plot(y, proba, best_model.classes_, test_pred[best], best, test,
                                FIGURES / "sentiment_diagnostics.png")
    words_fig = words_plot(word_model, FIGURES / "sentiment_top_words.png")

    # Business read-out on the whole sample with the production model (refit on all labelled reviews)
    final = candidates(C_word=best_C["tfidf_logreg"], C_wc=best_C["tfidf_word_char_logreg"])[best].fit(df.doc, df.label)
    themes = theme_table(df, test, test_pred[best])
    themes_fig = themes_plot(themes, FIGURES / "sentiment_themes.png")
    print(themes.round(3).to_string())

    errors = test.assign(pred=test_pred[best])
    errors = errors[(errors.label != errors.pred) & errors.label.isin(["negative", "positive"])
                    & errors.pred.isin(["negative", "positive"])]

    with mlflow.start_run(run_name=f"{best}-final"):
        mlflow.log_params({"model": best, "stage": "final", "C": best_C.get(best, "n/a"),
                           "train_reviews": len(trval), "test_reviews": len(test)})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_scores[best].items() if not isinstance(v, tuple)})
        for f in [data_fig, diag_fig, words_fig, themes_fig]:
            mlflow.log_artifact(str(f))

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"model": final, "classes": list(final.classes_), "task": "sentiment"}, ARTIFACTS / "sentiment_model.joblib")
    clean = lambda v: [round(x, 4) for x in v] if isinstance(v, tuple) else round(float(v), 4)
    report = {"reviews": len(df), "duplicates_dropped": df.attrs["duplicates_dropped"],
              "star_titles_removed": int(df.star_title.sum()),
              "splits": {"development_cv": len(dev), "test": len(test), "cv_folds": len(folds)},
              "label_share": df.label.value_counts(normalize=True).round(4).to_dict(),
              "C": best_C, "C_grid_cv": {n: {str(k): round(v, 4) for k, v in g.items()} for n, g in grids.items()},
              "best": best,
              "cv": valid_scores,
              "test": {n: {k: clean(v) for k, v in s.items()} for n, s in test_scores.items()},
              "themes": themes.round(4).to_dict(orient="index"),
              "overall_negative_rate_test_pred": round(themes.attrs["overall_negative_rate"], 4),
              "polarity_flips_in_test": int(len(errors))}
    (ROOT / "reports" / "sentiment_results.json").write_text(json.dumps(report, indent=2))
    print("wrote reports/sentiment_results.json, artifacts/sentiment_model.joblib")


if __name__ == "__main__":
    main()
