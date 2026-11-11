"""Phase 4: train, compare, explain and track the three predictive models.

For each task: candidate models are trained on stacked training cutoffs, compared on the validation cutoff,
the winner is refit on train + validation and reported on the out-of-time test cutoff.
Every run is logged to MLflow (mlruns/, see the README for the UI).

Run from the project root:  python -m src.models.train
"""
import json
import warnings

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import ROOT
from src.features.household_features import load_inputs
from src.models import evaluation as ev
from src.models.datasets import TASKS, Task, backtest_folds, build_split, summary, xy

warnings.filterwarnings("ignore", category=UserWarning)

ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"
SEED = 42


# Inverse L2 strength picked on the backtest folds (scratch grid 0.001-0.5). Coupon needs strong shrinkage:
# with C=0.5 the correlated spend features flip sign and the top decile is over-confident (0.84 vs 0.72 observed).
LOGREG_C = {"churn": 0.1, "coupon_propensity": 0.01}
# Tweedie handles zero-inflated, right-skewed spend: never negative, unbiased in total.
# Squared error predicted negative spend; L1 and log-target under-stated total CLV by 4-17% on the backtest.
TWEEDIE_POWER = 1.5


def candidates(task: Task) -> dict:
    if task.kind == "classification":
        def logreg():
            return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(max_iter=3000, C=LOGREG_C[task.name]))

        def xgboost():
            return xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=5,
                                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                     eval_metric="aucpr", random_state=SEED)
        return {
            "logistic_regression": logreg(),
            "lightgbm": lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                           reg_lambda=1.0, random_state=SEED, verbose=-1),
            "xgboost": xgboost(),
            # Linear and tree models make different mistakes; averaging their probabilities can beat both
            "logreg_xgb_average": VotingClassifier([("logreg", logreg()), ("xgb", xgboost())], voting="soft"),
        }
    return {
        "ridge_log": make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(signed_log1p),
                                   StandardScaler(), Ridge(alpha=1.0)),
        "lightgbm_tweedie": lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=TWEEDIE_POWER,
                                              n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                              random_state=SEED, verbose=-1),
        "xgboost_tweedie": xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=TWEEDIE_POWER,
                                            n_estimators=500, learning_rate=0.03, max_depth=4, min_child_weight=5,
                                            subsample=0.8, colsample_bytree=0.8, random_state=SEED),
    }


def signed_log1p(x):
    """log1p that keeps the sign: spend_trend and the savings columns can be negative."""
    return np.sign(x) * np.log1p(np.abs(x))


def predict(model, X: pd.DataFrame, task: Task) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if task.kind == "classification" else model.predict(X)


def naive_clv(X: pd.DataFrame, horizon: int) -> np.ndarray:
    """Baseline any analyst would try first: last 8 weeks' spend scaled to the horizon."""
    return X.spend_last_8w.values * horizon / 56


def shap_plots(model, X: pd.DataFrame, name: str) -> list:
    sample = X.sample(min(len(X), 2000), random_state=SEED)
    values = shap.TreeExplainer(model)(sample)
    paths = []
    for kind in ["beeswarm", "bar"]:
        plt.figure()
        getattr(shap.plots, kind)(values, max_display=15, show=False)
        path = FIGURES / f"{name}_shap_{kind}.png"
        plt.gcf().set_size_inches(10, 7)
        plt.tight_layout()
        plt.savefig(path, dpi=110)
        plt.close("all")
        paths.append(path)
    return paths


def select_model(task: Task, inputs: dict, cache: dict) -> tuple[str, dict, float | None]:
    """Rolling-origin backtest: each candidate is scored on every fold, the best mean score wins.
    One validation cutoff has ~120 churners, too few to separate models; several folds are steadier."""
    folds = backtest_folds(task)
    scores = {name: [] for name in candidates(task)}
    oof = {name: ([], []) for name in candidates(task)}
    for prior, cut in folds:
        Xa, ya = xy(build_split(inputs, task, prior, cache))
        Xb, yb = xy(build_split(inputs, task, [cut], cache))
        for name, model in candidates(task).items():
            p = predict(model.fit(Xa, ya), Xb, task)
            if task.kind == "classification":
                scores[name].append(average_precision_score(yb, p))
            else:
                scores[name].append(-mean_absolute_error(yb, p))
            oof[name][0].append(yb.values); oof[name][1].append(p)

    summary_table = {}
    for name, s in scores.items():
        with mlflow.start_run(run_name=f"{name}-backtest"):
            mlflow.log_params({"model": name, "stage": "backtest", "folds": str([c for _, c in folds])})
            metric = "pr_auc" if task.kind == "classification" else "mae"
            values = [v if task.kind == "classification" else -v for v in s]
            for step, v in enumerate(values):
                mlflow.log_metric(f"fold_{metric}", v, step=step)
            mlflow.log_metric(f"mean_{metric}", float(np.mean(values)))
            summary_table[name] = {f"mean_{metric}": np.mean(values), f"std_{metric}": np.std(values),
                                   "folds_won": 0, "per_fold": [round(v, 3) for v in values]}
    for i in range(len(folds)):
        winner = max(scores, key=lambda n: scores[n][i])
        summary_table[winner]["folds_won"] += 1
    best = max(scores, key=lambda n: np.mean(scores[n]))

    threshold = None
    if task.kind == "classification":  # decision threshold from pooled out-of-fold predictions, never from test
        threshold = ev.best_f1_threshold(np.concatenate(oof[best][0]), np.concatenate(oof[best][1]))
    print(f"  backtest folds {[c for _, c in folds]}")
    for name, row in summary_table.items():
        print(f"  {name:<20} " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    return best, summary_table, threshold


def run_task(task: Task, inputs: dict, cache: dict) -> dict:
    full = build_split(inputs, task, list(task.train_cutoffs) + [task.valid_cutoff], cache)
    test = build_split(inputs, task, [task.test_cutoff], cache)
    Xfull, yfull = xy(full)
    Xte, yte = xy(test)
    print(f"\n=== {task.name}: train+valid {summary(full, task)} | test {summary(test, task)}")

    mlflow.set_experiment(f"customer360-{task.name}")
    best, backtest, threshold = select_model(task, inputs, cache)
    models = candidates(task)

    # Refit every candidate on all cutoffs whose labels end before the test window; report on test
    test_preds, results = {}, {}
    for name in models:
        model = candidates(task)[name].fit(Xfull, yfull)
        test_preds[name] = predict(model, Xte, task)
        if name == best:
            best_model = model
    if task.kind == "regression":
        test_preds["naive_run_rate"] = naive_clv(Xte, task.horizon)

    for name, p in test_preds.items():
        if task.kind == "classification":
            results[name] = ev.classification_metrics(yte, p, threshold)
            results[name]["pr_auc_ci"] = ev.bootstrap_ci(average_precision_score, yte, p)
        else:
            results[name] = ev.regression_metrics(yte, p)
            results[name]["mae_ci"] = ev.bootstrap_ci(mean_absolute_error, yte, p)

    if task.kind == "classification":
        fig = ev.classification_plots(yte, test_preds, best, threshold, task.description, FIGURES / f"{task.name}_diagnostics.png")
    else:
        fig = ev.regression_plots(yte, test_preds, best, task.description, FIGURES / f"{task.name}_diagnostics.png")

    # SHAP needs a tree model; if a linear model won, explain the LightGBM challenger instead
    # SHAP needs a tree model: the winner itself, the XGBoost half of the average, or LightGBM if a linear model won
    if isinstance(best_model, (lgb.LGBMModel, xgb.XGBModel)):
        shap_source = best_model
    elif isinstance(best_model, VotingClassifier):
        shap_source = best_model.named_estimators_["xgb"]
    else:
        challenger = "lightgbm" if task.kind == "classification" else "lightgbm_tweedie"
        shap_source = candidates(task)[challenger].fit(Xfull, yfull)
    print(f"  SHAP explains: {type(shap_source).__name__}")
    shap_files = shap_plots(shap_source, Xte, task.name)

    with mlflow.start_run(run_name=f"{best}-final"):
        mlflow.log_params({"model": best, "stage": "final", "test_cutoff": task.test_cutoff,
                           "threshold": threshold if threshold is not None else "n/a"})
        for k, v in results[best].items():
            if isinstance(v, tuple):
                mlflow.log_metrics({f"test_{k}_lo": v[0], f"test_{k}_hi": v[1]})
            else:
                mlflow.log_metric(f"test_{k}", v)
        for path in [fig, *shap_files]:
            mlflow.log_artifact(str(path))

    # Production model: refit on every labelled cutoff so it has seen the most recent behaviour
    Xall = pd.concat([Xfull, Xte])
    yall = pd.concat([yfull, yte])
    final = candidates(task)[best].fit(Xall, yall)
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": list(Xall.columns), "threshold": threshold, "task": task.name},
                ARTIFACTS / f"{task.name}_model.joblib")

    def clean(v):
        return [round(x, 4) for x in v] if isinstance(v, tuple) else round(float(v), 4)

    table = pd.DataFrame({k: {m: clean(v) for m, v in r.items()} for k, r in results.items()}).T
    print(f"  selected by backtest: {best}\n  test results (cutoff {task.test_cutoff}):\n{table.to_string()}")
    return {"task": task.name, "best": best, "threshold": threshold, "backtest": backtest,
            "test": {k: {m: clean(v) for m, v in r.items()} for k, r in results.items()}}


def main() -> None:
    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    inputs, cache = load_inputs(), {}
    report = [run_task(task, inputs, cache) for task in TASKS.values()]
    (ROOT / "reports" / "model_results.json").write_text(json.dumps(report, indent=2))
    print("\nwrote reports/model_results.json and reports/figures/")


if __name__ == "__main__":
    main()
