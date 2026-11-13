"""Models, calibration, metrics, bootstrap intervals, cost analysis and drift checks."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler

from delivery.features import ALL_FEATURES, ALL_WITH_SCALE, CATEGORICAL, GROUPS, SCALE_COUNTS

LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=5.0, n_estimators=2000, verbosity=-1, n_jobs=-1)
HEAVY_TAILED = ["total_price", "total_freight", "total_weight_g", "total_volume_cm3", "payment_value", "distance_km", "seller_prior_n", "route_prior_n", "seller_orders_30d", "platform_orders_7d", "seller_load_ratio", "platform_load_ratio"]


def fit_lgbm(Xtr, ytr, Xva, yva, features, seed=0, params=None):
    cats = [c for c in CATEGORICAL if c in features]
    m = lgb.LGBMClassifier(**{**LGB_PARAMS, **(params or {})}, random_state=seed)
    m.fit(Xtr[features], ytr, eval_set=[(Xva[features], yva)], eval_metric="binary_logloss", categorical_feature=cats,
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def fit_logreg(Xtr, ytr, features, C=0.1, spline_cols=()):
    """Regularised logistic regression. Heavy-tailed numerics are log-transformed, categoricals one-hot encoded, and the
    `spline_cols` get smooth non-linear terms (cubic splines) so effects like 'very short promises are riskier' can be learned."""
    spline_cols = [c for c in spline_cols if c in features]
    num = [c for c in features if c not in CATEGORICAL and c not in spline_cols]
    cat = [c for c in features if c in CATEGORICAL]
    log = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    parts = [
        ("heavy", Pipeline([("imp", SimpleImputer(strategy="median")), ("log", log), ("sc", StandardScaler())]), [c for c in num if c in HEAVY_TAILED]),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), [c for c in num if c not in HEAVY_TAILED]),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100), cat)]
    if spline_cols:
        parts.append(("spline", Pipeline([("imp", SimpleImputer(strategy="median")), ("sp", SplineTransformer(n_knots=6, degree=3, knots="quantile", extrapolation="constant")), ("sc", StandardScaler())]), spline_cols))
    return Pipeline([("pre", ColumnTransformer(parts)), ("lr", LogisticRegression(C=C, max_iter=3000))]).fit(Xtr[features], ytr)


def proba(model, X, features):
    return model.predict_proba(X[features])[:, 1]


class Calibrator:
    """Maps raw scores to probabilities on the validation window. kind: 'platt' (2 parameters, robust) or 'isotonic'."""

    def __init__(self, kind="platt"):
        self.kind = kind

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p_raw, y):
        if self.kind == "platt":
            self.m = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p_raw), y)
        else:
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_raw, y)
        return self

    def predict(self, p_raw):
        return self.m.predict_proba(self._logit(p_raw))[:, 1] if self.kind == "platt" else self.m.predict(p_raw)


def ece(y, p, bins=10):
    """Expected calibration error with equal-count bins."""
    df = pd.DataFrame({"y": y, "p": p})
    df["b"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("b").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g["n"] * (g["y"] - g["p"]).abs()).sum() / g["n"].sum())


def capture_at(y, score, frac=0.10):
    """Share of all late orders found in the top `frac` of orders ranked by score."""
    k = int(len(y) * frac)
    top = np.argsort(-score, kind="stable")[:k]
    return float(np.asarray(y)[top].sum() / np.sum(y))


def metrics(y, p_rank, p_cal=None):
    """Ranking metrics use p_rank; probability metrics use p_cal (or p_rank when not given)."""
    p_cal = p_rank if p_cal is None else p_cal
    return {"roc_auc": roc_auc_score(y, p_rank), "pr_auc": average_precision_score(y, p_rank), "capture_top10": capture_at(y, p_rank, 0.10),
            "brier": brier_score_loss(y, p_cal), "log_loss": log_loss(y, np.clip(p_cal, 1e-6, 1 - 1e-6)), "ece": ece(y, p_cal)}


def bootstrap_ci(y, p, fn_names=("roc_auc", "pr_auc", "capture_top10"), b=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    fns = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score, "capture_top10": lambda a, s: capture_at(a, s, 0.10)}
    res = {n: [] for n in fn_names}
    for _ in range(b):
        i = rng.integers(0, len(y), len(y))
        for n in fn_names:
            res[n].append(fns[n](y[i], p[i]))
    return {n: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for n, v in res.items()}


def paired_bootstrap_diff(y, p_a, p_b, metric="pr_auc", b=1000, seed=0):
    """Difference metric(a) - metric(b) on the same resampled orders."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    f = average_precision_score if metric == "pr_auc" else roc_auc_score
    d = []
    for _ in range(b):
        i = rng.integers(0, len(y), len(y))
        d.append(f(y[i], p_a[i]) - f(y[i], p_b[i]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ----------------------------------------------------------------------------------------- decisions under costs
def expected_costs(y, p, threshold, c=1.0, L=10.0, success=0.5):
    """Cost per order when flagged orders get an intervention that costs c and prevents a late delivery with probability `success`.
    A late order that is not prevented costs L. Returns the model's cost and the 'never flag' and 'always flag' costs."""
    y = np.asarray(y)
    flag = p >= threshold
    n, late = len(y), y.sum()
    model = (c * flag.sum() + L * (late - success * y[flag].sum())) / n
    return {"model": model, "never": L * late / n, "always": (c * n + L * (1 - success) * late) / n,
            "flagged_share": float(flag.mean()), "precision": float(y[flag].mean()) if flag.any() else float("nan"), "recall": float(y[flag].sum() / late)}


def decision_threshold(c=1.0, L=10.0, success=0.5):
    """Flag an order when the expected saving exceeds the cost: p * success * L > c, i.e. p > c / (success * L) for calibrated p."""
    return c / (success * L)


def psi(expected: pd.Series, actual: pd.Series, bins=10) -> float:
    """Population stability index with bins from the expected (training) distribution. <0.1 stable, 0.1-0.25 moderate, >0.25 large shift."""
    edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(expected.dropna(), edges)[0] / expected.notna().sum()
    a = np.histogram(np.clip(actual.dropna(), edges[0], edges[-1]), edges)[0] / actual.notna().sum()
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(((a - e) * np.log(a / e)).sum())


# ----------------------------------------------------------------------------------------- rolling-origin cross-validation
# Three folds inside the training period, each with a 45-day maturity gap between the fitted window and the validation window.
# They cover calm (Aug to Sep 2017), Black Friday (Nov to Dec 2017) and the Feb 2018 spike. Used for model and feature selection,
# so the test set is touched only once.
CV_FOLDS = [("2017-06-30", "2017-08-15", "2017-09-30"), ("2017-09-30", "2017-11-15", "2017-12-31"), ("2017-11-30", "2018-01-15", "2018-02-28")]


def cv_masks(df: pd.DataFrame):
    p, deliv = df["purchase_ts"], df["delivered"]
    day = pd.Timedelta(days=1)
    for train_end, v0, v1 in CV_FOLDS:
        tr = deliv & (p >= "2017-01-01") & (p < pd.Timestamp(train_end) + day) & (df["delivered_ts"] < v0)
        va = deliv & (p >= v0) & (p < pd.Timestamp(v1) + day)
        yield tr, va


def cv_evaluate(df, X, y, make_scores, features):
    """make_scores(Xtr, ytr, Xva, features) -> validation scores. Returns per-fold ROC-AUC, AP lift (AP / base rate) and capture@10%."""
    rows = []
    for k, (tr, va) in enumerate(cv_masks(df), start=1):
        s = make_scores(X[tr], y[tr], X[va], features)
        base = float(y[va].mean())
        rows.append({"fold": k, "roc_auc": roc_auc_score(y[va], s), "ap_lift": average_precision_score(y[va], s) / base, "capture_top10": capture_at(y[va], s, 0.10), "base_rate": base})
    r = pd.DataFrame(rows)
    return r, r[["roc_auc", "ap_lift", "capture_top10"]].mean()
