"""End-to-end training and evaluation. `python -m delivery.pipeline` writes models/, reports/results.json and reports/figures/.

Order of events (so that the test set is touched exactly once):
  1. build features; define the time splits
  2. model and feature selection by rolling-origin cross-validation INSIDE the training period (reports/cv_*.csv)
  3. fit the chosen model on the training window
  4. evaluate calibration options and drift on the validation window; fit the calibrator there
  5. final evaluation on the test window
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from delivery import config as C
from delivery.data import load_orders
from delivery.explain import coefficient_table
from delivery.features import ALL_FEATURES, ALL_WITH_SCALE, CATEGORICAL, FINAL_FEATURES, GROUPS, CategoryEncoder, build_features
from delivery.modeling import (LGB_PARAMS, Calibrator, bootstrap_ci, capture_at, cv_evaluate, cv_masks, decision_threshold, ece, expected_costs, fit_logreg,
                               metrics, paired_bootstrap_diff, psi)
from delivery.split import make_splits, split_report

C_REG = 0.1


def prepare():
    df = load_orders()
    hol = pd.read_csv(C.EXTERNAL / "holidays_br.csv", parse_dates=["date"])
    wx = pd.read_csv(C.EXTERNAL / "weather_by_state.csv", parse_dates=["date"])
    X = build_features(df, hol, wx)
    masks = make_splits(df)
    enc = CategoryEncoder().fit(X[masks["train"]])
    return df, X, masks, enc, hol


def lr_scorer(C=C_REG, spline_cols=()):
    return lambda Xtr, ytr, Xva, feats: fit_logreg(Xtr, ytr, feats, C=C, spline_cols=spline_cols).predict_proba(Xva[feats])[:, 1]


def lgb_scorer(n, leaves, mcs):
    def f(Xtr, ytr, Xva, feats):
        p = {**LGB_PARAMS, "n_estimators": n, "num_leaves": leaves, "min_child_samples": mcs}
        m = lgb.LGBMClassifier(**p, random_state=0).fit(Xtr[feats], ytr, categorical_feature=[c for c in CATEGORICAL if c in feats])
        return m.predict_proba(Xva[feats])[:, 1]
    return f


SPLINE_COLS = ["promised_days", "distance_km", "freight_ratio", "seller_late_rate", "platform_late_rate_14d", "cust_state_late_rate_30d", "seller_mean_days_vs_promise", "days_to_next_holiday"]
CV_CANDIDATES = [
    ("LR, all 37 features + 4 growth counts", lr_scorer(), ALL_WITH_SCALE, "str"),
    ("LR, 37 stationary features", lr_scorer(), ALL_FEATURES, "str"),
    ("LR, 29 selected features (chosen)", lr_scorer(), FINAL_FEATURES, "str"),
    ("LR, 29 features + splines on 8 numerics", lr_scorer(spline_cols=SPLINE_COLS), FINAL_FEATURES, "str"),
    ("LightGBM, 37 features + 4 growth counts (300 trees, 31 leaves)", lgb_scorer(300, 31, 100), ALL_WITH_SCALE, "code"),
    ("LightGBM, 29 selected features (300 trees, 31 leaves)", lgb_scorer(300, 31, 100), FINAL_FEATURES, "code"),
    ("LightGBM, 29 selected features (300 trees, 8 leaves, min 300)", lgb_scorer(300, 8, 300), FINAL_FEATURES, "code"),
]


def run_cv_selection(df, X, y, enc):
    Xs, Xc = enc.transform(X, codes=False), enc.transform(X, codes=True)
    rows = []
    for name, fn, feats, kind in CV_CANDIDATES:
        r, mean = cv_evaluate(df, Xs if kind == "str" else Xc, y, fn, feats)
        rows.append({"model": name, "cv_roc_auc": mean.roc_auc, "cv_ap_lift": mean.ap_lift, "cv_capture_top10": mean.capture_top10, "auc_by_fold": " / ".join(f"{v:.3f}" for v in r.roc_auc)})
    cv = pd.DataFrame(rows)
    base = [r for r in rows if "37 stationary" in r["model"]][0]
    ab = []
    for g, cols in GROUPS.items():
        feats = [c for c in ALL_FEATURES if c not in cols]
        _, mean = cv_evaluate(df, Xs, y, lr_scorer(), feats)
        ab.append({"dropped_group": g, "n_features": len(cols), "cv_roc_auc": mean.roc_auc, "change_in_auc": mean.roc_auc - base["cv_roc_auc"], "change_in_ap_lift": mean.ap_lift - base["cv_ap_lift"]})
    return cv, pd.DataFrame(ab).sort_values("change_in_auc")


def calibration_options(p_tr_model, y_va, p_va, y_te, p_te):
    """Compare raw probabilities with calibrators fitted on the validation window, on validation (fit) and test (held out)."""
    out = {}
    for name, cal in (("raw (trained on the training window)", None), ("Platt scaling fitted on validation", Calibrator("platt")), ("isotonic fitted on validation", Calibrator("isotonic"))):
        if cal is not None:
            cal.fit(p_va, y_va)
        pv, pt = (p_va, p_te) if cal is None else (cal.predict(p_va), cal.predict(p_te))
        out[name] = {"val_mean_pred": float(pv.mean()), "val_actual": float(np.mean(y_va)), "test_mean_pred": float(pt.mean()), "test_actual": float(np.mean(y_te)),
                     "test_brier": float(((pt - y_te) ** 2).mean()), "test_ece": ece(y_te, pt)}
    return out


def main():
    t0 = time.time()
    tick = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
    df, X, masks, enc, hol = prepare()
    y = df["late"]
    tr, va, te = masks["train"], masks["val"], masks["test"]
    Xs, Xc = enc.transform(X, codes=False), enc.transform(X, codes=True)
    rep = split_report(df, masks)
    tick("features and splits ready\n" + rep.to_string(index=False))

    cv, cv_ab = run_cv_selection(df, X, y, enc)
    C.REPORTS.mkdir(exist_ok=True)
    cv.to_csv(C.REPORTS / "cv_model_selection.csv", index=False)
    cv_ab.to_csv(C.REPORTS / "cv_group_ablation.csv", index=False)
    tick("cross-validation done\n" + cv.round(4).to_string(index=False))

    # ---- fit the chosen model on the training window; predictions on validation and test
    lr = fit_logreg(Xs[tr], y[tr], FINAL_FEATURES, C=C_REG)
    p_tr, p_va, p_te = (lr.predict_proba(Xs.loc[m, FINAL_FEATURES])[:, 1] for m in (tr, va, te))
    yt, yv = y[te].values, y[va].values
    res = {"splits": rep.assign(first_purchase=rep.first_purchase.astype(str), last_purchase=rep.last_purchase.astype(str)).to_dict("records")}

    # ---- calibration and drift, using the validation window
    res["calibration"] = calibration_options(p_tr, yv, p_va, yt, p_te)
    cal = Calibrator("platt").fit(p_va, yv)
    res["validation_metrics"] = metrics(yv, p_va)

    # ---- the single test evaluation
    comparators = {}
    comparators["Base rate (no model)"] = np.full(te.sum(), p_tr.mean())
    comparators["Seller's historical late rate alone"] = Xs.loc[te, "seller_late_rate"].values
    for name, groups in (("Logistic regression: promise + calendar only", ["promise_calendar"]),
                         ("Logistic regression: promise + geography", ["promise_calendar", "geography"])):
        feats = [c for g in groups for c in GROUPS[g]]
        comparators[name] = fit_logreg(Xs[tr], y[tr], feats, C=C_REG).predict_proba(Xs.loc[te, feats])[:, 1]
    comparators["Logistic regression: 29 selected features"] = p_te
    best_lgb = cv[cv.model.str.startswith("LightGBM")].sort_values("cv_roc_auc", ascending=False).iloc[0].model
    cfg = [c for c in CV_CANDIDATES if c[0] == best_lgb][0]
    seeds = []
    for s in range(5):
        pp = {**LGB_PARAMS, "n_estimators": 300, "num_leaves": 8 if "8 leaves" in best_lgb else 31, "min_child_samples": 300 if "8 leaves" in best_lgb else 100}
        feats = cfg[2]
        mdl = lgb.LGBMClassifier(**pp, random_state=s).fit(Xc.loc[tr, feats], y[tr], categorical_feature=[c for c in CATEGORICAL if c in feats])
        seeds.append(mdl.predict_proba(Xc.loc[te, feats])[:, 1])
    comparators[f"LightGBM (best by CV: {best_lgb.split('(', 1)[1][:-1]})"] = seeds[0]
    res["lightgbm_seed_spread"] = {"roc_auc_mean": float(np.mean([metrics(yt, s)["roc_auc"] for s in seeds])), "roc_auc_sd": float(np.std([metrics(yt, s)["roc_auc"] for s in seeds], ddof=1)),
                                   "pr_auc_mean": float(np.mean([metrics(yt, s)["pr_auc"] for s in seeds])), "pr_auc_sd": float(np.std([metrics(yt, s)["pr_auc"] for s in seeds], ddof=1))}
    rows = []
    for name, p in comparators.items():
        m = metrics(yt, p)
        if name == "Base rate (no model)":      # a constant score carries no information: chance-level ranking, so state the chance values directly
            m = {**m, "roc_auc": 0.5, "pr_auc": float(yt.mean()), "capture_top10": 0.10}
            ci = {k: (m[k],) * 3 for k in ("roc_auc", "pr_auc", "capture_top10")}
        else:
            ci = bootstrap_ci(yt, p)
        rows.append({"model": name, **{k: m[k] for k in ("roc_auc", "pr_auc", "capture_top10")}, **{f"{k}_ci": [ci[k][1], ci[k][2]] for k in ci}})
    res["test_comparison"] = rows
    res["test_base_rate"] = float(yt.mean())
    final = "Logistic regression: 29 selected features"
    res["paired_differences_vs_final"] = {name: {m: paired_bootstrap_diff(yt, comparators[final], p, m) for m in ("roc_auc", "pr_auc")} for name, p in comparators.items() if name not in (final, "Base rate (no model)")}
    tick("test evaluation done")
    # sensitivity: the same model on the full window, including the truncated end of the data
    tf = masks["test_full"]
    pf = lr.predict_proba(Xs.loc[tf, FINAL_FEATURES])[:, 1]
    mf = metrics(y[tf].values, pf)
    res["sensitivity_full_test_window"] = {"orders": int(tf.sum()), "late_rate": float(y[tf].mean()), "roc_auc": mf["roc_auc"], "pr_auc": mf["pr_auc"], "capture_top10": mf["capture_top10"]}

    # ---- test-window ablation: drop one feature group at a time (the model is refit on the training window)
    ab = []
    for g in FINAL_FEATURES and [g for g in ("promise_calendar", "order_payment", "geography", "seller_history", "system_lateness", "holidays")]:
        feats = [c for c in FINAL_FEATURES if c not in GROUPS[g]]
        p = fit_logreg(Xs[tr], y[tr], feats, C=C_REG).predict_proba(Xs.loc[te, feats])[:, 1]
        ab.append({"dropped_group": g, "test_roc_auc": metrics(yt, p)["roc_auc"], "test_pr_auc": metrics(yt, p)["pr_auc"]})
    ab_df = pd.DataFrame(ab)
    ab_df["change_in_auc"] = ab_df.test_roc_auc - metrics(yt, p_te)["roc_auc"]
    res["test_ablation"] = ab_df.to_dict("records")

    # ---- decisions under costs (see modeling.expected_costs): a threshold policy and a capacity policy
    base_rate = float(yt.mean())
    pol = {}
    for L in (5, 10, 20, 40):
        for succ in (0.3, 0.5, 0.8):
            thr = decision_threshold(1.0, L, succ)
            r_raw = expected_costs(yt, p_te, thr, 1.0, L, succ)
            r_cal = expected_costs(yt, cal.predict(p_te), thr, 1.0, L, succ)
            curve = [(q, expected_costs(yt, p_te, np.quantile(p_te, 1 - q), 1.0, L, succ)["model"]) for q in np.linspace(0.005, 0.5, 100)]
            q_best, c_best = min(curve, key=lambda t: t[1])
            pol[f"L={L},success={succ}"] = {"threshold": thr, "never_flag_cost": r_raw["never"], "threshold_policy_raw": {k: r_raw[k] for k in ("model", "flagged_share", "precision", "recall")},
                                            "threshold_policy_platt": {k: r_cal[k] for k in ("model", "flagged_share", "precision", "recall")}, "best_capacity_share": q_best, "best_capacity_cost": c_best}
    res["cost_analysis"] = pol
    res["capture_curve_test"] = {f"top_{int(q * 100)}pct": {"late_orders_caught": capture_at(yt, p_te, q), "precision": float(yt[np.argsort(-p_te, kind='stable')[: int(len(yt) * q)]].mean())} for q in (0.01, 0.02, 0.05, 0.10, 0.20)}

    order = np.argsort(-p_te, kind="stable"); cum = np.cumsum(yt[order]); n_te = len(yt)
    res["gain_curve"] = [{"share": float(q), "precision": float(cum[int(n_te * q) - 1] / int(n_te * q)), "recall": float(cum[int(n_te * q) - 1] / yt.sum())} for q in np.linspace(0.01, 0.5, 50)]

    # ---- monthly behaviour and feature drift
    mdf = pd.DataFrame({"month": df.loc[te, "purchase_ts"].dt.to_period("M").astype(str).values, "y": yt, "p": p_te})
    monthly = []
    from sklearn.metrics import roc_auc_score
    for mth, g in mdf.groupby("month"):
        monthly.append({"month": mth, "orders": len(g), "late_rate": float(g.y.mean()), "mean_pred": float(g.p.mean()), "roc_auc": float(roc_auc_score(g.y, g.p)) if g.y.nunique() > 1 else None})
    res["test_by_month"] = monthly
    num = [c for c in FINAL_FEATURES if c not in CATEGORICAL]
    res["psi_train_to_test"] = {c: psi(X.loc[tr, c].astype(float), X.loc[te, c].astype(float)) for c in num}
    res["psi_train_to_val"] = {c: psi(X.loc[tr, c].astype(float), X.loc[va, c].astype(float)) for c in num}
    # calibration by CV fold for the chosen model (the regime-dependence that motivates a ranking-based policy)
    folds = []
    for k, (ftr, fva) in enumerate(cv_masks(df), start=1):
        pf = fit_logreg(Xs[ftr], y[ftr], FINAL_FEATURES, C=C_REG).predict_proba(Xs.loc[fva, FINAL_FEATURES])[:, 1]
        folds.append({"fold": k, "actual_late_rate": float(y[fva].mean()), "mean_predicted": float(pf.mean())})
    res["calibration_by_cv_fold"] = folds

    # ---- coefficients, figures, artifacts
    coef = coefficient_table(lr, FINAL_FEATURES)
    res["top_coefficients"] = coef.head(15).round(4).to_dict("records")
    from delivery import report, serve
    report.make_model_figures(df, X, masks, res, comparators, p_tr, p_va, p_te, yv, yt, cv, cv_ab, ab_df, coef, cal)
    serve.save_artifacts(df, X, masks, enc, lr, cal, res, hol, p_te)
    json.dump(res, open(C.REPORTS / "results.json", "w"), indent=2, default=float)
    tick("done: models/, reports/results.json, reports/figures/")


if __name__ == "__main__":
    main()
