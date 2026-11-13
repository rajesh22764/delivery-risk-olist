"""Figures for the modelling results (the EDA figures live in eda.py)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from delivery.config import FIGURES
from delivery.modeling import expected_costs

TEAL, GREY, RED, AMBER = "#0f766e", "#94a3b8", "#b91c1c", "#d97706"


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=130)
    plt.close(fig)


def make_model_figures(df, X, masks, res, comparators, p_tr, p_va, p_te, yv, yt, cv, cv_ab, ab_df, coef, cal):
    base = float(np.mean(yt))
    rows = [r for r in res["test_comparison"] if r["model"] != "Base rate (no model)"]
    names = [r["model"].replace("Logistic regression: ", "LR: ").replace("Seller's historical late rate alone", "Seller late rate alone") for r in rows]

    # 08: test comparison with bootstrap intervals
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for a, key, label, norm in ((ax[0], "roc_auc", "ROC-AUC (0.5 = chance)", 1.0), (ax[1], "pr_auc", "PR-AUC lift over base rate (1 = chance)", 1 / base)):
        v = np.array([r[key] for r in rows]) * norm
        lo = np.array([r[f"{key}_ci"][0] for r in rows]) * norm
        hi = np.array([r[f"{key}_ci"][1] for r in rows]) * norm
        colors = [TEAL if "29 selected" in r["model"] else GREY for r in rows]
        a.barh(names, v, xerr=[v - lo, hi - v], color=colors, capsize=3)
        a.set_xlabel(label); a.axvline(0.5 if key == "roc_auc" else 1.0, color=RED, ls="--", lw=1)
        a.set_xlim(0.45 if key == "roc_auc" else 0.5, None)
    ax[0].set_title("Test window, 95% bootstrap intervals"); ax[1].set_yticklabels([])
    _save(fig, "08_test_model_comparison.png")

    # 09: model selection by rolling cross-validation
    fig, ax = plt.subplots(figsize=(9, 3.8))
    c = cv.sort_values("cv_roc_auc")
    ax.barh(c["model"], c["cv_roc_auc"], color=[TEAL if "(chosen)" in m else GREY for m in c["model"]])
    ax.set_xlim(0.58, None); ax.set_xlabel("ROC-AUC, mean of 3 rolling folds inside the training period"); ax.set_title("Model selection (no test data used)")
    _save(fig, "09_cv_model_selection.png")

    # 10: feature-group ablation in CV
    fig, ax = plt.subplots(figsize=(7, 3.6))
    a = cv_ab.sort_values("change_in_auc", ascending=False)
    ax.barh(a["dropped_group"], a["change_in_auc"], color=[RED if v < 0 else AMBER for v in a["change_in_auc"]])
    ax.axvline(0, color="black", lw=.8); ax.set_xlabel("change in CV ROC-AUC when the group is dropped (negative = the group helps)"); ax.set_title("Which feature groups matter")
    _save(fig, "10_cv_group_ablation.png")

    # 11: reliability diagrams (raw model) on validation and test
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, y_, p_, title in ((ax[0], yv, p_va, "Validation window (a spike month)"), (ax[1], yt, p_te, "Test window (calmer)")):
        d = pd.DataFrame({"y": y_, "p": p_}); d["b"] = pd.qcut(d.p.rank(method="first"), 10, labels=False); g = d.groupby("b").agg(p=("p", "mean"), y=("y", "mean"))
        hi = max(g.p.max(), g.y.max()) * 1.1
        a.plot([0, hi], [0, hi], color=GREY, ls="--"); a.plot(g.p, g.y, "o-", color=TEAL)
        a.set_xlabel("mean predicted probability (decile)"); a.set_ylabel("observed late rate"); a.set_title(f"{title}\nmean predicted {d.p.mean():.3f} vs actual {d.y.mean():.3f}")
    _save(fig, "11_calibration.png")

    # 12: monthly behaviour on the test window
    m = pd.DataFrame(res["test_by_month"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(m))
    ax[0].bar(x - .2, m.late_rate * 100, .4, color=RED, label="actual"); ax[0].bar(x + .2, m.mean_pred * 100, .4, color=TEAL, label="mean predicted")
    ax[0].set_xticks(x); ax[0].set_xticklabels(m.month); ax[0].set_ylabel("% late"); ax[0].legend(); ax[0].set_title("Late rate by purchase month, test window")
    ax[1].plot(m.month, m.roc_auc, "o-", color=TEAL); ax[1].axhline(0.5, color=GREY, ls="--"); ax[1].set_ylim(0.4, 0.9); ax[1].set_title("Ranking quality (ROC-AUC) by month"); ax[1].set_ylabel("ROC-AUC")
    _save(fig, "12_monthly_behaviour.png")

    # 13: coefficients
    cf = coef.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(cf["column"].str.replace("__", ": "), cf["coef"], color=[RED if v > 0 else TEAL for v in cf["coef"]])
    ax.set_xlabel("standardised coefficient on the log-odds of being late (red = riskier)"); ax.set_title("Largest effects in the final model")
    _save(fig, "13_coefficients.png")

    # 14: expected cost vs how many orders are flagged
    fig, ax = plt.subplots(figsize=(8, 4))
    qs = np.linspace(0.005, 0.5, 100)
    for L, col in ((5, GREY), (10, AMBER), (20, TEAL), (40, RED)):
        cost = [expected_costs(yt, p_te, np.quantile(p_te, 1 - q), 1.0, L, 0.5)["model"] * 1000 for q in qs]
        never = expected_costs(yt, p_te, 2.0, 1.0, L, 0.5)["never"] * 1000
        ax.plot(qs * 100, np.array(cost) - never, color=col, label=f"late order costs {L}x an intervention")
    ax.axhline(0, color="black", lw=.8); ax.set_xlabel("% of orders flagged (highest risk first)"); ax.set_ylabel("expected cost per 1,000 orders,\nrelative to flagging nobody (negative = saves money)")
    ax.legend(fontsize=8); ax.set_title("Intervention success rate 50%; test window")
    _save(fig, "14_cost_vs_flagged_share.png")

    # 15: PSI
    psi_df = pd.DataFrame({"train_to_val": res["psi_train_to_val"], "train_to_test": res["psi_train_to_test"]}).sort_values("train_to_test", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    y_ = np.arange(len(psi_df)); ax.barh(y_ - .2, psi_df.train_to_val, .4, color=AMBER, label="training to validation"); ax.barh(y_ + .2, psi_df.train_to_test, .4, color=TEAL, label="training to test")
    ax.set_yticks(y_); ax.set_yticklabels(psi_df.index); ax.axvline(0.25, color=RED, ls="--", lw=1); ax.set_xlabel("population stability index (0.25 = large shift)"); ax.legend(fontsize=8)
    ax.set_title("Which inputs drifted most")
    _save(fig, "15_feature_drift_psi.png")

    # 16: ROC and PR curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, p in comparators.items():
        if name == "Base rate (no model)":
            continue
        short = name.replace("Logistic regression: ", "LR: ").split(" (best")[0]
        f, t, _ = roc_curve(yt, p); ax[0].plot(f, t, label=short, lw=2 if "29 selected" in name else 1, color=TEAL if "29 selected" in name else None)
        pr, rc, _ = precision_recall_curve(yt, p); ax[1].plot(rc, pr, lw=2 if "29 selected" in name else 1, color=TEAL if "29 selected" in name else None)
    ax[0].plot([0, 1], [0, 1], color=GREY, ls="--"); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate"); ax[0].legend(fontsize=7); ax[0].set_title("ROC, test window")
    ax[1].axhline(base, color=GREY, ls="--"); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_title(f"Precision-recall, test window (base rate {base:.3f})")
    _save(fig, "16_roc_and_pr_curves.png")
