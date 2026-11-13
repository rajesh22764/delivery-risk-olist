"""Streamlit demo: score an order for late-delivery risk, and see how good (and how limited) the model is.

    streamlit run app/streamlit_app.py
"""
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from delivery.serve import Predictor  # noqa: E402

FIG = ROOT / "reports" / "figures"
st.set_page_config(page_title="Late-delivery risk", layout="wide")


@st.cache_resource
def load():
    return Predictor(ROOT / "models")


@st.cache_data
def results():
    return json.loads((ROOT / "reports" / "results.json").read_text())


pred, res = load(), results()
samples = json.loads((ROOT / "models" / "lookups" / "sample_orders.json").read_text())
final = [r for r in res["test_comparison"] if "29 selected" in r["model"]][0]
base = res["test_base_rate"]

st.title("Will this order arrive late?")
st.caption("A model that scores an e-commerce order at the moment it is placed. Trained on about 46,000 Olist orders (2017 to early 2018), tested on later orders (May to August 2018).")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ranking quality (ROC-AUC)", f"{final['roc_auc']:.2f}", "0.50 = chance", delta_color="off", delta_arrow="off")
c2.metric("Late orders in the riskiest 10%", f"{final['capture_top10']:.0%}", "10% = chance", delta_color="off", delta_arrow="off")
c3.metric("Base late rate (test)", f"{base:.1%}", "orders delivered after the promised day", delta_color="off", delta_arrow="off")
c4.metric("Test orders", "20,874", "never used for training or tuning", delta_color="off", delta_arrow="off")
st.info("**Offline study, not a production system.** Olist data is Brazilian marketplace data from 2016 to 2018. The model finds risky orders better than chance but is far from a crystal ball; see the tabs for what it can and cannot do.")

tab_score, tab_good, tab_pay, tab_drift, tab_limits = st.tabs(["Score an order", "How good is it?", "Would acting on it pay?", "Drift and monitoring", "Data and limits"])

# ---------------------------------------------------------------- score an order
with tab_score:
    labels = [f"Order {i + 1}: {s['purchase_timestamp'][:10]}, promised {s['promised_delivery_date']}, price R${s['total_price']:.0f}" for i, s in enumerate(samples)]
    pick = st.selectbox("Start from a real order (from the test period)", range(len(samples)), format_func=lambda i: labels[i])
    s = samples[pick]
    left, right = st.columns([1, 1.25])
    with left:
        st.subheader("The order")
        purchase = st.date_input("Purchase date", datetime.fromisoformat(s["purchase_timestamp"]).date(), min_value=date(2017, 1, 1), max_value=date(2018, 9, 1))
        hour = st.slider("Purchase hour", 0, 23, datetime.fromisoformat(s["purchase_timestamp"]).hour)
        promised = st.date_input("Promised delivery date", date.fromisoformat(s["promised_delivery_date"]))
        price = st.number_input("Order price (R$)", 0.0, 10000.0, float(s["total_price"]), step=5.0)
        freight = st.number_input("Freight (R$)", 0.0, 1000.0, float(s["total_freight"]), step=1.0)
        zip_ = st.number_input("Customer zip prefix (5 digits)", 1000, 99999, int(s["customer_zip_prefix"]))
        seller = st.selectbox("Seller", pred.known_sellers(), index=pred.known_sellers().index(s["seller_id"]))
        ptype = st.selectbox("Payment", ["credit_card", "boleto", "voucher", "debit_card"], index=["credit_card", "boleto", "voucher", "debit_card"].index(s["payment_type"]) if s["payment_type"] in ("credit_card", "boleto", "voucher", "debit_card") else 0)
        flag_pct = st.slider("Flag the riskiest X% of orders", 1, 50, 10)
    order = {"seller_id": seller, "customer_zip_prefix": int(zip_), "n_items": s["n_items"], "total_price": price, "total_freight": freight, "payment_type": ptype,
             "max_installments": s["max_installments"], "purchase_timestamp": datetime.combine(purchase, datetime.min.time()).replace(hour=hour).isoformat(), "promised_delivery_date": promised.isoformat()}
    with right:
        st.subheader("The prediction")
        try:
            r = pred.predict(order, top_k=6, flag_percent=flag_pct)
            a, b, c = st.columns(3)
            a.metric("Late probability", f"{r['late_probability']:.1%}")
            b.metric("Risk percentile", f"{r['risk_percentile']:.0f}", "vs recent orders", delta_color="off", delta_arrow="off")
            c.metric("Flag", "FLAG" if r["flag"] else "no flag")
            st.caption(r["flag_rule"] + ". Probabilities are only well calibrated when conditions resemble the training period; the percentile is the safer signal.")
            f = pd.DataFrame(r["top_factors"])
            names = (f["label"] + " = " + f["value"].map(lambda v: f"{v:.2f}" if isinstance(v, float) else str(v))).tolist()
            chart = pd.DataFrame({"raises risk": f.log_odds_effect.clip(lower=0).to_numpy(), "lowers risk": f.log_odds_effect.clip(upper=0).to_numpy()}, index=names)   # .to_numpy(): a Series would be re-aligned to the new index and turn into NaN
            st.markdown("**Why:** the factors that moved this order's risk the most (log-odds)")
            st.bar_chart(chart, horizontal=True, color=["#b91c1c", "#0f766e"])
            for w in r["warnings"]:
                st.warning(w)
        except ValueError as e:
            st.error(str(e))
        if (promised.isoformat() == s["promised_delivery_date"] and price == s["total_price"] and purchase.isoformat() == s["purchase_timestamp"][:10]):
            st.markdown("**What actually happened:** " + (f":red[**delivered late**, {s['actual_days_vs_promise']} day(s) after the promised date]" if s["actual_late"] else f":green[**delivered on time**, {-s['actual_days_vs_promise']} day(s) before the promised date]"))
            st.caption(f"Score computed at the time of purchase in the batch pipeline: {s['score_at_purchase_time']:.1%}. The score above uses statistics as of {pred.snapshot[:10]}, so for old orders the two can differ.")

# ---------------------------------------------------------------- how good
with tab_good:
    st.markdown(f"On **20,874 test orders** (4.2% late), the selected model ranks orders with ROC-AUC **{final['roc_auc']:.3f}** (95% interval {final['roc_auc_ci'][0]:.3f} to {final['roc_auc_ci'][1]:.3f}). "
                f"Flagging the riskiest 10% of orders catches **{final['capture_top10']:.0%}** of late orders (a random 10% would catch 10%).")
    st.image(str(FIG / "08_test_model_comparison.png"), caption="A regularised linear model beat gradient boosting (best by cross-validation) and a seller-history rule. Bars show 95% bootstrap intervals.")
    cv = pd.read_csv(ROOT / "reports" / "cv_model_selection.csv")
    st.markdown("**How the model was chosen:** rolling cross-validation inside the training period (calm, Black Friday and February-spike regimes). The test set was used once, at the end.")
    st.dataframe(cv.rename(columns={"model": "candidate", "cv_roc_auc": "CV ROC-AUC", "cv_ap_lift": "CV PR lift", "cv_capture_top10": "CV top-10% capture", "auc_by_fold": "AUC by fold"}).round(3), hide_index=True, width="stretch")
    st.image(str(FIG / "16_roc_and_pr_curves.png"))

# ---------------------------------------------------------------- would it pay
with tab_pay:
    st.markdown("Acting on a flag costs something (an expedited carrier, a proactive message, a voucher). It pays off only where the chance of lateness is above the **break-even precision** = cost of acting / (success rate x cost of a late order). Try the numbers:")
    k1, k2, k3 = st.columns(3)
    L = k1.slider("A late order costs this many times an intervention", 2, 60, 20)
    succ = k2.slider("Chance an intervention prevents the lateness", 0.1, 1.0, 0.5, step=0.05)
    gc = pd.DataFrame(res["gain_curve"])
    gc["cost"] = 1000 * (gc.share + L * (base - succ * gc.recall * base))
    never = 1000 * L * base
    be = 1 / (succ * L)
    best = gc.loc[gc.cost.idxmin()]
    st.metric("Break-even precision", f"{be:.1%}", f"the model's precision: {gc.loc[(gc.share - 0.01).abs().idxmin(), 'precision']:.1%} for its riskiest 1%, {gc.loc[(gc.share - 0.10).abs().idxmin(), 'precision']:.1%} for its riskiest 10%", delta_color="off", delta_arrow="off")
    saving = (never - best.cost) / never
    st.markdown(f"Best policy on the test window: flag the riskiest **{best.share:.0%}** of orders, saving **{max(saving, 0):.1%}** of the expected cost" + ("" if saving > 0.001 else " (no flagging policy saves money at these assumptions)") + ". The share is picked with test data, so this is slightly optimistic.")
    saving_chart = pd.DataFrame({"expected saving per 1,000 orders, compared with flagging nobody (above 0 = saves money)": (never - gc.cost).to_numpy()}, index=pd.Index((gc.share * 100).round(0).astype(int).to_numpy(), name="% of orders flagged (riskiest first)"))
    st.caption("Reading the chart: above 0 the flagging policy saves money compared with flagging nobody; below 0 it costs more than it saves.")
    st.line_chart(saving_chart)
    st.image(str(FIG / "14_cost_vs_flagged_share.png"), caption="Across cost assumptions: flagging only pays when a late order is expensive relative to an intervention.")

# ---------------------------------------------------------------- drift
with tab_drift:
    st.markdown("Late-delivery rates swing from about 1% to 19% from month to month, and the model's inputs shift with them. Two honest findings:")
    st.markdown("- **Probabilities depend on the regime.** Raw probabilities are well calibrated on the test period (predicted 4.7% vs actual 4.2%). A calibrator fitted on the March 2018 spike (19% late) then predicts 14% on the calmer test period, and its Brier score gets worse. So the app uses the raw probability plus a rank-based flag.")
    st.markdown("- **Ranking quality decays.** ROC-AUC by purchase month: 0.75 (May), 0.75 (June), 0.67 (July), 0.61 (August). Retrain regularly and watch it.")
    a, b = st.columns(2)
    a.image(str(FIG / "11_calibration.png")); b.image(str(FIG / "12_monthly_behaviour.png"))
    st.image(str(FIG / "15_feature_drift_psi.png"))
    st.markdown("**Monitoring plan (not built here):** log every score and the eventual outcome; track ROC-AUC and the observed late rate by week; alert when the population stability index of an input passes 0.25 or weekly AUC falls below 0.60; retrain monthly on orders old enough to have resolved (45+ days).")

# ---------------------------------------------------------------- limits
with tab_limits:
    st.markdown(
        "- **Data:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0), plus public holidays (Nager.Date) and daily weather (Open-Meteo) that I collected. Weather added only about 0.002 AUC in cross-validation and is not used by the deployed model.\n"
        "- **Label:** an order is late when it is delivered on a later calendar day than promised (6.8% of delivered orders). Orders never delivered are not in the data.\n"
        "- **What the model uses:** only what is known when the order is placed: the promise, the route, the seller's and region's recent record, the calendar. Nothing from after purchase.\n"
        "- **Seller history is weak:** a seller's late rate barely persists over time (rank correlation 0.09 from training to test), so event-driven spikes dominate.\n"
        "- **Statistics shown here are as of 2018-08-22**, the end of the usable data. This is a demonstration, not a live system.\n"
        "- **The signal is modest.** Ranking quality is clearly better than chance, but at an intervention cost of 10x or less relative to a late order, acting on the flags saves almost nothing.")
