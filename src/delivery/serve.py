"""Serving: artifacts written at training time, and a Predictor that scores one order.

The history features (seller, state and platform statistics) are looked up from tables built "as of" a snapshot time with exactly the
same definitions as the batch pipeline (events strictly before the snapshot, same windows and smoothing), so training and serving agree.
tests/test_serving.py checks that equivalence against the batch features.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from delivery import config as C
from delivery.explain import contributions
from delivery.features import CATEGORICAL, FINAL_FEATURES, CategoryEncoder, calendar_features
from delivery.geo import haversine_km
from delivery.history import smoothed_rate

DAY = pd.Timedelta(days=1)
FEATURE_LABELS = {
    "promised_days": "days promised", "promised_on_weekend": "promised date is a weekend", "purchase_hour": "hour of purchase", "purchase_dow": "day of week of purchase",
    "is_weekend": "purchased on a weekend", "n_items": "number of items", "n_sellers": "number of sellers", "total_price": "order price", "total_freight": "freight cost",
    "freight_ratio": "freight relative to price", "payment_value": "payment value", "max_installments": "payment installments", "payment_type": "payment type",
    "distance_km": "seller-to-customer distance (km)", "same_state": "seller and customer in the same state", "customer_state": "customer state", "seller_state": "seller state",
    "seller_new": "seller has under 10 past deliveries", "seller_late_rate": "seller's historical late rate", "seller_late_rate_60d": "seller's late rate, last 60 days",
    "seller_mean_days_vs_promise": "seller's typical days vs promise", "seller_mean_actual_days": "seller's typical delivery time", "seller_load_ratio": "seller's current load vs usual",
    "cust_state_late_rate_30d": "late rate into the customer's state, last 30 days", "platform_late_rate_14d": "platform-wide late rate, last 14 days",
    "platform_load_ratio": "platform's current load vs usual", "holidays_in_window": "holidays before the promised date", "days_to_next_holiday": "days to next holiday",
    "black_friday_window": "purchased around Black Friday", "intercept": "baseline",
}


def build_lookups(df: pd.DataFrame, snapshot) -> dict:
    """Statistics as of `snapshot`, using only deliveries completed and orders placed strictly before it."""
    snapshot = pd.Timestamp(snapshot)
    ev = df[df["delivered"] & (df["delivered_ts"] < snapshot)]
    placed = df[df["purchase_ts"] < snapshot]
    seller = ev.groupby("seller_id").agg(n=("late", "size"), late=("late", "sum"), delay=("days_vs_promise", "sum"), days=("actual_days", "sum"))
    ev60 = ev[ev["delivered_ts"] >= snapshot - 60 * DAY].groupby("seller_id").agg(n60=("late", "size"), late60=("late", "sum"))
    v30 = placed[placed["purchase_ts"] >= snapshot - 30 * DAY].groupby("seller_id").size().rename("orders_30d")
    v90 = placed[placed["purchase_ts"] >= snapshot - 90 * DAY].groupby("seller_id").size().rename("orders_90d")
    geo = df.drop_duplicates("seller_id").set_index("seller_id")[["seller_zip_code_prefix", "seller_state"]]
    sellers = geo.join(seller).join(ev60).join(v30).join(v90).fillna({"n": 0, "late": 0, "delay": 0, "days": 0, "n60": 0, "late60": 0, "orders_30d": 0, "orders_90d": 0})
    state = ev[ev["delivered_ts"] >= snapshot - 30 * DAY].groupby("customer_state").agg(n30=("late", "size"), late30=("late", "sum"))
    last14 = ev[ev["delivered_ts"] >= snapshot - 14 * DAY]
    platform = {"n14": int(len(last14)), "late14": float(last14["late"].sum()),
                "orders_7d": int((placed["purchase_ts"] >= snapshot - 7 * DAY).sum()), "orders_28d": int((placed["purchase_ts"] >= snapshot - 28 * DAY).sum())}
    return {"sellers": sellers, "state": state, "platform": platform, "snapshot": str(snapshot)}


def save_artifacts(df, X, masks, enc, lr, cal, res, hol, p_te):
    from delivery.collect import geolocation_tables      # training-only dependency, imported lazily so the serving image stays small
    M = C.MODELS
    (M / "lookups").mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, M / "model.joblib")
    joblib.dump(cal, M / "calibrator_platt_val.joblib")
    snapshot = pd.Timestamp(C.TEST_END) + DAY
    look = build_lookups(df, snapshot)
    look["sellers"].to_parquet(M / "lookups" / "sellers.parquet")
    look["state"].to_parquet(M / "lookups" / "state_recent.parquet")
    zips, states = geolocation_tables()
    zips.to_parquet(M / "lookups" / "zips.parquet")
    states.to_parquet(M / "lookups" / "states.parquet")
    hol.to_csv(M / "lookups" / "holidays_br.csv", index=False)
    np.save(M / "lookups" / "reference_scores.npy", np.sort(p_te).astype("float32"))
    te = df[masks["test"]].reset_index(drop=True)
    pt = pd.Series(p_te)
    pick = np.random.default_rng(C.SEED).choice(len(te), 60, replace=False)
    sample = [{"seller_id": te.at[i, "seller_id"], "customer_zip_prefix": int(te.at[i, "customer_zip_code_prefix"]), "n_items": int(te.at[i, "n_items"]),
               "total_price": round(float(te.at[i, "total_price"]), 2), "total_freight": round(float(te.at[i, "total_freight"]), 2),
               "payment_type": te.at[i, "payment_type"], "max_installments": int(te.at[i, "max_installments"]),
               "purchase_timestamp": te.at[i, "purchase_ts"].isoformat(), "promised_delivery_date": te.at[i, "promised_date"].date().isoformat(),
               "actual_late": bool(te.at[i, "late"]), "actual_days_vs_promise": int(te.at[i, "days_vs_promise"]), "score_at_purchase_time": round(float(pt[i]), 4)} for i in pick]
    json.dump(sample, open(M / "lookups" / "sample_orders.json", "w"), indent=1)
    final = [r for r in res["test_comparison"] if r["model"] == "Logistic regression: 29 selected features"][0]
    import sklearn
    meta = {"sklearn_version": sklearn.__version__, "model": "regularised logistic regression (C=0.1), 29 features", "features": FINAL_FEATURES, "categorical_vocab": enc.vocab, "snapshot": look["snapshot"], "platform": look["platform"],
            "training_window": {"purchases": [C.TRAIN_START, C.TRAIN_END], "model_date": C.MODEL_DATE}, "validation_window": [C.VAL_START, C.VAL_END], "test_window": [C.TEST_START, C.TEST_END],
            "test_metrics": {k: final[k] for k in ("roc_auc", "pr_auc", "capture_top10")}, "test_base_rate": res["test_base_rate"], "flag_top_percent": 10, "priors": {
                "late": C.GLOBAL_LATE_PRIOR, "seller_strength": C.SELLER_PRIOR_STRENGTH, "state_strength": C.STATE_PRIOR_STRENGTH},
            "cost_defaults": {"intervention_cost": 1.0, "late_cost": 10.0, "success_rate": 0.5}}
    json.dump(meta, open(M / "meta.json", "w"), indent=2, default=float)


class Predictor:
    def __init__(self, models_dir: Path = C.MODELS):
        self.dir = Path(models_dir)
        self.meta = json.load(open(self.dir / "meta.json"))
        import sklearn
        if self.meta.get("sklearn_version") and self.meta["sklearn_version"] != sklearn.__version__:
            import warnings
            warnings.warn(f"model trained with scikit-learn {self.meta['sklearn_version']}, running {sklearn.__version__}; install the pinned version")
        self.model = joblib.load(self.dir / "model.joblib")
        L = self.dir / "lookups"
        self.sellers = pd.read_parquet(L / "sellers.parquet")
        self.state = pd.read_parquet(L / "state_recent.parquet")
        self.zips = pd.read_parquet(L / "zips.parquet").set_index("zip_prefix")
        self.states = pd.read_parquet(L / "states.parquet").set_index("state")
        self.holidays = pd.read_csv(L / "holidays_br.csv", parse_dates=["date"])
        self.reference = np.load(L / "reference_scores.npy")
        self.platform = self.meta["platform"]
        self.enc = CategoryEncoder()
        self.enc.vocab = self.meta["categorical_vocab"]
        self.snapshot = self.meta["snapshot"]

    def known_sellers(self):
        return self.sellers.index.tolist()

    def with_lookups(self, look: dict) -> "Predictor":
        """A copy that uses other 'as of' statistics (used to test that serving matches the batch features)."""
        import copy
        other = copy.copy(self)
        other.sellers, other.state, other.platform, other.snapshot = look["sellers"], look["state"], look["platform"], look["snapshot"]
        return other

    def features(self, o: dict):
        """One-row feature frame (raw categories as strings) and a list of warnings."""
        warnings = []
        purchase, promised = pd.Timestamp(o["purchase_timestamp"]), pd.Timestamp(o["promised_delivery_date"])
        if promised.normalize() < purchase.normalize():
            raise ValueError("promised_delivery_date is before purchase_timestamp")
        zip_ = int(o["customer_zip_prefix"])
        if zip_ not in self.zips.index:
            raise ValueError(f"unknown customer_zip_prefix {zip_}")
        c_state = self.zips.at[zip_, "state"]
        sid = o["seller_id"]
        if sid in self.sellers.index:
            s = self.sellers.loc[sid]
        else:
            if "seller_zip_prefix" not in o or int(o["seller_zip_prefix"]) not in self.zips.index:
                raise ValueError("unknown seller_id: give a valid seller_zip_prefix so the seller's location is known")
            s = pd.Series({"seller_zip_code_prefix": int(o["seller_zip_prefix"]), "seller_state": self.zips.at[int(o["seller_zip_prefix"]), "state"], **{k: 0.0 for k in ("n", "late", "delay", "days", "n60", "late60", "orders_30d", "orders_90d")}})
            warnings.append("seller not seen in the history: seller features fall back to the platform prior")
        lat1, lng1 = (self.zips.loc[int(s["seller_zip_code_prefix"]), ["lat", "lng"]] if int(s["seller_zip_code_prefix"]) in self.zips.index else self.states.loc[s["seller_state"], ["lat", "lng"]])
        lat2, lng2 = self.zips.loc[zip_, ["lat", "lng"]]
        cal = calendar_features(pd.Series([purchase]), pd.Series([promised]), self.holidays).iloc[0]
        prior, m = self.meta["priors"]["late"], self.meta["priors"]["seller_strength"]
        n = float(s["n"])
        st = self.state.loc[c_state] if c_state in self.state.index else pd.Series({"n30": 0.0, "late30": 0.0})
        pl = self.platform
        price, freight = float(o["total_price"]), float(o["total_freight"])
        row = {
            "promised_days": cal["promised_days"], "promised_on_weekend": cal["promised_on_weekend"], "purchase_hour": cal["purchase_hour"], "purchase_dow": cal["purchase_dow"], "is_weekend": cal["is_weekend"],
            "n_items": o.get("n_items", 1), "n_sellers": 1, "total_price": price, "total_freight": freight, "freight_ratio": freight / (price + 1.0),
            "payment_value": o.get("payment_value", price + freight), "max_installments": o.get("max_installments", 1), "payment_type": o.get("payment_type", "credit_card"),
            "distance_km": float(haversine_km(lat1, lng1, lat2, lng2)), "same_state": int(s["seller_state"] == c_state), "customer_state": c_state, "seller_state": s["seller_state"],
            "seller_new": int(n < 10), "seller_late_rate": float(smoothed_rate(s["late"], n, prior, m)), "seller_late_rate_60d": float(smoothed_rate(s["late60"], s["n60"], prior, 5)),
            "seller_mean_days_vs_promise": float((s["delay"] + m * -11.0) / (n + m)), "seller_mean_actual_days": float((s["days"] + m * 12.0) / (n + m)),
            "seller_load_ratio": float(s["orders_30d"] / (s["orders_90d"] / 3.0 + 1.0)),
            "cust_state_late_rate_30d": float(smoothed_rate(st["late30"], st["n30"], prior, self.meta["priors"]["state_strength"])),
            "platform_late_rate_14d": float(smoothed_rate(pl["late14"], pl["n14"], prior, 50)), "platform_load_ratio": float(pl["orders_7d"] / (pl["orders_28d"] / 4.0 + 1.0)),
            "holidays_in_window": cal["holidays_in_window"], "days_to_next_holiday": cal["days_to_next_holiday"], "black_friday_window": cal["black_friday_window"]}
        frame = pd.DataFrame([row])
        frame["product_category"] = "unknown"          # not a model input; the encoder expects the column
        frame = self.enc.transform(frame, codes=False)
        return frame[FINAL_FEATURES], warnings

    def predict(self, o: dict, top_k: int = 5, flag_percent: float | None = None) -> dict:
        X, warnings = self.features(o)
        p = float(self.model.predict_proba(X)[0, 1])
        pct = float(np.searchsorted(self.reference, p) / len(self.reference) * 100)
        cut = 100 - (flag_percent if flag_percent is not None else self.meta["flag_top_percent"])
        contrib = contributions(self.model, X, FINAL_FEATURES).iloc[0].drop("intercept")
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:top_k]
        factors = [{"feature": f, "label": FEATURE_LABELS.get(f, f), "value": (X.iloc[0][f].item() if hasattr(X.iloc[0][f], "item") else X.iloc[0][f]), "log_odds_effect": round(float(v), 4),
                    "direction": "raises risk" if v > 0 else "lowers risk"} for f, v in top.items()]
        return {"late_probability": round(p, 4), "risk_percentile": round(pct, 1), "flag": bool(pct >= cut), "flag_rule": f"flag orders in the top {100 - cut:.0f}% of recent risk scores",
                "top_factors": factors, "warnings": warnings, "stats_as_of": self.snapshot}
