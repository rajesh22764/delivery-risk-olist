"""Feature construction. Every feature is known when the order is placed.

Groups (used for the ablation study): promise_calendar, order_payment, product, geography, seller_history, route_history,
system_lateness, holidays, weather. Historical statistics (seller, route, system) use only deliveries completed before the
order's purchase time (see history.group_stats).
"""
import numpy as np
import pandas as pd

from delivery.config import GLOBAL_LATE_PRIOR, ROUTE_PRIOR_STRENGTH, SELLER_PRIOR_STRENGTH, STATE_PRIOR_STRENGTH
from delivery.history import group_stats, smoothed_rate

DAY_NS = 24 * 3600 * 10**9
CATEGORICAL = ["customer_state", "seller_state", "product_category", "payment_type"]
GROUPS = {
    "promise_calendar": ["promised_days", "promised_on_weekend", "purchase_hour", "purchase_dow", "is_weekend"],
    "order_payment": ["n_items", "n_sellers", "total_price", "total_freight", "freight_ratio", "payment_value", "max_installments", "payment_type"],
    "product": ["product_category", "total_weight_g", "total_volume_cm3"],
    "geography": ["distance_km", "same_state", "customer_state", "seller_state"],
    "seller_history": ["seller_new", "seller_late_rate", "seller_late_rate_60d", "seller_mean_days_vs_promise", "seller_mean_actual_days", "seller_load_ratio"],
    "route_history": ["route_late_rate"],
    "system_lateness": ["cust_state_late_rate_30d", "platform_late_rate_14d", "platform_load_ratio"],
    "holidays": ["holidays_in_window", "days_to_next_holiday", "black_friday_window"],
    "weather": ["cust_precip_1d_prior", "cust_precip_7d_prior", "sell_precip_1d_prior", "sell_precip_7d_prior"],
}
ALL_FEATURES = [c for cols in GROUPS.values() for c in cols]
# Counts that grow with the business. They behave like a calendar clock (platform volume rose through 2017), which tree models
# can memorise and which then fails out of sample. Kept only to demonstrate that in the cross-validation, not used by the final model.
SCALE_COUNTS = ["seller_prior_n", "route_prior_n", "seller_orders_30d", "platform_orders_7d"]
ALL_WITH_SCALE = ALL_FEATURES + SCALE_COUNTS
# Chosen by rolling cross-validation inside the training period: weather (adds ~0.002 AUC and needs a live API when serving),
# product and route history (removing them did not hurt) are left out of the deployed model.
FINAL_GROUPS = ["promise_calendar", "order_payment", "geography", "seller_history", "system_lateness", "holidays"]
FINAL_FEATURES = [c for g in FINAL_GROUPS for c in GROUPS[g]]


def national_holidays(holidays: pd.DataFrame) -> np.ndarray:
    """Sorted national holiday dates (Carnival included; state-only holidays excluded)."""
    return np.sort(holidays.loc[holidays["global"], "date"].dt.normalize().drop_duplicates().values.astype("datetime64[D]"))


def black_friday(year: int) -> np.datetime64:
    nov = pd.date_range(f"{year}-11-01", f"{year}-11-30")
    return np.datetime64(nov[nov.day_of_week == 4][3].date())      # the fourth Friday of November


def calendar_features(purchase_ts: pd.Series, promised_date: pd.Series, holidays: pd.DataFrame) -> pd.DataFrame:
    """Calendar facts that are public in advance. Also used when serving a single order."""
    p_day = purchase_ts.dt.normalize()
    hol = national_holidays(holidays)
    pd_d, pr_d = p_day.values.astype("datetime64[D]"), promised_date.dt.normalize().values.astype("datetime64[D]")
    in_window = np.searchsorted(hol, pr_d, side="right") - np.searchsorted(hol, pd_d, side="right")          # holidays in (purchase day, promised day]
    nxt = np.searchsorted(hol, pd_d, side="right")
    next_gap = np.where(nxt < len(hol), (hol[np.minimum(nxt, len(hol) - 1)] - pd_d).astype(int), 60)
    bf = np.array([black_friday(y) for y in p_day.dt.year])
    bf_window = ((pd_d - bf).astype(int) >= -4) & ((pd_d - bf).astype(int) <= 6)
    return pd.DataFrame({
        "promised_days": (promised_date.dt.normalize() - p_day).dt.days.values,
        "promised_on_weekend": (promised_date.dt.day_of_week >= 5).astype(int).values,
        "purchase_hour": purchase_ts.dt.hour.values, "purchase_dow": purchase_ts.dt.day_of_week.values,
        "is_weekend": (purchase_ts.dt.day_of_week >= 5).astype(int).values,
        "holidays_in_window": in_window, "days_to_next_holiday": np.minimum(next_gap, 60), "black_friday_window": bf_window.astype(int),
    }, index=purchase_ts.index)


def weather_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Rain over the days BEFORE the purchase day (the purchase day's total is not yet known when the order is placed)."""
    piv = weather.pivot(index="date", columns="state", values="precip_mm").sort_index()
    lag1 = piv.shift(1).stack().rename("p1")
    roll7 = piv.shift(1).rolling(7, min_periods=1).sum().stack().rename("p7")
    both = pd.concat([lag1, roll7], axis=1).reset_index().rename(columns={"level_0": "date", "level_1": "state"})
    day = df["purchase_ts"].dt.normalize()
    out = pd.DataFrame(index=df.index)
    for prefix, col in (("cust", "customer_state"), ("sell", "seller_state")):
        m = pd.DataFrame({"date": day.values, "state": df[col].values}).merge(both, on=["date", "state"], how="left")
        out[f"{prefix}_precip_1d_prior"], out[f"{prefix}_precip_7d_prior"] = m["p1"].values, m["p7"].values
    return out


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seller, route and system statistics from deliveries completed strictly before each order's purchase time."""
    q = df["purchase_ts"].values.astype("datetime64[ns]").astype("int64")
    ev = df[df["delivered"]]
    ev_t = ev["delivered_ts"].values.astype("datetime64[ns]").astype("int64")
    vals = {"late": ev["late"].values, "delay": ev["days_vs_promise"].values, "days": ev["actual_days"].values}
    out = pd.DataFrame(index=df.index)

    seller = df["seller_id"].astype("category").cat.codes.values
    ev_seller = seller[df["delivered"].values]
    n, s = group_stats(q, seller, ev_t, ev_seller, vals)
    out["seller_prior_n"] = n
    out["seller_late_rate"] = smoothed_rate(s["late"], n, GLOBAL_LATE_PRIOR, SELLER_PRIOR_STRENGTH)
    out["seller_mean_days_vs_promise"] = (s["delay"] + SELLER_PRIOR_STRENGTH * -11.0) / (n + SELLER_PRIOR_STRENGTH)     # prior: deliveries are typically ~11 days early
    out["seller_mean_actual_days"] = (s["days"] + SELLER_PRIOR_STRENGTH * 12.0) / (n + SELLER_PRIOR_STRENGTH)
    n60, s60 = group_stats(q, seller, ev_t, ev_seller, {"late": vals["late"]}, window_ns=60 * DAY_NS)
    out["seller_late_rate_60d"] = smoothed_rate(s60["late"], n60, GLOBAL_LATE_PRIOR, 5)

    route = (df["seller_state"].astype(str) + ">" + df["customer_state"].astype(str)).astype("category").cat.codes.values
    n, s = group_stats(q, route, ev_t, route[df["delivered"].values], {"late": vals["late"]})
    out["route_prior_n"] = n
    out["route_late_rate"] = smoothed_rate(s["late"], n, GLOBAL_LATE_PRIOR, ROUTE_PRIOR_STRENGTH)

    cstate = df["customer_state"].astype("category").cat.codes.values
    n, s = group_stats(q, cstate, ev_t, cstate[df["delivered"].values], {"late": vals["late"]}, window_ns=30 * DAY_NS)
    out["cust_state_late_rate_30d"] = smoothed_rate(s["late"], n, GLOBAL_LATE_PRIOR, STATE_PRIOR_STRENGTH)
    zeros_q, zeros_e = np.zeros(len(df), dtype=int), np.zeros(len(ev), dtype=int)
    n, s = group_stats(q, zeros_q, ev_t, zeros_e, {"late": vals["late"]}, window_ns=14 * DAY_NS)
    out["platform_late_rate_14d"] = smoothed_rate(s["late"], n, GLOBAL_LATE_PRIOR, 50)

    # volume uses purchase times of ALL orders (delivered or not): what a seller / the platform has been asked to ship recently
    all_t = q
    n_seller, _ = group_stats(q, seller, all_t, seller, {"x": np.zeros(len(df))}, window_ns=30 * DAY_NS)
    n_plat, _ = group_stats(q, zeros_q, all_t, zeros_q, {"x": np.zeros(len(df))}, window_ns=7 * DAY_NS)
    n_seller90, _ = group_stats(q, seller, all_t, seller, {"x": np.zeros(len(df))}, window_ns=90 * DAY_NS)
    n_plat28, _ = group_stats(q, zeros_q, all_t, zeros_q, {"x": np.zeros(len(df))}, window_ns=28 * DAY_NS)
    out["seller_orders_30d"], out["platform_orders_7d"] = n_seller, n_plat
    # scale-free versions: current load relative to the same entity's own recent average, so they do not trend with business growth
    out["seller_load_ratio"] = n_seller / (n_seller90 / 3.0 + 1.0)
    out["platform_load_ratio"] = n_plat / (n_plat28 / 4.0 + 1.0)
    out["seller_new"] = (out["seller_prior_n"] < 10).astype(int)
    return out


def build_features(df: pd.DataFrame, holidays: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Feature table for all orders in `df` (history is computed over the whole table, strictly causally)."""
    f = pd.DataFrame(index=df.index)
    cal = calendar_features(df["purchase_ts"], df["promised_date"], holidays)
    f = f.join(cal)
    for c in ("n_items", "n_sellers", "total_price", "total_freight", "total_weight_g", "total_volume_cm3", "payment_value", "max_installments", "distance_km"):
        f[c] = df[c].values
    f["freight_ratio"] = (df["total_freight"] / (df["total_price"] + 1.0)).values
    f["same_state"] = (df["seller_state"] == df["customer_state"]).astype(int).values
    for c in CATEGORICAL:
        f[c] = df[c].values
    f = f.join(history_features(df)).join(weather_features(df, weather))
    return f[ALL_WITH_SCALE]


class CategoryEncoder:
    """Fixed vocabulary per categorical column, learned on training rows only. Rare/unseen values map to 'other'."""

    def __init__(self, min_count: int = 100):
        self.min_count = min_count
        self.vocab: dict[str, list[str]] = {}

    def fit(self, X: pd.DataFrame):
        for c in CATEGORICAL:
            vc = X[c].astype(str).value_counts()
            self.vocab[c] = sorted(vc[vc >= self.min_count].index.tolist()) + ["other"]
        return self

    def transform(self, X: pd.DataFrame, codes: bool = True) -> pd.DataFrame:
        """codes=True: integer codes (for LightGBM). codes=False: readable strings (for the logistic regression and explanations)."""
        X = X.copy()
        for c in CATEGORICAL:
            v = X[c].astype(str)
            v = v.where(v.isin(set(self.vocab[c])), "other")
            X[c] = v.map({name: i for i, name in enumerate(self.vocab[c])}).astype(int) if codes else v
        return X
