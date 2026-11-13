"""Exploratory analysis: figures and the numbers quoted in the README. Uses delivered orders only (outcome known)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delivery.config import FIGURES, RAW

TEAL, GREY, RED = "#0f766e", "#94a3b8", "#b91c1c"


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=130)
    plt.close(fig)


def load_reviews() -> pd.DataFrame:
    r = pd.read_csv(RAW / "olist_order_reviews_dataset.csv", usecols=["order_id", "review_score"])
    return r.groupby("order_id", as_index=False)["review_score"].mean()


def run_eda(df: pd.DataFrame, feats: pd.DataFrame | None = None) -> dict:
    d = df[df["delivered"]].copy()
    d = d[(d["purchase_ts"] >= "2017-01-01") & (d["purchase_ts"] < "2018-09-01")]
    d["month"] = d["purchase_ts"].dt.to_period("M")
    out = {}

    # 1. lateness over time: bursty, not stable
    m = d.groupby("month").agg(orders=("late", "size"), late=("late", "mean"))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(m.index.astype(str), m["late"] * 100, color=TEAL)
    ax.set_ylabel("% of orders delivered late"); ax.set_title("Lateness comes in bursts (Black Friday 2017, Feb to Mar 2018), not at a steady rate")
    ax.tick_params(axis="x", rotation=60); ax.grid(axis="y", alpha=.3)
    _save(fig, "01_late_rate_by_month.png")
    out["late_rate_overall"] = float(d["late"].mean())
    out["late_rate_by_month"] = {str(k): round(float(v), 4) for k, v in m["late"].items()}

    # 2. promises are padded: actual delivery is far earlier than promised
    d["promised_days"] = (d["promised_date"].dt.normalize() - d["purchase_ts"].dt.normalize()).dt.days
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bins = np.arange(0, 61, 2)
    ax.hist(d["promised_days"].clip(upper=60), bins=bins, alpha=.7, color=GREY, label="promised")
    ax.hist(d["actual_days"].clip(upper=60), bins=bins, alpha=.7, color=TEAL, label="actual")
    ax.set_xlabel("days from purchase (clipped at 60)"); ax.set_ylabel("orders"); ax.legend(); ax.set_title("Promised vs actual delivery time")
    _save(fig, "02_promised_vs_actual.png")
    out["median_promised_days"] = float(d["promised_days"].median()); out["median_actual_days"] = float(d["actual_days"].median())

    # 3. where lateness lives: customer state
    s = d.groupby("customer_state").agg(orders=("late", "size"), late=("late", "mean")).query("orders >= 500").sort_values("late")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(s.index, s["late"] * 100, color=TEAL); ax.axvline(d["late"].mean() * 100, color=RED, ls="--", label="overall")
    ax.set_xlabel("% late"); ax.set_title("Late rate by customer state (states with 500+ orders)"); ax.legend()
    _save(fig, "03_late_rate_by_state.png")
    out["state_late_rate"] = {k: round(float(v), 4) for k, v in s["late"].items()}

    # 4. distance and promise length
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    db = pd.cut(d["distance_km"], [0, 100, 300, 600, 1000, 1500, 4000], include_lowest=True)
    g = d.groupby(db, observed=True)["late"].agg(["mean", "size"])
    ax[0].bar(g.index.astype(str), g["mean"] * 100, color=TEAL); ax[0].set_title("Late rate by seller-to-customer distance (km)"); ax[0].tick_params(axis="x", rotation=30); ax[0].set_ylabel("% late")
    pb = pd.cut(d["promised_days"], [0, 10, 15, 20, 25, 30, 40, 200])
    g2 = d.groupby(pb, observed=True)["late"].mean()
    ax[1].bar(g2.index.astype(str), g2 * 100, color=TEAL); ax[1].set_title("Late rate by promised days"); ax[1].tick_params(axis="x", rotation=30)
    _save(fig, "04_distance_and_promise.png")
    out["late_rate_by_distance"] = {str(k): round(float(v), 4) for k, v in g["mean"].items()}
    out["late_rate_by_promised_days"] = {str(k): round(float(v), 4) for k, v in g2.items()}

    # 5. why it matters: reviews
    rv = d.merge(load_reviews(), on="order_id", how="inner")
    rs = rv.groupby("late")["review_score"].agg(["mean", "size"])
    one_star = rv.assign(one=rv["review_score"] <= 1.5).groupby("late")["one"].mean()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(["on time", "late"], rs["mean"], color=[TEAL, RED]); ax.set_ylim(0, 5); ax.set_ylabel("average review score (1 to 5)")
    for i, v in enumerate(rs["mean"]): ax.text(i, v + .08, f"{v:.2f}", ha="center")
    ax.set_title("Late delivery costs about 2 review stars")
    _save(fig, "05_review_score_by_lateness.png")
    out["review_score_on_time"], out["review_score_late"] = float(rs.loc[0.0, "mean"]), float(rs.loc[1.0, "mean"])
    out["one_star_share_on_time"], out["one_star_share_late"] = float(one_star.loc[0.0]), float(one_star.loc[1.0])

    # 6. sellers: how concentrated is the problem?
    sl = d.groupby("seller_id").agg(orders=("late", "size"), late_orders=("late", "sum")).sort_values("late_orders", ascending=False)
    cum = sl["late_orders"].cumsum() / sl["late_orders"].sum()
    top10 = float(cum.iloc[int(len(sl) * 0.10) - 1])
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum.values * 100, color=TEAL); ax.plot([0, 100], [0, 100], color=GREY, ls="--")
    ax.set_xlabel("% of sellers (most late orders first)"); ax.set_ylabel("% of all late orders"); ax.set_title(f"Top 10% of sellers: {top10:.0%} of late orders"); ax.grid(alpha=.3)
    _save(fig, "06_seller_concentration.png")
    out["late_orders_share_top10pct_sellers"] = top10

    # 7. weather and holidays: is there any raw signal?
    if feats is not None:
        f = feats.loc[d.index]
        wb = pd.qcut(f["cust_precip_7d_prior"], 5, duplicates="drop")
        wg = d.groupby(wb, observed=True)["late"].mean()
        hg = d.groupby(f["holidays_in_window"].clip(upper=3))["late"].mean()
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        ax[0].bar([f"Q{i+1}" for i in range(len(wg))], wg.values * 100, color=TEAL); ax[0].set_title("Late rate by rain in the 7 days before purchase\n(customer state, quintiles)")
        ax[1].bar([str(i) for i in hg.index], hg.values * 100, color=TEAL); ax[1].set_title("Late rate by national holidays between purchase\nand promised date"); ax[1].set_xlabel("holidays (3 = 3 or more)")
        _save(fig, "07_weather_and_holidays.png")
        out["late_rate_by_rain_quintile"] = [round(float(v), 4) for v in wg.values]
        out["late_rate_by_holidays_in_window"] = {int(k): round(float(v), 4) for k, v in hg.items()}
    return out
