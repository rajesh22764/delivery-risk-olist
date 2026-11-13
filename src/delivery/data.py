"""Build the order-level base table from the raw Olist files.

One row per order (with at least one item), holding only what is known when the order is placed, plus outcome columns
(`delivered_ts`, `late`, `days_vs_promise`) that are used for labels and for *past* statistics, never as inputs for the same order.

Label: an order is late when it is delivered on a later calendar day than the promised date. (The promised date has no time of day,
so comparing it with the delivery timestamp would wrongly mark same-day deliveries as late: 8.1% instead of 6.8%.)
"""
import numpy as np
import pandas as pd

from delivery.collect import geolocation_tables
from delivery.config import RAW
from delivery.geo import haversine_km


def load_orders() -> pd.DataFrame:
    """All orders that have items. `delivered` marks orders with an actual delivery date (the only ones with a label)."""
    o = pd.read_csv(RAW / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
    items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    prods = pd.read_csv(RAW / "olist_products_dataset.csv")
    trans = pd.read_csv(RAW / "product_category_name_translation.csv")
    pay = pd.read_csv(RAW / "olist_order_payments_dataset.csv")
    cust = pd.read_csv(RAW / "olist_customers_dataset.csv")
    sell = pd.read_csv(RAW / "olist_sellers_dataset.csv")

    prods = prods.merge(trans, on="product_category_name", how="left")
    prods["product_category"] = prods["product_category_name_english"].fillna("unknown")
    for c in ("product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"):
        prods[c] = prods[c].fillna(prods[c].median())          # a handful of missing product dimensions
    prods["volume_cm3"] = prods.product_length_cm * prods.product_height_cm * prods.product_width_cm
    items = items.merge(prods[["product_id", "product_category", "product_weight_g", "volume_cm3"]], on="product_id", how="left")

    agg = items.groupby("order_id").agg(n_items=("order_item_id", "size"), n_sellers=("seller_id", "nunique"), total_price=("price", "sum"),
                                        total_freight=("freight_value", "sum"), total_weight_g=("product_weight_g", "sum"), total_volume_cm3=("volume_cm3", "sum"))
    primary = items.sort_values(["order_id", "price"], ascending=[True, False]).drop_duplicates("order_id")[["order_id", "seller_id", "product_category"]]

    pay_main = pay.sort_values(["order_id", "payment_value"], ascending=[True, False]).drop_duplicates("order_id")[["order_id", "payment_type"]]
    pay_agg = pay.groupby("order_id").agg(payment_value=("payment_value", "sum"), max_installments=("payment_installments", "max"))

    df = (o.merge(agg, on="order_id").merge(primary, on="order_id").merge(pay_main, on="order_id", how="left").merge(pay_agg, on="order_id", how="left")
           .merge(cust[["customer_id", "customer_zip_code_prefix", "customer_state"]], on="customer_id", how="left")
           .merge(sell[["seller_id", "seller_zip_code_prefix", "seller_state"]], on="seller_id", how="left"))
    df["payment_type"] = df["payment_type"].fillna("unknown")
    df = df.rename(columns={"order_purchase_timestamp": "purchase_ts", "order_estimated_delivery_date": "promised_date", "order_delivered_customer_date": "delivered_ts"})

    zips, states = geolocation_tables()
    zc = zips.set_index("zip_prefix")[["lat", "lng"]]
    sc = states.set_index("state")[["lat", "lng"]]
    def coords(zip_col, state_col):
        z = df[zip_col].map(zc["lat"]), df[zip_col].map(zc["lng"])
        s = df[state_col].map(sc["lat"]), df[state_col].map(sc["lng"])
        return z[0].fillna(s[0]), z[1].fillna(s[1])          # unknown zip -> state centroid
    clat, clng = coords("customer_zip_code_prefix", "customer_state")
    slat, slng = coords("seller_zip_code_prefix", "seller_state")
    df["distance_km"] = haversine_km(slat, slng, clat, clng)

    df["delivered"] = (df["order_status"] == "delivered") & df["delivered_ts"].notna()
    d_del, d_prom = df["delivered_ts"].dt.normalize(), df["promised_date"].dt.normalize()
    df["late"] = np.where(df["delivered"], (d_del > d_prom).astype("float"), np.nan)
    df["days_vs_promise"] = np.where(df["delivered"], (d_del - d_prom).dt.days, np.nan)      # positive = days late, negative = days early
    df["actual_days"] = np.where(df["delivered"], (d_del - df["purchase_ts"].dt.normalize()).dt.days, np.nan)
    return df.sort_values("purchase_ts").reset_index(drop=True)
