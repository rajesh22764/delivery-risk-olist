"""Train / validation / test masks with maturity gaps (see config).

train : purchased TRAIN_START..TRAIN_END, and delivered before MODEL_DATE (the label must have been known when the model was fitted)
val   : purchased VAL_START..VAL_END, and delivered before TEST_START (known when the calibrator and threshold were fitted)
test  : purchased TEST_START..TEST_END (outcomes used only for the final evaluation)
Purchases in the gaps between these windows are dropped, so almost every training/validation order has resolved by the next boundary.
"""
import pandas as pd

from delivery import config as C


def make_splits(df: pd.DataFrame) -> dict:
    p, deliv = df["purchase_ts"], df["delivered"]
    end = lambda d: pd.Timestamp(d) + pd.Timedelta(days=1)
    train = deliv & (p >= C.TRAIN_START) & (p < end(C.TRAIN_END)) & (df["delivered_ts"] < C.MODEL_DATE)
    val = deliv & (p >= C.VAL_START) & (p < end(C.VAL_END)) & (df["delivered_ts"] < C.TEST_START)
    test = deliv & (p >= C.TEST_START) & (p < end(C.TEST_END))
    test_full = deliv & (p >= C.TEST_START) & (p < end(C.TEST_END_FULL))      # sensitivity check only: includes the truncated tail
    return {"train": train, "val": val, "test": test, "test_full": test_full}


def split_report(df: pd.DataFrame, masks: dict) -> pd.DataFrame:
    rows = []
    for name, m in masks.items():
        if name == "test_full":
            continue
        d = df[m]
        rows.append({"split": name, "orders": len(d), "late_orders": int(d["late"].sum()), "late_rate": d["late"].mean(),
                     "first_purchase": d["purchase_ts"].min().date(), "last_purchase": d["purchase_ts"].max().date()})
    return pd.DataFrame(rows)
