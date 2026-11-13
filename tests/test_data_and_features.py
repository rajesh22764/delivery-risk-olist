import numpy as np
import pandas as pd

from delivery import config as C
from delivery.features import DAY_NS, calendar_features, black_friday
from delivery.history import group_stats, smoothed_rate
from delivery.split import make_splits


def test_group_stats_uses_only_earlier_events():
    # one key; events at t = 10, 20, 30; queries at 5, 10, 25, 31
    n, s = group_stats(np.array([5, 10, 25, 31]), np.zeros(4, int), np.array([10, 20, 30]), np.zeros(3, int), {"v": np.array([1.0, 2.0, 4.0])})
    assert n.tolist() == [0, 0, 2, 3]            # strictly before the query: an event at the same instant is excluded
    assert s["v"].tolist() == [0, 0, 3, 7]


def test_group_stats_window_and_keys():
    n, s = group_stats(np.array([100, 100]), np.array([0, 1]), np.array([50, 95, 99]), np.array([0, 0, 1]), {"v": np.ones(3)}, window_ns=10)
    assert n.tolist() == [1, 1]                   # key 0: only the event at 95 is within 10 of 100; key 1: the event at 99
    assert smoothed_rate(0, 0, 0.06, 10) == 0.06  # no history -> the prior


def test_label_is_calendar_day_comparison(orders):
    d = orders[orders.delivered]
    assert abs(d["late"].mean() - 0.0677) < 0.001
    same_day = d[d["delivered_ts"].dt.normalize() == d["promised_date"].dt.normalize()]
    assert len(same_day) > 0 and (same_day["late"] == 0).all()          # delivered on the promised day is on time


def test_seller_history_matches_brute_force_and_is_causal(orders):
    from delivery.features import history_features
    feats = history_features(orders)
    rng = np.random.default_rng(0)
    sample = rng.choice(orders.index[orders.delivered], size=300, replace=False)
    for i in sample:
        o = orders.loc[i]
        prior = orders[(orders.seller_id == o.seller_id) & orders.delivered & (orders.delivered_ts < o.purchase_ts)]
        assert feats.at[i, "seller_prior_n"] == len(prior)
        expected = smoothed_rate(prior["late"].sum(), len(prior), C.GLOBAL_LATE_PRIOR, C.SELLER_PRIOR_STRENGTH)
        assert abs(feats.at[i, "seller_late_rate"] - expected) < 1e-9
        recent = prior[prior.delivered_ts >= o.purchase_ts - pd.Timedelta(days=60)]
        assert abs(feats.at[i, "seller_late_rate_60d"] - smoothed_rate(recent["late"].sum(), len(recent), C.GLOBAL_LATE_PRIOR, 5)) < 1e-9
        assert o.delivered_ts > o.purchase_ts        # an order is never part of its own history


def test_split_gaps_and_label_availability(orders):
    m = make_splits(orders)
    tr, va, te = (orders[m[k]] for k in ("train", "val", "test"))
    assert tr.purchase_ts.max() < pd.Timestamp(C.TRAIN_END) + pd.Timedelta(days=1) and va.purchase_ts.min() >= pd.Timestamp(C.VAL_START)
    assert (va.purchase_ts.min() - tr.purchase_ts.max()).days >= C.MATURITY_DAYS      # gap between train and validation purchases
    assert (te.purchase_ts.min() - va.purchase_ts.max()).days >= C.MATURITY_DAYS      # gap between validation and test purchases
    assert (tr.delivered_ts < C.MODEL_DATE).all() and (va.delivered_ts < C.TEST_START).all()      # labels known when each stage was fitted
    assert not (m["train"] & m["val"]).any() and not (m["val"] & m["test"]).any()


def test_calendar_features_are_known_in_advance():
    hol = pd.DataFrame({"date": pd.to_datetime(["2017-11-15", "2017-12-25"]), "global": [True, True]})
    purchase = pd.Series(pd.to_datetime(["2017-11-10 10:00", "2017-11-24 23:30"]))
    promised = pd.Series(pd.to_datetime(["2017-11-20", "2017-12-20"]))
    f = calendar_features(purchase, promised, hol)
    assert f["holidays_in_window"].tolist() == [1, 0]         # Nov 15 lies in (Nov 10, Nov 20]; Dec 25 is after Dec 20
    assert f["days_to_next_holiday"].tolist() == [5, 31]      # Nov 10 -> Nov 15; Nov 24 -> Dec 25
    assert f["black_friday_window"].tolist() == [0, 1]        # Black Friday 2017 was Nov 24
    assert black_friday(2017) == np.datetime64("2017-11-24")
