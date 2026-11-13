"""Serving must reproduce the features the model was trained on, and the predictor must behave sensibly."""
import json

import numpy as np
import pandas as pd
import pytest

from delivery import config as C
from delivery.features import FINAL_FEATURES, CATEGORICAL, build_features

pytestmark = pytest.mark.skipif(not (C.MODELS / "model.joblib").exists(), reason="run python -m delivery.pipeline first")


@pytest.fixture(scope="module")
def predictor():
    from delivery.serve import Predictor
    return Predictor()


def order_dict(row):
    return {"seller_id": row.seller_id, "customer_zip_prefix": int(row.customer_zip_code_prefix), "n_items": int(row.n_items), "total_price": float(row.total_price),
            "total_freight": float(row.total_freight), "payment_value": float(row.payment_value), "payment_type": row.payment_type, "max_installments": int(row.max_installments),
            "purchase_timestamp": row.purchase_ts.isoformat(), "promised_delivery_date": row.promised_date.date().isoformat()}


def test_serving_features_equal_batch_features(orders, predictor):
    """For real orders: lookups 'as of the purchase time' must give the same features as the batch pipeline."""
    from delivery.serve import build_lookups
    hol = pd.read_csv(C.EXTERNAL / "holidays_br.csv", parse_dates=["date"])
    wx = pd.read_csv(C.EXTERNAL / "weather_by_state.csv", parse_dates=["date"])
    batch = build_features(orders, hol, wx)
    batch = predictor.enc.transform(batch.assign(product_category="unknown"), codes=False)[FINAL_FEATURES]
    cand = orders[orders.delivered & (orders.n_sellers == 1) & (orders.purchase_ts >= C.TEST_START) & (orders.purchase_ts < "2018-08-15") & orders.max_installments.notna()]
    sample = cand.sample(6, random_state=1)
    for i, row in sample.iterrows():
        pred = predictor.with_lookups(build_lookups(orders, row.purchase_ts))
        X, _ = pred.features(order_dict(row))
        for col in FINAL_FEATURES:
            a, b = X.iloc[0][col], batch.loc[i, col]
            if col in CATEGORICAL:
                assert a == b, (i, col, a, b)
            else:
                assert np.isclose(float(a), float(b), rtol=1e-6, atol=1e-8), (i, col, a, b)


def test_predict_is_consistent_and_explanations_add_up(predictor):
    s = json.load(open(C.MODELS / "lookups" / "sample_orders.json"))[0]
    o = {k: v for k, v in s.items() if not k.startswith("actual") and k != "score_at_purchase_time"}
    r = predictor.predict(o, top_k=29)
    assert 0 <= r["late_probability"] <= 1 and 0 <= r["risk_percentile"] <= 100
    assert len(r["top_factors"]) == 29 and all(f["direction"] in ("raises risk", "lowers risk") for f in r["top_factors"])
    X, _ = predictor.features(o)
    from delivery.explain import contributions
    logit = contributions(predictor.model, X, FINAL_FEATURES).iloc[0].sum()
    assert np.isclose(1 / (1 + np.exp(-logit)), r["late_probability"], atol=1e-4)


def test_tighter_promise_is_riskier_all_else_equal(predictor):
    base = {"seller_id": predictor.known_sellers()[0], "customer_zip_prefix": 1310, "total_price": 90.0, "total_freight": 20.0, "purchase_timestamp": "2018-07-10T10:00:00"}
    short = predictor.predict({**base, "promised_delivery_date": "2018-07-18"})["late_probability"]
    medium = predictor.predict({**base, "promised_delivery_date": "2018-08-03"})["late_probability"]
    assert short != medium


def test_invalid_inputs_raise(predictor):
    ok = {"seller_id": predictor.known_sellers()[0], "customer_zip_prefix": 1310, "total_price": 10.0, "total_freight": 5.0, "purchase_timestamp": "2018-07-10T10:00:00", "promised_delivery_date": "2018-07-30"}
    with pytest.raises(ValueError):
        predictor.predict({**ok, "promised_delivery_date": "2018-07-01"})          # promise before purchase
    with pytest.raises(ValueError):
        predictor.predict({**ok, "customer_zip_prefix": 99999999})                 # unknown zip
    with pytest.raises(ValueError):
        predictor.predict({**ok, "seller_id": "no-such-seller"})                   # unknown seller without a location


def test_unknown_seller_falls_back_to_prior_with_warning(predictor):
    ok = {"seller_id": "brand-new-seller", "seller_zip_prefix": 13023, "customer_zip_prefix": 1310, "total_price": 10.0, "total_freight": 5.0,
          "purchase_timestamp": "2018-07-10T10:00:00", "promised_delivery_date": "2018-07-30"}
    r = predictor.predict(ok)
    assert any("not seen" in w for w in r["warnings"])
