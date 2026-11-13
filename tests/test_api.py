import json

import pytest
from fastapi.testclient import TestClient

from delivery import config as C

pytestmark = pytest.mark.skipif(not (C.MODELS / "model.joblib").exists(), reason="run python -m delivery.pipeline first")


@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app)


def sample():
    s = json.load(open(C.MODELS / "lookups" / "sample_orders.json"))[0]
    return {k: v for k, v in s.items() if not k.startswith("actual") and k != "score_at_purchase_time"}


def test_health_and_model(client):
    assert client.get("/health").json()["status"] == "ok"
    m = client.get("/model").json()
    assert m["n_features"] == 29 and m["test_metrics"]["roc_auc"] > 0.6


def test_predict_returns_a_valid_response(client):
    r = client.post("/predict", json=sample())
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"late_probability", "risk_percentile", "flag", "top_factors", "warnings", "stats_as_of"}
    assert 0 <= body["late_probability"] <= 1 and len(body["top_factors"]) == 5


def test_flag_follows_the_percentile_rule(client):
    for pct in (1, 10, 50):
        b = client.post(f"/predict?flag_top_percent={pct}", json=sample()).json()
        assert b["flag"] == (b["risk_percentile"] >= 100 - pct)


@pytest.mark.parametrize("patch,status", [({"promised_delivery_date": "2010-01-01"}, 422), ({"customer_zip_prefix": 123}, 422), ({"total_price": -5}, 422),
                                          ({"seller_id": "unknown-seller"}, 422), ({"payment_type": "bitcoin"}, 422)])
def test_bad_requests_are_rejected_not_crashed(client, patch, status):
    assert client.post("/predict", json={**sample(), **patch}).status_code == status


def test_batch_reports_per_item_errors(client):
    good, bad = sample(), {**sample(), "customer_zip_prefix": 99998}
    out = client.post("/predict/batch", json=[good, bad]).json()
    assert "late_probability" in out[0] and "error" in out[1]
