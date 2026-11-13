"""FastAPI service: POST /predict scores one order for late-delivery risk.

    uvicorn app.main:app --port 8000        (from the repository root, with PYTHONPATH=src)
"""
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from delivery.serve import Predictor  # noqa: E402

predictor = Predictor()
app = FastAPI(title="Late-delivery risk API", version="1.0.0",
              description="Estimates, at the moment an order is placed, how likely it is to arrive after its promised date. "
                          "Trained on Olist orders (2017 to 2018). Offline study, not a production system.")


class OrderIn(BaseModel):
    seller_id: str = Field(..., description="Olist seller id (see GET /sample-orders for examples)")
    customer_zip_prefix: int = Field(..., ge=1000, le=99999, description="First 5 digits of the customer's zip code")
    seller_zip_prefix: int | None = Field(None, description="Only needed for a seller not seen in the history")
    n_items: int = Field(1, ge=1, le=50)
    total_price: float = Field(..., ge=0)
    total_freight: float = Field(..., ge=0)
    payment_value: float | None = Field(None, ge=0)
    payment_type: Literal["credit_card", "boleto", "voucher", "debit_card", "not_defined"] = "credit_card"
    max_installments: int = Field(1, ge=1, le=24)
    purchase_timestamp: datetime
    promised_delivery_date: date

    model_config = {"json_schema_extra": {"examples": [{
        "seller_id": "4a3ca9315b744ce9f8e9374361493884", "customer_zip_prefix": 1310, "n_items": 1, "total_price": 89.9, "total_freight": 22.5,
        "payment_type": "credit_card", "max_installments": 3, "purchase_timestamp": "2018-08-20T14:30:00", "promised_delivery_date": "2018-09-12"}]}}


class Factor(BaseModel):
    feature: str
    label: str
    value: float | int | str
    log_odds_effect: float
    direction: str


class PredictionOut(BaseModel):
    late_probability: float = Field(..., description="Model probability that the order is delivered after the promised date")
    risk_percentile: float = Field(..., description="Where this score ranks among recent orders (100 = riskiest)")
    flag: bool = Field(..., description="True when the order is in the riskiest slice (default top 10%)")
    flag_rule: str
    top_factors: list[Factor]
    warnings: list[str]
    stats_as_of: str


@app.get("/health")
def health():
    return {"status": "ok", "stats_as_of": predictor.snapshot}


@app.get("/model")
def model_info():
    m = predictor.meta
    return {"model": m["model"], "n_features": len(m["features"]), "features": m["features"], "training_window": m["training_window"], "test_window": m["test_window"],
            "test_metrics": m["test_metrics"], "test_base_rate": m["test_base_rate"], "stats_as_of": m["snapshot"],
            "caveats": ["probabilities are only calibrated when conditions resemble the training period; use the percentile for decisions",
                        "ranking quality decays over months; retrain regularly"]}


@app.get("/sample-orders")
def sample_orders(n: int = Query(5, ge=1, le=60)):
    import json
    return json.load(open(predictor.dir / "lookups" / "sample_orders.json"))[:n]


@app.post("/predict", response_model=PredictionOut)
def predict(order: OrderIn, flag_top_percent: float | None = Query(None, ge=1, le=50, description="Flag the riskiest X% (default 10)"), top_k: int = Query(5, ge=1, le=15)):
    try:
        payload = order.model_dump(mode="json")
        payload = {k: v for k, v in payload.items() if v is not None}
        return predictor.predict(payload, top_k=top_k, flag_percent=flag_top_percent)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch")
def predict_batch(orders: list[OrderIn], flag_top_percent: float | None = Query(None, ge=1, le=50)):
    if len(orders) > 500:
        raise HTTPException(status_code=413, detail="at most 500 orders per request")
    out = []
    for i, order in enumerate(orders):
        try:
            payload = {k: v for k, v in order.model_dump(mode="json").items() if v is not None}
            r = predictor.predict(payload, top_k=3, flag_percent=flag_top_percent)
            out.append({"index": i, "late_probability": r["late_probability"], "risk_percentile": r["risk_percentile"], "flag": r["flag"]})
        except ValueError as e:
            out.append({"index": i, "error": str(e)})
    return out
