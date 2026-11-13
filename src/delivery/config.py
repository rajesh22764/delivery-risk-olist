"""Paths and the evaluation protocol constants (all dates are purchase dates unless stated)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
EXTERNAL = ROOT / "data" / "external"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Time protocol. Orders resolve slowly (60% of late orders are still undelivered after 4 weeks), so every evaluation
# boundary is preceded by a 45-day maturity gap: purchases in the gap are dropped from training/validation.
MATURITY_DAYS = 45
TRAIN_START, TRAIN_END = "2017-01-01", "2018-01-14"     # purchases; model date = 2018-03-01 (45 days after the last training day)
VAL_START, VAL_END = "2018-03-01", "2018-03-30"         # purchases; 45 days later (2018-05-15) the labels are known
TEST_START, TEST_END = "2018-05-15", "2018-08-21"       # purchases; never touched until the final evaluation
# Daily volume collapses after Aug 21 2018 (about 250 orders a day to 66, then 11): an extraction cutoff, not demand, and the late
# deliveries of those orders are right-censored. The tail is excluded from the test window and reported as a sensitivity check.
TEST_END_FULL = "2018-08-31"
MODEL_DATE = "2018-03-01"                               # the model sees only outcomes delivered before this date

# Smoothing of historical late rates toward a prior (pseudo-observations)
SELLER_PRIOR_STRENGTH = 10
ROUTE_PRIOR_STRENGTH = 20
STATE_PRIOR_STRENGTH = 30
GLOBAL_LATE_PRIOR = 0.057       # late rate of the training window; used only as the smoothing prior

# Brazil bounding box, used to drop geolocation outliers
BR_LAT = (-34.0, 6.0)
BR_LNG = (-74.5, -34.0)

SEED = 0
