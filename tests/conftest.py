import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def orders():
    from delivery.config import RAW
    if not (RAW / "olist_orders_dataset.csv").exists():
        pytest.skip("Olist data not downloaded (python -m delivery.collect)")
    from delivery.data import load_orders
    return load_orders()
