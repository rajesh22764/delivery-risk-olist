"""Data collection: Olist from Kaggle, plus two free public APIs (no keys): Brazilian public holidays and daily weather.

    python -m delivery.collect            # everything that is missing
    python -m delivery.collect --force    # re-download

Weather is requested per state, at the centroid of the state's Olist zip-code locations (27 requests), as daily rainfall and
maximum temperature. Only weather that is already known when an order is placed (the purchase day and the days before it) may be used as a feature.
"""
import argparse
import subprocess
import time

import pandas as pd
import requests

from delivery.config import BR_LAT, BR_LNG, EXTERNAL, RAW

HOLIDAYS_URL = "https://date.nager.at/api/v3/PublicHolidays/{year}/BR"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_START, WEATHER_END = "2016-08-01", "2018-10-31"


def download_olist(force: bool = False):
    if (RAW / "olist_orders_dataset.csv").exists() and not force:
        return
    RAW.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", "olistbr/brazilian-ecommerce", "-p", str(RAW), "--unzip", "-q"], check=True)


def collect_holidays(force: bool = False) -> pd.DataFrame:
    path = EXTERNAL / "holidays_br.csv"
    if path.exists() and not force:
        return pd.read_csv(path, parse_dates=["date"])
    rows = []
    for year in (2016, 2017, 2018, 2019):
        r = requests.get(HOLIDAYS_URL.format(year=year), timeout=30)
        r.raise_for_status()
        rows += [{"date": h["date"], "name": h["name"], "local_name": h["localName"], "global": h["global"], "types": ",".join(h["types"])} for h in r.json()]
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(path, index=False)
    return df


def geolocation_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Centroids per zip-code prefix and per state, from Olist's own geolocation file (outliers outside Brazil removed)."""
    g = pd.read_csv(RAW / "olist_geolocation_dataset.csv", dtype={"geolocation_zip_code_prefix": int})
    g = g[g.geolocation_lat.between(*BR_LAT) & g.geolocation_lng.between(*BR_LNG)]
    zips = g.groupby("geolocation_zip_code_prefix").agg(lat=("geolocation_lat", "mean"), lng=("geolocation_lng", "mean"), state=("geolocation_state", lambda s: s.mode().iat[0])).reset_index()
    zips = zips.rename(columns={"geolocation_zip_code_prefix": "zip_prefix"})
    states = zips.groupby("state").agg(lat=("lat", "mean"), lng=("lng", "mean"), n_zip=("zip_prefix", "size")).reset_index()
    return zips, states


def collect_weather(force: bool = False) -> pd.DataFrame:
    path = EXTERNAL / "weather_by_state.csv"
    if path.exists() and not force:
        return pd.read_csv(path, parse_dates=["date"])
    _, states = geolocation_tables()
    frames = []
    for _, s in states.iterrows():
        for attempt in range(5):
            r = requests.get(WEATHER_URL, params={"latitude": round(s.lat, 3), "longitude": round(s.lng, 3), "start_date": WEATHER_START, "end_date": WEATHER_END,
                                                   "daily": "precipitation_sum,temperature_2m_max", "timezone": "America/Sao_Paulo"}, timeout=60)
            if r.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        d = r.json()["daily"]
        frames.append(pd.DataFrame({"state": s.state, "date": pd.to_datetime(d["time"]), "precip_mm": d["precipitation_sum"], "tmax_c": d["temperature_2m_max"]}))
        time.sleep(1.0)    # be polite to a free API
        print(f"weather {s.state}: {len(frames[-1])} days", flush=True)
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, index=False)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    download_olist(args.force)
    h = collect_holidays(args.force)
    w = collect_weather(args.force)
    print(f"holidays: {len(h)} rows | weather: {len(w):,} rows, {w.state.nunique()} states, {w.date.min().date()} to {w.date.max().date()}, missing precip: {int(w.precip_mm.isna().sum())}")


if __name__ == "__main__":
    main()
