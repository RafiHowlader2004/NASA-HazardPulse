import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional, List, Tuple

import pandas as pd
import requests
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def get_db_conn():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )


def clamp_date_window(start_dt: Optional[datetime], end_dt: Optional[datetime]) -> Tuple[date, date]:
    """
    Create a safe date window [start, end] for weather fetching.
    - If end_dt is missing: use today
    - Cap to last 30 days in v1 (keeps it fast and avoids huge pulls)
    """
    today = datetime.now(timezone.utc).date()

    if start_dt is None:
        start = today - timedelta(days=7)
    else:
        start = start_dt.date()

    if end_dt is None:
        end = today
    else:
        end = end_dt.date()

    if end < start:
        start, end = end, start

    # Cap to last 30 days to keep first version simple + quick
    if (end - start).days > 30:
        start = end - timedelta(days=30)

    return start, end


def fetch_events_to_enrich(limit: int = 20) -> List[Tuple[str, float, float, Optional[datetime], Optional[datetime]]]:
    """
    Pull a small batch of latest events with coordinates.
    We'll scale later.
    """
    sql = """
    SELECT event_id, latitude, longitude, start_date, end_date
    FROM events_raw
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    ORDER BY updated_at DESC
    LIMIT %s;
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            return cur.fetchall()


def fetch_open_meteo_daily(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """
    Fetch daily weather from Open-Meteo.
    Returns columns compatible with weather_daily.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "timezone": "UTC",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        return pd.DataFrame()

    df = pd.DataFrame(daily)
    df.rename(
        columns={
            "time": "date",
            "temperature_2m_mean": "temp_mean",
            "temperature_2m_max": "temp_max",
            "temperature_2m_min": "temp_min",
            "precipitation_sum": "precipitation",
            "windspeed_10m_max": "wind_max",
        },
        inplace=True,
    )
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Keep schema-aligned columns only
    return df[["date", "temp_mean", "temp_max", "temp_min", "precipitation", "wind_max"]]


def upsert_weather(event_id: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    rows = []
    for _, r in df.iterrows():
        rows.append(
            (
                event_id,
                r["date"],
                float(r["temp_mean"]) if pd.notna(r["temp_mean"]) else None,
                float(r["temp_max"]) if pd.notna(r["temp_max"]) else None,
                float(r["temp_min"]) if pd.notna(r["temp_min"]) else None,
                float(r["precipitation"]) if pd.notna(r["precipitation"]) else None,
                float(r["wind_max"]) if pd.notna(r["wind_max"]) else None,
                None,  # humidity_mean (we'll add later)
                None,  # temp_anom (later)
                None,  # precip_anom (later)
            )
        )

    sql = """
    INSERT INTO weather_daily (
      event_id, date, temp_mean, temp_max, temp_min, precipitation, wind_max,
      humidity_mean, temp_anom, precip_anom
    )
    VALUES %s
    ON CONFLICT (event_id, date) DO UPDATE SET
      temp_mean = EXCLUDED.temp_mean,
      temp_max = EXCLUDED.temp_max,
      temp_min = EXCLUDED.temp_min,
      precipitation = EXCLUDED.precipitation,
      wind_max = EXCLUDED.wind_max,
      humidity_mean = EXCLUDED.humidity_mean,
      temp_anom = EXCLUDED.temp_anom,
      precip_anom = EXCLUDED.precip_anom;
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=200)

    return len(rows)


def main():
    events = fetch_events_to_enrich(limit=20)
    if not events:
        print("No events with lat/lon found. (Unexpected) Check events_raw.")
        return

    total_rows = 0
    for (event_id, lat, lon, start_dt, end_dt) in events:
        start, end = clamp_date_window(start_dt, end_dt)
        df = fetch_open_meteo_daily(lat, lon, start, end)
        n = upsert_weather(event_id, df)
        total_rows += n
        print(f"{event_id}: saved {n} daily rows ({start} -> {end})")

    print(f"✅ Done. Inserted/updated {total_rows} rows into weather_daily.")


if __name__ == "__main__":
    main()
