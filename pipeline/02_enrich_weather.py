import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional, List, Tuple

import pandas as pd
import requests
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# --- Controls for building a good ML dataset ---
RECENT_DAYS = int(os.getenv("RECENT_DAYS", "14"))          # focus on new-ish events
FIRST_N_DAYS = int(os.getenv("FIRST_N_DAYS", "2"))         # leakage-safe window
FORCE_REFRESH_RECENT = os.getenv("FORCE_REFRESH_RECENT", "1") == "1"
LIMIT_EVENTS = int(os.getenv("ENRICH_LIMIT", "1200"))      # can be higher after bigger ingest


def get_db_conn():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )


def clamp_to_first_days(start_dt: Optional[datetime], end_dt: Optional[datetime], first_n_days: int = 2) -> Tuple[date, date]:
    today = datetime.now(timezone.utc).date()

    if start_dt is None:
        end = today
        start = today - timedelta(days=1)
        return start, end

    start = start_dt.date()
    end_candidate = start + timedelta(days=first_n_days - 1)

    if end_dt is not None:
        end = min(end_dt.date(), end_candidate)
    else:
        end = min(today, end_candidate)

    if end < start:
        start, end = end, start

    return start, end


def fetch_events_to_enrich(limit: int = 800, recent_days: int = 14, force_refresh_recent: bool = False):
    """
    Two modes:
    - force_refresh_recent=False: only events missing weather_daily rows
    - force_refresh_recent=True: refresh events that started in last `recent_days`
    """
    if force_refresh_recent:
        sql = """
        SELECT event_id, latitude, longitude, start_date, end_date
        FROM events_raw
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
          AND start_date IS NOT NULL
          AND (CURRENT_DATE - start_date::date) <= %s
        ORDER BY start_date DESC
        LIMIT %s;
        """
        params = (recent_days, limit)
    else:
        sql = """
        WITH candidates AS (
          SELECT
            e.event_id,
            e.latitude,
            e.longitude,
            e.start_date,
            e.end_date
          FROM events_raw e
          WHERE e.latitude IS NOT NULL AND e.longitude IS NOT NULL
            AND e.start_date IS NOT NULL
            AND NOT EXISTS (
              SELECT 1 FROM weather_daily w WHERE w.event_id = e.event_id
            )
        )
        SELECT event_id, latitude, longitude, start_date, end_date
        FROM candidates
        ORDER BY start_date DESC
        LIMIT %s;
        """
        params = (limit,)

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def delete_weather_for_events(event_ids: List[str]) -> None:
    """Optional cleanup: keep the refresh clean and deterministic."""
    if not event_ids:
        return
    sql = "DELETE FROM weather_daily WHERE event_id = ANY(%s);"
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (event_ids,))
    # commit happens on context manager exit


def fetch_open_meteo_daily(lat: float, lon: float, start: date, end: date, retries: int = 3) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "timezone": "UTC",
    }

    last_err = None
    for attempt in range(retries):
        try:
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
            return df[["date", "temp_mean", "temp_max", "temp_min", "precipitation", "wind_max"]]
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))

    raise last_err


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
                None,
                None,
                None,
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
    events = fetch_events_to_enrich(
        limit=LIMIT_EVENTS,
        recent_days=RECENT_DAYS,
        force_refresh_recent=FORCE_REFRESH_RECENT,
    )

    if not events:
        print("No events to enrich for current mode/settings.")
        return

    if FORCE_REFRESH_RECENT:
        event_ids = [e[0] for e in events]
        print(f"Force-refresh enabled: deleting existing weather for {len(event_ids)} events...")
        delete_weather_for_events(event_ids)

    total_rows = 0
    for (event_id, lat, lon, start_dt, end_dt) in events:
        start, end = clamp_to_first_days(start_dt, end_dt, first_n_days=FIRST_N_DAYS)
        df = fetch_open_meteo_daily(lat, lon, start, end)
        n = upsert_weather(event_id, df)
        total_rows += n
        print(f"{event_id}: saved {n} daily rows ({start} -> {end})")
        time.sleep(0.1)

    print(f"✅ Done. Inserted/updated {total_rows} rows into weather_daily.")


if __name__ == "__main__":
    main()
