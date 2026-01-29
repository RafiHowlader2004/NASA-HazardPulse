import os
from datetime import datetime, timezone, date, timedelta
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()


def get_db_conn():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )


def compute_escalates_7d(start_dt: Optional[datetime], end_dt: Optional[datetime]) -> Optional[int]:
    """
    Label:
      1 if total event duration >= 7 days (closed events only)
      0 if < 7 days (closed events only)
      None for ongoing events (end_dt is null) in this simple v1
    """
    if start_dt is None:
        return None
    if end_dt is None:
        return None
    duration_days = (end_dt.date() - start_dt.date()).days + 1
    return 1 if duration_days >= 7 else 0


def main():
    # Pull only events that have weather (so we can build features)
    sql = """
    SELECT
      e.event_id,
      e.category,
      e.latitude,
      e.longitude,
      e.start_date,
      e.end_date,
      w.date,
      w.temp_mean,
      w.precipitation,
      w.wind_max
    FROM events_raw e
    JOIN weather_daily w ON w.event_id = e.event_id
    ORDER BY e.event_id, w.date;
    """

    with get_db_conn() as conn:
        df = pd.read_sql(sql, conn)

    if df.empty:
        print("No joined event+weather data found. Run weather enrichment first.")
        return

    # Ensure types
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce")

    today = datetime.now(timezone.utc).date()

    # Per-event aggregations (use last available 7 days in weather_daily)
    features = []
    for event_id, g in df.groupby("event_id"):
        g = g.sort_values("date")

        category = g["category"].iloc[0]
        lat = float(g["latitude"].iloc[0])
        lon = float(g["longitude"].iloc[0])
        start_dt = g["start_date"].iloc[0].to_pydatetime() if pd.notna(g["start_date"].iloc[0]) else None
        end_dt = g["end_date"].iloc[0].to_pydatetime() if pd.notna(g["end_date"].iloc[0]) else None

        snapshot_date = max(g["date"])
        # Duration so far (if ongoing, use snapshot_date)
        if start_dt:
            end_for_duration = end_dt.date() if end_dt else snapshot_date
            duration_days = (end_for_duration - start_dt.date()).days + 1
        else:
            duration_days = None

        # Rolling windows based on available weather
        last_7 = g[g["date"] >= (snapshot_date - timedelta(days=6))]
        last_3 = g[g["date"] >= (snapshot_date - timedelta(days=2))]

        precip_sum_7d = float(last_7["precipitation"].fillna(0).sum())
        precip_sum_3d = float(last_3["precipitation"].fillna(0).sum())

        wind_max_7d = float(last_7["wind_max"].max()) if not last_7["wind_max"].isna().all() else None
        wind_max_3d = float(last_3["wind_max"].max()) if not last_3["wind_max"].isna().all() else None

        temp_anom_mean_3d = None  # we’ll add anomalies later properly

        # Simple “extreme rain day” count over 7 days (threshold v1)
        extreme_precip_days_7d = int((last_7["precipitation"].fillna(0) >= 10.0).sum())

        escalates_7d = compute_escalates_7d(start_dt, end_dt)

        features.append(
            {
                "event_id": event_id,
                "snapshot_date": snapshot_date,
                "category": category,
                "latitude": lat,
                "longitude": lon,
                "escalates_7d": escalates_7d,
                "duration_days": duration_days,
                "precip_sum_3d": precip_sum_3d,
                "precip_sum_7d": precip_sum_7d,
                "wind_max_3d": wind_max_3d,
                "wind_max_7d": wind_max_7d,
                "temp_anom_mean_3d": temp_anom_mean_3d,
                "extreme_precip_days_7d": extreme_precip_days_7d,
                "risk_score": None,  # model later
            }
        )

    feat_df = pd.DataFrame(features)
    print(f"Built features for {len(feat_df)} events.")

    # Upsert into events_features
    rows = []
    for _, r in feat_df.iterrows():
        rows.append(
            (
                r["event_id"],
                r["snapshot_date"],
                r["category"],
                r["latitude"],
                r["longitude"],
                r["escalates_7d"],
                r["duration_days"],
                r["precip_sum_3d"],
                r["precip_sum_7d"],
                r["wind_max_3d"],
                r["wind_max_7d"],
                r["temp_anom_mean_3d"],
                r["extreme_precip_days_7d"],
                r["risk_score"],
            )
        )

    upsert_sql = """
    INSERT INTO events_features (
      event_id, snapshot_date, category, latitude, longitude,
      escalates_7d, duration_days,
      precip_sum_3d, precip_sum_7d,
      wind_max_3d, wind_max_7d,
      temp_anom_mean_3d, extreme_precip_days_7d,
      risk_score
    )
    VALUES %s
    ON CONFLICT (event_id) DO UPDATE SET
      snapshot_date = EXCLUDED.snapshot_date,
      category = EXCLUDED.category,
      latitude = EXCLUDED.latitude,
      longitude = EXCLUDED.longitude,
      escalates_7d = EXCLUDED.escalates_7d,
      duration_days = EXCLUDED.duration_days,
      precip_sum_3d = EXCLUDED.precip_sum_3d,
      precip_sum_7d = EXCLUDED.precip_sum_7d,
      wind_max_3d = EXCLUDED.wind_max_3d,
      wind_max_7d = EXCLUDED.wind_max_7d,
      temp_anom_mean_3d = EXCLUDED.temp_anom_mean_3d,
      extreme_precip_days_7d = EXCLUDED.extreme_precip_days_7d,
      risk_score = EXCLUDED.risk_score;
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, upsert_sql, rows, page_size=200)

    print("✅ Upserted into events_features.")


if __name__ == "__main__":
    main()
