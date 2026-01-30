import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

# Controls
RECENT_DAYS = int(os.getenv("RECENT_DAYS", "14"))   # focus on events started recently
EARLY_DAYS = int(os.getenv("EARLY_DAYS", "2"))      # use first 2 days only


def get_db_conn():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )


def compute_escalates_7d_from_start(start_dt: Optional[pd.Timestamp]) -> Optional[int]:
    """
    Proxy label:
    1 if event age today >= 7 days, else 0.
    """
    if start_dt is None or pd.isna(start_dt):
        return None
    today = datetime.now(timezone.utc).date()
    age_days = (today - start_dt.date()).days + 1
    return 1 if age_days >= 7 else 0


def main():
    # We only need early-stage weather (first 2 days from start_date)
    # and we only want recently started events to get some 0 labels.
    sql = """
    SELECT
      e.event_id,
      e.category,
      e.latitude,
      e.longitude,
      e.start_date,
      w.date,
      w.temp_mean,
      w.precipitation,
      w.wind_max
    FROM events_raw e
    JOIN weather_daily w ON w.event_id = e.event_id
    WHERE e.start_date IS NOT NULL
      AND (CURRENT_DATE - e.start_date::date) <= %s
      AND w.date <= (e.start_date::date + INTERVAL '1 day')
    ORDER BY e.event_id, w.date;
    """

    with get_db_conn() as conn:
        df = pd.read_sql(sql, conn, params=(RECENT_DAYS,))

    if df.empty:
        print("No joined event+weather data found for the recent window. Run ingest + enrich first.")
        return

    # Types
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")

    features = []
    for event_id, g in df.groupby("event_id"):
        g = g.sort_values("date")

        category = g["category"].iloc[0]
        lat = float(g["latitude"].iloc[0])
        lon = float(g["longitude"].iloc[0])
        start_ts = g["start_date"].iloc[0]

        # Snapshot is "day 2" (start_date + 1 day) but clamp to last available in the 2-day window
        start_date = start_ts.date() if pd.notna(start_ts) else None
        if start_date is None:
            continue

        intended_snapshot = start_date + timedelta(days=EARLY_DAYS - 1)
        available_snapshot = max(g["date"])
        snapshot_date = min(intended_snapshot, available_snapshot)

        # Early window (first 2 days)
        early = g[g["date"] <= intended_snapshot]

        # Early features
        precip_sum_2d = float(early["precipitation"].fillna(0).sum())
        wind_max_2d = float(early["wind_max"].max()) if not early["wind_max"].isna().all() else None
        temp_mean_2d = float(early["temp_mean"].mean()) if not early["temp_mean"].isna().all() else None
        extreme_precip_days_2d = int((early["precipitation"].fillna(0) >= 10.0).sum())

        escalates_7d = compute_escalates_7d_from_start(start_ts)

        # duration_days is "age so far" at snapshot time (day 2)
        duration_days_at_snapshot = (snapshot_date - start_date).days + 1

        features.append(
            {
                "event_id": event_id,
                "snapshot_date": snapshot_date,
                "category": category,
                "latitude": lat,
                "longitude": lon,
                "escalates_7d": escalates_7d,
                "duration_days": duration_days_at_snapshot,
                "precip_sum_3d": precip_sum_2d,  # keep column name stable for now
                "precip_sum_7d": precip_sum_2d,  # keep column name stable for now
                "wind_max_3d": wind_max_2d,
                "wind_max_7d": wind_max_2d,
                "temp_anom_mean_3d": None,
                "extreme_precip_days_7d": extreme_precip_days_2d,  # keep name stable
                "risk_score": None,
            }
        )

    feat_df = pd.DataFrame(features)
    print(f"Built early-stage features for {len(feat_df)} events (recent_days={RECENT_DAYS}).")

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
            execute_values(cur, upsert_sql, rows, page_size=500)

    print("✅ Upserted into events_features.")


if __name__ == "__main__":
    main()
