import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

EONET_EVENTS_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"


def parse_iso_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def pick_lat_lon(event: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Pick a representative (lat, lon) from the latest geometry.
    EONET uses GeoJSON coordinates = [lon, lat]
    """
    geoms = event.get("geometry", [])
    if not geoms:
        return None, None

    latest = geoms[-1]
    g = latest.get("geometry", {})
    gtype = g.get("type")
    coords = g.get("coordinates")

    if not coords:
        return None, None

    if gtype == "Point":
        lon, lat = coords
        return float(lat), float(lon)

    # Simple fallback for LineString / Polygon
    try:
        # Polygon: [ [ [lon,lat], ... ] , ... ]
        if isinstance(coords[0], list) and isinstance(coords[0][0], list):
            lon, lat = coords[0][0]
            return float(lat), float(lon)

        # LineString: [ [lon,lat], ... ]
        if isinstance(coords[0], list) and len(coords[0]) == 2:
            lon, lat = coords[0]
            return float(lat), float(lon)
    except Exception:
        return None, None

    return None, None


def fetch_eonet_events(days: int = 90, limit: int = 500) -> List[Dict[str, Any]]:
    params = {"days": days, "limit": limit, "status": "all"}
    r = requests.get(EONET_EVENTS_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("events", [])


def get_db_conn():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )


def upsert_events(events: List[Dict[str, Any]]) -> int:
    now = datetime.now(timezone.utc)
    rows = []

    for e in events:
        event_id = e.get("id")
        title = e.get("title")

        categories = e.get("categories", [])
        category = categories[0]["title"] if categories else None

        status = e.get("status")

        sources = e.get("sources", [])
        source = sources[0]["url"] if sources else None

        geoms = e.get("geometry", [])
        geom_dates = [parse_iso_dt(g.get("date")) for g in geoms if g.get("date")]
        geom_dates = [d for d in geom_dates if d is not None]

        start_date = min(geom_dates) if geom_dates else None
        end_date = max(geom_dates) if (geom_dates and status == "closed") else None

        updated_at = parse_iso_dt(e.get("updated")) if e.get("updated") else now

        lat, lon = pick_lat_lon(e)

        rows.append(
            (
                event_id,
                title,
                category,
                status,
                start_date,
                end_date,
                updated_at,
                lat,
                lon,
                source,
                json.dumps(e),
            )
        )

    if not rows:
        return 0

    sql = """
    INSERT INTO events_raw (
      event_id, title, category, status, start_date, end_date, updated_at,
      latitude, longitude, source, raw_json
    )
    VALUES %s
    ON CONFLICT (event_id) DO UPDATE SET
      title = EXCLUDED.title,
      category = EXCLUDED.category,
      status = EXCLUDED.status,
      start_date = EXCLUDED.start_date,
      end_date = EXCLUDED.end_date,
      updated_at = EXCLUDED.updated_at,
      latitude = EXCLUDED.latitude,
      longitude = EXCLUDED.longitude,
      source = EXCLUDED.source,
      raw_json = EXCLUDED.raw_json;
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=200)

    return len(rows)


def main():
    print("Fetching NASA EONET events...")
    events = fetch_eonet_events(days=90, limit=500)
    print(f"Fetched {len(events)} events. Inserting into Postgres...")

    n = upsert_events(events)
    print(f"✅ Upserted {n} events into events_raw.")


if __name__ == "__main__":
    main()
