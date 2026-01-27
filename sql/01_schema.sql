CREATE TABLE IF NOT EXISTS events_raw (
  event_id       TEXT PRIMARY KEY,
  title          TEXT,
  category       TEXT,
  status         TEXT,
  start_date     TIMESTAMPTZ,
  end_date       TIMESTAMPTZ,
  updated_at     TIMESTAMPTZ,
  latitude       DOUBLE PRECISION,
  longitude      DOUBLE PRECISION,
  source         TEXT,
  raw_json       JSONB,
  inserted_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS weather_daily (
  event_id       TEXT REFERENCES events_raw(event_id) ON DELETE CASCADE,
  date           DATE NOT NULL,
  temp_mean      DOUBLE PRECISION,
  temp_max       DOUBLE PRECISION,
  temp_min       DOUBLE PRECISION,
  precipitation  DOUBLE PRECISION,
  wind_max       DOUBLE PRECISION,
  humidity_mean  DOUBLE PRECISION,
  temp_anom      DOUBLE PRECISION,
  precip_anom    DOUBLE PRECISION,
  PRIMARY KEY (event_id, date)
);

CREATE TABLE IF NOT EXISTS events_features (
  event_id              TEXT PRIMARY KEY REFERENCES events_raw(event_id) ON DELETE CASCADE,
  snapshot_date         DATE NOT NULL,
  category              TEXT,
  latitude              DOUBLE PRECISION,
  longitude             DOUBLE PRECISION,
  escalates_7d          INTEGER,
  duration_days         INTEGER,
  precip_sum_3d         DOUBLE PRECISION,
  precip_sum_7d         DOUBLE PRECISION,
  wind_max_3d           DOUBLE PRECISION,
  wind_max_7d           DOUBLE PRECISION,
  temp_anom_mean_3d     DOUBLE PRECISION,
  extreme_precip_days_7d INTEGER,
  risk_score            DOUBLE PRECISION,
  created_at            TIMESTAMPTZ DEFAULT NOW()
);
