import os
import json
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def get_db_conn():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )


FEATURE_COLS = [
    "duration_days",
    "precip_sum_3d",
    "precip_sum_7d",
    "wind_max_3d",
    "wind_max_7d",
    "extreme_precip_days_7d",
]


def load_training(view_name: str = "events_train_balanced"):
    sql = f"""
    SELECT
      event_id,
      category,
      {", ".join(FEATURE_COLS)},
      escalates_7d
    FROM {view_name};
    """
    with get_db_conn() as conn:
        df = pd.read_sql(sql, conn)

    df = pd.get_dummies(df, columns=["category"], dummy_na=True)

    y = df["escalates_7d"].astype(int)
    X = df.drop(columns=["escalates_7d", "event_id"])
    return X, y, list(X.columns)



def load_events_to_score(limit: int = 500):
    """
    Score recent events that already have engineered features.
    NOTE: We'll score from events_features (same table your dashboard will read).
    """
    sql = """
    SELECT
      event_id,
      category,
      duration_days,
      precip_sum_3d,
      precip_sum_7d,
      wind_max_3d,
      wind_max_7d,
      extreme_precip_days_7d
    FROM events_features
    WHERE duration_days IS NOT NULL
    ORDER BY snapshot_date DESC
    LIMIT %s;
    """
    with get_db_conn() as conn:
        return pd.read_sql(sql, conn, params=(limit,))


def align_columns(df_score: pd.DataFrame, model_columns: list[str]) -> pd.DataFrame:
    """
    One-hot encode category and align columns exactly to training columns.
    Missing columns -> add zeros. Extra columns -> drop.
    """
    df = df_score.copy()
    df = pd.get_dummies(df, columns=["category"], dummy_na=True)

    # Ensures all required cols exist
    for c in model_columns:
        if c not in df.columns:
            df[c] = 0

    # Keeps only model columns, in correct order
    df = df[model_columns]
    return df


def upsert_risk_scores(scores: pd.DataFrame) -> int:
    """
    Update events_features.risk_score for each event_id.
    """
    if scores.empty:
        return 0

    rows = [(row["event_id"], float(row["risk_score"])) for _, row in scores.iterrows()]

    sql = """
    UPDATE events_features AS ef
    SET risk_score = v.risk_score
    FROM (VALUES %s) AS v(event_id, risk_score)
    WHERE ef.event_id = v.event_id;
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=500)

    return len(rows)


def main():
    # Loads training data and train model (small-data friendly settings)
    X, y, model_cols = load_training("events_train_balanced")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        n_jobs=4,
    )

    model.fit(X_train, y_train)

    # Loads events to score
    df_score = load_events_to_score(limit=800)
    if df_score.empty:
        print("No events_features rows found to score. Run pipeline/03_build_features.py first.")
        return

    # Aligns columns and score
    event_ids = df_score["event_id"].copy()
    X_score = align_columns(df_score.drop(columns=["event_id"]), model_cols)


    probs = model.predict_proba(X_score)[:, 1]
    out = pd.DataFrame({"event_id": event_ids, "risk_score": probs})

    # Writes back to Postgres
    n = upsert_risk_scores(out)
    print(f"✅ Wrote risk_score for {n} events into events_features.")

    # Saves a small preview locally (optional)
    os.makedirs("ml/outputs", exist_ok=True)
    preview = out.sort_values("risk_score", ascending=False).head(10)
    with open("ml/outputs/risk_score_preview.json", "w") as f:
        json.dump(
            {
                "run_at_utc": datetime.now(timezone.utc).isoformat(),
                "top10": preview.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    print("Top 5 risk scores:")
    print(preview.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
