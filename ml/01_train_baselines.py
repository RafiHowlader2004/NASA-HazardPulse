import os
import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import psycopg2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_db_conn():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )


def load_data():
    sql = """
    SELECT
      category,
      duration_days,
      precip_sum_3d,
      precip_sum_7d,
      wind_max_3d,
      wind_max_7d,
      extreme_precip_days_7d,
      escalates_7d
    FROM events_train_balanced;
    """
    with get_db_conn() as conn:
        df = pd.read_sql(sql, conn)

    # one-hot encode category
    df = pd.get_dummies(df, columns=["category"], dummy_na=True)

    y = df["escalates_7d"].astype(int)
    X = df.drop(columns=["escalates_7d"])

    return X, y


def evaluate(name, model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    ap = float(average_precision_score(y_test, probs))
    cm = confusion_matrix(y_test, preds).tolist()
    report = classification_report(y_test, preds, output_dict=True)

    return {
        "model": name,
        "pr_auc": ap,
        "confusion_matrix": cm,
        "report": report,
    }


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    results = []

    # Baseline 1: Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    results.append(evaluate("logistic_regression", lr, X_test, y_test))

    # Baseline 2: Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate("random_forest", rf, X_test, y_test))

    out = {
        "run_at_utc": datetime.utcnow().isoformat(),
        "n_rows": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "results": results,
    }

    os.makedirs("ml/outputs", exist_ok=True)
    with open("ml/outputs/baselines.json", "w") as f:
        json.dump(out, f, indent=2)

    print("✅ Saved metrics -> ml/outputs/baselines.json")
    for r in results:
        print(f"{r['model']}: PR-AUC={r['pr_auc']:.4f} | CM={r['confusion_matrix']}")


if __name__ == "__main__":
    main()
