import os
import json
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import psycopg2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier


def get_db_conn():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )


def load_data(view_name: str = "events_train_balanced"):
    sql = f"""
    SELECT
      category,
      duration_days,
      precip_sum_3d,
      precip_sum_7d,
      wind_max_3d,
      wind_max_7d,
      extreme_precip_days_7d,
      escalates_7d
    FROM {view_name};
    """
    with get_db_conn() as conn:
        df = pd.read_sql(sql, conn)

    df = pd.get_dummies(df, columns=["category"], dummy_na=True)

    y = df["escalates_7d"].astype(int)
    X = df.drop(columns=["escalates_7d"])
    return X, y


def pick_threshold(y_true, probs, target_precision: float = 0.80):
    """
    Choose a threshold that tries to hit at least target_precision (if possible),
    while keeping recall as high as possible under that constraint.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)

    # precision/recall arrays are length N+1, thresholds length N
    best = None
    for i in range(len(thresholds)):
        p = precision[i]
        r = recall[i]
        t = thresholds[i]
        if p >= target_precision:
            # keep the highest recall among thresholds meeting precision
            if best is None or r > best["recall"]:
                best = {"threshold": float(t), "precision": float(p), "recall": float(r)}

    # fallback: default 0.5 if we can't reach target precision
    if best is None:
        best = {"threshold": 0.5, "precision": float(precision[0]), "recall": float(recall[0])}
    return best


def main():
    X, y = load_data("events_train_balanced")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Small-data friendly settings (avoid huge trees)
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

    probs = model.predict_proba(X_test)[:, 1]
    pr_auc = float(average_precision_score(y_test, probs))

    # Pick a threshold (nice for CV: tune for precision)
    choice = pick_threshold(y_test, probs, target_precision=0.80)
    thr = choice["threshold"]

    preds = (probs >= thr).astype(int)

    cm = confusion_matrix(y_test, preds).tolist()
    report = classification_report(y_test, preds, output_dict=True)

    out = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_view": "events_train_balanced",
        "n_rows": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "model": {
            "type": "xgboost",
            "params": model.get_params(),
        },
        "metrics": {
            "pr_auc": pr_auc,
            "threshold": float(thr),
            "threshold_selection": choice,
            "confusion_matrix": cm,
            "classification_report": report,
        },
    }

    os.makedirs("ml/outputs", exist_ok=True)
    with open("ml/outputs/xgb_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    print("✅ Saved -> ml/outputs/xgb_metrics.json")
    print(f"xgboost: PR-AUC={pr_auc:.4f} | threshold={thr:.3f} | CM={cm}")
    print(f"threshold picked: precision={choice['precision']:.3f}, recall={choice['recall']:.3f}")


if __name__ == "__main__":
    main()
