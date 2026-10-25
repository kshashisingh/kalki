import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.config import MODEL_DIR, RANDOM_STATE, PROCESSED_DIR


def train_graph():
    print("\n" + "=" * 60)
    print("MODEL 2: GRAPH ANOMALY (Isolation Forest)")
    print("=" * 60)

    data = pd.read_csv(f"{PROCESSED_DIR}/graph_features.csv")
    feats = data.drop(columns=["acct", "community_id"], errors="ignore")

    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=RANDOM_STATE)
    iso.fit(feats_scaled)

    raw = -iso.score_samples(feats_scaled)
    risk = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    result = data[["acct"]].copy()
    result["graph_risk"] = risk

    joblib.dump(iso, f"{MODEL_DIR}/model_graph.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler_graph.pkl")
    result.to_csv(f"{PROCESSED_DIR}/graph_scores.csv", index=False)

    print(f"  📊 Scored: {len(result):,} | Mean risk: {risk.mean():.3f}")
    print(f"  💾 Saved: model_graph.pkl")


if __name__ == "__main__":
    train_graph()