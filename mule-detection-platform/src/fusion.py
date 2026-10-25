import pandas as pd
import numpy as np
import joblib
from src.config import (
    W_BEHAVIOURAL, W_GRAPH, W_IDENTITY,
    RISK_HIGH, RISK_MEDIUM, PROCESSED_DIR, MODEL_DIR
)


def fuse_scores():
    print("\n" + "=" * 60)
    print("FUSION LAYER")
    print("=" * 60)

    behav = pd.read_csv(f"{PROCESSED_DIR}/behavioural_features.csv")
    graph = pd.read_csv(f"{PROCESSED_DIR}/graph_scores.csv")

    model_b = joblib.load(f"{MODEL_DIR}/model_behavioural.pkl")
    feats_b = joblib.load(f"{MODEL_DIR}/features_behavioural.pkl")
    behav["p_behavioural"] = model_b.predict_proba(behav[feats_b])[:, 1]

    unified = behav[["acct", "p_behavioural", "label"]].merge(
        graph[["acct", "graph_risk"]], on="acct", how="left"
    )
    unified["graph_risk"] = unified["graph_risk"].fillna(0)
    unified["p_identity"] = 0

    unified["risk_score"] = (
        W_BEHAVIOURAL * unified.p_behavioural +
        W_GRAPH * unified.graph_risk +
        W_IDENTITY * unified.p_identity
    ) * 100

    unified["band"] = pd.cut(
        unified.risk_score,
        bins=[-1, RISK_MEDIUM, RISK_HIGH, 101],
        labels=["LOW", "MEDIUM", "HIGH"]
    )

    unified = unified.sort_values("risk_score", ascending=False).reset_index(drop=True)
    unified["rank"] = unified.index + 1

    unified.to_csv(f"{PROCESSED_DIR}/unified_scores.csv", index=False)

    print(f"  📊 Scored: {len(unified):,}")
    print(f"  🔴 HIGH: {(unified.band == 'HIGH').sum():,}")
    print(f"  🟡 MEDIUM: {(unified.band == 'MEDIUM').sum():,}")
    print(f"  🟢 LOW: {(unified.band == 'LOW').sum():,}")
    print(f"\n  Top 5:")
    print(unified.head(5)[
        ["acct", "risk_score", "band", "p_behavioural", "graph_risk"]
    ].to_string())
    print(f"\n  💾 Saved: unified_scores.csv")

    return unified


if __name__ == "__main__":
    fuse_scores()