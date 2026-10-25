import numpy as np
import shap
import joblib
from src.config import MODEL_DIR

REASON_TEXT = {
    "fan_in_score": "Credits from many unrelated sources",
    "pass_through_ratio": "Funds passed through quickly (near 1:1)",
    "peak_hour_txns": "Unusual transaction burst within 1 hour",
    "structuring_ratio": "Amounts just below reporting thresholds",
    "unique_sources": "High number of unique credit sources",
    "unique_dests": "High number of unique debit destinations",
    "night_txn_ratio": "High proportion of late-night transactions",
    "first_in_to_out_hours": "Rapid fund movement after receiving money",
    "round_number_ratio": "Many round-number transactions",
    "counterparty_concentration": "Low counterparty diversity",
    "dist_to_bad": "Close to a complaint-linked account",
    "comm_bad_rate": "Belongs to a high-risk network community",
    "pagerank": "High centrality in fund-flow network",
    "betweenness": "Key intermediary in fund flows",
    "device_fraud_count": "Device previously linked to fraud",
    "device_reuse_risk": "Device shared across identities",
    "velocity_accel": "Transaction velocity accelerating",
}


class RiskExplainer:
    def __init__(self, model_name="behavioural"):
        self.model = joblib.load(f"{MODEL_DIR}/model_{model_name}.pkl")
        self.features = joblib.load(f"{MODEL_DIR}/features_{model_name}.pkl")
        self.explainer = shap.TreeExplainer(self.model)

    def get_reasons(self, X_row, top_k=5):
        X = X_row[self.features]
        sv = self.explainer.shap_values(X)[0]
        idx = np.argsort(-np.abs(sv))[:top_k]
        reasons = []
        for i in idx:
            feat = self.features[i]
            reasons.append({
                "feature": feat,
                "description": REASON_TEXT.get(feat, feat),
                "impact": round(float(sv[i]), 3),
                "direction": "increases risk" if sv[i] > 0 else "decreases risk",
            })
        return reasons


def format_reasons(reasons):
    return "\n".join(
        f"  • {r['description']}  ({r['direction']})"
        for r in reasons
    )