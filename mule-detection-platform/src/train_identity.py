import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from src.config import MODEL_DIR, RANDOM_STATE, PROCESSED_DIR


def train_identity():
    print("\n" + "=" * 60)
    print("MODEL 3: IDENTITY FRAUD (XGBoost)")
    print("=" * 60)

    data = pd.read_csv(f"{PROCESSED_DIR}/identity_features.csv")
    if "fraud_bool" not in data.columns:
        print("  ⚠️ fraud_bool not found — skipping")
        return

    FEATURES = [c for c in data.columns if c != "fraud_bool"]
    X = data[FEATURES]
    y = data.fraud_bool.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, eval_metric="aucpr",
        early_stopping_rounds=30, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    prob = model.predict_proba(X_test)[:, 1]
    print(f"  📊 PR-AUC:  {average_precision_score(y_test, prob):.4f}")
    print(f"  📊 ROC-AUC: {roc_auc_score(y_test, prob):.4f}")

    joblib.dump(model, f"{MODEL_DIR}/model_identity.pkl")
    joblib.dump(FEATURES, f"{MODEL_DIR}/features_identity.pkl")
    print(f"  💾 Saved: model_identity.pkl")


if __name__ == "__main__":
    train_identity()