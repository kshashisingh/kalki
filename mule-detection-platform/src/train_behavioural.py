import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from src.config import MODEL_DIR, RANDOM_STATE, PROCESSED_DIR


def train_behavioural():
    print("\n" + "=" * 60)
    print("MODEL 1: BEHAVIOURAL (XGBoost)")
    print("=" * 60)

    data = pd.read_csv(f"{PROCESSED_DIR}/behavioural_features.csv")
    FEATURES = [c for c in data.columns if c not in ("acct", "label")]
    print(f"  Features: {len(FEATURES)}")
    print(f"  Accounts: {len(data):,} | Mules: {data.label.sum():,}")

    X = data[FEATURES]
    y = data.label.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, eval_metric="aucpr",
        early_stopping_rounds=30, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    prob = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, prob)
    roc_auc = roc_auc_score(y_test, prob)

    print(f"\n  📊 PR-AUC:  {pr_auc:.4f}")
    print(f"  📊 ROC-AUC: {roc_auc:.4f}")
    for k in [10, 25, 50, 100]:
        top_k = np.argsort(-prob)[:k]
        print(f"  📊 Precision@{k}: {y_test.iloc[top_k].mean():.3f}")

    print(classification_report(y_test, (prob > 0.5).astype(int), zero_division=0))

    joblib.dump(model, f"{MODEL_DIR}/model_behavioural.pkl")
    joblib.dump(FEATURES, f"{MODEL_DIR}/features_behavioural.pkl")
    print(f"  💾 Saved: model_behavioural.pkl")


if __name__ == "__main__":
    train_behavioural()