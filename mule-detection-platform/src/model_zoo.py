"""
MODEL ZOO — 13+ ML algorithms ek jagah
Research paper ke liye: same interface, fair comparison
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    IsolationForest,
    VotingClassifier,
    BaggingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


MODELS = {
    # ===== SIMPLE / BASELINE =====
    "lr": {
        "name": "Logistic Regression",
        "short": "LR",
        "type": "baseline",
        "make": lambda **kw: LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, **kw
        ),
    },
    "dt": {
        "name": "Decision Tree",
        "short": "DT",
        "type": "baseline",
        "make": lambda **kw: DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=42, **kw
        ),
    },
    "nb": {
        "name": "Gaussian Naive Bayes",
        "short": "NB",
        "type": "baseline",
        "make": lambda **kw: GaussianNB(**kw),
    },
    "knn": {
        "name": "K-Nearest Neighbors",
        "short": "KNN",
        "type": "baseline",
        "make": lambda **kw: KNeighborsClassifier(n_neighbors=5, **kw),
    },

    # ===== ENSEMBLE =====
    "rf": {
        "name": "Random Forest",
        "short": "RF",
        "type": "ensemble",
        "make": lambda **kw: RandomForestClassifier(
            n_estimators=300, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1, **kw
        ),
    },
    "et": {
        "name": "Extra Trees",
        "short": "ET",
        "type": "ensemble",
        "make": lambda **kw: ExtraTreesClassifier(
            n_estimators=300, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1, **kw
        ),
    },
    "gb": {
        "name": "Gradient Boosting",
        "short": "GB",
        "type": "ensemble",
        "make": lambda **kw: GradientBoostingClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.1, random_state=42, **kw
        ),
    },
    "ada": {
        "name": "AdaBoost",
        "short": "AB",
        "type": "ensemble",
        "make": lambda **kw: AdaBoostClassifier(
            n_estimators=200, learning_rate=0.1,
            random_state=42, **kw
        ),
    },
    "bag": {
        "name": "Bagging",
        "short": "Bag",
        "type": "ensemble",
        "make": lambda **kw: BaggingClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, **kw
        ),
    },

    # ===== GRADIENT BOOSTING (usually best) =====
    "xgb": {
        "name": "XGBoost",
        "short": "XGB",
        "type": "boosting",
        "make": lambda **kw: XGBClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="aucpr", random_state=42,
            use_label_encoder=False, **kw
        ),
    },
    "lgbm": {
        "name": "LightGBM",
        "short": "LGBM",
        "type": "boosting",
        "make": lambda **kw: LGBMClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42,
            verbose=-1, **kw
        ) if HAS_LIGHTGBM else None,
    },
    "cat": {
        "name": "CatBoost",
        "short": "CatBoost",
        "type": "boosting",
        "make": lambda **kw: CatBoostClassifier(
            iterations=500, depth=7, learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=42, verbose=0, **kw
        ) if HAS_CATBOOST else None,
    },

    # ===== NEURAL NETWORK =====
    "mlp": {
        "name": "MLP Neural Network",
        "short": "MLP",
        "type": "neural",
        "make": lambda **kw: MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500, random_state=42, **kw
        ),
    },

    # ===== SVM =====
    "svm_rbf": {
        "name": "SVM (RBF)",
        "short": "SVM-RBF",
        "type": "svm",
        "make": lambda **kw: SVC(
            kernel="rbf", probability=True,
            class_weight="balanced", random_state=42, **kw
        ),
    },

    # ===== ANOMALY DETECTION =====
    "iforest": {
        "name": "Isolation Forest",
        "short": "IF",
        "type": "anomaly",
        "make": lambda **kw: IsolationForest(
            n_estimators=300, contamination=0.05,
            random_state=42, **kw
        ),
    },
}


def get_model(key, **kwargs):
    if key not in MODELS:
        raise ValueError(f"'{key}' not found. Available: {list(MODELS.keys())}")
    model = MODELS[key]["make"](**kwargs)
    if model is None:
        raise ImportError(f"Install library for '{key}'")
    return model


def get_model_name(key):
    return MODELS.get(key, {}).get("name", key)


def get_model_short(key):
    return MODELS.get(key, {}).get("short", key)


def list_all():
    for key, info in MODELS.items():
        print(f"  {key:15s} | {info['short']:12s} | {info['name']}")