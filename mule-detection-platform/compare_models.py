"""
=====================================================================
MODEL COMPARISON — Mule Account Detection Platform
=====================================================================
Saare ML algorithms try karo, compare karo, best dhoondho.

Run:
    python compare_models.py                    # saare models
    python compare_models.py --models xgb,rf,lr # sirf specific models
    python compare_models.py --features graph   # graph features pe
=====================================================================
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)

from src.config import PROCESSED_DIR, RANDOM_STATE
from src.model_zoo import get_model, get_model_name, get_model_short

# ============================================================
#  OUTPUT FOLDER
# ============================================================
RESULTS_DIR = os.path.join(PROCESSED_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
#  METRICS CALCULATE KARO
# ============================================================
def compute_metrics(y_true, y_prob, threshold=0.5):
    """Saare evaluation metrics ek jagah"""
    y_pred = (y_prob > threshold).astype(int)

    metrics = {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1-Score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "PR-AUC": round(float(average_precision_score(y_true, y_prob)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
    }

    # Precision @ K (investigation budget)
    for k in [10, 25, 50, 100]:
        k_actual = min(k, len(y_prob))
        top_k_idx = np.argsort(-y_prob)[:k_actual]
        metrics[f"P@{k}"] = round(float(np.array(y_true)[top_k_idx].mean()), 3)

    return metrics


# ============================================================
#  SINGLE MODEL TRAIN + EVALUATE
# ============================================================
def train_and_evaluate(model_key, X_train, X_test, y_train, y_test):
    """Ek model train karo, evaluate karo, results return karo"""
    name = get_model_name(model_key)
    short = get_model_short(model_key)

    start = time.time()
    model = get_model(model_key)

    # ---- Special handling for different model types ----
    if model_key == "xgb":
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model.set_params(scale_pos_weight=scale_pos)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    elif model_key == "lgbm":
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)], verbose=False)

    elif model_key == "cat":
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)], verbose=False)

    else:
        model.fit(X_train, y_train)

    elapsed = time.time() - start

    # ---- Predictions ----
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.values, y_prob)

    return model, y_prob, metrics, elapsed, short, name


# ============================================================
#  MAIN COMPARISON FUNCTION
# ============================================================
def compare_models(feature_set="behavioural", model_keys=None):
    """
    Saare models ko ek hi dataset pe train + compare karo.

    feature_set: "behavioural" ya "graph" ya "identity"
    model_keys:  list of model keys to try (None = sab try karo)
    """

    print("\n" + "=" * 80)
    print("   MODEL COMPARISON — Mule Account Detection Platform")
    print("=" * 80)

    # ============ LOAD DATA ============
    if feature_set == "behavioural":
        csv_path = os.path.join(PROCESSED_DIR, "behavioural_features.csv")
        label_col = "label"
    elif feature_set == "graph":
        csv_path = os.path.join(PROCESSED_DIR, "graph_features.csv")
        label_col = "label"
    elif feature_set == "identity":
        csv_path = os.path.join(PROCESSED_DIR, "identity_features.csv")
        label_col = "fraud_bool"
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    if not os.path.exists(csv_path):
        print(f"\n  ❌ File nahi mili: {csv_path}")
        print(f"     Pehle ye chalao: python run_pipeline.py")
        sys.exit(1)

    data = pd.read_csv(csv_path)
    print(f"\n  📂 Dataset: {csv_path}")
    print(f"  📊 Shape: {data.shape}")
    print(f"  📊 Columns: {list(data.columns[:10])}...")

    # Features + Label
    drop_cols = ["acct", label_col, "community_id"]
    FEATURES = [c for c in data.columns if c not in drop_cols]
    X = data[FEATURES]
    y = data[label_col].astype(int)

    # Handle missing
    X = X.fillna(0)

    print(f"  📊 Features: {len(FEATURES)}")
    print(f"  📊 Total samples: {len(data):,}")
    print(f"  📊 Positive (mule/fraud): {y.sum():,} ({y.mean()*100:.2f}%)")
    print(f"  📊 Negative (normal): {(y == 0).sum():,}")

    # ============ TRAIN/TEST SPLIT ============
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  📊 Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ============ MODELS TO TRY ============
    if model_keys is None:
        model_keys = [
            # Baseline models
            "lr", "dt", "nb", "knn",
            # Ensemble models
            "rf", "et", "gb", "ada", "bag",
            # Gradient boosting (best for tabular)
            "xgb", "lgbm", "cat",
            # Neural network
            "mlp",
        ]

    print(f"\n  🤖 Models to try: {len(model_keys)}")
    for k in model_keys:
        print(f"     • {get_model_name(k)}")

    # ============ TRAIN EACH MODEL ============
    print(f"\n{'─' * 80}")
    print(f"  {'Model':<20} {'PR-AUC':>8} {'ROC-AUC':>8} {'F1':>8} "
          f"{'P@50':>8} {'Time':>8}")
    print(f"{'─' * 80}")

    all_results = []
    all_predictions = {}
    trained_models = {}

    for model_key in model_keys:
        name = get_model_name(model_key)
        short = get_model_short(model_key)

        try:
            model, y_prob, metrics, elapsed, short, name = train_and_evaluate(
                model_key, X_train, X_test, y_train, y_test
            )

            # Store results
            row = {
                "Model": short,
                "Full Name": name,
                "Key": model_key,
                "PR-AUC": metrics["PR-AUC"],
                "ROC-AUC": metrics["ROC-AUC"],
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1-Score": metrics["F1-Score"],
                "P@10": metrics["P@10"],
                "P@25": metrics["P@25"],
                "P@50": metrics["P@50"],
                "P@100": metrics["P@100"],
                "Time (s)": round(elapsed, 2),
            }
            all_results.append(row)
            all_predictions[short] = (y_test.values, y_prob)
            trained_models[short] = model

            # Print row
            print(f"  {short:<20} {metrics['PR-AUC']:>8.4f} "
                  f"{metrics['ROC-AUC']:>8.4f} {metrics['F1-Score']:>8.4f} "
                  f"{metrics['P@50']:>8.3f} {elapsed:>7.1f}s")

        except Exception as e:
            err_msg = str(e)[:60]
            print(f"  {short:<20} {'❌ FAILED':>8} — {err_msg}")
            all_results.append({
                "Model": short,
                "Full Name": name,
                "Key": model_key,
                "PR-AUC": 0,
                "ROC-AUC": 0,
                "Accuracy": 0,
                "Precision": 0,
                "Recall": 0,
                "F1-Score": 0,
                "P@10": 0,
                "P@25": 0,
                "P@50": 0,
                "P@100": 0,
                "Time (s)": 0,
            })

    print(f"{'─' * 80}")

    # ============ RESULTS TABLE ============
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(
        "PR-AUC", ascending=False
    ).reset_index(drop=True)
    results_df.index = results_df.index + 1  # Rank 1 se start

    # ============ FINAL DISPLAY ============
    print(f"\n\n{'=' * 80}")
    print("   🏆 FINAL RESULTS (Sorted by PR-AUC)")
    print(f"{'=' * 80}\n")

    # Display columns for paper
    display_cols = [
        "Model", "PR-AUC", "ROC-AUC", "Accuracy",
        "Precision", "Recall", "F1-Score",
        "P@10", "P@50", "Time (s)"
    ]
    display_df = results_df[[c for c in display_cols if c in results_df.columns]]
    print(display_df.to_string())

    # ============ BEST MODEL ============
    best = results_df.iloc[0]
    print(f"\n{'=' * 80}")
    print(f"   🥇 BEST MODEL: {best['Full Name']} ({best['Model']})")
    print(f"{'=' * 80}")
    print(f"   PR-AUC:       {best['PR-AUC']:.4f}")
    print(f"   ROC-AUC:      {best['ROC-AUC']:.4f}")
    print(f"   F1-Score:     {best['F1-Score']:.4f}")
    print(f"   Precision@50: {best['P@50']:.3f}")
    print(f"   Time:         {best['Time (s)']:.1f}s")

    # ============ TOP-3 COMPARISON ============
    print(f"\n   📊 TOP 3 MODELS:")
    for i in range(min(3, len(results_df))):
        r = results_df.iloc[i]
        medal = ["🥇", "🥈", "🥉"][i]
        print(f"   {medal} {r['Model']:15s} | PR-AUC: {r['PR-AUC']:.4f} | "
              f"ROC-AUC: {r['ROC-AUC']:.4f} | P@50: {r['P@50']:.3f}")

    # ============ DETAILED REPORT FOR BEST MODEL ============
    best_key = best["Key"]
    best_short = best["Model"]

    if best_short in all_predictions:
        y_true_best, y_prob_best = all_predictions[best_short]
        y_pred_best = (y_prob_best > 0.5).astype(int)

        print(f"\n   📋 DETAILED CLASSIFICATION REPORT ({best_short}):")
        print(classification_report(
            y_true_best, y_pred_best,
            target_names=["Normal", "Mule/Fraud"],
            zero_division=0
        ))

    # ============ GENERATE GRAPHS (optional) ============
    print(f"   📊 Generating comparison graphs...")
    try:
        generate_comparison_graphs(results_df, all_predictions, y_test.values)
    except Exception as e:
        print(f"   ⚠️ Graph generation error: {e}")

    # ============ SAVE RESULTS ============
    csv_path = os.path.join(RESULTS_DIR, f"model_comparison_{feature_set}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n   💾 Results saved: {csv_path}")

    # Save best model info
    best_info = {
        "best_model": best["Key"],
        "best_model_name": best["Full Name"],
        "best_pr_auc": best["PR-AUC"],
        "best_roc_auc": best["ROC-AUC"],
        "feature_set": feature_set,
        "n_features": len(FEATURES),
        "n_samples": len(data),
    }
    best_path = os.path.join(RESULTS_DIR, "best_model_info.txt")
    with open(best_path, "w") as f:
        for k, v in best_info.items():
            f.write(f"{k}: {v}\n")
    print(f"   💾 Best model info: {best_path}")

    print(f"\n{'=' * 80}")
    print(f"   ✅ COMPARISON COMPLETE!")
    print(f"{'=' * 80}\n")

    return results_df, all_predictions, trained_models


# ============================================================
#  GRAPH GENERATION (Paper ke liye)
# ============================================================
def generate_comparison_graphs(results_df, all_predictions, y_test):
    """Research paper ke liye comparison graphs banao"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    # ---- 1. Bar Chart: PR-AUC Comparison ----
    fig, ax = plt.subplots(figsize=(12, 6))
    df = results_df.sort_values("PR-AUC", ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(df)))
    bars = ax.barh(df["Model"], df["PR-AUC"], color=colors, edgecolor="white")

    for bar, val in zip(bars, df["PR-AUC"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel("PR-AUC Score")
    ax.set_title("Model Comparison — PR-AUC (Higher is Better)")
    ax.set_xlim(0, max(df["PR-AUC"]) * 1.12)
    plt.savefig(os.path.join(RESULTS_DIR, "fig_comparison_pr_auc.png"))
    plt.close()
    print(f"     📊 Saved: fig_comparison_pr_auc.png")

    # ---- 2. Bar Chart: ROC-AUC Comparison ----
    fig, ax = plt.subplots(figsize=(12, 6))
    df = results_df.sort_values("ROC-AUC", ascending=True)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(df["Model"], df["ROC-AUC"], color=colors, edgecolor="white")

    for bar, val in zip(bars, df["ROC-AUC"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel("ROC-AUC Score")
    ax.set_title("Model Comparison — ROC-AUC")
    ax.set_xlim(0, max(df["ROC-AUC"]) * 1.12)
    plt.savefig(os.path.join(RESULTS_DIR, "fig_comparison_roc_auc.png"))
    plt.close()
    print(f"     📊 Saved: fig_comparison_roc_auc.png")

    # ---- 3. Training Time Comparison ----
    fig, ax = plt.subplots(figsize=(12, 6))
    df = results_df.sort_values("Time (s)", ascending=True)
    colors = plt.cm.Oranges(np.linspace(0.3, 0.8, len(df)))
    ax.barh(df["Model"], df["Time (s)"], color=colors, edgecolor="white")
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Training Time Comparison")
    plt.savefig(os.path.join(RESULTS_DIR, "fig_comparison_time.png"))
    plt.close()
    print(f"     📊 Saved: fig_comparison_time.png")

    # ---- 4. ROC Curves (Top 5 models) ----
    fig, ax = plt.subplots(figsize=(8, 6))
    from sklearn.metrics import roc_curve, precision_recall_curve

    top5 = results_df.head(5)["Model"].tolist()
    colors = plt.cm.Set1(np.linspace(0, 1, len(top5)))

    for name, color in zip(top5, colors):
        if name in all_predictions:
            y_true, y_prob = all_predictions[name]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{name} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Top 5 Models")
    ax.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, "fig_roc_curves.png"))
    plt.close()
    print(f"     📊 Saved: fig_roc_curves.png")

    # ---- 5. PR Curves (Top 5 models) ----
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, color in zip(top5, colors):
        if name in all_predictions:
            y_true, y_prob = all_predictions[name]
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            ax.plot(recall, precision, color=color, linewidth=2,
                    label=f'{name} (AP={ap:.3f})')

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Top 5 Models")
    ax.legend(loc="upper right")
    plt.savefig(os.path.join(RESULTS_DIR, "fig_pr_curves.png"))
    plt.close()
    print(f"     📊 Saved: fig_pr_curves.png")

    # ---- 6. Precision@K ----
    fig, ax = plt.subplots(figsize=(8, 6))
    k_values = list(range(10, min(201, len(y_test)), 10))

    for name, color in zip(top5, colors):
        if name in all_predictions:
            y_true, y_prob = all_predictions[name]
            p_at_k = []
            for k in k_values:
                k_actual = min(k, len(y_prob))
                top_k_idx = np.argsort(-y_prob)[:k_actual]
                p_at_k.append(np.array(y_true)[top_k_idx].mean())
            ax.plot(k_values, p_at_k, marker='o', color=color,
                    linewidth=2, markersize=4, label=name)

    ax.set_xlabel("Top-K Accounts (Investigation Budget)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision@K — True Mules in Top-K Alerts")
    ax.legend()
    ax.set_ylim([0, 1.05])
    plt.savefig(os.path.join(RESULTS_DIR, "fig_precision_at_k.png"))
    plt.close()
    print(f"     📊 Saved: fig_precision_at_k.png")

    # ---- 7. Multi-metric radar/bar ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, metric in enumerate(["F1-Score", "P@50", "Time (s)"]):
        df = results_df.sort_values(metric, ascending=True)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
        axes[idx].barh(df["Model"], df[metric], color=colors, edgecolor="white")
        axes[idx].set_title(metric)
        axes[idx].set_xlabel(metric)

    plt.suptitle("Multi-Metric Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_multi_metric.png"))
    plt.close()
    print(f"     📊 Saved: fig_multi_metric.png")

    print(f"\n   📊 All graphs saved in: {RESULTS_DIR}/")


# ============================================================
#  COMMAND LINE INTERFACE
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compare ML models for mule account detection"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="behavioural",
        choices=["behavioural", "graph", "identity"],
        help="Which feature set to use (default: behavioural)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model keys (e.g., xgb,rf,lr). Default: all"
    )

    args = parser.parse_args()

    # Parse model keys
    model_keys = None
    if args.models:
        model_keys = [m.strip() for m in args.models.split(",")]

    # Run comparison
    results_df, predictions, models = compare_models(
        feature_set=args.features,
        model_keys=model_keys,
    )

    # Print suggestion
    best_key = results_df.iloc[0]["Key"]
    print(f"   💡 Tip: Best model '{best_key}' use karne ke liye:")
    print(f"      python -m src.train_behavioural {best_key}")
    print(f"      (ya run_pipeline.py mein model_name='{best_key}' set karo)")
    print()


if __name__ == "__main__":
    main()