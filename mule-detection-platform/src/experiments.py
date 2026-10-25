"""
EXPERIMENTS — Research paper ke saare experiments
Run: python -m src.experiments
"""
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler

from src.model_zoo import get_model, get_model_name, get_model_short
from src.evaluate import (
    compute_metrics,
    plot_roc_curves,
    plot_pr_curves,
    plot_model_comparison_bar,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_shap_summary,
    plot_precision_at_k,
    generate_latex_table,
    RESULTS_DIR,
)
from src.config import PROCESSED_DIR, MODEL_DIR, RANDOM_STATE


# ============================================================
#  EXPERIMENT 1: MODEL COMPARISON (Paper Table 1)
# ============================================================
def experiment_1_model_comparison():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: MODEL COMPARISON")
    print("=" * 70)

    data = pd.read_csv(f"{PROCESSED_DIR}/behavioural_features.csv")
    FEATURES = [c for c in data.columns if c not in ("acct", "label")]
    X = data[FEATURES]
    y = data.label.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print(f"\n  Dataset: {len(data):,} accounts | {len(FEATURES)} features")
    print(f"  Mule ratio: {y.mean()*100:.2f}%")
    print(f"  Train/Test: {len(X_train):,} / {len(X_test):,}")

    # Models to compare (13 algorithms)
    model_keys = [
        "lr", "dt", "nb", "knn",
        "rf", "et", "gb", "ada", "bag",
        "xgb", "lgbm", "cat", "mlp",
    ]

    all_results = []
    all_predictions = {}  # For ROC/PR curves
    best_model = None
    best_pr_auc = 0
    best_name = ""

    for key in model_keys:
        name = get_model_name(key)
        short = get_model_short(key)
        print(f"\n  [{short:12s}] Training...", end=" ")
        start = time.time()

        try:
            model = get_model(key)

            # XGBoost special handling
            if key == "xgb":
                scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
                model.set_params(scale_pos_weight=scale_pos)
                model.fit(X_train, y_train,
                          eval_set=[(X_test, y_test)], verbose=False)
            elif key in ("lgbm", "cat"):
                model.fit(X_train, y_train,
                          eval_set=[(X_test, y_test)], verbose=False)
            else:
                model.fit(X_train, y_train)

            elapsed = time.time() - start
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test.values, y_prob)
            metrics["Model"] = short
            metrics["Full Name"] = name
            metrics["Time (s)"] = round(elapsed, 2)
            all_results.append(metrics)
            all_predictions[short] = (y_test.values, y_prob)

            print(f"✅ PR-AUC: {metrics['PR-AUC']:.4f} | "
                  f"ROC-AUC: {metrics['ROC-AUC']:.4f} | "
                  f"P@50: {metrics['P@50']:.3f} | "
                  f"{elapsed:.1f}s")

            # Track best
            if metrics['PR-AUC'] > best_pr_auc:
                best_pr_auc = metrics['PR-AUC']
                best_model = model
                best_name = short

        except Exception as e:
            print(f"❌ {str(e)[:60]}")

    # ===== RESULTS TABLE =====
    results_df = pd.DataFrame(all_results)
    col_order = ["Model", "Full Name", "Accuracy", "Precision", "Recall",
                 "F1-Score", "PR-AUC", "ROC-AUC",
                 "P@10", "P@25", "P@50", "P@100", "Time (s)"]
    results_df = results_df[[c for c in col_order if c in results_df.columns]]
    results_df = results_df.sort_values("PR-AUC", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("🏆 FINAL RESULTS TABLE (Paper Table 1)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # ===== PAPER GRAPHS =====
    print("\n  📊 Generating paper graphs...")

    # ROC Curves
    plot_roc_curves(all_predictions, "fig_roc_curves.png")

    # PR Curves
    plot_pr_curves(all_predictions, "fig_pr_curves.png")

    # Bar comparison
    plot_model_comparison_bar(results_df, "fig_model_comparison.png")

    # Precision@K
    plot_precision_at_k(all_predictions, y_test.values, "fig_precision_at_k.png")

    # Confusion matrix for best model
    best_pred = all_predictions[best_name]
    plot_confusion_matrix(
        best_pred[0], best_pred[1], best_name,
        filename="fig_confusion_matrix_best.png"
    )

    # Feature importance for best
    plot_feature_importance(
        best_model, FEATURES, best_name,
        filename="fig_feature_importance.png"
    )

    # SHAP for best
    plot_shap_summary(
        best_model, X_test.head(500), FEATURES,
        filename="fig_shap_summary.png"
    )

    # LaTeX table
    generate_latex_table(results_df, "table_model_comparison.tex")

    # Save results CSV
    results_df.to_csv(f"{RESULTS_DIR}/experiment_1_results.csv", index=False)

    # Save best model
    joblib.dump(best_model, f"{MODEL_DIR}/model_best.pkl")
    print(f"\n  🥇 Best model: {best_name} (PR-AUC: {best_pr_auc:.4f})")
    print(f"  💾 All results in: {RESULTS_DIR}/")

    return results_df, all_predictions, best_model


# ============================================================
#  EXPERIMENT 2: CROSS-VALIDATION (Paper Table 2)
# ============================================================
def experiment_2_cross_validation():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: 5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    data = pd.read_csv(f"{PROCESSED_DIR}/behavioural_features.csv")
    FEATURES = [c for c in data.columns if c not in ("acct", "label")]
    X = data[FEATURES]
    y = data.label.astype(int)

    # Top 5 models for CV (fast + accurate)
    top_models = ["xgb", "lgbm", "rf", "et", "gb"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_results = []
    for key in top_models:
        name = get_model_name(key)
        print(f"\n  [{name:12s}] 5-Fold CV...", end=" ")

        try:
            model = get_model(key)

            # PR-AUC ke liye custom scoring
            scores = cross_val_score(
                model, X, y, cv=cv,
                scoring="average_precision", n_jobs=-1
            )

            cv_results.append({
                "Model": name,
                "Mean PR-AUC": round(float(scores.mean()), 4),
                "Std PR-AUC": round(float(scores.std()), 4),
                "Min": round(float(scores.min()), 4),
                "Max": round(float(scores.max()), 4),
            })

            print(f"✅ {scores.mean():.4f} ± {scores.std():.4f}")

        except Exception as e:
            print(f"❌ {str(e)[:50]}")

    cv_df = pd.DataFrame(cv_results).sort_values(
        "Mean PR-AUC", ascending=False
    ).reset_index(drop=True)

    print("\n  📊 Cross-Validation Results (Paper Table 2):")
    print(cv_df.to_string(index=False))

    cv_df.to_csv(f"{RESULTS_DIR}/experiment_2_cv_results.csv", index=False)
    generate_latex_table(cv_df, "table_cross_validation.tex")

    return cv_df


# ============================================================
#  EXPERIMENT 3: ABLATION STUDY (Paper Table 3)
# ============================================================
def experiment_3_ablation():
    """
    Har feature group hata ke dekho — kaunsa feature group sabse important hai
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: ABLATION STUDY (Feature Groups)")
    print("=" * 70)

    data = pd.read_csv(f"{PROCESSED_DIR}/behavioural_features.csv")
    FEATURES = [c for c in data.columns if c not in ("acct", "label")]

    # Feature groups define karo
    feature_groups = {
        "All Features": FEATURES,
        "Without Fan-in/Out": [f for f in FEATURES if f not in
            ("fan_in_score", "fan_out_score", "unique_sources", "unique_dests")],
        "Without Pass-through": [f for f in FEATURES if f not in
            ("pass_through_ratio", "first_in_to_out_hours")],
        "Without Velocity": [f for f in FEATURES if f not in
            ("peak_hour_txns", "night_txn_ratio")],
        "Without Structuring": [f for f in FEATURES if f not in
            ("structuring_ratio", "round_number_ratio")],
        "Without Amount Stats": [f for f in FEATURES if f not in
            ("amount_mean", "amount_std", "amount_max", "amount_median",
             "in_amount", "out_amount")],
        "Basic Features Only": [
            "txn_count", "in_count", "out_count", "in_amount", "out_amount"
        ],
    }

    ablation_results = []

    for group_name, feat_subset in feature_groups.items():
        print(f"\n  [{group_name:30s}] Training XGBoost...", end=" ")

        X = data[feat_subset]
        y = data.label.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = get_model("xgb")
        model.set_params(scale_pos_weight=scale_pos)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.values, y_prob)

        ablation_results.append({
            "Feature Group": group_name,
            "N Features": len(feat_subset),
            "PR-AUC": round(metrics["PR-AUC"], 4),
            "ROC-AUC": round(metrics["ROC-AUC"], 4),
            "F1-Score": round(metrics["F1-Score"], 4),
            "P@50": round(metrics["P@50"], 3),
        })

        print(f"✅ PR-AUC: {metrics['PR-AUC']:.4f}")

    ablation_df = pd.DataFrame(ablation_results).sort_values(
        "PR-AUC", ascending=False
    ).reset_index(drop=True)

    print("\n  📊 Ablation Study Results (Paper Table 3):")
    print(ablation_df.to_string(index=False))

    ablation_df.to_csv(f"{RESULTS_DIR}/experiment_3_ablation.csv", index=False)
    generate_latex_table(ablation_df, "table_ablation.tex")

    return ablation_df


# ============================================================
#  EXPERIMENT 4: FUSION WEIGHT ANALYSIS (Paper Table 4)
# ============================================================
def experiment_4_fusion_weights():
    """
    Alag-alag fusion weights try karo
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: FUSION WEIGHT ANALYSIS")
    print("=" * 70)

    behav = pd.read_csv(f"{PROCESSED_DIR}/behavioural_features.csv")
    graph = pd.read_csv(f"{PROCESSED_DIR}/graph_scores.csv")

    merged = behav.merge(graph[["acct", "graph_risk"]], on="acct", how="left")
    merged = merged.fillna({"graph_risk": 0})

    # Behavioural probabilities
    model_b = joblib.load(f"{MODEL_DIR}/model_best.pkl")
    feats_b = joblib.load(f"{MODEL_DIR}/features_behavioural.pkl")
    available_feats = [f for f in feats_b if f in merged.columns]
    merged["p_behav"] = model_b.predict_proba(merged[available_feats])[:, 1]

    y = merged.label.astype(int).values

    weight_combos = [
        (0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
        (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
        (0.8, 0.2), (0.9, 0.1), (1.0, 0.0),
    ]

    fusion_results = []
    for w_b, w_g in weight_combos:
        fused = w_b * merged.p_behav.values + w_g * merged.graph_risk.values
        metrics = compute_metrics(y, fused)

        fusion_results.append({
            "W_behavioural": w_b,
            "W_graph": w_g,
            "PR-AUC": round(metrics["PR-AUC"], 4),
            "ROC-AUC": round(metrics["ROC-AUC"], 4),
            "F1": round(metrics["F1-Score"], 4),
            "P@50": round(metrics["P@50"], 3),
        })

    fusion_df = pd.DataFrame(fusion_results)

    print("\n  📊 Fusion Weight Analysis (Paper Table 4):")
    print(fusion_df.to_string(index=False))

    # Graph
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    import matplotlib.pyplot as plt
    ax.plot(fusion_df.W_behavioural, fusion_df["PR-AUC"],
            'bo-', linewidth=2, markersize=6, label='PR-AUC')
    ax.plot(fusion_df.W_behavioural, fusion_df["ROC-AUC"],
            'rs--', linewidth=2, markersize=6, label='ROC-AUC')
    ax.set_xlabel('Weight: Behavioural (1 - Graph)')
    ax.set_ylabel('Score')
    ax.set_title('Fusion Weight Analysis')
    ax.legend()
    plt.savefig(f"{RESULTS_DIR}/fig_fusion_weights.png")
    plt.close()

    fusion_df.to_csv(f"{RESULTS_DIR}/experiment_4_fusion.csv", index=False)
    generate_latex_table(fusion_df, "table_fusion_weights.tex")

    return fusion_df


# ============================================================
#  MASTER — Saare Experiments Run Karo
# ============================================================
def run_all_experiments():
    print("\n" + "🔬" * 35)
    print("   RESEARCH EXPERIMENTS — Mule Account Detection")
    print("🔬" * 35)

    # Experiment 1: Model Comparison
    results_df, predictions, best_model = experiment_1_model_comparison()

    # Experiment 2: Cross-Validation
    cv_df = experiment_2_cross_validation()

    # Experiment 3: Ablation Study
    ablation_df = experiment_3_ablation()

    # Experiment 4: Fusion Weights
    fusion_df = experiment_4_fusion_weights()

    # Final Summary
    print("\n" + "=" * 70)
    print("📊 ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"\n  📁 Results folder: {RESULTS_DIR}/")
    print(f"\n  📄 Paper files generated:")
    print(f"     ├── table_model_comparison.tex")
    print(f"     ├── table_cross_validation.tex")
    print(f"     ├── table_ablation.tex")
    print(f"     ├── table_fusion_weights.tex")
    print(f"     ├── fig_roc_curves.png")
    print(f"     ├── fig_pr_curves.png")
    print(f"     ├── fig_model_comparison.png")
    print(f"     ├── fig_precision_at_k.png")
    print(f"     ├── fig_confusion_matrix_best.png")
    print(f"     ├── fig_feature_importance.png")
    print(f"     ├── fig_shap_summary.png")
    print(f"     └── fig_fusion_weights.png")

    # Print key findings for paper
    print(f"\n  🏆 KEY FINDINGS FOR PAPER:")
    best = results_df.iloc[0]
    print(f"     • Best Model: {best['Full Name']}")
    print(f"     • PR-AUC: {best['PR-AUC']:.4f}")
    print(f"     • ROC-AUC: {best['ROC-AUC']:.4f}")
    print(f"     • Precision@50: {best['P@50']:.3f}")

    best_cv = cv_df.iloc[0]
    print(f"     • CV Score: {best_cv['Mean PR-AUC']:.4f} ± {best_cv['Std PR-AUC']:.4f}")

    best_ablation = ablation_df.iloc[0]
    worst_ablation = ablation_df.iloc[-1]
    print(f"     • Most Important Features: {best_ablation['Feature Group']}")
    print(f"     • Least Important: {worst_ablation['Feature Group']}")

    best_fusion = fusion_df.loc[fusion_df['PR-AUC'].idxmax()]
    print(f"     • Best Fusion Weights: w_behav={best_fusion['W_behavioural']}, "
          f"w_graph={best_fusion['W_graph']}")


if __name__ == "__main__":
    run_all_experiments()