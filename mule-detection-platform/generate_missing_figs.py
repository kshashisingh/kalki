"""
Generate missing figures for research paper
Run: python generate_missing_figs.py
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

RESULTS_DIR = "data/processed/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("GENERATING MISSING FIGURES")
print("=" * 60)

# ── Load data + model ──
print("\n[1/3] Loading data + model...")
data = pd.read_csv("data/processed/behavioural_features.csv")
model = joblib.load("models/model_behavioural.pkl")
features = joblib.load("models/features_behavioural.pkl")

X = data[features]
y = data.label.astype(int)

print(f"  Data: {X.shape}")
print(f"  Model: {type(model).__name__}")

# ── FIGURE 1: Feature Importance ──
print("\n[2/3] Generating feature importance chart...")

if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
elif hasattr(model, 'coef_'):
    importances = np.abs(model.coef_[0])
elif hasattr(model, 'estimators_'):
    # Bagging/ensemble — average importances
    all_imp = []
    for est in model.estimators_:
        if hasattr(est, 'feature_importances_'):
            all_imp.append(est.feature_importances_)
    if all_imp:
        importances = np.mean(all_imp, axis=0)
    else:
        importances = np.zeros(len(features))
else:
    importances = np.zeros(len(features))
    print("  ⚠️ Model has no feature_importances_ — using zeros")

top_n = 15
idx = np.argsort(importances)[-top_n:]

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(idx)))
ax.barh(
    [features[i].replace('_', ' ') for i in idx],
    importances[idx],
    color=colors,
    edgecolor='white',
    linewidth=0.5
)
ax.set_xlabel('Feature Importance')
ax.set_title(f'Top {top_n} Features — Mule Account Detection')
ax.tick_params(axis='y', labelsize=11)

plt.savefig(f"{RESULTS_DIR}/fig_feature_importance.png")
plt.close()
print(f"  ✅ Saved: fig_feature_importance.png")

# ── FIGURE 2: SHAP Summary ──
print("\n[3/3] Generating SHAP summary plot...")

try:
    import shap

    # Sample 500 rows for speed
    X_sample = X.sample(min(500, len(X)), random_state=42)

    # TreeExplainer works for tree-based models
    if type(model).__name__ in ('XGBClassifier', 'GradientBoostingClassifier',
                                 'RandomForestClassifier', 'ExtraTreesClassifier',
                                 'DecisionTreeClassifier'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        # For Bagging or other models, try generic
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=features,
        show=False,
        max_display=15,
        plot_size=(10, 8)
    )
    plt.title('SHAP Feature Impact — Mule Account Detection', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/fig_shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: fig_shap_summary.png")

except Exception as e:
    print(f"  ⚠️ SHAP error: {e}")
    print(f"  Generating fallback feature importance plot...")

    # Fallback: correlation-based importance
    fig, ax = plt.subplots(figsize=(10, 7))
    correlations = X_sample.corrwith(y.iloc[X_sample.index] if hasattr(y.iloc[X_sample.index], 'corrwith') else y.iloc[X_sample.index]).abs().sort_values(ascending=True).tail(15)

    if len(correlations) > 0:
        correlations.plot.barh(ax=ax, color=plt.cm.RdYlGn(np.linspace(0.3, 0.8, len(correlations))))
        ax.set_xlabel('Absolute Correlation with Mule Label')
        ax.set_title('Feature Correlation — Mule Account Detection')
    else:
        ax.text(0.5, 0.5, 'SHAP not available\nSee feature importance chart',
                ha='center', va='center', fontsize=16)

    plt.savefig(f"{RESULTS_DIR}/fig_shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: fig_shap_summary.png (fallback)")

# ── Verify all figures ──
print(f"\n{'=' * 60}")
print("ALL FIGURES IN results/:")
print(f"{'=' * 60}")
for f in sorted(os.listdir(RESULTS_DIR)):
    if f.endswith('.png'):
        size = os.path.getsize(os.path.join(RESULTS_DIR, f))
        print(f"  ✅ {f:45s} ({size:,} bytes)")

print(f"\n✅ DONE!")