"""
EVALUATE — Saare metrics + graphs jo research paper mein chahiye
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI ke bina kaam karega
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from src.config import PROCESSED_DIR

# Professional style for paper-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

RESULTS_DIR = f"{PROCESSED_DIR}/results"
import os
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Saare metrics ek jagah calculate karo"""
    y_pred = (y_prob > threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }

    # Precision @ K (investigation budget ke liye)
    for k in [10, 25, 50, 100]:
        top_k = np.argsort(-y_prob)[:k]
        metrics[f"P@{k}"] = np.array(y_true)[top_k].mean()

    return metrics


def plot_roc_curves(results_dict, filename="roc_curves.png"):
    """
    Sabhi models ka ROC curve ek graph mein
    results_dict = {"xgboost": (y_true, y_prob), "random_forest": (y_true, y_prob), ...}
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))

    for (name, (y_true, y_prob)), color in zip(results_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Model Comparison')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.savefig(f"{RESULTS_DIR}/{filename}")
    plt.close()
    print(f"  📊 Saved: {filename}")


def plot_pr_curves(results_dict, filename="pr_curves.png"):
    """Sabhi models ka Precision-Recall curve"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))

    for (name, (y_true, y_prob)), color in zip(results_dict.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, color=color, linewidth=2,
                label=f'{name} (AP = {ap:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves — Model Comparison')
    ax.legend(loc='upper right')

    plt.savefig(f"{RESULTS_DIR}/{filename}")
    plt.close()
    print(f"  📊 Saved: {filename}")


def plot_model_comparison_bar(results_df, filename="model_comparison_bar.png"):
    """Bar chart — sabhi models ka PR-AUC comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    df_sorted = results_df.sort_values('PR-AUC', ascending=True)

    # PR-AUC
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df_sorted)))
    axes[0].barh(df_sorted['Model'], df_sorted['PR-AUC'], color=colors)
    axes[0].set_xlabel('PR-AUC')
    axes[0].set_title('PR-AUC Comparison')

    # ROC-AUC
    df_sorted2 = results_df.sort_values('ROC-AUC', ascending=True)
    axes[1].barh(df_sorted2['Model'], df_sorted2['ROC-AUC'], color=colors)
    axes[1].set_xlabel('ROC-AUC')
    axes[1].set_title('ROC-AUC Comparison')

    # Training Time
    df_sorted3 = results_df.sort_values('Time (s)', ascending=True)
    axes[2].barh(df_sorted3['Model'], df_sorted3['Time (s)'], color=colors)
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_title('Training Time Comparison')

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{filename}")
    plt.close()
    print(f"  📊 Saved: {filename}")


def plot_confusion_matrix(y_true, y_prob, model_name, threshold=0.5,
                          filename=None):
    """Confusion matrix for best model"""
    y_pred = (y_prob > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['Normal', 'Mule'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix — {model_name}')

    if filename is None:
        filename = f"confusion_matrix_{model_name}.png"
    plt.savefig(f"{RESULTS_DIR}/{filename}")
    plt.close()
    print(f"  📊 Saved: {filename}")


def plot_feature_importance(model, features, model_name,
                             top_n=15, filename="feature_importance.png"):
    """Feature importance — research paper ka key chart"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print(f"  ⚠️ {model_name} has no feature_importances_")
        return

    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(idx)))
    ax.barh(
        [features[i].replace('_', ' ') for i in idx],
        importances[idx],
        color=colors
    )
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Features — {model_name}')

    plt.savefig(f"{RESULTS_DIR}/{filename}")
    plt.close()
    print(f"  📊 Saved: {filename}")


def plot_shap_summary(model, X, features, filename="shap_summary.png"):
    """SHAP summary plot — explainability ke liye"""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=features,
                          show=False, max_display=15)
        plt.savefig(f"{RESULTS_DIR}/{filename}")
        plt.close()
        print(f"  📊 Saved: {filename}")
    except Exception as e:
        print(f"  ⚠️ SHAP plot error: {e}")


def plot_precision_at_k(results_dict, y_true, filename="precision_at_k.png"):
    """Precision@K curve — investigation budget analysis"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    k_values = list(range(10, 201, 10))
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))

    for (name, (yt, yp)), color in zip(results_dict.items(), colors):
        p_at_k = []
        for k in k_values:
            top_k = np.argsort(-yp)[:k]
            p_at_k.append(np.array(yt)[top_k].mean())
        ax.plot(k_values, p_at_k, marker='o', color=color,
                linewidth=2, markersize=4, label=name)

    ax.set_xlabel('Top-K Accounts (Investigation Budget)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision@K — How Many True Mules in Top-K Alerts?')
    ax.legend()
    ax.set_ylim([0, 1.05])

    plt.savefig(f"{RESULTS_DIR}/{filename}")
    plt.close()
    print(f"  📊 Saved: {filename}")


def generate_latex_table(results_df, filename="results_table.tex"):
    """Research paper ke liye LaTeX table"""
    latex = results_df.to_latex(
        index=False,
        float_format="%.4f",
        caption="Model Comparison Results on Mule Account Detection",
        label="tab:results",
        column_format="l" + "c" * (len(results_df.columns) - 1),
    )
    with open(f"{RESULTS_DIR}/{filename}", 'w') as f:
        f.write(latex)
    print(f"  📄 Saved: {filename}")