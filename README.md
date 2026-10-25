<div align="center">

# 🛡️ MuleDetect

### Intelligent Mule Account Detection & Financial Fraud Investigation Platform

**AI/ML + Graph Analytics for Cyber-Enabled Financial Fraud Detection**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Team KALKI · Cyber Hackathon 2026**

[Features](#-key-features) · [Architecture](#-architecture) · [Results](#-results) · [Dashboard](#-investigation-dashboard) · [Setup](#-getting-started) · [Team](#-team-kalki)

</div>

---

## 📌 Overview

Mule accounts are the backbone of cyber-enabled financial fraud — bank accounts used to receive illicit funds, layer them through networks, and withdraw them to conceal their origin. Current detection relies on static thresholds and manual review, taking days to weeks — by which time funds are already gone.

**MuleDetect** is an end-to-end platform that:

- 🔍 **Detects** mule accounts across **8.8M transactions / 560,913 accounts** using a 7-model ML ensemble
- 🕸️ **Uncovers** organised fraud rings with graph analytics (PageRank, betweenness, Louvain communities)
- 🧠 **Explains** every alert with SHAP reason codes — *why*, not just *what*
- 📊 **Gives investigators** a full investigation dashboard — risk queue → money trail → network → SAR report

### The Numbers at a Glance

| 8.8M | 560,913 | 64 | 7 |
|:---:|:---:|:---:|:---:|
| transactions analysed | accounts covered | engineered features | ML algorithms compared |

| 0.5795 | 1.00 | 0.9910 | 22s |
|:---:|:---:|:---:|:---:|
| PR-AUC (best) | P@50 (Bagging) | ROC-AUC (XGBoost) | XGBoost training |

> **80× better than random** on an extremely imbalanced dataset (only 0.69% mule accounts)

---

## 🧩 Key Features

| Feature | Description |
|---|---|
| 🤖 **Multi-Model Ensemble** | 7 algorithms benchmarked — GB, Bagging, XGBoost, Random Forest, Extra Trees, Decision Tree, Logistic Regression |
| 🕸️ **Graph Analytics** | PageRank · Betweenness centrality · Louvain community detection · distance-to-bad-account |
| 🧠 **Explainable AI** | SHAP-based human-readable reason codes on every risk score (GDPR "right to explanation") |
| ⚖️ **Score Fusion** | `R = 100 × (0.50×B + 0.30×G + 0.20×I)` — behavioural + graph + identity domain fusion |
| 🎯 **Precision-First Evaluation** | PR-AUC & Precision@K on 0.69% prevalence — metrics that reflect real investigator workload |
| 🖥️ **Investigation Dashboard** | Risk Queue · Account Profile · Money Trail (5 hops) · Network View · Complaints · Reports |
| 📄 **SAR Report Generation** | Auto-generated investigation reports — TXT · HTML (print-ready) · PDF |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         📊 DATA LAYER                           │
│        SAML-D (996MB) │ HI-Small (475MB) │ Base (213MB)         │
│   8.8M transactions · 560,913 accounts · 3,864 mule (0.69%)     │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ⚙️ FEATURE ENGINEERING                      │
│     Behavioural (20) │ Graph (9) │ Identity (35) → 64 features  │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     🤖 MULTI-MODEL ENSEMBLE                     │
│        GB │ Bagging │ XGB │ RF │ ET │ DT │ LR (7 algorithms)    │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  🔍 FUSION + EXPLAINABILITY                     │
│   R = 0.50×B + 0.30×G + 0.20×I   │   SHAP Reason Codes         │
│   Bands: HIGH ≥70 │ MEDIUM 40–69 │ LOW <40                      │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│        🖥️ INVESTIGATION DASHBOARD (FastAPI + HTML/CSS/JS + SVG) │
│     Risk Queue │ Money Trail │ Network │ Complaints │ Reports    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Datasets

Three publicly available benchmark datasets, merged and deduplicated:

| Dataset | Volume | Fraud Rate | Domain | Size |
|---|---|---|---|---|
| **SAML-D** | 9,504,852 transactions | 0.107% | Money Laundering (UK/Intl) | 996 MB |
| **HI-Small** | 5,078,345 transactions | 0.105% | AML Bank Transactions (US) | 475 MB |
| **Base** | 1,000,000 applications | 1.103% | Identity Fraud (Device) | 213 MB |
| **Combined** | 8.8M transactions | 0.69% mule | 560,913 accounts · 3,864 mule | — |

> ⚠️ **Extreme class imbalance** — only 3,864 of 560,913 accounts (0.69%) are mules. This is why we report **PR-AUC** and **Precision@K** instead of plain accuracy.

---

## 🔧 Feature Engineering — 64 Features

| Domain | Count | Key Features |
|---|---|---|
| 🔹 **Behavioural** | 20 | fan-in / fan-out, pass-through ratio, velocity, structuring ratio, dwell time |
| 🔹 **Graph** | 9 | PageRank, betweenness centrality, Louvain communities, distance-to-bad-account |
| 🔹 **Identity** | 35 | device reuse, KYC velocity, email similarity, phone validity |

### Molecular Signatures of a Mule Account

| Feature | Formula | Pattern Detected |
|---|---|---|
| `fan_in_score` | unique_sources / in_count | ≈1.0 → collecting from many unrelated people (collection layer) |
| `pass_through_ratio` | outflow / inflow | ≈1.0 → money received is immediately forwarded (layering) |
| `structuring_ratio` | txns just below thresholds | Clusters at 9,500–9,999 / 45,000–49,999 (smurfing) |
| `dist_to_bad` | shortest path → complaint-linked acct | 1–2 hops → highly suspicious |

---

## 📈 Results

### Model Comparison — 7 Algorithms

| # | Model | PR-AUC | ROC-AUC | F1 | P@50 | P@10 | Time |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Gradient Boosting** | **0.5795** | 0.9796 | 0.5097 | 0.86 | 0.70 | 333s |
| 🥈 | **Bagging** | 0.5701 | 0.9558 | 0.4946 | **1.00** | **1.00** | 62s |
| 🥉 | **XGBoost** | 0.5497 | **0.9910** | 0.2655 | 0.88 | **1.00** | **22s** |
| 4 | Decision Tree | 0.4193 | 0.9702 | 0.1652 | 0.64 | 0.80 | 3s |
| 5 | Random Forest | 0.3953 | 0.9895 | 0.1670 | 0.62 | 0.70 | 40s |
| 6 | Extra Trees | 0.2720 | 0.9853 | 0.2858 | 0.40 | 0.30 | 28s |
| 7 | Logistic Regression | 0.0744 | 0.9202 | 0.1233 | 0.20 | 0.70 | 24s |

### Key Insights

- ✅ **Bagging P@50 = 1.00** → if investigators review the top-50 alerts daily, all 50 are genuine mule accounts (zero false positives)
- ✅ **PR-AUC 0.58 vs 0.007 random** → ~80× lift over chance on 0.69% prevalence
- ✅ **XGBoost 22s** → near-optimal accuracy at real-time scoring speed
- ✅ **Ensembles dominate** → Logistic Regression is 7.8× worse than Gradient Boosting
- ✅ **PR-AUC reveals true gaps** that ROC-AUC obscures on heavily imbalanced data

### Results Visuals

| Figure | File |
|---|---|
| PR-AUC comparison | `results/fig_comparison_pr_auc.png` |
| ROC curves | `results/fig_roc_curves.png` |
| PR curves | `results/fig_pr_curves.png` |
| Precision@K | `results/fig_precision_at_k.png` |
| SHAP summary | `results/fig_shap_summary.png` |
| Feature importance | `results/fig_feature_importance.png` |
| Training time | `results/fig_comparison_time.png` |

---

## 🧠 Explainable AI — SHAP Reason Codes

Every risk score ships with human-readable explanations. Example for **ACC-4471** (risk score **92.4 — HIGH**):

| Impact | Feature | Reason Code |
|:---:|---|---|
| 🔴 +2.41 | `fan_in_score` | "Credits from many unrelated sources" |
| 🔴 +1.87 | `pass_through_ratio` | "Funds passed through quickly (near 1:1)" |
| 🔴 +1.12 | `peak_hour_burst` | "Transaction burst within 1 hour" |
| 🟢 −0.34 | `account_age` | "Account age > 1 year" |

> Investigators understand **WHY**, not just **WHAT** — essential for building evidence-based cases and GDPR-compliant "right to explanation".

---

## 🕸️ Graph Analytics

| Algorithm | Purpose |
|---|---|
| **PageRank** | Identify hub accounts in money flow |
| **Betweenness centrality** | Find key intermediary accounts |
| **Louvain communities** | Detect organised fraud rings |
| **Distance-to-bad** | Proximity to complaint-linked accounts |
| **Force-directed layout** | Visualise hidden patterns (interactive: zoom · pan · click) |

---

## 🖥️ Investigation Dashboard

FastAPI backend · custom HTML/CSS/JS frontend · SVG graph engine

| Section | What It Does |
|---|---|
| **Risk Queue** | Ranked leads with colour-coded risk bands (HIGH / MEDIUM / LOW) + filters |
| **Account Profile** | Detailed SHAP explanations + inflow/outflow summary |
| **Money Trail** | Fund flow tracing up to 5 hops |
| **Network View** | Interactive SVG graph — zoom, pan, click to inspect |
| **Complaints** | Hotspot accounts + complaint correlation |
| **Reports** | SAR-style investigation reports — TXT · HTML (print) · PDF |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- 8 GB+ RAM (datasets are large)

### Installation

```bash
# 1. Clone the repository (private — you must be added as a collaborator)
git clone https://github.com/<leader-username>/kalki.git
cd kalki

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
# Start the FastAPI backend (update the module path to match your project)
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

---

## ⚡ Tech Stack

| Layer | Technologies |
|---|---|
| **ML / Data** | Python · scikit-learn · XGBoost · SHAP · pandas · NumPy |
| **Graph Analytics** | NetworkX · Louvain community detection |
| **Backend** | FastAPI · Uvicorn |
| **Frontend** | HTML · CSS · JavaScript · SVG Graph Engine |
| **Reports** | TXT · HTML (print-ready) · PDF |

---

## 🔮 Future Scope

- 🔮 **Graph Neural Networks (GNN)** for higher-order relational patterns
- 🔮 **Real-time streaming** (Apache Kafka) for live transaction scoring
- 🔮 **Federated learning** across banks — privacy-preserving collaboration
- 🔮 **LSTM / Transformer** models for time-aware sequence detection
- 🔮 **Active learning** loops with investigator feedback
- 🔮 **Mobile app** for field investigators

---

## 🆚 Comparison with Existing Methods

| Capability | Rule-Based | Single ML Model | **MuleDetect** |
|---|---|---|---|
| Network Analysis | ✕ | ✕ | ✅ Graph analytics (9 features) |
| Explainability | ✅ Rules | ✕ Black box | ✅ SHAP reason codes |
| Adaptability | ✕ Static | ✅ | ✅ 7-model ensemble |
| False Positives | High | Medium | **P@50 = 1.00** |
| Multi-dataset Validation | ✕ | ✕ | ✅ 3 benchmark datasets |
| Dashboard | ✕ | ✕ | ✅ Full workflow |
| Report Generation | Manual | ✕ | ✅ Auto SAR (TXT/HTML/PDF) |
| Speed | Fast | Medium | **22s (XGBoost)** |

---

## 🔐 Repository & Team Access

| Item | Details |
|---|---|
| **Repository Name** | `kalki` |
| **Visibility** | 🔒 Private |
| **Owner** | Team Leader — Shashi Ranjan Kumar ([@kshashisingh](https://github.com/kshashisingh)) |
| **Event** | Cyber Hackathon 2026 |
| **Institute** | National Institute of Electronics and Information Technology (NIELIT), Patna |

### 👥 Collaborators

| Role | Member | GitHub |
|---|---|---|
| 🎖️ Team Leader | Shashi Ranjan Kumar | [@kshashisingh](https://github.com/kshashisingh) |
| 👤 Member | Chhavi Pandey | [@chhavi-pandey-05](https://github.com/chhavi-pandey-05) |
| 👤 Member | Utsav Kumar Chaudhary | [@Utsav070207](https://github.com/Utsav070207) |
| 👤 Member | Anurag Kumar | [@Anurag7007](https://github.com/Anurag7007) |

### ✅ Submission Checklist

- [x] Private repository `kalki` created by the team leader
- [x] All team members added as collaborators (verified GitHub usernames from the portal)
- [x] Iterative commits throughout development (no single final code dump)
- [ ] Ownership transfer to organisers before the Grand Final (if shortlisted)

---

## 👥 Team KALKI

| Role | Member |
|---|---|
| 🎖️ **Team Lead** | **Shashi Ranjan Kumar** |
| 👤 Member | **Chhavi Pandey** |
| 👤 Member | **Utsav Kumar Chaudhary** |
| 👤 Member | **Anurag Kumar** |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you like this project, give it a star on GitHub!**

Made with ❤️ by **Team KALKI**

</div>
