"""
FastAPI Backend — Mule Detection Platform
Run: python -m uvicorn api.main:app --reload --port 8000
"""
import os
import sys
import traceback
import pandas as pd
import networkx as nx

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PROCESSED_DIR, MODEL_DIR, DASHBOARD_DIR

# ============ APP INIT ============
app = FastAPI(title="Mule Detection Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ SERVE DASHBOARD FILES ============
# Root pe index.html
@app.get("/")
def serve_index():
    index_path = os.path.join(DASHBOARD_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        {"error": "index.html not found", "path": index_path},
        status_code=404
    )

# Static files (CSS, JS, images)
static_path = os.path.join(DASHBOARD_DIR)
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    print(f"✅ Static files mounted: {static_path}")
else:
    print(f"❌ Dashboard folder not found: {static_path}")


# ============ LOAD DATA (Once at startup) ============
print("\n📂 Loading data files...")

SCORES = None
COMPLAINTS = None
BEHAV = None

def load_data():
    global SCORES, COMPLAINTS, BEHAV

    scores_path = os.path.join(PROCESSED_DIR, "unified_scores.csv")
    complaints_path = os.path.join(PROCESSED_DIR, "complaints.csv")
    behav_path = os.path.join(PROCESSED_DIR, "behavioural_features.csv")

    if os.path.exists(scores_path):
        SCORES = pd.read_csv(scores_path)
        print(f"  ✅ Scores loaded: {SCORES.shape}")
    else:
        print(f"  ❌ Not found: {scores_path}")

    if os.path.exists(complaints_path):
        COMPLAINTS = pd.read_csv(complaints_path)
        print(f"  ✅ Complaints loaded: {COMPLAINTS.shape}")
    else:
        print(f"  ❌ Not found: {complaints_path}")

    if os.path.exists(behav_path):
        BEHAV = pd.read_csv(behav_path)
        print(f"  ✅ Behavioural features loaded: {BEHAV.shape}")
    else:
        print(f"  ❌ Not found: {behav_path}")

load_data()


# ============ API ENDPOINTS ============

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "scores_loaded": SCORES is not None,
        "complaints_loaded": COMPLAINTS is not None,
        "behav_loaded": BEHAV is not None,
    }


@app.get("/api/dashboard/stats")
def dashboard_stats():
    """Overview metrics"""
    try:
        if SCORES is None:
            return {"error": "Data not loaded"}

        complaints_count = len(COMPLAINTS) if COMPLAINTS is not None else 0
        unique_reported = COMPLAINTS.reported_acct.nunique() if COMPLAINTS is not None else 0

        return {
            "total_accounts": int(len(SCORES)),
            "high_risk": int((SCORES.band == "HIGH").sum()),
            "medium_risk": int((SCORES.band == "MEDIUM").sum()),
            "low_risk": int((SCORES.band == "LOW").sum()),
            "total_complaints": complaints_count,
            "unique_reported": int(unique_reported),
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


@app.get("/api/risk-queue")
def risk_queue(band: str = None, limit: int = 50):
    """Top risky accounts"""
    try:
        if SCORES is None:
            return {"accounts": [], "total": 0, "error": "Data not loaded"}

        df = SCORES.copy()
        if band and band != "ALL":
            df = df[df.band == band]

        result = df.head(limit).to_dict(orient="records")

        # Convert numpy types to Python types
        for r in result:
            for k, v in r.items():
                if hasattr(v, 'item'):
                    r[k] = v.item()

        return {"accounts": result, "total": int(len(df))}
    except Exception as e:
        return {"accounts": [], "total": 0, "error": str(e)}


@app.get("/api/account/{acct_id}")
def account_profile(acct_id: str):
    """Account profile + SHAP reasons"""
    try:
        if SCORES is None:
            return {"error": "Data not loaded"}

        row = SCORES[SCORES.acct.astype(str) == acct_id]
        if len(row) == 0:
            return {"error": f"Account {acct_id} not found"}

        row = row.iloc[0].to_dict()

        # Convert numpy types
        for k, v in row.items():
            if hasattr(v, 'item'):
                row[k] = v.item()

        # SHAP reasons (optional — fail gracefully)
        reasons = []
        try:
            if BEHAV is not None:
                from src.explain import RiskExplainer
                X_row = BEHAV[BEHAV.acct.astype(str) == acct_id]
                if len(X_row) > 0:
                    exp = RiskExplainer("behavioural")
                    reasons = exp.get_reasons(X_row)
        except Exception as e:
            reasons = [{"feature": "N/A", "description": f"SHAP error: {str(e)[:50]}",
                        "impact": 0, "direction": "N/A"}]

        # Complaints
        complaints_list = []
        if COMPLAINTS is not None:
            linked = COMPLAINTS[COMPLAINTS.reported_acct.astype(str) == acct_id]
            for _, c in linked.iterrows():
                complaints_list.append({
                    "complaint_id": str(c.get("complaint_id", "")),
                    "category": str(c.get("category", "")),
                    "amount_lost": float(c.get("amount_lost", 0)),
                    "complaint_date": str(c.get("complaint_date", ""))[:10],
                })

        return {
            "account": row,
            "reasons": reasons,
            "complaints": complaints_list,
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


@app.get("/api/account/{acct_id}/trail")
def account_trail(acct_id: str, depth: int = 3):
    """Money trail"""
    try:
        from src.money_trail import trace_funds

        # Graph load karo (lazy — sirf jab zaroorat ho)
        from src.data_loader import load_samld
        from src.features_graph import build_graph_features
        txns = load_samld()
        _, G = build_graph_features(txns)

        trail = trace_funds(G, acct_id, depth=depth)
        if len(trail) == 0:
            return {"trail": [], "summary": {"total_transactions": 0, "total_outflow": 0, "unique_destinations": 0}}

        result = trail.head(100).to_dict(orient="records")
        for r in result:
            for k, v in r.items():
                if hasattr(v, 'item'):
                    r[k] = v.item()

        return {
            "trail": result,
            "summary": {
                "total_transactions": int(len(trail)),
                "total_outflow": float(trail.amount.sum()),
                "unique_destinations": int(trail["to"].nunique()),
            },
        }
    except Exception as e:
        return {"trail": [], "summary": {"total_transactions": 0}, "error": str(e)}


@app.get("/api/account/{acct_id}/network")
def account_network(acct_id: str, depth: int = 2):
    """Network graph for vis.js"""
    try:
        from src.data_loader import load_samld
        from src.features_graph import build_graph_features
        txns = load_samld()
        _, G = build_graph_features(txns)

        bad = set(COMPLAINTS.reported_acct.astype(str).unique()) if COMPLAINTS is not None else set()

        # Collect neighbors
        acct_str = str(acct_id)
        nbrs = {acct_str}
        for n in G.nodes():
            if str(n) == acct_str:
                nbrs.add(n)
                for _ in range(depth):
                    new_nbrs = set()
                    for node in list(nbrs):
                        if node in G:
                            new_nbrs.update([str(x) for x in G.successors(node)][:5])
                            new_nbrs.update([str(x) for x in G.predecessors(node)][:5])
                    nbrs.update(new_nbrs)
                break

        sub = G.subgraph([n for n in G.nodes() if str(n) in nbrs])

        nodes = []
        for n in sub.nodes():
            n_str = str(n)
            is_bad = n_str in bad
            is_selected = (n_str == acct_str)
            nodes.append({
                "id": n_str,
                "label": n_str[:15],
                "color": "#e5484d" if is_bad else ("#ffa726" if is_selected else "#3b82f6"),
                "size": 35 if is_selected else (25 if is_bad else 15),
                "title": f"Account: {n_str}",
                "is_suspicious": is_bad,
                "is_selected": is_selected,
            })

        edges = []
        for u, v, e in sub.edges(data=True):
            edges.append({
                "from": str(u),
                "to": str(v),
                "value": max(float(e.get("amount", 1)), 1),
                "title": f"Amount: {e.get('amount', 0)}",
            })

        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}


@app.get("/api/complaints/hotspots")
def complaint_hotspots(limit: int = 20):
    try:
        if COMPLAINTS is None:
            return {"hotspots": []}

        hotspots = (
            COMPLAINTS.groupby("reported_acct")
            .agg(
                n_complaints=("complaint_id", "count"),
                total_loss=("amount_lost", "sum"),
                categories=("category", lambda x: ", ".join(set(x))),
            )
            .sort_values("n_complaints", ascending=False)
            .head(limit)
            .reset_index()
        )
        return {"hotspots": hotspots.to_dict(orient="records")}
    except Exception as e:
        return {"hotspots": [], "error": str(e)}


@app.get("/api/complaints/rings")
def complaint_rings():
    try:
        return {"rings": {}, "total_rings": 0, "note": "Requires graph. Use after first trail request."}
    except Exception as e:
        return {"rings": {}, "error": str(e)}


@app.get("/api/report/{acct_id}")
def generate_report(acct_id: str):
    try:
        if SCORES is None:
            return {"report_text": "Data not loaded", "error": "Data not loaded"}

        row = SCORES[SCORES.acct.astype(str) == acct_id]
        if len(row) == 0:
            return {"report_text": f"Account {acct_id} not found"}

        row = row.iloc[0]

        n_complaints = 0
        if COMPLAINTS is not None:
            linked = COMPLAINTS[COMPLAINTS.reported_acct.astype(str) == acct_id]
            n_complaints = len(linked)

        report = f"""
══════════════════════════════════════════════════
       SUSPICIOUS ACTIVITY INVESTIGATION REPORT
══════════════════════════════════════════════════

ACCOUNT ID:       {acct_id}
RISK SCORE:       {float(row.risk_score):.1f} / 100
RISK BAND:        {str(row.band)}
GLOBAL RANK:      #{int(row['rank'])}
BEHAVIOURAL RISK: {float(row.p_behavioural):.3f}
GRAPH RISK:       {float(row.graph_risk):.3f}

──────────────────────────────────────────────────

LINKED COMPLAINTS: {n_complaints}

RECOMMENDATION: {
    'ESCALATE — High priority investigation'
    if str(row.band) == 'HIGH'
    else 'REVIEW — Monitor and gather more data'
    if str(row.band) == 'MEDIUM'
    else 'LOW PRIORITY — Routine monitoring'
}

══════════════════════════════════════════════════
Generated by: Mule Detection Platform (AI/ML + Graph)
"""
        return {
            "account_id": acct_id,
            "report_text": report,
            "risk_score": float(row.risk_score),
            "band": str(row.band),
            "n_complaints": n_complaints,
        }
    except Exception as e:
        return {"report_text": f"Error: {str(e)}", "error": str(e)}


# ============ RUN ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)