import pandas as pd
import numpy as np
import networkx as nx
from src.config import PROCESSED_DIR


def build_complaints(txns, n=800, seed=7):
    """Synthetic complaints from laundering transactions"""
    rng = np.random.default_rng(seed)
    bad_txns = txns[txns.is_laundering == 1]

    if len(bad_txns) == 0:
        print("  ⚠️ No laundering transactions found!")
        return pd.DataFrame(columns=[
            "complaint_id", "victim_acct", "reported_acct",
            "amount_lost", "complaint_date", "category"
        ])

    if len(bad_txns) < n:
        n = len(bad_txns)
        print(f"  Reduced to {n} complaints (available bad txns)")

    sample = bad_txns.sample(n, random_state=seed)

    complaints = pd.DataFrame({
        "complaint_id":   [f"C{i:05d}" for i in range(n)],
        "victim_acct":    rng.choice(txns.src.unique(), n),
        "reported_acct":  sample.dst.values,
        "amount_lost":    sample.amount.values,
        "complaint_date": (
            sample.datetime.values +
            pd.to_timedelta(rng.integers(1, 14, n), unit="D")
        ),
        "category": rng.choice(
            ["UPI fraud", "Phishing", "Investment scam",
             "OTP fraud", "Loan fraud"],
            n
        ),
    })
    return complaints


def correlate_complaints(complaints, G):
    """Multiple complaints ek hi ring se linked hain?"""
    if complaints.empty:
        print("  ⚠️ No complaints to correlate")
        return pd.DataFrame(), {}

    # Hotspot accounts (most complained against)
    hotspots = (
        complaints.groupby("reported_acct")
        .agg(
            n_complaints=("complaint_id", "count"),
            total_loss=("amount_lost", "sum"),
            categories=("category", lambda x: ", ".join(set(x))),
        )
        .sort_values("n_complaints", ascending=False)
    )

    # Complaint clustering via graph communities
    rings = {}
    try:
        communities = nx.connected_components(G.to_undirected())
        for i, comp in enumerate(communities):
            linked = complaints[complaints.reported_acct.isin(comp)]
            if len(linked) > 1:
                rings[f"RING_{i}"] = {
                    "n_complaints": int(len(linked)),
                    "complaint_ids": linked.complaint_id.tolist()[:10],
                    "accounts": linked.reported_acct.unique().tolist()[:10],
                    "total_loss": round(float(linked.amount_lost.sum()), 2),
                }
    except Exception as e:
        print(f"  ⚠️ Ring detection error: {e}")

    return hotspots, rings


if __name__ == "__main__":
    from src.data_loader import load_samld
    from src.features_graph import build_graph_features

    print("=" * 60)
    print("COMPLAINTS — TEST")
    print("=" * 60)

    txns = load_samld()
    _, G = build_graph_features(txns)

    complaints = build_complaints(txns)
    complaints.to_csv(f"{PROCESSED_DIR}/complaints.csv", index=False)

    hotspots, rings = correlate_complaints(complaints, G)

    print(f"\n  Complaints: {len(complaints)}")
    print(f"  Hotspots: {len(hotspots)}")
    print(f"  Rings: {len(rings)}")
    print(f"\n  Top 5 hotspots:")
    print(hotspots.head(5).to_string())
    print(f"\n  ✅ Saved: data/processed/complaints.csv")