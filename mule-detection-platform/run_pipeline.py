"""
MASTER PIPELINE — All 3 Datasets
Run: python run_pipeline.py
"""
import time


def main():
    start = time.time()
    print("🚀 Mule Detection Platform — Full Pipeline (3 Datasets)\n")

    print("=" * 60)
    print("PHASE 1: DATA LOADING")
    print("=" * 60)
    from src.data_loader import load_samld, load_hi_small, load_base

    txns = load_samld()
    try:
        hi_txns = load_hi_small()
    except Exception as e:
        print(f"  ⚠️ HI-Small_Trans: {e}")
        hi_txns = None
    base = load_base()

    print("\n" + "=" * 60)
    print("PHASE 2: FEATURE ENGINEERING (3 Datasets)")
    print("=" * 60)
    from src.save_features import save_all_features
    save_all_features()

    print("\n" + "=" * 60)
    print("PHASE 3: MODEL TRAINING")
    print("=" * 60)
    from src.train_behavioural import train_behavioural
    from src.train_graph import train_graph
    from src.train_identity import train_identity

    train_behavioural()
    train_graph()
    train_identity()

    print("\n" + "=" * 60)
    print("PHASE 4: FUSION")
    print("=" * 60)
    from src.fusion import fuse_scores
    unified = fuse_scores()

    print("\n" + "=" * 60)
    print("PHASE 5: COMPLAINTS")
    print("=" * 60)
    from src.complaints import build_complaints, correlate_complaints
    from src.features_graph import build_graph_features

    # Combined transactions se complaints banao
    import pandas as pd
    common_cols = ["datetime", "src", "dst", "amount", "channel", "is_laundering"]
    if hi_txns is not None:
        cols_s = [c for c in common_cols if c in txns.columns]
        cols_h = [c for c in common_cols if c in hi_txns.columns]
        txns_all = pd.concat([txns[cols_s], hi_txns[cols_h]], ignore_index=True)
    else:
        txns_all = txns

    _, G = build_graph_features(txns_all)
    complaints = build_complaints(txns_all)
    complaints.to_csv("data/processed/complaints.csv", index=False)
    hotspots, rings = correlate_complaints(complaints, G)
    print(f"  Complaints: {len(complaints)}")
    print(f"  Rings: {len(rings)}")

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE! (All 3 Datasets)")
    print("=" * 60)
    print(f"  ⏱ Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  📊 Scored: {len(unified):,}")
    print(f"  🔴 HIGH: {(unified.band == 'HIGH').sum():,}")
    print(f"\n  Next: python -m uvicorn api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()