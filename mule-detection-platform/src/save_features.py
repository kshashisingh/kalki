
import time
import gc
from src.data_loader import load_samld, load_hi_small, load_base
from src.features_behavioural import build_behavioural_features, get_labels
from src.features_graph import build_graph_features
from src.features_identity import build_identity_features
from src.config import PROCESSED_DIR


def save_all_features():
    total_start = time.time()

    print("=" * 60)
    print("FEATURE ENGINEERING (All 3 Datasets)")
    print("=" * 60)

    # ── 1. SAML-D ──
    print("\n[1/5] Loading SAML-D...")
    t = time.time()
    txns = load_samld()
    print(f"  ⏱ {time.time()-t:.1f}s")

    # ── 2. HI-Small_Trans ──
    print("\n[2/5] Loading HI-Small_Trans...")
    t = time.time()
    try:
        hi_txns = load_hi_small()
        print(f"  ⏱ {time.time()-t:.1f}s | {hi_txns.shape}")
    except Exception as e:
        print(f"  ⚠️ HI-Small_Trans load failed: {e}")
        hi_txns = None
    gc.collect()

    # ── 3. Behavioural Features (SAML-D + HI-Small COMBINED) ──
    print("\n[3/5] Behavioural features (SAML-D + HI-Small)...")
    t = time.time()

    # Dono datasets ke transactions combine karo
    if hi_txns is not None:
        # Common columns select karo
        common_cols = ["datetime", "src", "dst", "amount", "channel", "is_laundering"]
        cols_samld = [c for c in common_cols if c in txns.columns]
        cols_hi = [c for c in common_cols if c in hi_txns.columns]

        txns_combined = pd.concat([
            txns[cols_samld],
            hi_txns[cols_hi]
        ], ignore_index=True)
        print(f"  Combined: {len(txns_combined):,} transactions")
        print(f"    SAML-D: {len(txns):,} + HI-Small: {len(hi_txns):,}")
    else:
        txns_combined = txns
        print(f"  Using SAML-D only: {len(txns_combined):,}")

    behav = build_behavioural_features(txns_combined)
    labels = get_labels(txns_combined)
    behav = behav.merge(labels, on="acct", how="left").fillna({"label": 0})
    behav.to_csv(f"{PROCESSED_DIR}/behavioural_features.csv", index=False)
    print(f"  ⏱ {time.time()-t:.1f}s | Saved: {behav.shape}")
    print(f"  Mules: {behav.label.sum():,} ({behav.label.mean()*100:.2f}%)")
    gc.collect()

    # ── 4. Graph Features (Combined transactions) ──
    print("\n[4/5] Graph features (combined)...")
    t = time.time()
    bad = behav[behav.label == 1].acct.tolist()
    graph_feats, G = build_graph_features(txns_combined, bad_accounts=bad)
    graph_feats.to_csv(f"{PROCESSED_DIR}/graph_features.csv", index=False)
    print(f"  ⏱ {time.time()-t:.1f}s | Saved: {graph_feats.shape}")
    del G
    gc.collect()

    # ── 5. Identity Features (Base) ──
    print("\n[5/5] Identity features (Base)...")
    t = time.time()
    base = load_base()
    ident = build_identity_features(base)
    ident.to_csv(f"{PROCESSED_DIR}/identity_features.csv", index=False)
    print(f"  ⏱ {time.time()-t:.1f}s | Saved: {ident.shape}")
    del base
    gc.collect()

    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"✅ ALL FEATURES SAVED! ({total:.0f}s / {total/60:.1f} min)")
    print(f"{'=' * 60}")


import pandas as pd  # top pe add karo agar nahi hai

if __name__ == "__main__":
    save_all_features()