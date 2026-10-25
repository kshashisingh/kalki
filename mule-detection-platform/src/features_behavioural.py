import pandas as pd
import numpy as np
import gc


def build_behavioural_features(txns):
    """
    FAST VERSION — Vectorized operations (10x faster)
    Slow lambdas hata diye, sab numpy/pandas vectorized hai
    """
    print("  Building behavioural features (fast mode)...")

    # ── Step 1: In/Out split ──
    out = txns.rename(columns={"src": "acct"}).assign(dir="out")
    inc = txns.rename(columns={"dst": "acct"}).assign(dir="in")
    long = pd.concat([out, inc], ignore_index=True)

    del out, inc
    gc.collect()

    # ── Step 2: Basic aggregations (FAST — no apply) ──
    print("    Aggregating basic stats...")

    # Ek hi groupby mein sab karo (multiple .agg ek saath)
    agg_dict = {
        "amount": ["size", "sum", "mean", "std", "max", "median"],
    }
    basic = long.groupby("acct").agg(agg_dict)
    basic.columns = [
        "txn_count", "total_amount", "amount_mean",
        "amount_std", "amount_max", "amount_median"
    ]
    basic = basic.reset_index()

    # ── Step 3: In/Out separately (FAST) ──
    print("    Aggregating inflow/outflow...")

    in_mask = long.dir == "in"
    out_mask = long.dir == "out"

    in_agg = long[in_mask].groupby("acct").agg(
        in_count=("amount", "size"),
        in_amount=("amount", "sum"),
        unique_sources=("src", "nunique"),
    ).reset_index()

    out_agg = long[out_mask].groupby("acct").agg(
        out_count=("amount", "size"),
        out_amount=("amount", "sum"),
        unique_dests=("dst", "nunique"),
    ).reset_index()

    del in_mask, out_mask
    gc.collect()

    # ── Step 4: Merge ──
    feats = basic.merge(in_agg, on="acct", how="left")
    feats = feats.merge(out_agg, on="acct", how="left")
    feats = feats.fillna(0)

    del in_agg, out_agg, basic
    gc.collect()

    # ── Step 5: Derived features (FAST — vectorized) ──
    print("    Computing derived features...")

    feats["pass_through_ratio"] = (
        feats.out_amount / feats.in_amount.clip(lower=1)
    ).clip(0, 10)

    feats["fan_in_score"] = feats.unique_sources / feats.in_count.clip(lower=1)
    feats["fan_out_score"] = feats.unique_dests / feats.out_count.clip(lower=1)

    feats["counterparty_concentration"] = 1 - (
        (feats.unique_sources + feats.unique_dests) /
        (feats.in_count + feats.out_count).clip(lower=1)
    )

    # ── Step 6: Night ratio (FAST — vectorized) ──
    print("    Computing night ratio + velocity...")

    long["hour"] = long.datetime.dt.hour
    long["is_night"] = long.hour.isin([0, 1, 2, 3, 4, 22, 23]).astype("int8")

    night = long.groupby("acct")["is_night"].mean().rename("night_txn_ratio")
    feats = feats.merge(night.reset_index(), on="acct", how="left")

    # ── Step 7: Peak hour velocity (FAST) ──
    long["hour_bin"] = long.datetime.dt.floor("h")
    peak = (
        long.groupby(["acct", "hour_bin"])
        .size()
        .groupby("acct")
        .max()
        .rename("peak_hour_txns")
        .reset_index()
    )
    feats = feats.merge(peak, on="acct", how="left")

    del peak
    gc.collect()

    # ── Step 8: Dwell time (FAST) ──
    in_first = long[long.dir == "in"].groupby("acct").datetime.min()
    out_first = long[long.dir == "out"].groupby("acct").datetime.min()

    dwell = ((out_first - in_first).dt.total_seconds() / 3600).rename(
        "first_in_to_out_hours"
    )
    feats = feats.merge(dwell.reset_index(), on="acct", how="left")

    del in_first, out_first, dwell
    gc.collect()

    # ── Step 9: Structuring + Round numbers (FAST — vectorized) ──
    print("    Computing structuring signals...")

    # Pehle per-row flag banao, phir mean
    long["is_structuring"] = (
        ((long.amount >= 9500) & (long.amount <= 9999)) |
        ((long.amount >= 45000) & (long.amount <= 49999))
    ).astype("int8")

    struct = (
        long.groupby("acct")["is_structuring"]
        .mean()
        .rename("structuring_ratio")
        .reset_index()
    )
    feats = feats.merge(struct, on="acct", how="left")

    # Round numbers
    long["is_round"] = (long.amount % 1000 == 0).astype("int8")
    round_r = (
        long.groupby("acct")["is_round"]
        .mean()
        .rename("round_number_ratio")
        .reset_index()
    )
    feats = feats.merge(round_r, on="acct", how="left")

    # ── Step 10: Final cleanup ──
    feats = feats.fillna(0)
    feats["first_in_to_out_hours"] = feats["first_in_to_out_hours"].fillna(-1)

    # Unnecessary columns drop
    drop_cols = ["total_amount"]
    feats = feats.drop(columns=[c for c in drop_cols if c in feats.columns])

    del long
    gc.collect()

    print(f"  ✅ Features built: {feats.shape}")
    return feats


def get_labels(txns):
    """Account-level label"""
    src_lab = txns.groupby("src").is_laundering.max()
    dst_lab = txns.groupby("dst").is_laundering.max()
    lab = pd.concat([src_lab, dst_lab], axis=1).max(axis=1).rename("label")
    lab = lab.reset_index()
    lab.columns = ["acct", "label"]
    return lab