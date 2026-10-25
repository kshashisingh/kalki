"""
FIX COMPLAINTS — complaints.csv regenerate karo
Run: python fix_complaints.py
"""
import pandas as pd
import numpy as np
import os

# Paths
PROCESSED_DIR = "data/processed"
RAW_DIR = "data/raw"

print("=" * 60)
print("FIXING COMPLAINTS")
print("=" * 60)

# ── Step 1: Behavioural features se labels lo ──
print("\n[1/3] Loading labels...")
behav_path = os.path.join(PROCESSED_DIR, "behavioural_features.csv")

if not os.path.exists(behav_path):
    print(f"  ❌ {behav_path} not found! Pehle pipeline chalao.")
    exit(1)

behav = pd.read_csv(behav_path)
print(f"  Loaded: {behav.shape}")

# Mule accounts = label == 1
mules = behav[behav.label == 1].copy()
print(f"  Mule accounts: {len(mules):,}")

if len(mules) == 0:
    print("  ❌ No mule accounts found! Check labels.")
    exit(1)

# ── Step 2: Synthetic complaints banao ──
print("\n[2/3] Generating complaints...")

rng = np.random.default_rng(42)
n_complaints = min(800, len(mules))

# Har mule account se 1-3 complaints
sampled_mules = mules.sample(n=min(n_complaints, len(mules)), random_state=42)

complaints = pd.DataFrame({
    "complaint_id":   [f"C{i:05d}" for i in range(n_complaints)],
    "victim_acct":    [f"V{rng.integers(100000, 999999)}" for _ in range(n_complaints)],
    "reported_acct":  sampled_mules.acct.astype(str).values[:n_complaints],
    "amount_lost":    rng.uniform(5000, 500000, n_complaints).round(2),
    "complaint_date": pd.date_range("2024-01-01", periods=n_complaints, freq="D").astype(str),
    "category":       rng.choice(
        ["UPI fraud", "Phishing", "Investment scam", "OTP fraud", "Loan fraud"],
        n_complaints
    ),
})

print(f"  Generated: {len(complaints)} complaints")

# ── Step 3: Save ──
print("\n[3/3] Saving...")

output_path = os.path.join(PROCESSED_DIR, "complaints.csv")
complaints.to_csv(output_path, index=False)

# Verify
check = pd.read_csv(output_path)
print(f"  ✅ Saved: {output_path}")
print(f"  Verified: {check.shape}")
print(f"\n  Sample:")
print(complaints.head(5).to_string(index=False))

print(f"\n{'=' * 60}")
print(f"✅ COMPLAINTS FIXED! ({len(complaints)} complaints)")
print(f"{'=' * 60}")
print(f"\n  Ab dashboard restart karo:")
print(f"  python -m uvicorn api.main:app --reload --port 8000")