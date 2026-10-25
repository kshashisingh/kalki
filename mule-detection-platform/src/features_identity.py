import pandas as pd
import numpy as np


def build_identity_features(base_df):
    df = base_df.copy()
    df.columns = df.columns.str.strip()

    if "device_fraud_count" in df.columns and "device_distinct_emails_8w" in df.columns:
        df["device_reuse_risk"] = df.device_fraud_count * df.device_distinct_emails_8w

    if "velocity_24h" in df.columns and "velocity_6h" in df.columns:
        df["velocity_accel"] = df.velocity_6h / df.velocity_24h.clip(lower=1)

    if "name_email_similarity" in df.columns and "email_is_free" in df.columns:
        df["identity_mismatch_risk"] = (1 - df.name_email_similarity) * (1 - df.email_is_free)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    print(f"  Identity features: {df.shape}")
    return df