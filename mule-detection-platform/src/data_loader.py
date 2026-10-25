import pandas as pd
import gc
from src.config import SAMLD_PATH, HI_PATH, BASE_PATH


def load_samld():
    print("  Loading SAML-D.csv...")
    usecols = [
        "Time", "Date", "Sender_account", "Receiver_account",
        "Amount", "Payment_type", "Is_laundering", "Laundering_type"
    ]

    df = pd.read_csv(
        SAMLD_PATH,
        usecols=usecols,
        low_memory=True,
        dtype={
            "Sender_account": str,
            "Receiver_account": str,
            "Amount": "float32",
            "Is_laundering": "int8",
        }
    )

    df.columns = df.columns.str.strip()
    print(f"  SAML-D raw shape: {df.shape}")

    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True, errors="coerce"
    )

    df = df.rename(columns={
        "Sender_account": "src",
        "Receiver_account": "dst",
        "Amount": "amount",
        "Payment_type": "channel",
        "Is_laundering": "is_laundering",
        "Laundering_type": "laundering_type",
    })

    before = len(df)
    df = df.dropna(subset=["datetime", "src", "dst", "amount"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["is_laundering"] = pd.to_numeric(
        df["is_laundering"], errors="coerce"
    ).fillna(0).astype(int)

    print(f"  Dropped {before - len(df)} rows | Final: {df.shape}")
    print(f"  Laundering %: {df.is_laundering.mean()*100:.3f}%")

    if "Date" in df.columns:
        df = df.drop(columns=["Date", "Time"], errors="ignore")
    gc.collect()

    return df[[
        "datetime", "src", "dst", "amount", "channel",
        "is_laundering", "laundering_type"
    ]]


def load_hi_small():
    print("  Loading HI-Small_Trans.csv...")
    df = pd.read_csv(HI_PATH, low_memory=True)
    df.columns = df.columns.str.strip()
    print(f"  HI-Small raw shape: {df.shape}")
    print(f"  HI-Small columns: {list(df.columns)}")

    # Duplicate "Account" columns handle karo
    cols = list(df.columns)
    new_cols = []
    acct_count = 0
    for c in cols:
        if c.lower() == "account":
            acct_count += 1
            new_cols.append(f"account_{acct_count}")
        else:
            new_cols.append(c)
    df.columns = new_cols

    print(f"  After account rename: {list(df.columns)}")

    # Flexible rename
    rename_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("timestamp", "time"):
            rename_map[c] = "datetime"
        elif cl in ("from bank", "from_bank", "src_bank"):
            rename_map[c] = "src_bank"
        elif cl in ("to bank", "to_bank", "dst_bank"):
            rename_map[c] = "dst_bank"
        elif cl == "account_1":
            rename_map[c] = "src"
        elif cl == "account_2":
            rename_map[c] = "dst"
        elif cl in ("amount received", "amount_received", "amount"):
            rename_map[c] = "amount"
        elif cl in ("payment format", "payment_format", "payment_type"):
            rename_map[c] = "channel"
        elif cl in ("is laundering", "is_laundering"):
            rename_map[c] = "is_laundering"

    df = df.rename(columns=rename_map)
    print(f"  Renamed columns: {list(df.columns)}")

    # Agar src/dst nahi bane to manually map karo
    if "src" not in df.columns:
        possible = [c for c in df.columns if "account" in c.lower()]
        if len(possible) >= 2:
            df = df.rename(columns={possible[0]: "src", possible[1]: "dst"})
        elif len(possible) == 1:
            df["src"] = df[possible[0]]
            df["dst"] = df[possible[0]]
        print(f"  Manual map: {list(df.columns)}")

    # Datetime banao
    if "datetime" not in df.columns:
        first_col = df.columns[0]
        df["datetime"] = pd.to_datetime(df[first_col], errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Types convert karo
    df["amount"] = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0)

    if "is_laundering" in df.columns:
        df["is_laundering"] = pd.to_numeric(
            df["is_laundering"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        df["is_laundering"] = 0

    if "channel" not in df.columns:
        df["channel"] = "unknown"

    # Valid rows
    before = len(df)
    df = df.dropna(subset=["datetime", "src", "dst", "amount"])
    print(f"  Dropped {before - len(df)} | Final: {df.shape}")
    print(f"  Laundering %: {df.is_laundering.mean()*100:.3f}%")

    gc.collect()
    return df[["datetime", "src", "dst", "amount", "channel", "is_laundering"]]


def load_base():
    print("  Loading Base.csv...")
    df = pd.read_csv(BASE_PATH, low_memory=True)
    df.columns = df.columns.str.strip()
    print(f"  Base raw shape: {df.shape}")
    print(f"  Fraud %: {df.fraud_bool.mean()*100:.3f}%")

    miss = df.isnull().sum()
    if miss.any():
        print(f"  Missing values:\n{miss[miss > 0]}")

    gc.collect()
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("DATA LOADER — VALIDATION")
    print("=" * 60)

    print("\n[1/3] SAML-D:")
    s = load_samld()
    print(s.head(3).to_string())

    print("\n[2/3] HI-Small_Trans:")
    h = load_hi_small()
    print(h.head(3).to_string())

    print("\n[3/3] Base:")
    b = load_base()
    print(b.head(3).to_string())

    print("\n✅ All 3 datasets loaded successfully!")