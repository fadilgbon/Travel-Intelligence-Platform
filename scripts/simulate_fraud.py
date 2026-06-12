import pandas as pd
import numpy as np
import pyarrow
import fastparquet
from pathlib import Path
from datetime import datetime, timedelta

RAW_PATH = Path(__file__).parent.parent / "data" / "Clean_Dataset.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "flights_with_fraud.parquet"
RNG = np.random.default_rng(42)


def add_booking_behaviour(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    base_date = datetime(2024, 1, 1)
    booking_offsets = RNG.integers(0, 90 * 24 * 3600, size=n)
    df["booking_timestamp"] = [
        base_date + timedelta(seconds=int(s)) for s in booking_offsets
    ]

    # ~50k unique users across 300k bookings (avg 6 bookings per user)
    df["user_id"] = RNG.integers(1, 50_001, size=n)

    # Session id — most users have 1 session per booking, bots reuse sessions
    df["session_id"] = [f"sess_{RNG.integers(1, 200_001)}" for _ in range(n)]

    # Device type
    df["device"] = RNG.choice(["mobile", "desktop", "tablet"], size=n, p=[0.55, 0.35, 0.10])

    return df

def add_payment_columns(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    df["payment_method"] = RNG.choice(
        ["credit_card", "debit_card", "paypal"],
        size=n,
        p=[0.40,0.35,0.25],
    )

    # Card BIN country — mostly India, small % foreign (higher fraud risk)
    df["card_country"] = RNG.choice(
        ["CA", "US", "MX"],
        size=n,
        p=[0.40, 0.50, 0.10]
    )

    # Billing country matches card country most of the time
    mismatch_mask = RNG.random(n) < 0.04
    df["billing_country"] = df["card_country"].copy()
    df.loc[mismatch_mask, "billing_country"] = RNG.choice(
        ["US", "GB", "AE", "NG", "CN"], size=mismatch_mask.sum()
    )

    df["is_refunded"] = RNG.random(n) < 0.03

    return df

def inject_fraud(df: pd.DataFrame) -> pd.DataFrame:
    df["is_fraud"] = False
    df["fraud_type"] = "none"

    # Same user books many flights in a short window
    fraud_users = RNG.choice(df["user_id"].unique(), size=300, replace=False)
    velocity_mask = df["user_id"].isin(fraud_users)
    burst_time = datetime(2024, 2, 15, 3, 0, 0)
    df.loc[velocity_mask, "booking_timestamp"] = [
        burst_time + timedelta(seconds=int(s))
        for s in RNG.integers(0, 600, size=velocity_mask.sum())
    ]
    df.loc[velocity_mask, "is_fraud"] = True
    df.loc[velocity_mask, "fraud_type"] = "velocity_attack"

    # Business class tickets priced below typical Economy floor
    business_mask = df["class"] == "Business"
    economy_floor = df[df["class"] == "Economy"]["price"].quantile(0.05)
    cheap_business = business_mask & (df["price"] < economy_floor * 0.8)
    df.loc[cheap_business, "is_fraud"] = True
    df.loc[cheap_business, "fraud_type"] = "price_anomaly"

    #Bot scraping
    session_counts = df.groupby("session_id")["route"].nunique()
    bot_sessions = session_counts[session_counts > 8].index
    bot_mask = df["session_id"].isin(bot_sessions)
    df.loc[bot_mask, "is_fraud"] = True
    df.loc[bot_mask & (df["fraud_type"] == "none"), "fraud_type"] = "bot_activity"

    return df


def run():
    print("Loading data")
    df = pd.read_csv(RAW_PATH, index_col=0)
    print(f"  {len(df):,} rows")

    print("Adding booking behaviour columns")
    df = add_booking_behaviour(df)

    print("Adding payment columns")
    df = add_payment_columns(df)

    print("Adding route column")
    df["route"] = df["source_city"] + " → " + df["destination_city"]

    print("Injecting fraud patterns")
    df = inject_fraud(df)

    fraud_count = df["is_fraud"].sum()
    print(f"  {fraud_count:,} fraud rows ({fraud_count / len(df) * 100:.2f}%)")
    print(f"  Breakdown:\n{df['fraud_type'].value_counts().to_string()}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run()