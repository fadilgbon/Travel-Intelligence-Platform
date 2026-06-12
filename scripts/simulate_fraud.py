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

