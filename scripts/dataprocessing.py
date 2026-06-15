"""
Phase 1 — Ingestion & Validation Pipeline
Loads raw CSV, validates schema/values, engineers features, writes clean Parquet.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

RAW_PATH = Path(__file__).parent.parent / "data" / "processed"/"flights_with_fraud.parquet"

def load():
    df = pd.read_parquet(RAW_PATH)
    print(df.shape)
    print(df.columns.tolist())
    return df

def validate(df):
    if df.isnull().sum().sum() == 0:
        print("Null Check passed")
    else:
        print("Null Check failed")

    if(df["price"] <= 0).sum() == 0:
        print("Price Check passed")
    else:
        print("Price Check failed")

    if (df["days_left"] <= 0).sum() == 0:
        print("Days Check passed")
    else:
        print("Days Check failed")

    if df["is_fraud"].isin([True, False]).all():
        print("Fraud Check passed")
    else:
        print("Fraud Check failed")

def engineer_features(df):
    df["price_per_hour"]= df["price"]/df["duration"]
    df["route"] = df["source_city"] +" -> " + df["destination_city"]
    return df



if __name__ == "__main__":
    df = load()
    validate(df)
    df = engineer_features(df)
    print(df[["price", "duration", "price_per_hour", "route"]].head())
