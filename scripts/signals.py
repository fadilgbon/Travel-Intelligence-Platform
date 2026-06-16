import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "processed"/"flights_processed.parquet"

def load():
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} rows")
    return df

def cheapest_booking_window(df):
    result = df.groupby(["route","booking_window"])["price"].mean().reset_index()
    result.columns = ["route","booking_window","average_price"]
    result = result.sort_values(["route","average_price"])
    return result




if __name__ == "__main__":
    df = load()
    result = cheapest_booking_window(df)
    print(result.head(20))
