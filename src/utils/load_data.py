import os
import pandas as pd

def load_processed_data(repo_root=".", filename="car_sales_processed.csv"):
    processed_path = os.path.join(repo_root, "data", "processed", filename)
    df = pd.read_csv(processed_path)
    return df
