import os
import pandas as pd
import numpy as np

def generate_synthetic_electricity(output_path, rows=45312):
    print(f"Generating synthetic Electricity dataset ({rows} rows)...")
    np.random.seed(42)
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=rows, freq='5min'),
        'day': np.random.randint(1, 8, rows),
        'period': np.random.randint(1, 49, rows),
        'nswprice': np.random.uniform(0, 1, rows),
        'nswdemand': np.random.uniform(0, 1, rows),
        'vicprice': np.random.uniform(0, 1, rows),
        'vicdemand': np.random.uniform(0, 1, rows),
        'transfer': np.random.uniform(0, 1, rows),
        'class': np.random.choice(['UP', 'DOWN'], rows)
    })
    
    # Induce recurring drift
    for i in range(0, rows, 5000):
        if i + 1000 < rows:
            df.loc[i:i+1000, 'nswprice'] += 0.5
            df.loc[i:i+1000, 'nswprice'] = df.loc[i:i+1000, 'nswprice'].clip(0, 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

def generate_synthetic_cicids(output_path, rows=50000):
    print(f"Generating synthetic CICIDS-2017 dataset ({rows} rows)...")
    np.random.seed(42)
    # Using a subset of features for simplicity
    df = pd.DataFrame({
        'Flow Duration': np.random.randint(1, 1000000, rows),
        'Total Fwd Packets': np.random.randint(1, 1000, rows),
        'Total Backward Packets': np.random.randint(1, 1000, rows),
        'Label': np.random.choice(['BENIGN', 'DDoS', 'PortScan', 'Bot'], rows, p=[0.8, 0.1, 0.05, 0.05])
    })
    
    # Induce sudden drift (attacks)
    for i in range(10000, 15000):
        df.loc[i, 'Flow Duration'] *= 50
    for i in range(30000, 35000):
        df.loc[i, 'Total Fwd Packets'] *= 100

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    elec_path = os.path.join(base_dir, "datasets", "processed", "electricity.csv")
    cicids_path = os.path.join(base_dir, "datasets", "processed", "cicids2017.csv")
    
    generate_synthetic_electricity(elec_path, 45312)
    generate_synthetic_cicids(cicids_path, 2830743) # 100% full dataset stream
