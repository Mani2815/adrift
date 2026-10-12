import os
import glob
import pandas as pd
import numpy as np

def clean_column_names(df):
    """Standardizes column names by stripping spaces, replacing internal spaces with underscores, and lowercasing."""
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    return df

def preprocess_electricity(input_path, output_path):
    print(f"Processing Electricity dataset: {input_path}")
    df = pd.read_csv(input_path)
    df = clean_column_names(df)
    
    row_count = len(df)
    col_count = len(df.columns)
    
    # Rename class label for standard adapter parsing
    if 'class' in df.columns:
        df.rename(columns={'class': 'label'}, inplace=True)
        
    # Analyze missing before drop
    missing = df.isna().sum().sum()
    df.dropna(inplace=True)
    
    df.to_csv(output_path, index=False)
    
    label_dist = df['label'].value_counts().to_dict() if 'label' in df.columns else {}
    mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    return {
        "name": "Electricity",
        "rows": len(df),
        "cols": col_count,
        "labels": label_dist,
        "missing_removed": missing,
        "memory_mb": mem_usage
    }

def preprocess_cicids(input_dir, output_path):
    print(f"Processing CICIDS-2017 datasets from: {input_dir}")
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not all_files:
        print("No CSV files found in CICIDS directory.")
        return None
        
    df_list = []
    for file in all_files:
        print(f" Loading {os.path.basename(file)}...")
        temp_df = pd.read_csv(file, low_memory=False)
        df_list.append(temp_df)
        
    print(" Merging datasets...")
    df = pd.concat(df_list, ignore_index=True)
    df = clean_column_names(df)
    
    col_count = len(df.columns)
    
    print(" Cleaning NaNs and Infinites...")
    # Count missing before cleaning
    initial_rows = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    missing_removed = initial_rows - len(df)
    
    row_count = len(df)
    
    print(f" Saving merged CICIDS-2017 to {output_path} (This might take a minute...)")
    df.to_csv(output_path, index=False)
    
    label_dist = df['label'].value_counts().to_dict() if 'label' in df.columns else {}
    mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    return {
        "name": "CICIDS-2017",
        "rows": row_count,
        "cols": col_count,
        "labels": label_dist,
        "missing_removed": missing_removed,
        "memory_mb": mem_usage
    }

def generate_report(stats_list, report_path):
    report = "# ADRIFT Dataset Preprocessing Report\n\n"
    
    for stats in stats_list:
        if not stats:
            continue
        report += f"## {stats['name']} Dataset\n"
        report += f"- **Clean Row Count**: {stats['rows']:,}\n"
        report += f"- **Column Count**: {stats['cols']}\n"
        report += f"- **Corrupted/Missing Rows Removed**: {stats['missing_removed']:,}\n"
        report += f"- **Memory Usage**: {stats['memory_mb']:.2f} MB\n"
        report += f"- **Label Distribution**:\n"
        for k, v in stats['labels'].items():
            report += f"  - `{k}`: {v:,}\n"
        report += "\n"
        
    with open(report_path, "w") as f:
        f.write(report)
        
    print(f"\nReport generated at {report_path}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, "datasets")
    
    # User's physical location for the datasets
    raw_electricity = "/Users/mani/Downloads/research/Project/datasets/electricity-normalized.csv"
    raw_cicids = "/Users/mani/Downloads/research/Project/datasets/MachineLearningCVE"
    
    processed_dir = os.path.join(datasets_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    out_electricity = os.path.join(processed_dir, "electricity_stream.csv")
    out_cicids = os.path.join(processed_dir, "cicids2017_merged.csv")
    out_report = os.path.join(processed_dir, "dataset_report.md")
    
    stats = []
    
    if os.path.exists(raw_electricity):
        stats.append(preprocess_electricity(raw_electricity, out_electricity))
    else:
        print(f"Skipping Electricity: File not found at {raw_electricity}")
        
    if os.path.exists(raw_cicids):
        res = preprocess_cicids(raw_cicids, out_cicids)
        if res:
            stats.append(res)
    else:
        print(f"Skipping CICIDS: Dir not found at {raw_cicids}")
        
    if stats:
        generate_report(stats, out_report)
        
        print("\n=== Preprocessing Pipeline Summary ===")
        for s in stats:
            if s:
                print(f"-> {s['name']} | Rows: {s['rows']:,} | Cols: {s['cols']} | Missing Removed: {s['missing_removed']:,} | Size: {s['memory_mb']:.1f} MB")
    else:
        print("No datasets were processed.")

if __name__ == "__main__":
    main()
