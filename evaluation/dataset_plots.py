import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_dataset_plots():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    csv_path = os.path.join(results_dir, "dataset_drifts.csv")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Dataset drifts CSV is empty.")
        return
        
    # Group data for bar charts without seaborn
    grouped = df.groupby(['drift_type', 'dataset']).size().unstack(fill_value=0)
    
    # 1. Drift Type Distribution by Dataset
    ax = grouped.plot(kind='bar', figsize=(10, 6), color=['#1f77b4', '#ff7f0e'])
    plt.title('Drift Type Distribution: Electricity vs CICIDS-2017')
    plt.xlabel('Detected Drift Type')
    plt.ylabel('Frequency (Count)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'dataset_drift_types.png'))
    plt.close()
    
    # 2. Severity Score Boxplot
    plt.figure(figsize=(10, 6))
    # We create a simple boxplot using matplotlib natively
    datasets = df['dataset'].unique()
    data_to_plot = [df[df['dataset'] == d]['severity_score'].values for d in datasets]
    
    plt.boxplot(data_to_plot, labels=datasets)
    plt.title('Severity Score Distribution by Dataset')
    plt.ylabel('Severity Score')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'dataset_severity_boxplot.png'))
    plt.close()
    
    print("✅ Dataset Visualizations saved to evaluation/results/figures/")

if __name__ == "__main__":
    generate_dataset_plots()
