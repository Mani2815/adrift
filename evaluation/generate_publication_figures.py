import os
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "ieee_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set global aesthetics for IEEE
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def save_plot(filename):
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.{ext}"), format=ext)
    plt.close()

def save_graph(dot, filename):
    for ext in ["png", "pdf", "svg"]:
        dot.render(os.path.join(OUTPUT_DIR, filename), format=ext, cleanup=True)

def fig1_architecture():
    dot = Digraph(comment="ADRIFT Architecture")
    dot.attr(rankdir='LR', size='10,5')
    dot.node('A', 'Data Source\n(Kafka Stream)', shape='cylinder')
    dot.node('B', 'Phase 2:\nDrift Detectors\n(ADWIN, DDM, KSWIN)', shape='box', style='rounded,filled', fillcolor='lightblue')
    dot.node('C', 'Phase 3:\nIntelligence Core\n(Classification & Severity)', shape='box', style='rounded,filled', fillcolor='lightyellow')
    dot.node('D', 'Phase 4 & 5:\nRecovery Engines\n(Rule Baseline & PPO)', shape='box', style='rounded,filled', fillcolor='lightgreen')
    dot.node('E', 'Phase 6:\nExplainable AI\n(SHAP Surrogates)', shape='box', style='rounded,filled', fillcolor='lightpink')
    
    dot.edges(['AB', 'BC', 'CD', 'CE'])
    save_graph(dot, "Fig1_Architecture")

def fig2_workflow():
    dot = Digraph(comment="Overall Workflow")
    dot.attr(rankdir='TB')
    dot.node('1', 'Stream Ingestion', shape='parallelogram')
    dot.node('2', 'Statistical Monitoring', shape='rect')
    dot.node('3', 'Drift Detected?', shape='diamond')
    dot.node('4', 'Intelligence Extraction', shape='rect')
    dot.node('5', 'Recovery Execution', shape='rect')
    dot.node('6', 'Continue Processing', shape='ellipse')
    
    dot.edge('1', '2')
    dot.edge('2', '3')
    dot.edge('3', '4', label='Yes')
    dot.edge('3', '6', label='No')
    dot.edge('4', '5')
    dot.edge('5', '6')
    dot.edge('6', '1', label='Next Event')
    save_graph(dot, "Fig2_Workflow")

def fig3_classification():
    dot = Digraph(comment="Drift Classification")
    dot.node('1', 'Drift Triggered', shape='ellipse')
    dot.node('2', 'Compare w/ Memory', shape='rect')
    dot.node('3', 'Match Found?', shape='diamond')
    dot.node('4', 'Recurring Drift', shape='box', style='filled', fillcolor='#e1d5e7')
    dot.node('5', 'Calculate Rate of Change', shape='rect')
    dot.node('6', 'RoC > 0.5', shape='diamond')
    dot.node('7', 'Sudden Drift', shape='box', style='filled', fillcolor='#f8cecc')
    dot.node('8', 'RoC > 0.05', shape='diamond')
    dot.node('9', 'Gradual Drift', shape='box', style='filled', fillcolor='#fff2cc')
    dot.node('10', 'Incremental Drift', shape='box', style='filled', fillcolor='#d5e8d4')
    
    dot.edge('1', '2')
    dot.edge('2', '3')
    dot.edge('3', '4', label='Yes')
    dot.edge('3', '5', label='No')
    dot.edge('5', '6')
    dot.edge('6', '7', label='Yes')
    dot.edge('6', '8', label='No')
    dot.edge('8', '9', label='Yes')
    dot.edge('8', '10', label='No')
    save_graph(dot, "Fig3_Classification_Workflow")

def fig4_severity():
    dot = Digraph(comment="Severity Scoring")
    dot.attr(rankdir='LR')
    dot.node('A', 'Input Features:\nMagnitude, StdDev,\nSteps, Detectors', shape='note')
    dot.node('B', 'Base Formula:\n(|Mag|/StdDev)*10\n+ (1000/Steps)', shape='box')
    dot.node('C', 'Detector Multiplier:\n* (Count * 0.25)', shape='box')
    dot.node('D', 'Bounds Check\n[0, 100]', shape='diamond')
    dot.node('E', 'Final Severity Score', shape='ellipse', style='filled', fillcolor='lightgrey')
    
    dot.edges(['AB', 'BC', 'CD', 'DE'])
    save_graph(dot, "Fig4_Severity_Workflow")

def fig5_electricity():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "results", "dataset_drifts.csv"))
    counts = df[df['dataset'] == 'Electricity']['drift_type'].value_counts()
    labels = ['Recurring Drift', 'Incremental Drift', 'Sudden Drift', 'Gradual Drift']
    counts_arr = [counts.get(l, 0) for l in labels]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=counts_arr, hue=labels, palette='Blues_r', legend=False)
    plt.title('Electricity Dataset: Concept Drift Types')
    plt.ylabel('Event Count')
    for i, v in enumerate(counts_arr):
        plt.text(i, v + (max(counts_arr)*0.02 if max(counts_arr)>0 else 5), str(v), ha='center')
    save_plot("Fig5_Electricity_Results")

def fig6_cicids():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "results", "dataset_drifts.csv"))
    counts = df[df['dataset'] == 'CICIDS-2017']['drift_type'].value_counts()
    labels = ['Recurring Drift', 'Sudden Drift', 'Incremental Drift', 'Gradual Drift']
    counts_arr = [counts.get(l, 0) for l in labels]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=counts_arr, hue=labels, palette='Oranges_r', legend=False)
    plt.title('CICIDS-2017 Dataset: Concept Drift Types')
    plt.ylabel('Event Count')
    for i, v in enumerate(counts_arr):
        plt.text(i, v + (max(counts_arr)*0.02 if max(counts_arr)>0 else 5), str(v), ha='center')
    save_plot("Fig6_CICIDS_Results")

def fig7_ppo():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "results", "comparison_report.csv"))
    metrics = ['MTTR (s)', 'Success Rate (%)', 'Pipeline Uptime (%)']
    
    baseline_row = df[df['system_type'] == 'Baseline Rule Engine'].iloc[0]
    ppo_row = df[df['system_type'] == 'PPO Agent'].iloc[0]
    
    baseline = [baseline_row['mttr'], baseline_row['success_rate'], baseline_row['uptime']]
    ppo = [ppo_row['mttr'], ppo_row['success_rate'], ppo_row['uptime']]
    
    x = range(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width/2 for i in x], baseline, width, label='Baseline Rule Engine', color='#4c72b0')
    ax.bar([i + width/2 for i in x], ppo, width, label='PPO Agent', color='#dd8452')
    
    ax.set_ylabel('Scores')
    ax.set_title('Performance Comparison: Baseline vs PPO Agent')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    save_plot("Fig7_PPO_Evaluation")

def fig8_dataset_comparison():
    drifts_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "results", "dataset_drifts.csv"))
    tp_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "results", "dataset_throughput.csv"))
    
    datasets = ['Electricity', 'CICIDS-2017']
    dpm = []
    for d in datasets:
        total_drifts = len(drifts_df[drifts_df['dataset'] == d])
        try:
            rows = tp_df[tp_df['dataset'] == d]['rows_processed'].iloc[0]
            dpm.append((total_drifts / rows) * 1000000)
        except IndexError:
            dpm.append(0.0)
    
    plt.figure(figsize=(7, 5))
    sns.barplot(x=datasets, y=dpm, hue=datasets, palette='Set2', legend=False)
    plt.title('Drift Frequency Across Domains')
    plt.ylabel('Drifts per 1 Million Records')
    for i, v in enumerate(dpm):
        plt.text(i, v + (max(dpm)*0.02 if max(dpm)>0 else 100), f"{v:,.1f}", ha='center')
    save_plot("Fig8_Dataset_Comparison")

def fig9_dataset_pipeline():
    dot = Digraph(comment="Dataset Processing Pipeline")
    dot.attr(rankdir='LR')
    dot.node('A', 'Raw CSV Datasets\n(Electricity, CICIDS)', shape='note')
    dot.node('B', 'Data Cleaning\n(NaN removal, type casting)', shape='rect')
    dot.node('C', 'Standardization\n(Mapping to \'value\' and \'payload\')', shape='rect')
    dot.node('D', 'DatasetAdapter\n(CSV to Event Stream)', shape='cylinder')
    dot.node('E', 'ADRIFT Pipeline', shape='box3d')
    
    dot.edges(['AB', 'BC', 'CD', 'DE'])
    save_graph(dot, "Fig9_Dataset_Pipeline")

def fig10_evaluation_workflow():
    dot = Digraph(comment="Evaluation Workflow")
    dot.attr(rankdir='TB')
    dot.node('1', 'Lock Random Seed', shape='oval')
    dot.node('2', 'Inject Concept Drift Sequence', shape='rect')
    dot.node('3', 'Baseline Execution', shape='rect', style='filled', fillcolor='#dae8fc')
    dot.node('4', 'PPO Agent Execution', shape='rect', style='filled', fillcolor='#ffe6cc')
    dot.node('5', 'Record MTTR & Uptime', shape='cylinder')
    dot.node('6', 'Statistical Comparison', shape='box3d')
    
    dot.edge('1', '2')
    dot.edge('2', '3')
    dot.edge('2', '4')
    dot.edge('3', '5')
    dot.edge('4', '5')
    dot.edge('5', '6')
    save_graph(dot, "Fig10_Evaluation_Workflow")

def fig11_experimental_setup():
    dot = Digraph(comment="Experimental Setup")
    dot.attr(rankdir='LR')
    dot.node('Config', 'Config\n- DATA_SOURCE\n- SEED\n- FAST_FORWARD', shape='folder')
    dot.node('Env', 'Environment\n- Kafka (Topic)\n- SQLite DB', shape='cylinder')
    dot.node('Engine', 'ADRIFT Core\n- Detectors\n- Classifiers\n- Recovery', shape='component')
    
    dot.edge('Config', 'Engine')
    dot.edge('Engine', 'Env', dir='both')
    save_graph(dot, "Fig11_Experimental_Setup")

if __name__ == "__main__":
    print("Generating IEEE Figures...")
    fig1_architecture()
    fig2_workflow()
    fig3_classification()
    fig4_severity()
    fig5_electricity()
    fig6_cicids()
    fig7_ppo()
    fig8_dataset_comparison()
    fig9_dataset_pipeline()
    fig10_evaluation_workflow()
    fig11_experimental_setup()
    print(f"All figures generated in {OUTPUT_DIR} as PNG, PDF, and SVG at 300 DPI.")
