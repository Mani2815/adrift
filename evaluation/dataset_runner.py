import sys
import os
import time
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from producer.dataset_adapter import DatasetAdapter
from recovery.recovery_engine import RecoveryEngine
from data.history_db import SessionLocal, ClassifiedDrift
from consumer.drift_detector import DriftDetectionService

# Monkeypatch sleep for instant fast-forward execution
import time
time.sleep = lambda x: None

def clear_db():
    db = SessionLocal()
    db.query(ClassifiedDrift).delete()
    db.commit()
    db.close()

def extract_and_clear_db(dataset_name):
    db = SessionLocal()
    drifts = db.query(ClassifiedDrift).all()
    
    data = []
    for d in drifts:
        data.append({
            "dataset": dataset_name,
            "drift_type": d.drift_type,
            "confidence": d.confidence,
            "severity_score": d.severity_score,
            "severity_level": d.severity_level,
            "timestamp": d.timestamp
        })
        
    db.query(ClassifiedDrift).delete()
    db.commit()
    db.close()
    return pd.DataFrame(data)

def run_dataset(dataset_name, max_rows=None):
    print(f"\n🚀 Starting Fast-Forward Evaluation for: {dataset_name.upper()}")
    
    settings.DATA_SOURCE = dataset_name
    settings.PPO_ENABLED = False # Use baseline for speed
    
    # Wrap adapter to ensure it doesn't infinite loop when dataset is fully consumed
    class FiniteAdapter(DatasetAdapter):
        def generate_event(self):
            row = next(self.csv_reader)
            self.step += 1
            try:
                primary_val = float(row.get(self.value_col, 0.0))
            except ValueError:
                primary_val = 0.0
            return {"value": primary_val}
            
    try:
        adapter = FiniteAdapter(dataset_name=dataset_name)
    except FileNotFoundError:
        print(f"Skipping {dataset_name}. Processed file not found.")
        return 0, 0
        
    recovery_engine = RecoveryEngine()
    drift_detector = DriftDetectionService(recovery_engine=recovery_engine)
    
    start_time = time.time()
    rows_processed = 0
    
    try:
        while True:
            if max_rows and rows_processed >= max_rows:
                print(f"Reached max row limit ({max_rows}). Stopping.")
                break
                
            event = adapter.generate_event()
            drift_detector.process_value(event["value"])
            
            rows_processed += 1
            if rows_processed % 50000 == 0:
                print(f"Processed {rows_processed:,} rows...")
                
    except StopIteration:
        print(f"Finished dataset ({rows_processed:,} rows).")
    except Exception as e:
        print(f"Ended run unexpectedly: {e}")
        
    end_time = time.time()
    duration = end_time - start_time
    throughput = rows_processed / duration if duration > 0 else 0
    
    print("-" * 40)
    print(f"Completed {dataset_name.upper()}")
    print(f"Total Rows: {rows_processed:,}")
    print(f"Time Taken: {duration:.2f} seconds")
    print(f"Throughput: {throughput:,.2f} rows/sec")
    print("-" * 40)
    
    return rows_processed, throughput

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    
    clear_db()
    
    # 1. CICIDS-2017
    c_rows, c_thr = run_dataset("cicids2017", max_rows=None)
    final_df = extract_and_clear_db("CICIDS-2017")
    
    csv_path = os.path.join(out_dir, "dataset_drifts.csv")
    final_df.to_csv(csv_path, index=False)
    
    # Save throughput stats
    stats_df = pd.DataFrame([
        {"dataset": "CICIDS-2017", "rows_processed": c_rows, "throughput_rps": c_thr}
    ])
    stats_df.to_csv(os.path.join(out_dir, "dataset_throughput.csv"), index=False)
    
    print(f"\n✅ Evaluation complete. Data saved to {csv_path}")

if __name__ == "__main__":
    main()
