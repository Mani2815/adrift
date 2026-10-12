from data.database import SessionLocal
from data.models import Event, PipelineMetrics
from sqlalchemy import func
from datetime import datetime, timezone, timedelta

def update_pipeline_metrics(kafka_connected: bool = True):
    """
    Calculates current pipeline metrics and updates the database.
    This is designed to be called periodically by the consumer.
    """
    db = SessionLocal()
    try:
        # Get total events
        total_events = db.query(func.count(Event.id)).scalar()
        
        # Calculate rate over last minute
        one_min_ago = datetime.now(timezone.utc) - timedelta(minutes=1)
        recent_events = db.query(func.count(Event.id)).filter(Event.processed_at >= one_min_ago).scalar()
        
        metrics = db.query(PipelineMetrics).first()
        if not metrics:
            metrics = PipelineMetrics()
            db.add(metrics)
            
        metrics.total_received = total_events
        metrics.total_processed = total_events
        metrics.processing_rate = float(recent_events)
        metrics.kafka_connected = kafka_connected
        metrics.timestamp = datetime.now(timezone.utc)
        
        db.commit()
    except Exception as e:
        print(f"Failed to update metrics: {e}")
    finally:
        db.close()
