import sys
import os
import json
from datetime import datetime, timezone

# Ensure we can import from config and data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from config.logging_config import setup_logger
from data.database import SessionLocal, init_db
from data.models import Event
from monitoring.metrics_tracker import update_pipeline_metrics
from consumer.drift_detector import DriftDetectionService
from recovery.recovery_engine import RecoveryEngine
from recovery.fault_monitor import FaultMonitor

logger = setup_logger("Consumer", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "consumer.log"))

try:
    from kafka import KafkaConsumer
except ImportError:
    logger.error("kafka-python is not installed. Please run 'pip install -r requirements.txt'")
    sys.exit(1)

def json_deserializer(data):
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        logger.error(f"Error deserializing data: {e}")
        return None

def main():
    logger.info("Initializing DB for Consumer...")
    init_db()
    
    logger.info("Initializing Recovery Engine & Fault Monitor...")
    recovery_engine = RecoveryEngine()
    fault_monitor = FaultMonitor(recovery_engine)
    
    logger.info("Initializing Drift Detection Service...")
    drift_detector = DriftDetectionService(recovery_engine=recovery_engine)
    
    logger.info("Initializing Kafka Consumer...")
    try:
        consumer = KafkaConsumer(
            settings.TOPIC_NAME,
            bootstrap_servers=[settings.KAFKA_BROKER],
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='adrift_consumer_group',
            value_deserializer=json_deserializer
        )
        logger.info(f"Connected to Kafka broker at {settings.KAFKA_BROKER}, subscribed to {settings.TOPIC_NAME}")
        update_pipeline_metrics(kafka_connected=True)
    except Exception as e:
        logger.error(f"Failed to connect to Kafka broker: {e}")
        update_pipeline_metrics(kafka_connected=False)
        return

    db = SessionLocal()
    events_processed = 0
    
    try:
        for message in consumer:
            event_data = message.value
            if not event_data:
                continue
                
            # Check for simulated system faults
            fault_monitor.check_for_faults(event_data)
            
            # Save event to DB
            db_event = Event(
                timestamp=event_data.get("timestamp"),
                event_id=event_data.get("event_id"),
                value=event_data.get("value"),
                source=event_data.get("source"),
                status=event_data.get("status")
            )
            db.add(db_event)
            events_processed += 1
            
            # Commit every 10 events or immediately to simulate real-time
            db.commit()
            
            # Feed value to drift detector
            val = event_data.get("value")
            if val is not None:
                drift_detector.process_value(float(val))
            
            # Update metrics every 10 events
            if events_processed % 10 == 0:
                update_pipeline_metrics(kafka_connected=True)
                logger.info(f"Processed {events_processed} events so far in this session.")
                
    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user. Shutting down...")
    except Exception as e:
        logger.error(f"Error during message consumption: {e}")
    finally:
        consumer.close()
        update_pipeline_metrics(kafka_connected=False)
        db.close()
        logger.info("Consumer closed.")

if __name__ == "__main__":
    main()
