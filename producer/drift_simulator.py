import sys
import os
import time
import json
import uuid
import random
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("DriftSimulator", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "producer.log"))

try:
    from kafka import KafkaProducer
except ImportError:
    logger.error("kafka-python is not installed. Please run 'pip install -r requirements.txt'")
    sys.exit(1)

def json_serializer(data):
    return json.dumps(data).encode("utf-8")

class DriftSimulator:
    def __init__(self, scenario="Sudden"):
        self.scenario = scenario
        self.step = 0
        self.base_mean = 50.0
        self.current_mean = self.base_mean
        
    def generate_event(self):
        self.step += 1
        
        # Apply drift scenario logic to update the mean
        if self.scenario == "Sudden":
            if self.step == 500:
                self.current_mean = 90.0
                logger.info(f"--- SUDDEN DRIFT TRIGGERED at step {self.step} ---")
                
        elif self.scenario == "Gradual":
            if 500 <= self.step <= 1000:
                self.current_mean += (90.0 - 50.0) / 500.0
                if self.step == 500:
                    logger.info(f"--- GRADUAL DRIFT STARTED at step {self.step} ---")
                    
        elif self.scenario == "Recurring":
            if self.step > 0 and self.step % 500 == 0:
                self.current_mean = 90.0 if self.current_mean == 50.0 else 50.0
                logger.info(f"--- RECURRING DRIFT FLIPPED to mean {self.current_mean} at step {self.step} ---")
                
        elif self.scenario == "Incremental":
            self.current_mean += 0.05 # small continuous shift
            if self.step == 1:
                logger.info(f"--- INCREMENTAL DRIFT STARTED ---")
                
        value = round(random.gauss(self.current_mean, 10.0), 2)
        
        sources = ["sensor_A", "sensor_B", "network_node_1"]
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": str(uuid.uuid4()),
            "value": value,
            "source": random.choice(sources),
            "status": "OK"
        }
        return event

def main():
    logger.info("Initializing Kafka Producer (Drift Simulator)...")
    
    try:
        producer = KafkaProducer(
            bootstrap_servers=[settings.KAFKA_BROKER],
            value_serializer=json_serializer,
            retries=5,
            request_timeout_ms=5000
        )
        logger.info(f"Connected to Kafka broker at {settings.KAFKA_BROKER}")
    except Exception as e:
        logger.error(f"Failed to connect to Kafka broker: {e}")
        return

    topic = settings.TOPIC_NAME
    delay = settings.STREAM_DELAY
    scenario = settings.DRIFT_SCENARIO
    
    logger.info(f"Starting Drift Simulator for topic '{topic}' with scenario '{scenario}'")
    simulator = DriftSimulator(scenario=scenario)
    
    try:
        while True:
            event = simulator.generate_event()
            producer.send(topic, value=event)
            # Log with less verbosity for standard events to avoid spamming
            if simulator.step % 50 == 0:
                logger.info(f"Published event {simulator.step} | Value: {event['value']} | Mean: {simulator.current_mean:.2f}")
            time.sleep(delay)
    except KeyboardInterrupt:
        logger.info("Simulator interrupted by user. Shutting down...")
    except Exception as e:
        logger.error(f"Error during event generation: {e}")
    finally:
        producer.close()
        logger.info("Producer closed.")

if __name__ == "__main__":
    main()
