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
from producer.dataset_adapter import DatasetAdapter

logger = setup_logger("FaultSimulator", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "producer.log"))

try:
    from kafka import KafkaProducer
except ImportError:
    logger.error("kafka-python is not installed. Please run 'pip install -r requirements.txt'")
    sys.exit(1)

def json_serializer(data):
    return json.dumps(data).encode("utf-8")

class FaultSimulator:
    def __init__(self, scenario="Sudden"):
        self.scenario = scenario
        self.step = 0
        self.base_mean = 50.0
        self.current_mean = self.base_mean
        
    def generate_event(self):
        self.step += 1
        fault_payload = None
        
        # Apply drift & fault scenario logic
        if self.scenario == "Sudden Drift":
            if self.step == 500:
                self.current_mean = 90.0
        elif self.scenario == "Gradual Drift":
            if 500 <= self.step <= 1000:
                self.current_mean += (90.0 - 50.0) / 500.0
        elif self.scenario == "Recurring Drift":
            if self.step > 0 and self.step % 500 == 0:
                self.current_mean = 90.0 if self.current_mean == 50.0 else 50.0
        elif self.scenario == "Incremental Drift":
            self.current_mean += 0.05
            
        # Phase 4 System Faults Injection
        if self.scenario == "Consumer Crash" and self.step == 500:
            fault_payload = "CONSUMER_CRASH"
        elif self.scenario == "High Latency" and 500 <= self.step <= 600:
            fault_payload = "HIGH_LATENCY"
        elif self.scenario == "Data Processing Failure" and self.step == 500:
            fault_payload = "PROCESSING_FAILURE"

        value = round(random.gauss(self.current_mean, 10.0), 2)
        
        sources = ["sensor_A", "sensor_B", "network_node_1"]
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": str(uuid.uuid4()),
            "value": value,
            "source": random.choice(sources),
            "status": "OK",
            "simulated_fault": fault_payload
        }
        return event

def main():
    logger.info("Initializing Kafka Producer (Fault Simulator)...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=[settings.KAFKA_BROKER],
            value_serializer=json_serializer,
            retries=5,
            request_timeout_ms=5000
        )
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        return

    topic = settings.TOPIC_NAME
    delay = settings.STREAM_DELAY
    scenario = settings.DRIFT_SCENARIO
    data_source = settings.DATA_SOURCE.lower()
    
    if data_source == "simulated":
        logger.info(f"Starting Fault Simulator for topic '{topic}' with scenario '{scenario}'")
        generator = FaultSimulator(scenario=scenario)
    else:
        logger.info(f"Starting Dataset Adapter for topic '{topic}' with data source '{data_source}'")
        generator = DatasetAdapter(dataset_name=data_source)
    
    try:
        while True:
            event = generator.generate_event()
            producer.send(topic, value=event)
            
            if generator.step % 50 == 0 or event.get("simulated_fault"):
                logger.info(f"Step {generator.step} | Mean: {generator.current_mean:.2f} | Fault: {event.get('simulated_fault')}")
            
            time.sleep(delay)
    except KeyboardInterrupt:
        logger.info("Simulator interrupted.")
    finally:
        producer.close()

if __name__ == "__main__":
    main()
