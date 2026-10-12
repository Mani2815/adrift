import sys
import os
import time
import json
import uuid
import random
from datetime import datetime, timezone

# Ensure we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from config.logging_config import setup_logger

# Initialize logger
logger = setup_logger("Producer", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "producer.log"))

# We import KafkaProducer late in case the library is not installed yet (e.g. initial setup)
try:
    from kafka import KafkaProducer
except ImportError:
    logger.error("kafka-python is not installed. Please run 'pip install -r requirements.txt'")
    sys.exit(1)

def generate_synthetic_event():
    """Generates a random event simulating a stream"""
    sources = ["sensor_A", "sensor_B", "network_node_1", "web_server", "db_server"]
    statuses = ["OK", "WARN", "ERROR"]
    
    # We will skew towards OK for a normal pipeline
    status = random.choices(statuses, weights=[0.8, 0.15, 0.05])[0]
    
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_id": str(uuid.uuid4()),
        "value": round(random.gauss(50, 15), 2), # Random normal distribution value
        "source": random.choice(sources),
        "status": status
    }
    return event

def json_serializer(data):
    return json.dumps(data).encode("utf-8")

def main():
    logger.info("Initializing Kafka Producer...")
    
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
    
    logger.info(f"Starting event generation for topic '{topic}' with delay {delay}s")
    
    try:
        while True:
            event = generate_synthetic_event()
            producer.send(topic, value=event)
            logger.info(f"Published event: {event['event_id']} - Source: {event['source']} - Value: {event['value']}")
            
            time.sleep(delay)
    except KeyboardInterrupt:
        logger.info("Producer interrupted by user. Shutting down...")
    except Exception as e:
        logger.error(f"Error during event generation: {e}")
    finally:
        producer.close()
        logger.info("Producer closed.")

if __name__ == "__main__":
    main()
