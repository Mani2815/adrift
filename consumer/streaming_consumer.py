#!/usr/bin/env python3
"""
streaming_consumer.py
=====================
Containerized Kafka streaming consumer for ADRIFT Phase 4.

Key Responsibilities:
1. Connects to Kafka broker at $KAFKA_BROKER (default: localhost:9092 or kafka:29092).
2. Subscribes to topic $TOPIC_NAME (default: adrift_events) with group $GROUP_ID.
3. Consumes messages and processes them deterministically (JSON parse, running statistics, offset tracking).
4. Serves HTTP telemetry on port $HTTP_PORT (default: 8080):
   - /health:  200 OK when the process is alive.
   - /ready:   200 OK when Kafka consumer is actively connected and consuming, 503 otherwise.
   - /metrics: JSON telemetry with processed count, offsets, and uptime.
5. Handles clean shutdown on SIGTERM and SIGINT.
"""

import os
import sys
import json
import time
import signal
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("StreamingConsumer")

# Configuration via environment variables
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC_NAME = os.getenv("TOPIC_NAME", "adrift_events")
GROUP_ID = os.getenv("GROUP_ID", "adrift_consumer_group")
HTTP_PORT = int(os.getenv("HTTP_PORT", "8080"))
AUTO_OFFSET_RESET = os.getenv("AUTO_OFFSET_RESET", "earliest")

try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("kafka-python not installed; running in degraded standalone mode.")


class ConsumerState:
    """Thread-safe state tracker for consumer telemetry and readiness."""
    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.is_connected = False
        self.is_ready = False
        self.is_shutting_down = False
        self.messages_processed = 0
        self.messages_failed = 0
        self.last_offset: Optional[int] = None
        self.last_partition: Optional[int] = None
        self.last_event_id: Optional[str] = None
        self.last_heartbeat = time.time()
        self.running_sum = 0.0
        self.running_count = 0

    def set_connected(self, connected: bool):
        with self._lock:
            self.is_connected = connected
            self.is_ready = connected and not self.is_shutting_down
            self.last_heartbeat = time.time()

    def set_shutting_down(self):
        with self._lock:
            self.is_shutting_down = True
            self.is_ready = False

    def record_message(self, event_id: str, value: float, partition: int, offset: int):
        with self._lock:
            self.messages_processed += 1
            self.last_offset = offset
            self.last_partition = partition
            self.last_event_id = event_id
            self.running_sum += value
            self.running_count += 1
            self.last_heartbeat = time.time()

    def record_failure(self):
        with self._lock:
            self.messages_failed += 1
            self.last_heartbeat = time.time()

    def get_telemetry(self) -> Dict[str, Any]:
        with self._lock:
            uptime = round(time.time() - self.start_time, 2)
            mean_val = (self.running_sum / self.running_count) if self.running_count > 0 else 0.0
            return {
                "status": "READY" if self.is_ready else "NOT_READY",
                "connected": self.is_connected,
                "uptime_seconds": uptime,
                "messages_processed": self.messages_processed,
                "messages_failed": self.messages_failed,
                "last_offset": self.last_offset,
                "last_partition": self.last_partition,
                "last_event_id": self.last_event_id,
                "running_mean": round(mean_val, 4),
                "last_heartbeat_age_s": round(time.time() - self.last_heartbeat, 2),
                "broker": KAFKA_BROKER,
                "topic": TOPIC_NAME,
                "group_id": GROUP_ID,
            }


consumer_state = ConsumerState()


class TelemetryHTTPHandler(BaseHTTPRequestHandler):
    """Exposes /health, /ready, and /metrics endpoints."""
    def log_message(self, format, *args):
        # Suppress noisy HTTP access logs for readiness polling
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "UP"}).encode("utf-8"))
        elif self.path == "/ready":
            telemetry = consumer_state.get_telemetry()
            if telemetry["status"] == "READY":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(telemetry).encode("utf-8"))
            else:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(telemetry).encode("utf-8"))
        elif self.path == "/metrics":
            telemetry = consumer_state.get_telemetry()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(telemetry).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def start_http_server(port: int) -> HTTPServer:
    server = HTTPServer(("0.0.0.0", port), TelemetryHTTPHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    logger.info(f"Telemetry HTTP server listening on 0.0.0.0:{port} (/health, /ready, /metrics)")
    return server


def run_consumer_loop(stop_event: threading.Event):
    """Main consumer loop with reconnection handling."""
    if not KAFKA_AVAILABLE:
        logger.error("kafka-python not available, consumer loop aborted.")
        return

    logger.info(f"Connecting to Kafka broker at {KAFKA_BROKER}, topic: {TOPIC_NAME}...")
    consumer = None

    while not stop_event.is_set():
        try:
            consumer = KafkaConsumer(
                TOPIC_NAME,
                bootstrap_servers=[KAFKA_BROKER],
                auto_offset_reset=AUTO_OFFSET_RESET,
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                group_id=GROUP_ID,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                consumer_timeout_ms=1000,  # Unblocks poll every 1s to check stop_event
            )
            # Initial poll triggers group coordinator join and partition assignment
            consumer.poll(timeout_ms=500)
            consumer_state.set_connected(True)
            logger.info(f"Successfully subscribed to {TOPIC_NAME} on {KAFKA_BROKER}")
            break
        except Exception as e:
            consumer_state.set_connected(False)
            logger.warning(f"Waiting for Kafka broker ({KAFKA_BROKER}): {e}")
            time.sleep(2)

    if stop_event.is_set() or consumer is None:
        return

    try:
        while not stop_event.is_set():
            try:
                for message in consumer:
                    if stop_event.is_set():
                        break
                    
                    payload = message.value
                    if not isinstance(payload, dict):
                        consumer_state.record_failure()
                        continue

                    event_id = str(payload.get("event_id", ""))
                    val = float(payload.get("value", 0.0))
                    
                    # Deterministic processing step
                    consumer_state.record_message(
                        event_id=event_id,
                        value=val,
                        partition=message.partition,
                        offset=message.offset,
                    )

                    if consumer_state.messages_processed % 50 == 0:
                        logger.info(
                            f"Processed {consumer_state.messages_processed} messages. "
                            f"Last offset={message.offset}, partition={message.partition}"
                        )
            except Exception as e:
                if not stop_event.is_set():
                    logger.warning(f"Error during message consumption poll: {e}")
                    time.sleep(0.5)
    finally:
        consumer_state.set_shutting_down()
        if consumer:
            try:
                consumer.close(timeout=2.0)
            except Exception:
                pass
        logger.info("Kafka consumer cleanly closed.")


def main():
    stop_event = threading.Event()

    def handle_signal(sig, frame):
        logger.info(f"Received termination signal ({sig}). Initiating clean shutdown...")
        consumer_state.set_shutting_down()
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Start telemetry HTTP server
    server = start_http_server(HTTP_PORT)

    # Run consumer in worker thread
    consumer_thread = threading.Thread(target=run_consumer_loop, args=(stop_event,))
    consumer_thread.start()

    # Wait for stop signal
    while not stop_event.is_set():
        time.sleep(0.5)

    consumer_thread.join(timeout=5.0)
    server.shutdown()
    logger.info("Streaming consumer process exited.")


if __name__ == "__main__":
    main()
