FROM python:3.9-slim

WORKDIR /app

# Install minimal dependencies for Kafka streaming consumer
RUN pip install --no-cache-dir kafka-python==2.0.2 six

# Copy consumer service script
COPY consumer/streaming_consumer.py /app/streaming_consumer.py

# Expose HTTP readiness and telemetry port
EXPOSE 8080

# Environment defaults
ENV KAFKA_BROKER=kafka:29092 \
    TOPIC_NAME=adrift_events \
    GROUP_ID=adrift_consumer_group \
    HTTP_PORT=8080 \
    AUTO_OFFSET_RESET=earliest

# Run unbuffered Python
ENV PYTHONUNBUFFERED=1

CMD ["python", "streaming_consumer.py"]
