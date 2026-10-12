from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()

class Event(Base):
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String) # Keep as string for simplicity since it's JSON from producer
    event_id = Column(String, index=True)
    value = Column(Float)
    source = Column(String)
    status = Column(String)
    processed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class PipelineMetrics(Base):
    __tablename__ = 'pipeline_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    total_received = Column(Integer, default=0)
    total_processed = Column(Integer, default=0)
    processing_rate = Column(Float, default=0.0) # Events per minute
    kafka_connected = Column(Boolean, default=False)
