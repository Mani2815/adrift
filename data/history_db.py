import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'drift_history.db')

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ClassifiedDrift(Base):
    __tablename__ = 'classified_drifts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    drift_type = Column(String)  # Sudden, Gradual, Incremental, Recurring
    confidence = Column(Float)
    severity_score = Column(Float)
    severity_level = Column(String)  # Low, Moderate, High, Critical
    detectors_triggered = Column(String)  # comma separated list e.g., 'ADWIN,DDM'
    magnitude_shift = Column(Float)

def init_history_db():
    """Initialize the tables for the drift intelligence repository."""
    Base.metadata.create_all(bind=engine)

def get_history_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
