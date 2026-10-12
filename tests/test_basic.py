import unittest
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from producer.producer import generate_synthetic_event
from data.models import Base

class TestPipelineBasics(unittest.TestCase):
    
    def test_synthetic_event_generation(self):
        """Test that synthetic events are correctly structured."""
        event = generate_synthetic_event()
        self.assertIn("timestamp", event)
        self.assertIn("event_id", event)
        self.assertIn("value", event)
        self.assertIn("source", event)
        self.assertIn("status", event)
        
        # Check types and allowed values
        self.assertIsInstance(event["value"], float)
        self.assertIn(event["status"], ["OK", "WARN", "ERROR"])
        self.assertIsInstance(event["event_id"], str)

    def test_models_exist(self):
        """Test that database models are defined and registered."""
        tables = Base.metadata.tables.keys()
        self.assertIn("events", tables)
        self.assertIn("pipeline_metrics", tables)

if __name__ == '__main__':
    unittest.main()
