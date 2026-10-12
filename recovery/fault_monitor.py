from config.settings import settings
import logging

class FaultMonitor:
    def __init__(self, recovery_engine):
        self.recovery_engine = recovery_engine
        self.logger = logging.getLogger("FaultMonitor")
        
    def check_for_faults(self, event_data):
        """
        Inspects incoming event payloads for simulated faults or 
        calculates real latency anomalies.
        """
        if not settings.FAULT_MONITOR_ENABLED:
            return
            
        fault_type = event_data.get("simulated_fault")
        if fault_type:
            self.logger.warning(f"SYSTEM FAULT DETECTED: {fault_type}")
            
            # Map the simulated string to standard fault names for the rule engine
            if fault_type == "CONSUMER_CRASH":
                self.recovery_engine.handle_fault("Consumer Failure")
            elif fault_type == "HIGH_LATENCY":
                self.recovery_engine.handle_fault("High Latency")
            elif fault_type == "PROCESSING_FAILURE":
                self.recovery_engine.handle_fault("Data Processing Failure")
