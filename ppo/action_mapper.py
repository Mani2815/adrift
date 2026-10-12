class ActionMapper:
    """
    Maps PPO discrete integer actions to ADRIFT pipeline recovery commands.
    """
    def __init__(self):
        self.mapping = {
            0: "Ignore Event",
            1: "Retrain Model",
            2: "Restart Consumer",
            3: "Rollback Previous Model",
            4: "Scale Processing Resources"
        }
        
    def get_action_string(self, action_int: int) -> str:
        return self.mapping.get(int(action_int), "Ignore Event")
        
    def get_action_space_size(self) -> int:
        return len(self.mapping)
