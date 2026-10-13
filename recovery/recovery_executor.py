import time
import random
import logging
from typing import Optional, Tuple

class RecoveryExecutor:
    """
    Simulates the execution of recovery actions.
    Supports explicit, isolated random number generators (RNG) and 
    Common Random Numbers (CRN) variates for reproducible evaluation.
    """
    # Action parameter definitions: (t_min, t_max, success_prob)
    ACTION_PARAMS = {
        "Retrain Model": (2.0, 5.0, 0.85),
        "Restart Consumer": (0.5, 1.5, 0.95),
        "Rollback Previous Model": (1.0, 3.0, 0.90),
        "Scale Processing Resources": (1.5, 4.0, 0.80),
        "Ignore Event": (0.1, 0.1, 1.0),
    }
    DEFAULT_PARAMS = (1.0, 1.0, 0.5)

    def __init__(self, rng: Optional[random.Random] = None, seed: Optional[int] = None, simulate_sleep: bool = True):
        self.logger = logging.getLogger("RecoveryExecutor")
        self.simulate_sleep = simulate_sleep
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = None

    def execute(
        self,
        action: str,
        rng: Optional[random.Random] = None,
        u_rand: Optional[float] = None,
        v_rand: Optional[float] = None,
        simulate_sleep: Optional[bool] = None,
    ) -> Tuple[bool, float]:
        """
        Simulates the execution of a recovery action.

        Parameters:
        -----------
        action : str
            The name of the recovery action to execute.
        rng : Optional[random.Random]
            An explicit, isolated random generator instance for this execution.
        u_rand : Optional[float]
            Exogenous uniform random variate in [0.0, 1.0) for recovery success determination
            under Common Random Numbers (CRN). If provided, success = (u_rand <= success_prob).
        v_rand : Optional[float]
            Exogenous uniform random variate in [0.0, 1.0) for execution duration determination
            under Common Random Numbers (CRN). If provided, exec_time = t_min + v_rand * (t_max - t_min).
        simulate_sleep : Optional[bool]
            If provided, overrides instance-level simulate_sleep. When False, time.sleep is bypassed.

        Returns:
        --------
        success : bool
            Whether the simulated recovery action succeeded.
        actual_time : float
            Simulated or measured execution duration in seconds.
        """
        self.logger.info(f"Initiating Recovery Action: {action}")
        start_time = time.time()

        # Select RNG instance
        active_rng = rng if rng is not None else self.rng

        # Lookup action profile
        t_min, t_max, success_prob = self.ACTION_PARAMS.get(action, self.DEFAULT_PARAMS)

        # 1. Determine execution duration
        if v_rand is not None:
            # Deterministic Common Random Number scaling
            v_clamped = max(0.0, min(1.0, float(v_rand)))
            exec_time = t_min + v_clamped * (t_max - t_min)
        elif active_rng is not None:
            exec_time = active_rng.uniform(t_min, t_max) if t_min < t_max else t_min
        else:
            # Fallback to module-level random for backward compatibility
            exec_time = random.uniform(t_min, t_max) if t_min < t_max else t_min

        # Simulate blocking operational work
        should_sleep = self.simulate_sleep if simulate_sleep is None else simulate_sleep
        if should_sleep:
            time.sleep(exec_time)

        # 2. Determine recovery success
        if u_rand is not None:
            # Deterministic Common Random Number thresholding
            u_clamped = max(0.0, min(1.0, float(u_rand)))
            success = u_clamped <= success_prob
        elif active_rng is not None:
            success = active_rng.random() <= success_prob
        else:
            # Fallback to module-level random for backward compatibility
            success = random.random() <= success_prob

        end_time = time.time()
        actual_time = (end_time - start_time) if should_sleep else exec_time

        if success:
            self.logger.info(f"Action '{action}' COMPLETED SUCCESSFULLY in {actual_time:.2f}s (prob={success_prob})")
        else:
            self.logger.error(f"Action '{action}' FAILED after {actual_time:.2f}s (prob={success_prob})")

        return success, actual_time

