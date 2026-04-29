class PIDController:
    """
    PID controller that outputs a single drive signal given a temperature error.

    The signal is clamped to [0, 1] and is intended to be applied to both
    fan speed and auger feed rate together (coupled operation). The NN
    controller can learn to vary them independently for better performance.

    Output is clamped to [0, 1] with integral anti-windup to prevent
    the I term from accumulating while the output is saturated.
    """

    def __init__(
        self,
        kp: float = 0.015,
        ki: float = 0.002,
        kd: float = 0.2,
        timestep: float = 1.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.timestep = timestep

        self._integral: float = 0.0
        self._prev_error: float = 0.0

    def step(self, current_temp: float, target_temp: float) -> float:
        """
        Compute drive signal for this timestep.

        Args:
            current_temp: Current grill temperature (°F).
            target_temp:  Desired grill temperature (°F).

        Returns:
            drive signal in [0, 1] — apply to both fan speed and auger feed rate.
        """
        error = target_temp - current_temp

        # Derivative on error (not output, to avoid derivative kick on setpoint changes)
        derivative = (error - self._prev_error) / self.timestep

        # Tentative output before clamping
        output = (
            self.kp * error
            + self.ki * self._integral
            + self.kd * derivative
        )
        output_clamped = max(0.0, min(1.0, output))

        # Anti-windup: only accumulate integral when output is not saturated
        if output == output_clamped:
            self._integral += error * self.timestep

        self._prev_error = error
        return output_clamped

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
