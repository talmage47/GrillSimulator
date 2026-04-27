import numpy as np
import gymnasium as gym
from gymnasium import spaces

from grill_simulator import PelletGrillSimulator


class GrillEnv(gym.Env):
    """
    Gymnasium environment wrapping PelletGrillSimulator.

    Observation: [grill_temperature, delta_temperature, temperature_acceleration,
                  auger_feed_rate, target_temperature, error, elapsed_fraction]
    Action:      auger_feed_rate in [0, 1]

    The agent does not observe ambient temperature or lid state — it must
    infer disturbances from temperature response alone.
    """

    MAX_TEMP = 600.0    # °F, upper bound for observation space
    MIN_TEMP = 0.0      # °F, lower bound for observation space

    # Target temperature range the agent will be trained across
    TARGET_TEMP_MIN = 180.0
    TARGET_TEMP_MAX = 500.0

    # Ambient temperature range sampled at each reset
    AMBIENT_TEMP_MIN = 20.0
    AMBIENT_TEMP_MAX = 100.0

    # Episode length in timesteps (30 simulated minutes at 1s/step)
    EPISODE_STEPS = 1800

    # Lid event parameters
    LID_OPEN_PROB = 0.0005      # probability per step of lid opening
    LID_CLOSE_PROB = 0.05       # probability per step of lid closing (once open)

    def __init__(self, timestep: float = 1.0):
        super().__init__()

        self.timestep = timestep

        self.observation_space = spaces.Box(
            low=np.array( [self.MIN_TEMP, -600.0, -600.0, 0.0, self.TARGET_TEMP_MIN, -600.0, 0.0], dtype=np.float32),
            high=np.array([self.MAX_TEMP,  600.0,  600.0, 1.0, self.TARGET_TEMP_MAX,  600.0, 1.0], dtype=np.float32),
        )

        self.action_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(1,),
        )

        self.simulator = PelletGrillSimulator(timestep=timestep)
        self.target_temperature: float = 225.0
        self._steps: int = 0
        self._delta_temperature: float = 0.0
        self._temperature_acceleration: float = 0.0

    # ------------------------------------------------------------------
    # Core gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        ambient = self.np_random.uniform(self.AMBIENT_TEMP_MIN, self.AMBIENT_TEMP_MAX)
        self.simulator.ambient_temperature = float(ambient)
        self.simulator.reset()

        self.target_temperature = float(
            self.np_random.uniform(self.TARGET_TEMP_MIN, self.TARGET_TEMP_MAX)
        )
        self._steps = 0
        self._delta_temperature = 0.0
        self._temperature_acceleration = 0.0

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        auger_feed_rate = float(action[0])

        self._maybe_toggle_lid()

        prev_temp = self.simulator.grill_temperature
        grill_temp, auger_rate = self.simulator.step(auger_feed_rate)
        self._steps += 1

        delta_temp = grill_temp - prev_temp
        self._temperature_acceleration = delta_temp - self._delta_temperature
        self._delta_temperature = delta_temp

        obs = self._get_obs()
        reward = self._compute_reward(grill_temp)
        terminated = False
        truncated = self._steps >= self.EPISODE_STEPS

        info = {
            "grill_temperature": grill_temp,
            "target_temperature": self.target_temperature,
            "ambient_temperature": self.simulator.ambient_temperature,
            "lid_open": self.simulator.lid_open,
            "fire_strength": self.simulator.fire_strength,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        temp, delta, accel, auger, target, error = self._get_obs()
        print(
            f"step={self._steps:4d} | "
            f"temp={temp:6.1f}°F | "
            f"target={target:6.1f}°F | "
            f"error={error:+6.1f}°F | "
            f"auger={auger:.2f} | "
            f"lid={'open' if self.simulator.lid_open else 'closed'}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        temp = self.simulator.grill_temperature
        error = self.target_temperature - temp
        elapsed_fraction = self._steps / self.EPISODE_STEPS
        return np.array(
            [temp, self._delta_temperature, self._temperature_acceleration,
             self.simulator.auger_feed_rate, self.target_temperature, error, elapsed_fraction],
            dtype=np.float32,
        )

    def _compute_reward(self, grill_temp: float) -> float:
        """Negative absolute error — agent is rewarded for staying on target."""
        return -abs(grill_temp - self.target_temperature)

    def _maybe_toggle_lid(self):
        if self.simulator.lid_open:
            if self.np_random.random() < self.LID_CLOSE_PROB:
                self.simulator.close_lid()
        else:
            if self.np_random.random() < self.LID_OPEN_PROB:
                self.simulator.open_lid()
