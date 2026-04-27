# GrillSimulator

Compares a **PID controller** and a **reinforcement learning neural network** (SAC) at managing temperature in a simulated pellet-fed grill. Both controllers are given identical scenarios and evaluated by mean absolute error (MAE) from the target temperature over the cook.

## Results

The NN outperforms the PID in every scenario, with the advantage growing for longer cooks:

| Scenario | PID MAE | NN MAE |
|---|---|---|
| Steaks — 30 min, 450°F | 8.9°F | 7.0°F |
| Nervous griller — chicken, 90 min, 375°F | 6.8°F | 2.0°F |
| Ribs — 3 hours, 275°F | 5.1°F | 1.4°F |
| Nervous griller — brisket, 6 hours, 225°F | 3.3°F | 0.7°F |

## How it works

### Simulator (`grill_simulator.py`)
Physics-based pellet grill model. Each timestep (1 second) computes:
- **Fire strength** — lags behind auger feed rate with a 15-second time constant (pellets take time to ignite/extinguish)
- **Heat in** — proportional to fire strength
- **Heat loss** — proportional to the difference between grill and ambient temperature (Newton's law of cooling); multiplied 4x when the lid is open
- **Temperature change** — net heat divided by thermal mass

The agent observes only `[grill_temperature, delta_temperature, temperature_acceleration, auger_feed_rate, target_temperature, error, elapsed_fraction]`. Ambient temperature and lid state are hidden — the agent must infer disturbances from temperature response alone, just like a real grill controller.

### PID Controller (`pid_controller.py`)
Standard PID with anti-windup. Tuned gains: `kp=0.005, ki=0.0002, kd=0.5`. The PID sees the same limited observation as the NN (current temp and target) and outputs an auger feed rate in `[0, 1]`.

### Neural Network (`grill_env.py`, `train.py`)
A [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) agent trained via reinforcement learning using [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). The gymnasium environment randomizes ambient temperature (20–100°F), target temperature (180–500°F), and introduces stochastic lid-open events during training so the agent learns robust control across varied conditions.

Reward: `-abs(grill_temperature - target_temperature)` per step.

### Comparison (`compare.py`)
Runs both controllers through four scenarios of varying duration (30 min to 6 hours), temperature (225–450°F), ambient conditions, and lid-open events. Cook timers start only after the grill has preheated to within 5°F of the target.

### SAC Analysis (`sac_analysis.py`)
Runs the same four scenarios with the SAC controller only and produces a 3-panel graph per scenario showing temperature, auger feed rate, and fan speed (fire strength) over time. Useful for inspecting how the NN modulates its two control outputs across different cooks.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Train the NN (saves best model to models/grill_sac/)
python train.py

# Run PID only with a plot
python run_pid.py

# Compare PID vs NN across all scenarios
python compare.py

# Run SAC-only analysis with feed rate and fan speed graphs
python sac_analysis.py
```
