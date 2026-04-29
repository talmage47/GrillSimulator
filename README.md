# GrillSimulator

Compares a **PID controller** and a **reinforcement learning neural network** (SAC) at managing temperature in a simulated pellet grill. Both controllers are evaluated across four cook scenarios by mean absolute error (MAE) from the target temperature.

## Results

The NN outperforms the PID on longer cooks, with the advantage growing with cook duration:

| Scenario | Duration | Target | PID MAE | NN MAE | Outcome |
|---|---|---|---|---|---|
| Steaks | 30 min | 450°F | 8.50°F | 10.10°F | PID wins |
| Chicken | 90 min | 375°F | 2.10°F | 1.90°F | NN wins |
| Ribs | 3 hours | 275°F | 1.09°F | 1.13°F | ≈ Tie |
| Brisket | 6 hours | 225°F | 0.50°F | 0.30°F | NN wins |

The PID is competitive on short, high-temperature cooks. The NN's advantage stems from its ability to independently modulate fan speed and auger duty cycle — something the coupled PID cannot do.

## How it works

### Simulator (`grill_simulator.py`)

Physics-based pellet grill model running at 1-second timesteps:

- **Fan speed** (continuous, 0–1) controls airflow into the firepot. Airflow responds with a 5-second time constant.
- **Auger** (binary on/off) feeds pellets into the firepot. Fuel supply responds with a 3-second time constant.
- **Combustion rate** = `min(airflow, fuel_supply)` — fire requires both air and fuel; excess of either is wasted.
- **Heat in** — proportional to combustion rate, max 45 heat units at full combustion.
- **Heat loss** — Newton's law of cooling (coefficient 0.1/°F above ambient); multiplied 4× when the lid is open. High fan speeds add a small forced-convection cooling effect.
- **Temperature change** — net heat divided by thermal mass (50 °F·s per heat unit).

### PID Controller (`pid_controller.py`)

Standard PID with integral anti-windup. Gains: `kp=0.015, ki=0.002, kd=0.2`. Outputs a single drive signal in `[0, 1]` applied to both fan speed and auger equally — auger switches on when the signal exceeds 0.5.

### Neural Network (`grill_env.py`, `train.py`)

A [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) agent trained via reinforcement learning using [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).

**Observation space (8D):** `[grill_temperature, delta_temperature, temperature_acceleration, fan_speed, auger_feed_rate, target_temperature, error, elapsed_fraction]`. Ambient temperature and lid state are hidden — the agent must infer disturbances from temperature response alone.

**Action space (2D):** `[fan_signal, auger_signal]` — both continuous `[0, 1]`. Fan signal maps directly to fan speed; auger signal is thresholded at 0.5 internally to produce binary on/off delivery.

**Reward:** `-abs(grill_temperature - target_temperature)` per step.

**Training:** 1,000,000 timesteps, learning rate 1×10⁻⁴, 3,600-step episodes (60 simulated minutes), 4 parallel environments. Best checkpoint saved at ~760,000 steps.

The environment randomizes ambient temperature (20–100°F), target temperature (180–500°F), and introduces stochastic lid-open events so the agent learns robust control across varied conditions.

### Comparison (`compare.py`)

Runs both controllers through four scenarios of varying duration (30 min to 6 hours), temperature (225–450°F), ambient conditions, and lid events. Cook timers start only after the grill preheats to within 5°F of the target (capped at 2 hours). Also generates a 3-panel controls graph for the ribs scenario showing temperature, fan speed, and auger duty cycle over time.

### SAC Analysis (`sac_analysis.py`)

Runs all four scenarios with the SAC controller and produces a 3-panel graph per scenario showing temperature, fan speed, and auger duty cycle. Useful for inspecting how the NN modulates its two outputs independently.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Train the NN (saves best model to models/grill_sac/)
python train.py

# Compare PID vs NN across all scenarios
python compare.py

# Run PID only with a plot
python run_pid.py

# Run SAC-only analysis with fan speed and auger duty cycle graphs
python sac_analysis.py

# Sweep PID gain combinations
python tune_pid.py
```
