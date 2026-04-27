import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from stable_baselines3 import SAC

from grill_env import GrillEnv
from grill_simulator import PelletGrillSimulator
from pid_controller import PIDController

GRAPHS_DIR = "graphs"


MODEL_PATH = "models/grill_sac/best_model.zip"

def _nervous_checks(duration_s, checks_per_hour=1, open_duration_s=30):
    """
    Lid opens at a rate of checks_per_hour, distributed across the middle
    50% of the cook (from 25% to 75% of total duration).
    Check count scales directly with cook duration.
    """
    interval_s = int(3600 / checks_per_hour)
    start_s = duration_s // 4           # 25% into the cook
    end_s   = (duration_s * 3) // 4    # 75% into the cook
    return [
        (s, s + open_duration_s)
        for s in range(start_s, end_s, interval_s)
        if s + open_duration_s <= duration_s
    ]


SCENARIOS = [
    {
        "label": "Steaks — 30 min, 450°F, mild day",
        "ambient": 65.0,
        "target": 450.0,
        "duration": 1800,       # 30 min
        "lid_events": [(900, 930)],     # one flip at 15 min
    },
    {
        "label": "Nervous griller — chicken, 90 min, 375°F, warm day",
        "ambient": 75.0,
        "target": 375.0,
        "duration": 5400,       # 90 min
        "lid_events": _nervous_checks(5400, checks_per_hour=2),
    },
    {
        "label": "Ribs — 3 hours, 275°F, cold day",
        "ambient": 40.0,
        "target": 275.0,
        "duration": 10800,      # 3 hours
        "lid_events": [(3600, 3660), (7200, 7260)],     # spritz at 1h and 2h
    },
    {
        "label": "Nervous griller — brisket, 6 hours, 225°F, freezing day",
        "ambient": 25.0,
        "target": 225.0,
        "duration": 21600,      # 6 hours
        "lid_events": _nervous_checks(21600, checks_per_hour=2),
    },
]


PREHEAT_THRESHOLD = 5.0     # °F within target to consider grill "up to temp"
PREHEAT_MAX_STEPS = 7200    # 2 hour cap on preheat


def _preheat_pid(sim, pid, target):
    """Run PID until within PREHEAT_THRESHOLD of target. Returns preheat temps for plotting."""
    temps, augers = [], []
    for _ in range(PREHEAT_MAX_STEPS):
        auger = pid.step(sim.grill_temperature, target)
        sim.step(auger)
        temps.append(sim.grill_temperature)
        augers.append(sim.auger_feed_rate)
        if sim.grill_temperature >= target - PREHEAT_THRESHOLD:
            break
    return temps, augers


def _preheat_nn(env, model, target):
    """Run NN until within PREHEAT_THRESHOLD of target. Returns preheat temps for plotting."""
    temps, augers = [], []
    obs = env._get_obs()  # elapsed_fraction=0 during preheat
    for _ in range(PREHEAT_MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        prev_temp = env.simulator.grill_temperature
        env.simulator.step(float(action[0]))
        delta_temp = env.simulator.grill_temperature - prev_temp
        env._temperature_acceleration = delta_temp - env._delta_temperature
        env._delta_temperature = delta_temp
        # keep _steps=0 so elapsed_fraction stays 0 throughout preheat
        obs = env._get_obs()
        temps.append(env.simulator.grill_temperature)
        augers.append(env.simulator.auger_feed_rate)
        if env.simulator.grill_temperature >= target - PREHEAT_THRESHOLD:
            break
    return temps, augers


def run_pid(scenario) -> dict:
    sim = PelletGrillSimulator(ambient_temperature=scenario["ambient"])
    pid = PIDController()
    sim.reset()
    pid.reset()

    preheat_temps, preheat_augers = _preheat_pid(sim, pid, scenario["target"])

    times, temps, augers = [], [], []
    opens  = {o for o, _ in scenario["lid_events"]}
    closes = {c for _, c in scenario["lid_events"]}

    for step in range(scenario["duration"]):
        if step in opens:
            sim.open_lid()
        if step in closes:
            sim.close_lid()

        auger = pid.step(sim.grill_temperature, scenario["target"])
        sim.step(auger)

        times.append(step / 60)
        temps.append(sim.grill_temperature)
        augers.append(sim.auger_feed_rate)

    preheat_minutes = len(preheat_temps) / 60
    return {"times": times, "temps": temps, "augers": augers,
            "preheat_temps": preheat_temps, "preheat_minutes": preheat_minutes}


def run_nn(scenario, model) -> dict:
    env = GrillEnv()
    env.simulator.ambient_temperature = scenario["ambient"]
    env.target_temperature = scenario["target"]
    env.simulator.reset()
    env._delta_temperature = 0.0
    env._temperature_acceleration = 0.0

    preheat_temps, preheat_augers = _preheat_nn(env, model, scenario["target"])

    obs = env._get_obs()
    times, temps, augers = [], [], []
    opens  = {o for o, _ in scenario["lid_events"]}
    closes = {c for _, c in scenario["lid_events"]}

    for step in range(scenario["duration"]):
        if step in opens:
            env.simulator.open_lid()
        if step in closes:
            env.simulator.close_lid()

        action, _ = model.predict(obs, deterministic=True)
        prev_temp = env.simulator.grill_temperature
        env.simulator.step(float(action[0]))
        delta_temp = env.simulator.grill_temperature - prev_temp
        env._temperature_acceleration = delta_temp - env._delta_temperature
        env._delta_temperature = delta_temp
        obs = env._get_obs()

        times.append(step / 60)
        temps.append(env.simulator.grill_temperature)
        augers.append(env.simulator.auger_feed_rate)

    preheat_minutes = len(preheat_temps) / 60
    return {"times": times, "temps": temps, "augers": augers,
            "preheat_temps": preheat_temps, "preheat_minutes": preheat_minutes}


def mae(temps, target):
    return sum(abs(t - target) for t in temps) / len(temps)


def _slugify(text):
    """Convert a scenario label into a safe filename fragment."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s]+", "_", text.strip())
    return text[:60]


def plot(scenarios, pid_results, nn_results):
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.style.use("dark_background")

    # PowerPoint widescreen slide: 13.33" × 7.5"
    SLIDE_W, SLIDE_H = 13.33, 7.5

    for scenario, pid_res, nn_res in zip(scenarios, pid_results, nn_results):
        fig, ax = plt.subplots(figsize=(SLIDE_W, SLIDE_H))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        target = scenario["target"]

        pid_preheat_min = pid_res["preheat_minutes"]
        nn_preheat_min  = nn_res["preheat_minutes"]
        preheat_end = max(pid_preheat_min, nn_preheat_min)

        pid_preheat_times = [-pid_preheat_min + i/60 for i in range(len(pid_res["preheat_temps"]))]
        nn_preheat_times  = [-nn_preheat_min  + i/60 for i in range(len(nn_res["preheat_temps"]))]

        ax.plot(pid_preheat_times, pid_res["preheat_temps"], color="#4fc3f7", linewidth=1.5, alpha=0.35)
        ax.plot(nn_preheat_times,  nn_res["preheat_temps"],  color="#ff7043", linewidth=1.5, alpha=0.35)
        ax.axvspan(-preheat_end, 0, color="#2a2a4a", alpha=0.8, label="Preheat")
        ax.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")

        ax.plot(pid_res["times"], pid_res["temps"], color="#4fc3f7", linewidth=2.0, label="PID")
        ax.plot(nn_res["times"],  nn_res["temps"],  color="#ff7043", linewidth=2.0, label="NN (SAC)")
        ax.axhline(target, color="#b0b0b0", linestyle="--", linewidth=1, label=f"Target ({target}°F)")

        if scenario["lid_events"]:
            for open_t, close_t in scenario["lid_events"]:
                ax.axvline(open_t / 60,  color="#ce93d8", linestyle=":", linewidth=1.0, alpha=0.7)
                ax.axvline(close_t / 60, color="#ce93d8", linestyle=":", linewidth=1.0, alpha=0.7)
            lid_patch = mpatches.Patch(color="#ce93d8", label="Lid open/close")
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=[*handles, lid_patch], loc="lower right", fontsize=10,
                      facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white")
        else:
            ax.legend(loc="lower right", fontsize=10,
                      facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white")

        pid_mae = mae(pid_res["temps"], target)
        nn_mae  = mae(nn_res["temps"],  target)
        ax.text(0.99, 0.93, f"MAE — PID: {pid_mae:.1f}°F   NN: {nn_mae:.1f}°F",
                transform=ax.transAxes, ha="right", fontsize=11,
                color="#dddddd", bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e", ec="#444466", alpha=0.9))

        fig.suptitle("PID vs Neural Network — Pellet Grill Temperature Control",
                     fontsize=13, color="#e0e0e0", y=0.98)
        ax.set_title(scenario["label"], fontsize=12, color="#c0c0ff", pad=8)
        ax.set_ylabel("Temperature (°F)", color="#cccccc", fontsize=11)
        ax.set_xlabel("Time (minutes)", color="#cccccc", fontsize=11)
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333355")
        ax.grid(True, alpha=0.2, color="#445566")

        plt.tight_layout()

        slug = _slugify(scenario["label"])
        filename = f"{timestamp}_{slug}.png"
        filepath = os.path.join(GRAPHS_DIR, filename)
        plt.savefig(filepath, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved: {filepath}")
        plt.close(fig)


def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = SAC.load(MODEL_PATH)

    pid_results, nn_results = [], []
    for scenario in SCENARIOS:
        print(f"Running: {scenario['label']}")
        pid_results.append(run_pid(scenario))
        nn_results.append(run_nn(scenario, model))

    plot(SCENARIOS, pid_results, nn_results)

    print("\n--- Summary ---")
    for scenario, pid_res, nn_res in zip(SCENARIOS, pid_results, nn_results):
        pid_mae = mae(pid_res["temps"], scenario["target"])
        nn_mae  = mae(nn_res["temps"],  scenario["target"])
        winner = "NN" if nn_mae < pid_mae else "PID"
        print(f"{scenario['label']}: PID={pid_mae:.1f}°F  NN={nn_mae:.1f}°F  → {winner} wins")


if __name__ == "__main__":
    main()
