import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from stable_baselines3 import SAC

from grill_env import GrillEnv

GRAPHS_DIR = "graphs"
MODEL_PATH = "models/grill_sac/best_model.zip"


def _nervous_checks(duration_s, checks_per_hour=1, open_duration_s=30):
    interval_s = int(3600 / checks_per_hour)
    start_s = duration_s // 4
    end_s   = (duration_s * 3) // 4
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
        "duration": 1800,
        "lid_events": [(900, 930)],
    },
    {
        "label": "Nervous griller — chicken, 90 min, 375°F, warm day",
        "ambient": 75.0,
        "target": 375.0,
        "duration": 5400,
        "lid_events": _nervous_checks(5400, checks_per_hour=2),
    },
    {
        "label": "Ribs — 3 hours, 275°F, cold day",
        "ambient": 40.0,
        "target": 275.0,
        "duration": 10800,
        "lid_events": [(3600, 3660), (7200, 7260)],
    },
    {
        "label": "Nervous griller — brisket, 6 hours, 225°F, freezing day",
        "ambient": 25.0,
        "target": 225.0,
        "duration": 21600,
        "lid_events": _nervous_checks(21600, checks_per_hour=2),
    },
]


PREHEAT_THRESHOLD = 5.0
PREHEAT_MAX_STEPS = 7200


def _preheat_nn(env, model, target):
    temps, fan_speeds, auger_rates = [], [], []
    obs = env._get_obs()
    for _ in range(PREHEAT_MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        prev_temp = env.simulator.grill_temperature
        env.simulator.step(float(action[0]), float(action[1]))
        delta_temp = env.simulator.grill_temperature - prev_temp
        env._temperature_acceleration = delta_temp - env._delta_temperature
        env._delta_temperature = delta_temp
        obs = env._get_obs()
        temps.append(env.simulator.grill_temperature)
        fan_speeds.append(env.simulator.fan_speed)
        auger_rates.append(env.simulator.auger_feed_rate)
        if env.simulator.grill_temperature >= target - PREHEAT_THRESHOLD:
            break
    return temps, fan_speeds, auger_rates


def run_nn(scenario, model) -> dict:
    env = GrillEnv()
    env.simulator.ambient_temperature = scenario["ambient"]
    env.target_temperature = scenario["target"]
    env.simulator.reset()
    env._delta_temperature = 0.0
    env._temperature_acceleration = 0.0

    preheat_temps, preheat_fan_speeds, preheat_auger_rates = _preheat_nn(env, model, scenario["target"])

    obs = env._get_obs()
    times, temps, fan_speeds, auger_rates = [], [], [], []
    opens  = {o for o, _ in scenario["lid_events"]}
    closes = {c for _, c in scenario["lid_events"]}

    for step in range(scenario["duration"]):
        if step in opens:
            env.simulator.open_lid()
        if step in closes:
            env.simulator.close_lid()

        action, _ = model.predict(obs, deterministic=True)
        prev_temp = env.simulator.grill_temperature
        env.simulator.step(float(action[0]), float(action[1]))
        delta_temp = env.simulator.grill_temperature - prev_temp
        env._temperature_acceleration = delta_temp - env._delta_temperature
        env._delta_temperature = delta_temp
        obs = env._get_obs()

        times.append(step / 60)
        temps.append(env.simulator.grill_temperature)
        fan_speeds.append(env.simulator.fan_speed)
        auger_rates.append(env.simulator.auger_feed_rate)

    preheat_minutes = len(preheat_temps) / 60
    return {
        "times": times,
        "temps": temps,
        "fan_speeds": fan_speeds,
        "auger_rates": auger_rates,
        "preheat_temps": preheat_temps,
        "preheat_fan_speeds": preheat_fan_speeds,
        "preheat_auger_rates": preheat_auger_rates,
        "preheat_minutes": preheat_minutes,
    }


def mae(temps, target):
    return sum(abs(t - target) for t in temps) / len(temps)


def _slugify(text):
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s]+", "_", text.strip())
    return text[:60]


def plot(scenarios, nn_results):
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.style.use("dark_background")

    BG_OUTER = "#1a1a2e"
    BG_INNER = "#16213e"
    PREHEAT_COLOR = "#2a2a4a"
    TEMP_COLOR = "#ff7043"
    FEED_COLOR = "#4fc3f7"
    FAN_COLOR  = "#a5d6a7"
    LID_COLOR  = "#ce93d8"
    TARGET_COLOR = "#b0b0b0"

    SLIDE_W, SLIDE_H = 13.33, 9.5

    for scenario, nn_res in zip(scenarios, nn_results):
        fig, (ax_temp, ax_feed, ax_fan) = plt.subplots(
            3, 1, figsize=(SLIDE_W, SLIDE_H),
            gridspec_kw={"height_ratios": [3, 1, 1]},
            sharex=True,
        )
        fig.patch.set_facecolor(BG_OUTER)
        for ax in (ax_temp, ax_feed, ax_fan):
            ax.set_facecolor(BG_INNER)

        target = scenario["target"]
        preheat_min = nn_res["preheat_minutes"]

        preheat_times = [-preheat_min + i / 60 for i in range(len(nn_res["preheat_temps"]))]

        # --- Temperature panel ---
        ax_temp.plot(preheat_times, nn_res["preheat_temps"],
                     color=TEMP_COLOR, linewidth=1.5, alpha=0.35)
        ax_temp.axvspan(-preheat_min, 0, color=PREHEAT_COLOR, alpha=0.8, label="Preheat")
        ax_temp.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
        ax_temp.plot(nn_res["times"], nn_res["temps"],
                     color=TEMP_COLOR, linewidth=2.0, label="SAC (NN)")
        ax_temp.axhline(target, color=TARGET_COLOR, linestyle="--", linewidth=1,
                        label=f"Target ({target:.0f}°F)")

        if scenario["lid_events"]:
            for open_t, close_t in scenario["lid_events"]:
                for ax in (ax_temp, ax_feed, ax_fan):
                    ax.axvline(open_t / 60,  color=LID_COLOR, linestyle=":", linewidth=1.0, alpha=0.7)
                    ax.axvline(close_t / 60, color=LID_COLOR, linestyle=":", linewidth=1.0, alpha=0.7)
            lid_patch = mpatches.Patch(color=LID_COLOR, label="Lid open/close")
            handles, labels = ax_temp.get_legend_handles_labels()
            ax_temp.legend(handles=[*handles, lid_patch], loc="lower right", fontsize=10,
                           facecolor=BG_OUTER, edgecolor="#444466", labelcolor="white")
        else:
            ax_temp.legend(loc="lower right", fontsize=10,
                           facecolor=BG_OUTER, edgecolor="#444466", labelcolor="white")

        nn_mae = mae(nn_res["temps"], target)
        ax_temp.text(0.99, 0.93, f"MAE: {nn_mae:.1f}°F",
                     transform=ax_temp.transAxes, ha="right", fontsize=11,
                     color="#dddddd",
                     bbox=dict(boxstyle="round,pad=0.4", fc=BG_OUTER, ec="#444466", alpha=0.9))

        ax_temp.set_ylabel("Temperature (°F)", color="#cccccc", fontsize=11)
        ax_temp.tick_params(colors="#aaaaaa")
        ax_temp.spines[:].set_color("#333355")
        ax_temp.grid(True, alpha=0.2, color="#445566")

        # --- Fan speed panel ---
        ax_feed.plot(preheat_times, nn_res["preheat_fan_speeds"],
                     color=FEED_COLOR, linewidth=1.2, alpha=0.35)
        ax_feed.axvspan(-preheat_min, 0, color=PREHEAT_COLOR, alpha=0.8)
        ax_feed.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
        ax_feed.plot(nn_res["times"], nn_res["fan_speeds"],
                     color=FEED_COLOR, linewidth=1.5, label="Fan Speed")
        ax_feed.set_ylabel("Fan Speed", color="#cccccc", fontsize=10)
        ax_feed.set_ylim(-0.05, 1.05)
        ax_feed.legend(loc="upper right", fontsize=9, facecolor=BG_OUTER, edgecolor="#444466", labelcolor="white")
        ax_feed.tick_params(colors="#aaaaaa")
        ax_feed.spines[:].set_color("#333355")
        ax_feed.grid(True, alpha=0.2, color="#445566")

        # --- Auger feed rate panel ---
        ax_fan.plot(preheat_times, nn_res["preheat_auger_rates"],
                    color=FAN_COLOR, linewidth=1.2, alpha=0.35)
        ax_fan.axvspan(-preheat_min, 0, color=PREHEAT_COLOR, alpha=0.8)
        ax_fan.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
        ax_fan.plot(nn_res["times"], nn_res["auger_rates"],
                    color=FAN_COLOR, linewidth=1.5, label="Auger Feed Rate")
        ax_fan.set_ylabel("Auger Feed Rate", color="#cccccc", fontsize=10)
        ax_fan.set_ylim(-0.05, 1.05)
        ax_fan.set_xlabel("Time (minutes)", color="#cccccc", fontsize=11)
        ax_fan.legend(loc="upper right", fontsize=9, facecolor=BG_OUTER, edgecolor="#444466", labelcolor="white")
        ax_fan.tick_params(colors="#aaaaaa")
        ax_fan.spines[:].set_color("#333355")
        ax_fan.grid(True, alpha=0.2, color="#445566")

        fig.suptitle("SAC Neural Network — Pellet Grill Temperature Control",
                     fontsize=13, color="#e0e0e0", y=0.99)
        ax_temp.set_title(scenario["label"], fontsize=12, color="#c0c0ff", pad=8)

        plt.tight_layout()

        slug = _slugify(scenario["label"])
        filename = f"{timestamp}_sac_{slug}.png"
        filepath = os.path.join(GRAPHS_DIR, filename)
        plt.savefig(filepath, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved: {filepath}")
        plt.close(fig)


def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = SAC.load(MODEL_PATH)

    nn_results = []
    for scenario in SCENARIOS:
        print(f"Running: {scenario['label']}")
        nn_results.append(run_nn(scenario, model))

    plot(SCENARIOS, nn_results)

    print("\n--- Summary ---")
    for scenario, nn_res in zip(SCENARIOS, nn_results):
        nn_mae = mae(nn_res["temps"], scenario["target"])
        print(f"{scenario['label']}: MAE={nn_mae:.1f}°F")


if __name__ == "__main__":
    main()
