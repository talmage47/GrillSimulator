import matplotlib.pyplot as plt

from grill_simulator import PelletGrillSimulator
from pid_controller import PIDController


AMBIENT_TEMP = 55.0
TARGET_TEMP = 225.0
EPISODE_STEPS = 3600
LID_OPEN_AT = 1800
LID_CLOSE_AT = 1920

GAIN_SETS = [
    {"label": "kp=0.005, ki=0.0002, kd=0.5", "kp": 0.005, "ki": 0.0002, "kd": 0.5},
    {"label": "kp=0.005, ki=0.0002, kd=0.7", "kp": 0.005, "ki": 0.0002, "kd": 0.7},
    {"label": "kp=0.005, ki=0.0003, kd=0.5", "kp": 0.005, "ki": 0.0003, "kd": 0.5},
    {"label": "kp=0.004, ki=0.0002, kd=0.5", "kp": 0.004, "ki": 0.0002, "kd": 0.5},
]


def run_gains(kp, ki, kd):
    sim = PelletGrillSimulator(ambient_temperature=AMBIENT_TEMP)
    pid = PIDController(kp=kp, ki=ki, kd=kd)
    sim.reset()
    pid.reset()

    times, temps, fan_speeds = [], [], []

    for step in range(EPISODE_STEPS):
        if step == LID_OPEN_AT:
            sim.open_lid()
        elif step == LID_CLOSE_AT:
            sim.close_lid()

        drive = pid.step(sim.grill_temperature, TARGET_TEMP)
        sim.step(drive, drive)  # PID drives fan and auger equally

        times.append(step / 60)
        temps.append(sim.grill_temperature)
        fan_speeds.append(sim.fan_speed)

    return times, temps, fan_speeds


def main():
    fig, axes = plt.subplots(len(GAIN_SETS), 1, figsize=(13, 3.5 * len(GAIN_SETS)), sharex=True)
    fig.suptitle("PID Gain Comparison", fontsize=13)

    for ax, gains in zip(axes, GAIN_SETS):
        times, temps, fan_speeds = run_gains(gains["kp"], gains["ki"], gains["kd"])

        ax2 = ax.twinx()
        ax2.plot(times, fan_speeds, color="steelblue", alpha=0.4, linewidth=1, label="fan")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("Fan Speed", color="steelblue", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=7)

        ax.plot(times, temps, color="tomato", linewidth=1.5)
        ax.axhline(TARGET_TEMP, color="gray", linestyle="--", linewidth=1)
        ax.axvline(LID_OPEN_AT / 60, color="purple", linestyle=":", linewidth=1.5)
        ax.axvline(LID_CLOSE_AT / 60, color="purple", linestyle=":", linewidth=1.5)
        ax.set_ylabel("Temp (°F)")
        ax.set_title(gains["label"], fontsize=9)
        ax.grid(True, alpha=0.3)

        mae = sum(abs(t - TARGET_TEMP) for t in temps) / len(temps)
        ax.text(0.99, 0.05, f"MAE={mae:.1f}°F", transform=ax.transAxes,
                ha="right", fontsize=8, color="dimgray")

    axes[-1].set_xlabel("Time (minutes)")
    plt.tight_layout()
    plt.savefig("pid_tuning.png", dpi=150)
    print("Saved to pid_tuning.png")


if __name__ == "__main__":
    main()
