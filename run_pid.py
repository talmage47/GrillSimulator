import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from grill_simulator import PelletGrillSimulator
from pid_controller import PIDController


AMBIENT_TEMP = 55.0     # cold day
TARGET_TEMP = 225.0     # low-and-slow smoke
EPISODE_STEPS = 3600    # 60 simulated minutes
LID_OPEN_AT = 1800      # open lid at 30 min
LID_CLOSE_AT = 1920     # close lid at 32 min


def run():
    sim = PelletGrillSimulator(ambient_temperature=AMBIENT_TEMP)
    pid = PIDController()  # kp=0.002, ki=0.0004, kd=3.0
    sim.reset()
    pid.reset()

    times, temps, auger_rates, fire_strengths = [], [], [], []
    lid_events = []

    for step in range(EPISODE_STEPS):
        t = step * sim.timestep

        if step == LID_OPEN_AT:
            sim.open_lid()
            lid_events.append(("open", t))
        elif step == LID_CLOSE_AT:
            sim.close_lid()
            lid_events.append(("close", t))

        auger = pid.step(sim.grill_temperature, TARGET_TEMP)
        sim.step(auger)

        times.append(t / 60)  # convert to minutes
        temps.append(sim.grill_temperature)
        auger_rates.append(sim.auger_feed_rate)
        fire_strengths.append(sim.fire_strength)

    _plot(times, temps, auger_rates, fire_strengths, lid_events)


def _plot(times, temps, auger_rates, fire_strengths, lid_events):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("PID Controller — Pellet Grill Simulation", fontsize=13)

    # -- Temperature plot --
    ax1.plot(times, temps, color="tomato", label="Grill temp")
    ax1.axhline(TARGET_TEMP, color="gray", linestyle="--", linewidth=1, label=f"Target ({TARGET_TEMP}°F)")
    ax1.set_ylabel("Temperature (°F)")
    ax1.set_ylim(0, max(temps) * 1.15)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # -- Auger / fire plot --
    ax2.plot(times, auger_rates, color="steelblue", label="Auger feed rate")
    ax2.plot(times, fire_strengths, color="orange", linestyle="--", label="Fire strength")
    ax2.set_ylabel("Rate (0–1)")
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Mark lid events on both axes
    for ax in (ax1, ax2):
        for event, t in lid_events:
            color = "purple"
            ax.axvline(t / 60, color=color, linestyle=":", linewidth=1.5)

    if lid_events:
        lid_patch = mpatches.Patch(color="purple", label="Lid open/close")
        ax1.legend(handles=[*ax1.get_legend().legend_handles, lid_patch], loc="lower right")

    plt.tight_layout()
    plt.savefig("pid_run.png", dpi=150)
    print("Saved plot to pid_run.png")
    plt.show()


if __name__ == "__main__":
    run()
