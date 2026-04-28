import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from grill_env import GrillEnv


TIMESTEPS = 1_000_000
MODEL_PATH = "models/grill_sac"
LOG_PATH = "logs/"


def main():
    # Vectorized training env (4 parallel envs)
    train_env = make_vec_env(GrillEnv, n_envs=4)

    # Separate env for evaluation during training
    eval_env = make_vec_env(GrillEnv, n_envs=1)

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=0.0001,
        verbose=1,
        tensorboard_log=LOG_PATH,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_PATH,
        log_path=LOG_PATH,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f"Training for {TIMESTEPS:,} timesteps...")
    model.learn(total_timesteps=TIMESTEPS, callback=eval_callback, progress_bar=True)

    model.save(MODEL_PATH + "_final")
    print(f"Saved final model to {MODEL_PATH}_final.zip")


if __name__ == "__main__":
    main()
