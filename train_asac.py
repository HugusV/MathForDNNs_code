from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from envs import Ackermann2DEnv, SafeDWATTCLyapEnv

import os
from stable_baselines3.common.monitor import Monitor

def make_env():
    base = Ackermann2DEnv(
    )

    # No Hybrid A* here
    wrapped = SafeDWATTCLyapEnv(
        base,
        dwa_enable=True,
        dwa_horizon_steps=12,
        dwa_dt=0.1,
        dwa_n_steer=7,
        dwa_n_throttle=5,
        dwa_weight=0.6,
        ttc_enable=True,
        ttc_threshold=1.2,
        lyap_enable=True,
        lyap_c=1.5,
    )
    return wrapped


def main():
    venv = DummyVecEnv([make_env])

    # ASAC = SAC with adaptive entropy temperature (ent_coef="auto")
    model = SAC(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        buffer_size=80_000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",     # adaptive entropy => ASAC
        verbose=1,
    )

    model.learn(total_timesteps=100_000)
    model.save("asac_ackermann_20x20_dwa_ttc_lyap")

    # enregistrement des rewards
    log_dir = "./logs/asac_pure/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = make_env()
    env = Monitor(env, log_dir) # Enregistre les rewards par épisode
    
    model = SAC("MlpPolicy", env, ent_coef="auto", verbose=1)
    model.learn(total_timesteps=100000)
    model.save("asac_ackermann_model")


if __name__ == "__main__":
    main()
