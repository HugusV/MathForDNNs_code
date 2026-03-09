from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from envs import Ackermann2DEnv, SafeGuidedDWAEnv

import os
from stable_baselines3.common.monitor import Monitor

def make_env():
    base = Ackermann2DEnv(
    )

    wrapped = SafeGuidedDWAEnv(
        base,
        # Hybrid A* guidance
        guide_max_steps=250,
        guide_beta0=0.85,
        guide_beta_min=0.0,
        planner_grid_res=0.5,
        planner_yaw_bins=24,
        planner_step=0.6,
        lookahead=1.2,
        target_speed=1.5,

        # DWA
        dwa_enable=True,
        dwa_horizon_steps=12,
        dwa_dt=0.1,
        dwa_n_steer=7,
        dwa_n_throttle=5,

        # TTC
        ttc_enable=True,
        ttc_threshold=1.2,

        # Lyapunov
        lyap_enable=True,
        lyap_c=1.5,

        # Blend
        blend_with_sac=True,
        dwa_weight=0.6,
    )

    return wrapped


def main():
    venv = DummyVecEnv([make_env])

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
        ent_coef="auto",
        verbose=1,
    )

    model.learn(total_timesteps=100_000)
    model.save("sac_ackermann_20x20_hybrid_astar_dwa_ttc_lyap")

    # enregistrement des rewards
    log_dir = "./logs/asac_hybrid/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = make_env()
    env = Monitor(env, log_dir)
    
    model = SAC("MlpPolicy", env, ent_coef="auto", verbose=1)
    model.learn(total_timesteps=100000)
    model.save("sac_hybrid_astar_model")


if __name__ == "__main__":
    main()
