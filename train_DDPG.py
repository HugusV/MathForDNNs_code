from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from envs import Ackermann2DEnv


def main():
    env = Ackermann2DEnv(
        world_size=20.0,
        n_obstacles=5,
        n_rays=31,
        ray_fov_deg=180.0,
        ray_max=7.0,
        dt=0.1,
        v_min=0.2,
    )

    venv = DummyVecEnv([lambda: env])

    # DDPG needs exploration noise (SAC explores via entropy, DDPG does not)
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)   # you can try 0.2 if exploration is weak
    )

    model = DDPG(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        buffer_size=80_000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        verbose=1,
    )

    model.learn(total_timesteps=300_000)
    model.save("ddpg_ackermann_20x20")


if __name__ == "__main__":
    main()
