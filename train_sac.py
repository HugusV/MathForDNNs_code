from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
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
    model.save("sac_ackermann_20x20")


if __name__ == "__main__":
    main()
