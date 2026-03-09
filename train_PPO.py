from stable_baselines3 import PPO
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

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,   # same LR as SAC
        gamma=0.99,           # same gamma as SAC
        gae_lambda=0.95,
        n_steps=2048,         # rollout length (on-policy equivalent of "collecting data")
        batch_size=128,       # similar minibatch size
        n_epochs=10,          # PPO standard
        clip_range=0.2,
        ent_coef=0.0,         # you can try 0.01 if exploration is weak
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
    )

    model.learn(total_timesteps=300_000)
    model.save("ppo_ackermann_20x20")


if __name__ == "__main__":
    main()
