import time
import numpy as np
from stable_baselines3 import DDPG

from envs import Ackermann2DEnv


def main():
    env = Ackermann2DEnv(
        world_size=20.0,
        n_obstacles=0,
        n_rays=31,           # MUST match training
        ray_fov_deg=180.0,
        ray_max=7.0,
        dt=0.1,
        v_min=0.2,
    )

    model = DDPG.load("ddpg_ackermann_20x20")

    obs, _ = env.reset()

    for step in range(3000):
        action, _ = model.predict(obs, deterministic=True)

        # Optional: force forward-only throttle for visualization stability
        action = action.copy()
        action = np.clip(action, -1.0, 1.0)

        # Optional prints every 20 steps
        if step % 20 == 0:
            print(f"Pose -> x={env.x:6.2f}, y={env.y:6.2f}, v={env.v:4.2f} m/s")
            print(f"Action -> throttle: {action[0]:+.3f}, steering: {action[1]:+.3f}")

        obs, reward, terminated, truncated, info = env.step(action)

        env.render()
        time.sleep(0.03)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
