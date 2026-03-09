import time
from stable_baselines3 import SAC
from envs import Ackermann2DEnv


def main():
    env = Ackermann2DEnv(
        world_size=20.0,
        n_obstacles=7,
        n_rays=31,           
        ray_max=7.0,
        dt=0.1,
        v_min=0.2,         
    )

    model = SAC.load("sac_ackermann_20x20")

    obs, _ = env.reset()

    for step in range(3000):
        action, _ = model.predict(obs, deterministic=True)

        # ✅ Force forward-only throttle for demo/visualization
        action = action.copy()
        action[0] = max(action[0], 0.0)

        # Optional prints every 10 steps (avoid terminal spam)
        if step % 10 == 0:
            print(f"Pose -> x={env.x:6.2f}, y={env.y:6.2f}, yaw={env.yaw:6.2f} rad, v={env.v:4.2f} m/s")
            print(f"Action -> throttle: {action[0]:+.3f}, steering: {action[1]:+.3f}")

        obs, reward, terminated, truncated, info = env.step(action)

        env.render()
        time.sleep(0.03)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
