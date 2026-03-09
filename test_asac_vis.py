import time
from stable_baselines3 import SAC
import numpy as np

from envs import Ackermann2DEnv, SafeDWATTCLyapEnv


def main():
    base = Ackermann2DEnv()
    env = SafeDWATTCLyapEnv(base)

    model = SAC.load("asac_ackermann_20x20_dwa_ttc_lyap")

    obs, _ = env.reset()
    for _ in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        action = action.copy()
        action = np.clip(action, -1.0, 1.0) # décélaration ou accélération 

        obs, r, term, trunc, info = env.step(action)
        env.render()
        time.sleep(0.03)
        if term or trunc:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
