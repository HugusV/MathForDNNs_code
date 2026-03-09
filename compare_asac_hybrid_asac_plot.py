import math
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from envs import Ackermann2DEnv, SafeDWATTCLyapEnv, SafeGuidedDWAEnv


def get_base_env(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    while hasattr(env, "env"):
        env = env.env
    return env


def run_episode(env, model, max_steps=2000, clamp_forward=False, seed=None):
    # NOTE: clamp_forward est passé à False par défaut pour permettre le freinage

    obs, _ = env.reset(seed=seed)
    base = get_base_env(env)

    traj = [(float(base.x), float(base.y))]
    dist_traveled = 0.0
    reached_goal = False
    time_sec = np.nan

    x_prev, y_prev = float(base.x), float(base.y)

    # --- NOUVEAU : Suivi de la récompense cumulée ---
    cumulative_rewards = [0.0]
    current_score = 0.0
    # ------------------------------------------------

    for k in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)

        if clamp_forward:
            action = action.copy()
            action[0] = max(action[0], 0.0)

        # On récupère la récompense (reward) à chaque étape
        obs, reward, terminated, truncated, info = env.step(action)

        # --- NOUVEAU : Addition et stockage de la récompense ---
        current_score += float(reward)
        cumulative_rewards.append(current_score)
        # -------------------------------------------------------

        base = get_base_env(env)
        x, y = float(base.x), float(base.y)

        traj.append((x, y))
        dist_traveled += math.hypot(x - x_prev, y - y_prev)
        x_prev, y_prev = x, y

        if info.get("reached_goal", False):
            reached_goal = True
            time_sec = (k + 1) * float(base.dt)
            break

        if terminated or truncated:
            break

    base = get_base_env(env)
    goal = np.array(base.goal, dtype=float)
    obstacles = list(base.obstacles)

    # On retourne cumulative_rewards à la fin
    return np.array(traj), reached_goal, time_sec, dist_traveled, goal, obstacles, cumulative_rewards


def draw_obstacles(ax, obstacles):
    for shape in obstacles:
        if shape.geom_type == "Polygon":
            x, y = shape.exterior.xy
            ax.fill(x, y, color="black") # Ajout de la couleur noire pour les obstacles
        elif shape.geom_type == "MultiPolygon":
            for poly in shape.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, color="black")


def fmt_time(ok, t):
    return f"{t:.2f} s" if ok and np.isfinite(t) else "FAIL"


def main():
    # Seed aléatoire unique pour s'assurer que les deux modèles affrontent la même carte
    episode_seed = np.random.randint(0, 10_000)
    ENV_KW = dict()

    # ================
    # 1) ASAC
    # ================
    base_asac = Ackermann2DEnv(**ENV_KW)
    env_asac = SafeDWATTCLyapEnv(base_asac)

    model_asac = SAC.load("asac_ackermann_20x20_dwa_ttc_lyap")

    # On récupère la liste cumulative_rewards_asac
    traj_asac, ok_asac, t_asac, d_asac, goal_a, obs_a, cumulative_rewards_asac = run_episode(
        env_asac, model_asac, clamp_forward=False, seed=episode_seed
    )
    env_asac.close()

    # ================
    # 2) Hybrid SAC
    # ================
    base_h = Ackermann2DEnv(**ENV_KW)
    env_hybrid = SafeGuidedDWAEnv(base_h)

    model_hybrid = SAC.load("sac_ackermann_20x20_hybrid_astar_dwa_ttc_lyap")

    # On récupère la liste cumulative_rewards_h
    traj_h, ok_h, t_h, d_h, goal_h, obs_h, cumulative_rewards_h = run_episode(
        env_hybrid, model_hybrid, clamp_forward=False, seed=episode_seed
    )
    env_hybrid.close()

    # Même carte pour les deux
    goal = goal_h
    obstacles = obs_h
    ws = float(base_h.world_size)

    # =========================
    # Plot : 1 figure, 2 graphiques (1 ligne, 2 colonnes)
    # =========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- GRAPHIQUE 1 : Trajectoires ---
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlim(0, ws)
    ax1.set_ylim(0, ws)
    ax1.set_title(f"Comparing ASAC and Hybrid A* ASAC")

    draw_obstacles(ax1, obstacles)
    ax1.plot(goal[0], goal[1], marker="*", markersize=14, color="gold")

    # Tracé des trajectoires
    ax1.plot(traj_asac[:, 0], traj_asac[:, 1], linewidth=2, color="blue", label="ASAC")
    ax1.plot(traj_h[:, 0], traj_h[:, 1], linewidth=2, color="orange", label="ASAC + Hybrid A*")
    ax1.scatter(traj_asac[0, 0], traj_asac[0, 1], s=60, color="green", label="Start")

    text = (
        f"Seed = {episode_seed}\n"
        f"ASAC : time={fmt_time(ok_asac, t_asac)} | dist={d_asac:.2f} m\n"
        f"Hybrid : time={fmt_time(ok_h, t_h)} | dist={d_h:.2f} m"
    )

    ax1.text(
        0.02, 0.98,
        text,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax1.legend()

    # --- GRAPHIQUE 2 : Récompense Cumulée ---
    ax2.set_title("Reward comparison of both models in testing performance")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Cumulative Reward")

    # Tracé des récompenses (en gardant les mêmes couleurs pour la cohérence)
    ax2.plot(cumulative_rewards_asac, linewidth=2, color="blue", label="ASAC")
    ax2.plot(cumulative_rewards_h, linewidth=2, color="orange", label="ASAC + Hybrid A*")
    
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()