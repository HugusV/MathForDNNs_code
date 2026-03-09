import math
import numpy as np
import gymnasium as gym
from shapely.geometry import Point


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class SafeDWATTCLyapEnv(gym.Wrapper):
    """
    Gymnasium wrapper:
      - DWA local action refinement
      - TTC safety layer (from lidar rays)
      - Lyapunov-like stability filter (heading to goal)

    No Hybrid A*.
    Does NOT change observation space.
    """

    def __init__(
        self,
        env,
        # DWA
        dwa_enable: bool = True,
        dwa_horizon_steps: int = 12,
        dwa_dt: float = 0.1,
        dwa_n_steer: int = 7,
        dwa_n_throttle: int = 5,
        dwa_weight: float = 0.6,          # blending between SAC action and DWA action

        # TTC
        ttc_enable: bool = True,
        ttc_threshold: float = 1.2,
        ttc_brake_strength: float = 0.8,

        # Lyapunov-like
        lyap_enable: bool = True,
        lyap_c: float = 1.5,
        lyap_max_steer_correction: float = 0.8,
    ):
        super().__init__(env)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.dwa_enable = bool(dwa_enable)
        self.dwa_horizon_steps = int(dwa_horizon_steps)
        self.dwa_dt = float(dwa_dt)
        self.dwa_n_steer = int(dwa_n_steer)
        self.dwa_n_throttle = int(dwa_n_throttle)
        self.dwa_weight = float(dwa_weight)

        self.ttc_enable = bool(ttc_enable)
        self.ttc_threshold = float(ttc_threshold)
        self.ttc_brake_strength = float(ttc_brake_strength)

        self.lyap_enable = bool(lyap_enable)
        self.lyap_c = float(lyap_c)
        self.lyap_max_steer_correction = float(lyap_max_steer_correction)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, sac_action):
        sac_action = np.asarray(sac_action, dtype=np.float32).copy()
        sac_action = np.clip(sac_action, -1.0, 1.0)

        # (A) DWA
        if self.dwa_enable:
            a_dwa = self._dwa_best_action(sac_action)
            mixed = (1.0 - self.dwa_weight) * sac_action + self.dwa_weight * a_dwa
        else:
            mixed = sac_action

        mixed = np.clip(mixed, -1.0, 1.0)

        # (B) TTC safety
        if self.ttc_enable:
            mixed = self._apply_ttc_safety(mixed)

        # (C) Lyapunov-like filter
        if self.lyap_enable:
            mixed = self._apply_lyapunov_filter(mixed)

        mixed = np.clip(mixed, -1.0, 1.0)
        return self.env.step(mixed)

    def render(self):
        return self.env.render()

    # --------------------------
    # DWA (dynamic window search)
    # --------------------------
    def _dwa_best_action(self, center_action):
        ca = np.asarray(center_action, dtype=np.float32)

        steer_candidates = np.linspace(-1.0, 1.0, self.dwa_n_steer)
        thr_candidates = np.linspace(0.0, 1.0, self.dwa_n_throttle)

        # bias around center
        steer_candidates = 0.6 * steer_candidates + 0.4 * float(ca[1])
        thr_candidates = 0.7 * thr_candidates + 0.3 * max(0.0, float(ca[0]))

        best_score = -1e18
        best = ca

        for thr in thr_candidates:
            for st in steer_candidates:
                a = np.array([thr, st], dtype=np.float32)
                score = self._simulate_and_score(a)
                if score > best_score:
                    best_score = score
                    best = a

        return np.clip(best, -1.0, 1.0)

    def _simulate_and_score(self, action):
        x = float(self.env.x)
        y = float(self.env.y)
        yaw = float(self.env.yaw)
        v = float(self.env.v)

        throttle_cmd = float(action[0])
        steer_cmd = float(action[1])

        accel = 2.0 * throttle_cmd
        delta = float(getattr(self.env, "max_steer", math.radians(30))) * steer_cmd

        gx, gy = float(self.env.goal[0]), float(self.env.goal[1])
        start_dist = math.hypot(gx - x, gy - y)

        collided = False
        min_clear = 1e9

        for _ in range(self.dwa_horizon_steps):
            v = float(np.clip(v + accel * self.dwa_dt, self.env.v_min, self.env.v_max))
            yaw_rate = (v / self.env.L) * math.tan(delta)
            yaw = wrap_to_pi(yaw + yaw_rate * self.dwa_dt)
            x = x + v * math.cos(yaw) * self.dwa_dt
            y = y + v * math.sin(yaw) * self.dwa_dt

            if not (0.0 <= x <= self.env.world_size and 0.0 <= y <= self.env.world_size):
                collided = True
                break

            clear = self._clearance(x, y)
            min_clear = min(min_clear, clear)
            if clear <= 0.0:
                collided = True
                break

        end_dist = math.hypot(gx - x, gy - y)
        progress = start_dist - end_dist

        # scoring weights
        w_goal = 8.0
        w_speed = 0.2
        w_clear = 1.0
        w_col = 1000.0

        inv_clear = 1.0 / max(min_clear, 0.05)

        score = (
            w_goal * progress
            + w_speed * v
            - w_clear * inv_clear
            - (w_col if collided else 0.0)
            - 0.05 * abs(steer_cmd)
        )
        return score

    def _clearance(self, x, y):
        car = Point(x, y).buffer(self.env.car_radius)
        dmin = 1e9
        for obs in self.env.obstacles:
            dmin = min(dmin, car.distance(obs))
        return dmin

    # --------------------------
    # TTC safety
    # --------------------------
    def _apply_ttc_safety(self, action):
        a = np.asarray(action, dtype=np.float32).copy()

        if not hasattr(self.env, "_cast_rays"):
            return a

        rays = np.asarray(self.env._cast_rays(), dtype=np.float32)
        v = float(getattr(self.env, "v", 0.0))
        if v < 1e-3:
            return a

        fov = float(getattr(self.env, "ray_fov", math.radians(180.0)))
        n = len(rays)
        start = -fov / 2.0
        denom = (n - 1) if n > 1 else 1

        ttc_min = 1e9
        idx_min = int(np.argmin(rays))

        for i in range(n):
            ang = start + i * (fov / denom)
            closing = max(0.0, v * math.cos(ang))
            if closing < 1e-6:
                continue
            ttc = float(rays[i]) / closing
            ttc_min = min(ttc_min, ttc)

        if ttc_min < self.ttc_threshold:
            a[0] = max(0.0, a[0] * (1.0 - self.ttc_brake_strength))

            mid = (n - 1) / 2.0
            if idx_min < mid:
                a[1] = min(1.0, a[1] + 0.5)  # steer right
            else:
                a[1] = max(-1.0, a[1] - 0.5)  # steer left

        return np.clip(a, -1.0, 1.0)

    # --------------------------
    # Lyapunov-like stability
    # --------------------------
    def _apply_lyapunov_filter(self, action):
        a = np.asarray(action, dtype=np.float32).copy()

        # Reference = goal direction (no Hybrid A*)
        rx, ry = float(self.env.goal[0]), float(self.env.goal[1])

        x, y, yaw = float(self.env.x), float(self.env.y), float(self.env.yaw)
        desired = math.atan2(ry - y, rx - x)
        e = wrap_to_pi(desired - yaw)

        V = 0.5 * (e ** 2)

        v = float(getattr(self.env, "v", 0.0))
        accel = 2.0 * float(a[0])
        delta = float(getattr(self.env, "max_steer", math.radians(30))) * float(a[1])

        v_next = float(np.clip(v + accel * self.env.dt, self.env.v_min, self.env.v_max))
        yaw_rate = (v_next / self.env.L) * math.tan(delta)
        yaw_next = wrap_to_pi(yaw + yaw_rate * self.env.dt)

        e_next = wrap_to_pi(desired - yaw_next)
        V_next = 0.5 * (e_next ** 2)

        dV = V_next - V
        if dV > -self.lyap_c * V * self.env.dt:
            max_steer = float(getattr(self.env, "max_steer", math.radians(30)))
            steer_corr = np.clip(-e / max_steer, -1.0, 1.0)
            a[1] = np.clip(
                0.4 * float(a[1]) + 0.6 * steer_corr,
                -self.lyap_max_steer_correction,
                self.lyap_max_steer_correction,
            )

        return np.clip(a, -1.0, 1.0)
