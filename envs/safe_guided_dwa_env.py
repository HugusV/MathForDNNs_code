import math
import numpy as np
import gymnasium as gym
from shapely.geometry import Point

from planners import HybridAStarPlanner


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class SafeGuidedDWAEnv(gym.Wrapper):
    """
    Wrapper Gymnasium:
      - Hybrid A*: global reference path at reset
      - DWA: local short-horizon action search (dynamic window)
      - TTC: danger estimation from lidar rays + ego velocity (static obstacles)
      - Lyapunov-like stability: enforce decrease of a candidate V (heading/waypoint error)

    Does NOT change observation space.
    """

    def __init__(
        self,
        env,
        # Hybrid A* guidance (same as GuidedAckermannEnv)
        guide_max_steps: int = 250,
        guide_beta0: float = 0.85,
        guide_beta_min: float = 0.0,
        planner_grid_res: float = 0.5,
        planner_yaw_bins: int = 24,
        planner_step: float = 0.6,
        lookahead: float = 1.2,
        target_speed: float = 1.5,

        # DWA parameters
        dwa_enable: bool = True,
        dwa_horizon_steps: int = 12,      # simulate 12 steps ahead
        dwa_dt: float = 0.1,              # simulation dt (should match env.dt)
        dwa_n_steer: int = 7,
        dwa_n_throttle: int = 5,

        # TTC safety
        ttc_enable: bool = True,
        ttc_threshold: float = 1.2,       # seconds
        ttc_brake_strength: float = 0.8,  # 0..1 scale applied to throttle

        # Lyapunov-like filter
        lyap_enable: bool = True,
        lyap_c: float = 1.5,              # decay rate (bigger => more aggressive stabilization)
        lyap_max_steer_correction: float = 0.8,

        # General blending
        blend_with_sac: bool = True,
        dwa_weight: float = 0.6,          # how much DWA influences vs SAC after safety
    ):
        super().__init__(env)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        # Hybrid A*
        self.guide_max_steps = int(guide_max_steps)
        self.beta0 = float(guide_beta0)
        self.beta_min = float(guide_beta_min)
        self.planner_grid_res = float(planner_grid_res)
        self.planner_yaw_bins = int(planner_yaw_bins)
        self.planner_step = float(planner_step)
        self.lookahead = float(lookahead)
        self.target_speed = float(target_speed)

        # DWA
        self.dwa_enable = bool(dwa_enable)
        self.dwa_horizon_steps = int(dwa_horizon_steps)
        self.dwa_dt = float(dwa_dt)
        self.dwa_n_steer = int(dwa_n_steer)
        self.dwa_n_throttle = int(dwa_n_throttle)

        # TTC
        self.ttc_enable = bool(ttc_enable)
        self.ttc_threshold = float(ttc_threshold)
        self.ttc_brake_strength = float(ttc_brake_strength)

        # Lyapunov-like
        self.lyap_enable = bool(lyap_enable)
        self.lyap_c = float(lyap_c)
        self.lyap_max_steer_correction = float(lyap_max_steer_correction)

        # Blending
        self.blend_with_sac = bool(blend_with_sac)
        self.dwa_weight = float(dwa_weight)

        # Internal
        self._path = None
        self._wp_idx = 0

    # ---- passthroughs ----
    @property
    def x(self): return self.env.x
    @property
    def y(self): return self.env.y
    @property
    def yaw(self): return self.env.yaw
    @property
    def v(self): return self.env.v
    @property
    def goal(self): return self.env.goal
    @property
    def obstacles(self): return self.env.obstacles

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Hybrid A* plan
        planner = HybridAStarPlanner(
            world_size=self.env.world_size,
            obstacles=self.env.obstacles,
            car_radius=self.env.car_radius,
            grid_res=self.planner_grid_res,
            yaw_bins=self.planner_yaw_bins,
            step=self.planner_step,
            wheelbase=self.env.L,
            max_steer_deg=math.degrees(self.env.max_steer),
            goal_tolerance=max(0.7, getattr(self.env, "goal_radius", 0.5)),
        )
        start = (float(self.env.x), float(self.env.y), float(self.env.yaw))
        goal = (float(self.env.goal[0]), float(self.env.goal[1]))

        self._path = planner.plan(start=start, goal=goal)
        self._wp_idx = 0
        return obs, info

    def step(self, sac_action):
        sac_action = np.asarray(sac_action, dtype=np.float32).copy()

        # (A) Hybrid A* waypoint follower action
        a_astar = self._astar_action()

        # (B) Blend schedule beta(t) for A* guidance
        t = min(int(getattr(self.env, "step_count", 0)), self.guide_max_steps)
        if self.guide_max_steps <= 0:
            beta = 0.0
        else:
            frac = 1.0 - (t / self.guide_max_steps)
            beta = max(self.beta_min, self.beta0 * frac)

        guided_action = (1.0 - beta) * sac_action + beta * a_astar
        guided_action = np.clip(guided_action, -1.0, 1.0)

        # (C) DWA refinement around guided_action (optional)
        if self.dwa_enable:
            a_dwa = self._dwa_best_action(guided_action)
        else:
            a_dwa = guided_action

        # (D) Merge SAC with DWA result (optional)
        if self.blend_with_sac:
            mixed = (1.0 - self.dwa_weight) * guided_action + self.dwa_weight * a_dwa
        else:
            mixed = a_dwa

        mixed = np.clip(mixed, -1.0, 1.0)

        # (E) TTC safety layer (brake + steer away) (optional)
        if self.ttc_enable:
            mixed = self._apply_ttc_safety(mixed)

        # (F) Lyapunov-like stability filter (optional)
        if self.lyap_enable:
            mixed = self._apply_lyapunov_filter(mixed)

        mixed = np.clip(mixed, -1.0, 1.0)
        return self.env.step(mixed)

    def render(self):
        self.env.render()

        # Overlay planned path
        if self._path is None:
            return
        ax = getattr(self.env, "_ax", None)
        fig = getattr(self.env, "_fig", None)
        if ax is None or fig is None:
            return

        xs = [p[0] for p in self._path]
        ys = [p[1] for p in self._path]
        ax.plot(xs, ys, linewidth=1.5)  # default color (matplotlib decides)
        fig.canvas.draw()
        fig.canvas.flush_events()

    # --------------------------
    # Hybrid A* waypoint action
    # --------------------------
    def _astar_action(self):
        if not self._path or len(self._path) < 2:
            return np.array([0.3, 0.0], dtype=np.float32)

        x, y, yaw = float(self.env.x), float(self.env.y), float(self.env.yaw)

        # find lookahead waypoint
        while self._wp_idx < len(self._path) - 1:
            wx, wy, _ = self._path[self._wp_idx]
            if math.hypot(wx - x, wy - y) >= self.lookahead:
                break
            self._wp_idx += 1

        wx, wy, _ = self._path[self._wp_idx]
        desired = math.atan2(wy - y, wx - x)
        e = wrap_to_pi(desired - yaw)

        max_steer = float(getattr(self.env, "max_steer", math.radians(30)))
        steer_cmd = np.clip(e / max_steer, -1.0, 1.0)

        v = float(getattr(self.env, "v", 0.0))
        speed_err = self.target_speed - v
        throttle_cmd = np.clip(0.6 * speed_err, 0.0, 1.0)

        return np.array([throttle_cmd, steer_cmd], dtype=np.float32)

    # --------------------------
    # DWA (dynamic window search)
    # --------------------------
    def _dwa_best_action(self, center_action):
        """
        Sample candidate actions near center_action and pick best by:
          score = w_goal * progress_to_goal - w_col * collision - w_clear * (1/clearance) + w_speed * speed
        """
        ca = np.asarray(center_action, dtype=np.float32)
        # search window around center action
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
        """
        Short rollout from current state using env dynamics (bicycle model)
        without modifying the real env.
        """
        x = float(self.env.x)
        y = float(self.env.y)
        yaw = float(self.env.yaw)
        v = float(self.env.v)

        # action -> accel, steer
        throttle_cmd = float(action[0])
        steer_cmd = float(action[1])

        accel = 2.0 * throttle_cmd
        delta = float(getattr(self.env, "max_steer", math.radians(30))) * steer_cmd

        gx, gy = float(self.env.goal[0]), float(self.env.goal[1])
        start_dist = math.hypot(gx - x, gy - y)

        collided = False
        min_clear = 1e9

        for _ in range(self.dwa_horizon_steps):
            # update v, yaw, x,y like env
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

        # scoring weights (tuneable)
        w_goal = 8.0
        w_speed = 0.2
        w_clear = 1.0
        w_col = 1000.0

        # prefer clearance, but not too aggressive
        inv_clear = 1.0 / max(min_clear, 0.05)

        score = (
            w_goal * progress
            + w_speed * v
            - w_clear * inv_clear
            - (w_col if collided else 0.0)
            - 0.05 * abs(steer_cmd)  # smoother steering
        )
        return score

    def _clearance(self, x, y):
        """
        Positive distance between car disk and nearest obstacle; <=0 means collision.
        """
        car = Point(x, y).buffer(self.env.car_radius)
        dmin = 1e9
        for obs in self.env.obstacles:
            d = car.distance(obs)
            dmin = min(dmin, d)
        return dmin

    # --------------------------
    # TTC safety
    # --------------------------
    def _apply_ttc_safety(self, action):
        """
        Approx TTC from lidar rays for static obstacles:
          closing_speed_i = max(0, v * cos(ray_angle_offset))
          TTC_i = dist_i / closing_speed_i
        Take min TTC. If below threshold -> brake and steer away from closest ray.
        """
        a = np.asarray(action, dtype=np.float32).copy()

        # get rays from current obs by calling env._cast_rays if exists; else rebuild from obs
        # safest: call env's internal rays method if present
        if hasattr(self.env, "_cast_rays"):
            rays = np.asarray(self.env._cast_rays(), dtype=np.float32)
        else:
            # fallback: assume obs last part are rays (not recommended here)
            return a

        v = float(getattr(self.env, "v", 0.0))
        if v < 1e-3:
            return a

        # angles across FOV
        fov = float(getattr(self.env, "ray_fov", math.radians(180.0)))
        n = len(rays)
        start = -fov / 2.0
        denom = (n - 1) if n > 1 else 1

        ttc_min = 1e9
        idx_min = int(np.argmin(rays))
        for i in range(n):
            ang = start + i * (fov / denom)  # angle offset from heading
            closing = max(0.0, v * math.cos(ang))
            if closing < 1e-6:
                continue
            ttc = float(rays[i]) / closing
            ttc_min = min(ttc_min, ttc)

        if ttc_min < self.ttc_threshold:
            # brake strongly
            a[0] = max(0.0, a[0] * (1.0 - self.ttc_brake_strength))

            # steer away from closest obstacle direction
            # if obstacle is on left side (idx < mid), steer right (+)
            mid = (n - 1) / 2.0
            if idx_min < mid:
                a[1] = min(1.0, a[1] + 0.5)
            else:
                a[1] = max(-1.0, a[1] - 0.5)

        return np.clip(a, -1.0, 1.0)

    # --------------------------
    # Lyapunov-like stability filter
    # --------------------------
    def _apply_lyapunov_filter(self, action):
        """
        Practical discrete Lyapunov condition:
          V = 0.5 * e^2, where e = heading error to current A* waypoint (or goal)
          Want: ΔV <= -c * V * dt

        If condition violated, we correct steering toward reducing e.
        """
        a = np.asarray(action, dtype=np.float32).copy()

        # choose reference: waypoint if exists else goal
        ref = None
        if self._path and len(self._path) >= 2:
            ref = self._path[min(self._wp_idx, len(self._path) - 1)]
            rx, ry = float(ref[0]), float(ref[1])
        else:
            rx, ry = float(self.env.goal[0]), float(self.env.goal[1])

        x, y, yaw = float(self.env.x), float(self.env.y), float(self.env.yaw)
        desired = math.atan2(ry - y, rx - x)
        e = wrap_to_pi(desired - yaw)

        V = 0.5 * (e ** 2)

        # predict next yaw with current action (1 step)
        v = float(getattr(self.env, "v", 0.0))
        accel = 2.0 * float(a[0])
        delta = float(getattr(self.env, "max_steer", math.radians(30))) * float(a[1])

        v_next = float(np.clip(v + accel * self.env.dt, self.env.v_min, self.env.v_max))
        yaw_rate = (v_next / self.env.L) * math.tan(delta)
        yaw_next = wrap_to_pi(yaw + yaw_rate * self.env.dt)

        e_next = wrap_to_pi(desired - yaw_next)
        V_next = 0.5 * (e_next ** 2)

        dV = V_next - V
        # condition: dV <= -c * V * dt
        if dV > -self.lyap_c * V * self.env.dt:
            # apply steering correction to reduce e
            # steer_cmd should have opposite sign of e (turn toward desired heading)
            steer_correction = np.clip(-e / float(getattr(self.env, "max_steer", math.radians(30))), -1.0, 1.0)
            # blend correction into action steering
            a[1] = np.clip(
                0.4 * float(a[1]) + 0.6 * steer_correction,
                -self.lyap_max_steer_correction,
                self.lyap_max_steer_correction
            )

        return np.clip(a, -1.0, 1.0)
