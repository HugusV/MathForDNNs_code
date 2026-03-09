import math
import numpy as np
import gymnasium as gym

from planners import HybridAStarPlanner


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class GuidedAckermannEnv(gym.Wrapper):
    """
    Gymnasium-compatible wrapper around Ackermann2DEnv.

    Hybrid A* computes a global reference path at reset.
    A simple waypoint follower generates a planner action.
    We blend planner action with SAC action early in the episode:
        mixed = (1-beta)*agent + beta*planner
    where beta decays over time.

    IMPORTANT: Does NOT change observation shape.
    """

    def __init__(
        self,
        env,
        guide_max_steps: int = 200,
        guide_beta0: float = 0.8,
        guide_beta_min: float = 0.0,
        planner_grid_res: float = 0.5,
        planner_yaw_bins: int = 24,
        planner_step: float = 0.6,
        lookahead: float = 1.2,
        target_speed: float = 1.5,
        throttle_gain: float = 0.6,
    ):
        super().__init__(env)

        # SB3 needs these spaces on the wrapper
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.guide_max_steps = int(guide_max_steps)
        self.beta0 = float(guide_beta0)
        self.beta_min = float(guide_beta_min)

        self.planner_grid_res = float(planner_grid_res)
        self.planner_yaw_bins = int(planner_yaw_bins)
        self.planner_step = float(planner_step)

        self.lookahead = float(lookahead)
        self.target_speed = float(target_speed)
        self.throttle_gain = float(throttle_gain)

        self._path = None
        self._wp_idx = 0

    # ---- Convenience passthroughs (optional) ----
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

        # Build planner using current obstacles and vehicle parameters
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

    def step(self, action):
        planner_action = self._planner_action()

        # Beta decay based on base env step_count (exists in your Ackermann env)
        t = min(int(getattr(self.env, "step_count", 0)), self.guide_max_steps)
        if self.guide_max_steps <= 0:
            beta = 0.0
        else:
            frac = 1.0 - (t / self.guide_max_steps)  # 1->0
            beta = max(self.beta_min, self.beta0 * frac)

        a = np.asarray(action, dtype=np.float32).copy()
        pa = np.asarray(planner_action, dtype=np.float32)

        mixed = (1.0 - beta) * a + beta * pa
        mixed = np.clip(mixed, -1.0, 1.0)

        return self.env.step(mixed)

    def render(self):
        # Render base env
        self.env.render()

        # Overlay planned path if available and if base env uses matplotlib with _ax
        if self._path is None:
            return
        ax = getattr(self.env, "_ax", None)
        fig = getattr(self.env, "_fig", None)
        if ax is None or fig is None:
            return

        xs = [p[0] for p in self._path]
        ys = [p[1] for p in self._path]
        ax.plot(xs, ys, linewidth=1.5)
        fig.canvas.draw()
        fig.canvas.flush_events()

    # --------- planner action: simple waypoint follower ----------
    def _planner_action(self):
        # If no path found, just go gently forward
        if not self._path or len(self._path) < 2:
            return np.array([0.3, 0.0], dtype=np.float32)

        x, y, yaw = float(self.env.x), float(self.env.y), float(self.env.yaw)

        # advance waypoint to keep lookahead
        while self._wp_idx < len(self._path) - 1:
            wx, wy, _ = self._path[self._wp_idx]
            if math.hypot(wx - x, wy - y) >= self.lookahead:
                break
            self._wp_idx += 1

        wx, wy, _ = self._path[self._wp_idx]

        goal_heading = math.atan2(wy - y, wx - x)
        heading_err = wrap_to_pi(goal_heading - yaw)

        # steering normalized by max steer (use base env max_steer)
        max_steer = float(getattr(self.env, "max_steer", math.radians(30.0)))
        steer_cmd = np.clip(heading_err / max_steer, -1.0, 1.0)

        # throttle toward target speed (forward only)
        v = float(getattr(self.env, "v", 0.0))
        speed_err = self.target_speed - v
        throttle_cmd = np.clip(self.throttle_gain * speed_err, 0.0, 1.0)

        return np.array([throttle_cmd, steer_cmd], dtype=np.float32)
