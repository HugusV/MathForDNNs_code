# envs/ackermann_2d_env.py
# 2D kinematic Ackermann environment with:
# - front-only lidar rays
# - mixed obstacle shapes: circles, rectangles, triangles
# - walls/corridors: long thin rectangles (optionally rotated)
# - dynamic obstacles with linear/diagonal movement
# - all obstacles rendered in BLACK
#
# Dependencies:
#   pip install gymnasium numpy shapely matplotlib

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from shapely.geometry import Point, LineString, Polygon, box
from shapely.affinity import rotate, translate


def _wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class Ackermann2DEnv(gym.Env):
    """
    World: 20x20 meters (default), continuous 2D.
    Vehicle: kinematic Ackermann (x, y, yaw, v).
    Observation: [goal_dx, goal_dy, sin(yaw), cos(yaw), v, rays...]
    Action: [throttle_cmd, steering_cmd] in [-1,1]x[-1,1]
    Obstacles: shapely geometries (black): circles/rects/triangles + corridors/walls (long thin rects).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        world_size=20.0,
        dt=0.1,
        wheelbase=0.33,
        v_min=0.0,                 # Allow the vehicle to stop completely if needed
        v_max=3.0,
        max_steer_deg=30.0,
        n_rays=31,
        ray_fov_deg=40.0,
        ray_max=7.0,
        car_radius=0.25,
        n_obstacles=7,            # non-wall obstacles
        obstacle_radius_range=(0.25, 0.6),
        n_walls=3,                 # corridor/wall segments
        wall_thickness_range=(0.2, 0.35),
        wall_length_range=(6.0, 14.0),
        seed=0,

    ):
        super().__init__()

        # World / dynamics
        self.world_size = float(world_size)
        self.dt = float(dt)
        self.L = float(wheelbase)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.max_steer = math.radians(max_steer_deg)
        self.car_radius = float(car_radius)

        # Rays
        self.n_rays = int(n_rays)
        self.ray_fov = math.radians(ray_fov_deg)
        self.ray_max = float(ray_max)

        # Obstacles
        self.n_obstacles = int(n_obstacles)
        self.obs_r_min, self.obs_r_max = obstacle_radius_range

        # Dynamic obstacles properties
        self.obs_velocities = [] # Stores (vx, vy) for each obstacle
        self.dynamic_prob = 0.4  # chance for an obstacle to be dynamic

        # Walls / corridors
        self.n_walls = int(n_walls)
        self.wall_t_min, self.wall_t_max = wall_thickness_range
        self.wall_L_min, self.wall_L_max = wall_length_range

        # Termination
        self.goal_radius = 0.5
        self.max_steps = 500

        # RNG
        self.np_random = np.random.default_rng(seed)

        # Action space: throttle, steer in [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space = 5 + n_rays
        obs_low = np.array(
            [-self.world_size, -self.world_size, -1.0, -1.0, self.v_min] + [0.0] * self.n_rays,
            dtype=np.float32,
        )
        obs_high = np.array(
            [self.world_size, self.world_size, 1.0, 1.0, self.v_max] + [self.ray_max] * self.n_rays,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Internal state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.goal = np.zeros(2, dtype=np.float32)
        self.obstacles = []  # list[shapely geometry]
        self.step_count = 0

        # Reward shaping memory
        self._prev_dist = None

        # Rendering
        self._fig = None
        self._ax = None
        self._traj_x = []
        self._traj_y = []

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.step_count = 0
        self._traj_x = []
        self._traj_y = []

        # Spawn car away from walls/boundaries
        self.x, self.y = self._sample_free_point(margin=2.0)
        self.yaw = float(self.np_random.uniform(-np.pi, np.pi))
        self.v = float(self.v_min)

        # Spawn goal far enough from car
        for _ in range(300):
            gx, gy = self._sample_free_point(margin=2.0)
            if (gx - self.x) ** 2 + (gy - self.y) ** 2 > 6.0**2:
                self.goal = np.array([gx, gy], dtype=np.float32)
                break
        else:
            self.goal = np.array([self.world_size - 2.0, self.world_size - 2.0], dtype=np.float32)

        # Build obstacle set (shapely geometries)
        self.obstacles = []
        self.obs_velocities = [] # Reset velocities array
        car_shape = Point(self.x, self.y).buffer(self.car_radius)
        goal_shape = Point(self.goal[0], self.goal[1]).buffer(self.goal_radius)

        # 1) Add corridor/wall segments (long thin rectangles)
        # Walls are added first, their indices will be 0 to n_walls-1
        self._add_walls(car_shape=car_shape, goal_shape=goal_shape)

        # 2) Add mixed-shape obstacles (some will be dynamic)
        self._add_random_obstacles(car_shape=car_shape, goal_shape=goal_shape)

        dist_to_goal = float(np.linalg.norm(self.goal - np.array([self.x, self.y], dtype=np.float32)))
        self._prev_dist = dist_to_goal

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.step_count += 1

        a = np.asarray(action, dtype=np.float32)
        # Ensure action is clipped within limits. No max(0) to allow braking.
        a = np.clip(a, -1.0, 1.0)
        throttle_cmd, steer_cmd = float(a[0]), float(a[1])

        # Map commands to physical values
        accel = 2.0 * throttle_cmd
        delta = self.max_steer * steer_cmd

        # Ackermann update
        self.v = float(np.clip(self.v + accel * self.dt, self.v_min, self.v_max))
        yaw_rate = (self.v / self.L) * math.tan(delta)
        self.yaw = _wrap_to_pi(self.yaw + yaw_rate * self.dt)

        x_prev, y_prev = self.x, self.y
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt

        self._traj_x.append(self.x)
        self._traj_y.append(self.y)

        # -------------------------
        # Dynamic Obstacles Update
        # -------------------------
        new_obstacles = []
        
        for i, shape in enumerate(self.obstacles):
            # Walls are static. Assuming they are added first in the list
            if i < len(self.obstacles) - len(self.obs_velocities): 
                new_obstacles.append(shape)
                continue
                
            # Get velocity for random obstacles
            vel_idx = i - (len(self.obstacles) - len(self.obs_velocities))
            vel = self.obs_velocities[vel_idx]
            
            if np.any(vel != 0):
                # Linear/diagonal movement
                dx = vel[0] * self.dt
                dy = vel[1] * self.dt
                shape = translate(shape, xoff=dx, yoff=dy)
                
                # Bounce off the world boundaries
                minx, miny, maxx, maxy = shape.bounds
                if minx < 0 or maxx > self.world_size: 
                    self.obs_velocities[vel_idx][0] *= -1
                if miny < 0 or maxy > self.world_size: 
                    self.obs_velocities[vel_idx][1] *= -1
                
            new_obstacles.append(shape)
        
        self.obstacles = new_obstacles

        # Termination checks
        collided = self._check_collision()
        out_of_bounds = not (0.0 <= self.x <= self.world_size and 0.0 <= self.y <= self.world_size)
        dist_to_goal = float(np.linalg.norm(self.goal - np.array([self.x, self.y], dtype=np.float32)))
        reached_goal = dist_to_goal <= self.goal_radius
        time_up = self.step_count >= self.max_steps

        terminated = collided or out_of_bounds or reached_goal
        truncated = (not terminated) and time_up

        # Lidar rays
        rays = self._cast_rays()
        min_ray = float(np.min(rays))

        # -------------------------
        # Reward Logic
        # -------------------------
        progress = self._prev_dist - dist_to_goal
        self._prev_dist = dist_to_goal

        reward = 0.0

        # 1. Progress toward goal (Distance)
        reward += 2.5 * progress 

        # 2. Dynamic Steering Penalty (Prevent high-speed turns near obstacles)
        # Penalizes the product of speed and steering when an obstacle is close.
        # Forces the model to slow down to turn safely.
        steering_effort = abs(steer_cmd)
        if min_ray < 2.5: # Anticipate at 2.5 meters
            danger_factor = (2.5 - min_ray) / 2.5
            reward -= 2.0 * (self.v * steering_effort * danger_factor)

        # 3. Dynamic Steering Smoothness (General stability)
        # Prevents sharp turns at high speeds even in open space
        reward -= 0.1 * (steering_effort * self.v)

        # 4. Directional Alignment
        dx, dy = self.goal[0] - self.x, self.goal[1] - self.y
        target_angle = math.atan2(dy, dx)
        angle_diff = abs(_wrap_to_pi(target_angle - self.yaw))
        reward += 0.2 * (math.pi - angle_diff) 

        # 5. Immediate LiDAR Safety (Proximity sanction)
        if min_ray < 1.0:
            reward -= 2.0 * (1.0 - min_ray)

        # 6. Survival Penalties/Rewards
        if reached_goal:
            reward += 100.0
        if collided or out_of_bounds:
            reward -= 150.0 # Strongly discourage aggressiveness/collisions

        obs = self._get_obs(rays=rays)
        info = {
            "dist_to_goal": dist_to_goal,
            "collided": collided,
            "out_of_bounds": out_of_bounds,
            "reached_goal": reached_goal,
            "min_ray": min_ray,
            "dx_step": self.x - x_prev,
            "dy_step": self.y - y_prev,
        }
        return obs, reward, terminated, truncated, info

    # -------------------------
    # Obstacles generation
    # -------------------------
    def _add_walls(self, car_shape, goal_shape):
        """
        Adds long thin rectangles (corridor/wall segments), possibly rotated.
        Stored as shapely Polygons.
        """
        walls_added = 0
        for _ in range(400):
            if walls_added >= self.n_walls:
                break
                
            cx, cy = self._sample_free_point(margin=2.0)
            thickness = float(self.np_random.uniform(self.wall_t_min, self.wall_t_max))
            length = float(self.np_random.uniform(self.wall_L_min, self.wall_L_max))

            # axis-aligned wall around center (cx,cy)
            wall = box(cx - length / 2, cy - thickness / 2, cx + length / 2, cy + thickness / 2)

            # rotate around its center
            ang = float(self.np_random.uniform(0, 180))
            wall = rotate(wall, ang, origin=(cx, cy))

            # Keep walls inside world bounds by requiring its bbox within margins
            minx, miny, maxx, maxy = wall.bounds
            if minx < 0.5 or miny < 0.5 or maxx > self.world_size - 0.5 or maxy > self.world_size - 0.5:
                continue

            # clearance from car and goal
            if wall.distance(car_shape) < 1.2:
                continue
            if wall.distance(goal_shape) < 1.2:
                continue

            # avoid too-close overlaps with existing obstacles
            if not self._is_shape_clear(wall, min_clearance=0.25):
                continue

            self.obstacles.append(wall)
            walls_added += 1

    def _add_random_obstacles(self, car_shape, goal_shape):
        """
        Adds circles/rectangles/triangles, stored as shapely Polygons.
        Assigns velocities to make some dynamic.
        """
        obs_added = 0
        for _ in range(400):
            if obs_added >= self.n_obstacles:
                break
                
            ox, oy = self._sample_free_point(margin=1.0)
            shape = self._make_random_obstacle(ox, oy)

            # keep inside bounds a bit
            minx, miny, maxx, maxy = shape.bounds
            if minx < 0.3 or miny < 0.3 or maxx > self.world_size - 0.3 or maxy > self.world_size - 0.3:
                continue

            # clearance from car and goal
            if shape.distance(car_shape) < 1.0:
                continue
            if shape.distance(goal_shape) < 1.0:
                continue

            # avoid overlaps / too close to other obstacles
            if not self._is_shape_clear(shape, min_clearance=0.2):
                continue

            self.obstacles.append(shape)
            
            # --- Assign Velocity (Mouvement Addition) ---
            if self.np_random.random() < self.dynamic_prob:
                # Generate random velocity between -0.5 and 0.5 m/s for x and y
                vx = float(self.np_random.uniform(-0.5, 0.5))
                vy = float(self.np_random.uniform(-0.5, 0.5))
                self.obs_velocities.append(np.array([vx, vy]))
            else:
                self.obs_velocities.append(np.array([0.0, 0.0])) # Static
                
            obs_added += 1

    def _is_shape_clear(self, shape, min_clearance: float) -> bool:
        for other in self.obstacles:
            if shape.distance(other) < min_clearance:
                return False
        return True

    def _make_random_obstacle(self, ox: float, oy: float):
        """
        Returns a shapely geometry centered around (ox, oy):
        - circle
        - rectangle (rotated)
        - triangle (rotated)
        """
        kind = self.np_random.choice(["circle", "rect", "tri"], p=[0.4, 0.4, 0.2])

        if kind == "circle":
            r = float(self.np_random.uniform(self.obs_r_min, self.obs_r_max))
            return Point(ox, oy).buffer(r)

        if kind == "rect":
            w = float(self.np_random.uniform(0.6, 2.0))
            h = float(self.np_random.uniform(0.6, 2.0))
            rect = box(ox - w / 2, oy - h / 2, ox + w / 2, oy + h / 2)
            ang = float(self.np_random.uniform(0, 180))
            return rotate(rect, ang, origin=(ox, oy))

        # triangle
        s = float(self.np_random.uniform(0.8, 2.0))
        tri = Polygon([
            (ox, oy + s / 2),
            (ox - s / 2, oy - s / 2),
            (ox + s / 2, oy - s / 2),
        ])
        ang = float(self.np_random.uniform(0, 180))
        return rotate(tri, ang, origin=(ox, oy))

    # -------------------------
    # Observation / sensing
    # -------------------------
    def _get_obs(self, rays=None):
        if rays is None:
            rays = self._cast_rays()

        dx, dy = (self.goal - np.array([self.x, self.y], dtype=np.float32)).tolist()
        return np.array(
            [dx, dy, math.sin(self.yaw), math.cos(self.yaw), self.v] + list(rays),
            dtype=np.float32,
        )

    def _cast_rays(self):
        """
        Cast n_rays in a forward field-of-view centered on yaw.
        Each ray returns distance to nearest obstacle up to ray_max.
        Obstacles include mixed shapes + walls (all in self.obstacles).
        """
        origin = Point(self.x, self.y)
        rays = np.empty(self.n_rays, dtype=np.float32)

        start_ang = self.yaw - self.ray_fov / 2.0
        denom = (self.n_rays - 1) if self.n_rays > 1 else 1

        for i in range(self.n_rays):
            ang = start_ang + i * (self.ray_fov / denom)
            endx = self.x + self.ray_max * math.cos(ang)
            endy = self.y + self.ray_max * math.sin(ang)
            ray = LineString([(self.x, self.y), (endx, endy)])

            d = self.ray_max
            for shape in self.obstacles:
                inter = ray.intersection(shape)
                if inter.is_empty:
                    continue

                if inter.geom_type == "Point":
                    d = min(d, origin.distance(inter))
                else:
                    geoms = getattr(inter, "geoms", None)
                    if geoms is not None:
                        for g in geoms:
                            d = min(d, origin.distance(g))
                    else:
                        d = min(d, origin.distance(inter))

            rays[i] = float(np.clip(d, 0.0, self.ray_max))

        return rays

    # -------------------------
    # Collision
    # -------------------------
    def _check_collision(self):
        car = Point(self.x, self.y).buffer(self.car_radius)
        for shape in self.obstacles:
            if car.intersects(shape):
                return True
        return False

    # -------------------------
    # Sampling helpers
    # -------------------------
    def _sample_free_point(self, margin=1.0):
        x = float(self.np_random.uniform(margin, self.world_size - margin))
        y = float(self.np_random.uniform(margin, self.world_size - margin))
        return x, y

    # -------------------------
    # Rendering (Matplotlib)
    # -------------------------
    def render(self):
        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 6))

        self._ax.clear()
        self._ax.set_xlim(0, self.world_size)
        self._ax.set_ylim(0, self.world_size)
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.set_title("Simulation of Ackermann robot in 20x20m world")

        # Obstacles in black
        for shape in self.obstacles:
            self._draw_shape_black(shape)

        # Goal
        self._ax.plot(self.goal[0], self.goal[1], marker="*", markersize=14)

        # Trajectory
        if len(self._traj_x) > 2:
            self._ax.plot(self._traj_x, self._traj_y, linewidth=1)

        # Lidar rays visualization (semi-transparent cyan)
        current_rays = self._cast_rays()
        
        start_ang = self.yaw - self.ray_fov / 2.0
        denom = (self.n_rays - 1) if self.n_rays > 1 else 1

        for i, dist in enumerate(current_rays):
            ang = start_ang + i * (self.ray_fov / denom)
            end_x = self.x + dist * math.cos(ang)
            end_y = self.y + dist * math.sin(ang)
            
            self._ax.plot([self.x, end_x], [self.y, end_y], color='cyan', alpha=0.3, linewidth=1)
            
            # Optional: small red dot at the end of the ray if it hits an obstacle
            if dist < self.ray_max - 0.01:
                self._ax.plot(end_x, end_y, marker='.', color='red', markersize=3)
        # -------------------------------------------

        # Vehicle body + heading
        car = plt.Circle((self.x, self.y), self.car_radius, fill=False)
        self._ax.add_patch(car)
        hx = self.x + 0.8 * self.car_radius * math.cos(self.yaw)
        hy = self.y + 0.8 * self.car_radius * math.sin(self.yaw)
        self._ax.plot([self.x, hx], [self.y, hy], linewidth=2)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def _draw_shape_black(self, shape):
        """
        Draw shapely Polygon/MultiPolygon in black (filled).
        """
        if shape.geom_type == "Polygon":
            x, y = shape.exterior.xy
            patch = MplPolygon(
                np.column_stack([x, y]),
                closed=True,
                fill=True,
                edgecolor="black",
                facecolor="black",
                linewidth=1,
            )
            self._ax.add_patch(patch)
        elif shape.geom_type == "MultiPolygon":
            for poly in shape.geoms:
                x, y = poly.exterior.xy
                patch = MplPolygon(
                    np.column_stack([x, y]),
                    closed=True,
                    fill=True,
                    edgecolor="black",
                    facecolor="black",
                    linewidth=1,
                )
                self._ax.add_patch(patch)
        else:
            # Fallback: draw convex hull
            hull = shape.convex_hull
            x, y = hull.exterior.xy
            patch = MplPolygon(
                np.column_stack([x, y]),
                closed=True,
                fill=True,
                edgecolor="black",
                facecolor="black",
                linewidth=1,
            )
            self._ax.add_patch(patch)

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
        self._fig = None
        self._ax = None