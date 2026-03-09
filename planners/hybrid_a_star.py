import math
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from shapely.geometry import Point


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class NodeKey:
    ix: int
    iy: int
    it: int


@dataclass
class Node:
    x: float
    y: float
    yaw: float
    g: float
    h: float
    parent: Optional[NodeKey]
    parent_xyyaw: Optional[Tuple[float, float, float]]


class HybridAStarPlanner:
    def __init__(
        self,
        world_size: float,
        obstacles,                 # list of shapely geometries (polygons)
        car_radius: float,
        grid_res: float = 0.5,     # meters
        yaw_bins: int = 24,
        step: float = 0.6,         # meters per expansion
        wheelbase: float = 0.33,
        max_steer_deg: float = 30.0,
        steer_set: int = 3,        # number of steer samples per side
        goal_tolerance: float = 0.7,
    ):
        self.world_size = float(world_size)
        self.obstacles = list(obstacles)
        self.car_radius = float(car_radius)

        self.grid_res = float(grid_res)
        self.yaw_bins = int(yaw_bins)
        self.step = float(step)

        self.L = float(wheelbase)
        self.max_steer = math.radians(max_steer_deg)
        self.goal_tol = float(goal_tolerance)

        # motion primitives steering angles
        # includes 0 and symmetric angles
        if steer_set < 1:
            steer_set = 1
        angles = np.linspace(0.0, self.max_steer, steer_set + 1)  # include max
        self.steer_angles = sorted(set(list(angles) + list(-angles)))

    def plan(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float],
        max_expansions: int = 50_000,
    ) -> Optional[List[Tuple[float, float, float]]]:
        sx, sy, syaw = start
        gx, gy = goal

        if self._collides(sx, sy):
            return None

        start_key = self._key(sx, sy, syaw)
        start_node = Node(
            x=sx, y=sy, yaw=syaw,
            g=0.0,
            h=self._heuristic(sx, sy, gx, gy),
            parent=None,
            parent_xyyaw=None
        )

        open_heap = []
        heapq.heappush(open_heap, (start_node.g + start_node.h, 0, start_key, start_node))

        best = {start_key: start_node.g}
        came_from = {start_key: start_node}

        expansions = 0
        counter = 1

        while open_heap and expansions < max_expansions:
            _, _, key, cur = heapq.heappop(open_heap)

            # goal check in continuous space
            if math.hypot(cur.x - gx, cur.y - gy) <= self.goal_tol:
                return self._reconstruct_path(key, came_from)

            # if this is not the best g anymore, skip
            if cur.g > best.get(key, float("inf")) + 1e-9:
                continue

            expansions += 1

            for delta in self.steer_angles:
                nx, ny, nyaw, cost = self._forward_sim(cur.x, cur.y, cur.yaw, delta)
                if not self._in_bounds(nx, ny):
                    continue
                if self._collides(nx, ny):
                    continue

                nkey = self._key(nx, ny, nyaw)
                ng = cur.g + cost + 0.02 * abs(delta)  # small steer cost
                if ng < best.get(nkey, float("inf")):
                    best[nkey] = ng
                    nh = self._heuristic(nx, ny, gx, gy)
                    node = Node(
                        x=nx, y=ny, yaw=nyaw,
                        g=ng, h=nh,
                        parent=key,
                        parent_xyyaw=(cur.x, cur.y, cur.yaw),
                    )
                    came_from[nkey] = node
                    heapq.heappush(open_heap, (ng + nh, counter, nkey, node))
                    counter += 1

        return None

    # ---------- internals ----------
    def _forward_sim(self, x: float, y: float, yaw: float, delta: float) -> Tuple[float, float, float, float]:
        """
        Single-step bicycle model forward, constant steer.
        """
        v = 1.0  # arbitrary for geometric planning; step sets distance
        dt = self.step / max(v, 1e-6)
        yaw_rate = (v / self.L) * math.tan(delta)
        nyaw = wrap_to_pi(yaw + yaw_rate * dt)
        nx = x + v * math.cos(nyaw) * dt
        ny = y + v * math.sin(nyaw) * dt
        return nx, ny, nyaw, self.step

    def _heuristic(self, x: float, y: float, gx: float, gy: float) -> float:
        return math.hypot(gx - x, gy - y)

    def _collides(self, x: float, y: float) -> bool:
        car = Point(x, y).buffer(self.car_radius)
        for obs in self.obstacles:
            if car.intersects(obs):
                return True
        return False

    def _in_bounds(self, x: float, y: float) -> bool:
        return (0.0 <= x <= self.world_size) and (0.0 <= y <= self.world_size)

    def _key(self, x: float, y: float, yaw: float) -> NodeKey:
        ix = int(x / self.grid_res)
        iy = int(y / self.grid_res)
        # yaw bin
        t = (yaw + math.pi) / (2 * math.pi)  # [0,1)
        it = int(t * self.yaw_bins) % self.yaw_bins
        return NodeKey(ix, iy, it)

    def _reconstruct_path(self, goal_key: NodeKey, came_from) -> List[Tuple[float, float, float]]:
        path = []
        k = goal_key
        while True:
            node = came_from[k]
            path.append((node.x, node.y, node.yaw))
            if node.parent is None:
                break
            k = node.parent
        path.reverse()
        return path
