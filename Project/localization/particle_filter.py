import math
import random

import numpy as np

from sensor_model import template_match_score


class Particle:
    def __init__(self, x, y, theta, weight=1.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

    def pose(self): return (self.x, self.y, self.theta)


class ParticleFilter:
    def __init__(self, num_particles, free_indices, x_min, y_min, resolution, grid_height, grid_width,
                 wheel_base=0.160, prior_regions=None, prior_outside_scale=1e-3):
        self.num_particles = num_particles
        self.x_min = x_min
        self.y_min = y_min
        self.resolution = resolution
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.wheel_base = wheel_base
        self.prior_regions = prior_regions or []
        self.prior_outside_scale = prior_outside_scale
        self.particles = []
        for _ in range(num_particles):
            row, col = random.choice(free_indices)
            x = x_min + col * resolution
            y_max = y_min + grid_height * resolution
            y = y_max - (row + 1) * resolution
            theta = random.uniform(-math.pi, math.pi)
            self.particles.append(Particle(x, y, theta, 1.0 / num_particles))

    def _bounds(self):
        """World bounds from map meta."""
        x_max = self.x_min + self.grid_width * self.resolution
        y_max = self.y_min + self.grid_height * self.resolution
        return self.x_min, x_max, self.y_min, y_max

    def _clamp_to_bounds(self, x, y):
        xmin, xmax, ymin, ymax = self._bounds()
        return (min(max(x, xmin), xmax), min(max(y, ymin), ymax))

    def _in_bounds(self, x, y):
        xmin, xmax, ymin, ymax = self._bounds()
        return (xmin <= x <= xmax) and (ymin <= y <= ymax)

    def _prior_weight(self, x, y):
        if not self.prior_regions:
            return 1.0
        for (xmin, xmax, ymin, ymax) in self.prior_regions:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return 1.0
        return self.prior_outside_scale  # softly down-weight outside

    def _world_to_grid(self, x, y):
        H = self.grid_height
        W = self.grid_width
        y_max = self.y_min + H * self.resolution
        col = int((x - self.x_min) / self.resolution)
        row = int((y_max - y) / self.resolution)
        if 0 <= row < H and 0 <= col < W:
            return row, col
        return None

    def _is_free(self, occupancy_grid, x, y):
        idx = self._world_to_grid(x, y)
        if idx is None:
            return False
        r, c = idx
        return occupancy_grid[r, c] == 0

    def _snap_to_nearest_free(self, distance_map, x, y):
        """Snap to nearest free cell center using the distance transform peak-ascent."""
        idx = self._world_to_grid(x, y)
        if idx is None:  # out of bounds, just return original
            return x, y
        r, c = idx
        H, W = distance_map.shape
        best_r, best_c = r, c
        best_d = distance_map[r, c]
        # climb to local maximum
        for _ in range(5):
            r0, c0 = best_r, best_c
            for rr in (r0 - 1, r0, r0 + 1):
                for cc in (c0 - 1, c0, c0 + 1):
                    if 0 <= rr < H and 0 <= cc < W:
                        if distance_map[rr, cc] > best_d:
                            best_d = distance_map[rr, cc]
                            best_r, best_c = rr, cc
            if (best_r, best_c) == (r0, c0):
                break
        # convert back to world
        y_max = self.y_min + H * self.resolution
        xw = self.x_min + (best_c + 0.5) * self.resolution
        yw = y_max - (best_r + 0.5) * self.resolution
        return xw, yw

    # replace the whole sensor_update with template-only
    def sensor_update(self, scan_local_points, occupancy_grid, room_rects):
        ws = []
        total = 0.0
        for p in self.particles:
            s = template_match_score(
                scan_local_points, p.pose(),
                occupancy_grid, room_rects,
                self.x_min, self.y_min, self.resolution
            )
            s *= self._prior_weight(p.x, p.y)
            ws.append(s)
            total += s

        if total <= 0:
            ws = [1.0 / self.num_particles] * self.num_particles
        else:
            ws = [w / total for w in ws]

        for p, w in zip(self.particles, ws):
            p.weight = w

    def resample(self):
        weights = np.array([p.weight for p in self.particles], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights) / len(weights)
        idx = np.random.choice(len(self.particles), size=self.num_particles, p=weights)
        new = [self.particles[i] for i in idx]
        # reset weights uniform
        for p in new:
            p.weight = 1.0 / self.num_particles
        self.particles = new

    def reseed_in_free_indices(self, free_indices):
        """Hard reset: sample all particles from given free (row,col) list uniform headings."""
        import random, math
        self.particles = []
        H = self.grid_height
        for _ in range(self.num_particles):
            row, col = random.choice(free_indices)
            x = self.x_min + col * self.resolution
            y_max = self.y_min + H * self.resolution
            y = y_max - (row + 1) * self.resolution
            theta = random.uniform(-math.pi, math.pi)
            self.particles.append(Particle(x, y, theta, 1.0 / self.num_particles))

    def lock_to_room(self, room_rect, occupancy_grid, free_indices_func):
        self.prior_regions = [room_rect]
        self.prior_outside_scale = 1e-9  # effectively zero outside
        free_in_room = free_indices_func(occupancy_grid, self.x_min, self.y_min, self.resolution, [room_rect])
        if free_in_room:
            self.reseed_in_free_indices(free_in_room)

    def motion_update(self, delta_left, delta_right, occupancy_grid=None, distance_map=None, keep_free=True,
                      keep_in_bounds=True):
        dc = 0.5 * (delta_left + delta_right)
        dth = (delta_right - delta_left) / self.wheel_base
        for p in self.particles:
            nx = p.x + dc * math.cos(p.theta + 0.5 * dth)
            ny = p.y + dc * math.sin(p.theta + 0.5 * dth)
            nth = (p.theta + dth + math.pi) % (2 * math.pi) - math.pi

            # keep in bounds
            if keep_in_bounds:
                nx, ny = self._clamp_to_bounds(nx, ny)

            # keep on free cells
            if keep_free and (occupancy_grid is not None):
                if not self._is_free(occupancy_grid, nx, ny):
                    if distance_map is not None:
                        nx, ny = self._snap_to_nearest_free(distance_map, nx, ny)
                    else:
                        # reject step
                        nx, ny = p.x, p.y

            p.x, p.y, p.theta = nx, ny, nth

    def roughen(self, sigma_xy=0.01, sigma_theta=0.01, occupancy_grid=None, distance_map=None, keep_free=True,
                keep_in_bounds=True):
        for p in self.particles:
            nx = p.x + np.random.normal(0.0, sigma_xy)
            ny = p.y + np.random.normal(0.0, sigma_xy)
            nth = (p.theta + np.random.normal(0.0, sigma_theta) + math.pi) % (2 * math.pi) - math.pi

            if keep_in_bounds:
                nx, ny = self._clamp_to_bounds(nx, ny)

            if keep_free and (occupancy_grid is not None):
                if not self._is_free(occupancy_grid, nx, ny):
                    if distance_map is not None:
                        nx, ny = self._snap_to_nearest_free(distance_map, nx, ny)
                    else:
                        nx, ny = p.x, p.y

            p.x, p.y, p.theta = nx, ny, nth

    def estimate_pose(self):
        xs = np.array([p.x for p in self.particles])
        ys = np.array([p.y for p in self.particles])
        th = np.array([p.theta for p in self.particles])
        return float(xs.mean()), float(ys.mean()), float(th.mean())
