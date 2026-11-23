import math
import os
import sys

import matplotlib.pyplot as plt

sys.path.append('/usr/local/webots/lib/controller/python')
os.environ['WEBOTS_HOME'] = '/usr/local/webots/'

from controller import Robot

# --- Shared Constants ---
MAX_SPEED = 4.0

START_POS = (0.0, -2.7)
BUG_GOAL_POS = (0.0, 4.75)
CABINET_POS = (-3.4, 4.2)

# --- Thresholds ---
OBSTACLE_DISTANCE_THRESHOLD_FRONT = 0.10
TARGET_WALL_DISTANCE_SIDE = 0.14
WALL_FAR_DISTANCE_THRESHOLD = 0.20
GOAL_REACHED_THRESHOLD = 0.10
ALIGNMENT_TOLERANCE_RADIANS = 0.08
HIT_POINT_RETURN_TOLERANCE = 0.08
M_LINE_TOLERANCE_BUG2 = 0.15

LASER_LOOKUP_TABLE = [
    [0.0, 4095, 0.001],
    [0.05, 2133.33, 0.001],
    [0.10, 1465.73, 0.001],
    [0.15, 601.46, 0.001],
    [0.20, 383.84, 0.001],
    [0.30, 234.93, 0.001],
    [0.40, 158.03, 0.001],
    [0.50, 120.00, 0.001],
    [0.60, 104.09, 0.001],
    [0.70, 67.19, 0.001]
]


# --- Helper Functions ---
def sensor_raw_to_distance(raw, table):
    """Converts a raw sensor reading to a distance."""
    if not isinstance(raw, (int, float)) or raw < 0:
        return -1

    table = sorted(table, key=lambda x: x[1], reverse=True)

    if raw >= table[0][1] or raw <= table[-1][1]:
        return -1

    for i in range(len(table) - 1):
        d1, r1, _ = table[i]
        d2, r2, _ = table[i + 1]
        if r1 >= raw >= r2:
            t = (raw - r2) / (r1 - r2) if (r1 - r2) else 0
            return d2 + t * (d1 - d2)

    return -1


def normalize_angle(a):
    """Normalizes an angle to the range [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def dist(p1, p2):
    """Calculates the Euclidean distance between two points."""
    if p1 is None or p2 is None:
        return float('inf')
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class BaseBugRobotController:
    """A base class for bug algorithm robot controllers."""

    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

        # Motors
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        # Sensors
        self.sensors = [self.robot.getDevice(f'ps{i}') for i in range(8)]
        for s in self.sensors:
            s.enable(self.time_step)
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.time_step)
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.time_step)

        # Pose
        self.pos = START_POS
        self.th = math.pi / 2
        self.path = [list(START_POS)]
        self.last_logged = START_POS

        # State
        self.bug_state = "GOAL_SEEK"
        self.goal = BUG_GOAL_POS
        self.bug_done = False
        self.cab_done = False
        self.steps = 0

        # Sensor groupings
        self.front_ids = (0, 7)
        self.right_ids = (1, 2)
        self.left_ids = (5, 6)

        self._stop()

    def _set_speed(self, l, r):
        """Sets the speed of the left and right motors."""
        self.left_motor.setVelocity(l * MAX_SPEED)
        self.right_motor.setVelocity(r * MAX_SPEED)

    def _stop(self):
        """Stops the robot."""
        self._set_speed(0, 0)

    def _update_pose(self):
        """Updates the robot's position and orientation."""
        g = self.gps.getValues()
        self.pos = (g[0], g[1])
        c = self.compass.getValues()
        self.th = normalize_angle(math.atan2(c[0], c[1]))

        if dist(self.pos, self.last_logged) >= 0.02:
            self.path.append(list(self.pos))
            self.last_logged = self.pos

    def _sensor_distances(self):
        """Reads sensor values and converts them to distances."""
        raws = [s.getValue() for s in self.sensors]
        ds = [sensor_raw_to_distance(r, LASER_LOOKUP_TABLE) for r in raws]
        return [d if 0 <= d <= 0.6 else -1 for d in ds]

    def _front_blocked(self, ds):
        """Checks if the front of the robot is blocked."""
        return any(0 <= ds[i] < OBSTACLE_DISTANCE_THRESHOLD_FRONT for i in self.front_ids)

    def _wall_follow(self, ds, side='left'):
        """Implements the wall-following behavior."""
        if self._front_blocked(ds):
            if side == 'left':
                self._set_speed(0.25, -0.25)  # Pivot right
            else:
                self._set_speed(-0.25, 0.25)  # Pivot left
            return

        ids = self.left_ids if side == 'left' else self.right_ids
        valid_distances = [ds[i] for i in ids if ds[i] != -1]
        wall_dist = min(valid_distances) if valid_distances else -1

        if wall_dist < 0 or wall_dist > WALL_FAR_DISTANCE_THRESHOLD:
            if side == 'left':
                self._set_speed(0.2, 0.4)  # Curve left
            else:
                self._set_speed(0.4, 0.2)  # Curve right
            return

        error = TARGET_WALL_DISTANCE_SIDE - wall_dist
        Kp = 1.5
        correction = Kp * error

        sign = 1 if side == 'left' else -1
        base_speed = 0.5
        l_speed = base_speed - sign * correction
        r_speed = base_speed + sign * correction

        self._set_speed(max(-1, min(1, l_speed)), max(-1, min(1, r_speed)))

    def _angle_to(self, target):
        """Calculates the angle to a target point."""
        return normalize_angle(math.atan2(target[1] - self.pos[1],
                                          target[0] - self.pos[0]))

    def _align(self, target_angle):
        """Aligns the robot to a specific angle."""
        diff = normalize_angle(target_angle - self.th)
        if abs(diff) > ALIGNMENT_TOLERANCE_RADIANS:
            turn_speed = 0.30 if abs(diff) > 0.5 else 0.15
            self._set_speed(-turn_speed if diff > 0 else turn_speed,
                            turn_speed if diff > 0 else -turn_speed)
            return False
        self._stop()
        return True

    def _cabinet_step(self, ds):
        """Handles the final approach to the cabinet."""
        if dist(self.pos, CABINET_POS) < GOAL_REACHED_THRESHOLD * 1.5:
            self._stop()
            self.cab_done = True
            return
        self._wall_follow(ds, side='right')

    def run(self, max_steps=100000):
        """Main loop of the controller."""
        while self.robot.step(self.time_step) != -1 and self.steps < max_steps:
            self.steps += 1
            self._update_pose()
            ds = self._sensor_distances()

            if not self.bug_done:
                if dist(self.pos, self.goal) < GOAL_REACHED_THRESHOLD:
                    self._stop()
                    self.bug_done = True
                    self.goal = CABINET_POS
                    self.path.append(None)
                    continue
                self._bug_step(ds)
            else:
                if self.cab_done:
                    break
                self._cabinet_step(ds)

        self._stop()
        print(f"Simulation ended after {self.steps} steps.")
        self._plot_path()

    def _bug_step(self, ds):
        """A placeholder for the specific bug algorithm implementation."""
        raise NotImplementedError("This method should be implemented by a subclass.")

    def _plot_path(self):
        """Plots the robot's path."""
        plt.figure(figsize=(10, 8))
        seg_x, seg_y, x, y = [], [], [], []
        for p in self.path + [None]:
            if p is None:
                if x:
                    seg_x.append(x)
                    seg_y.append(y)
                    x, y = [], []
            else:
                x.append(p[0])
                y.append(p[1])

        colors = ['tab:blue', 'tab:green']
        labels = [f'{self.__class__.__name__} path', 'Cabinet approach']
        for i, (xs, ys) in enumerate(zip(seg_x, seg_y)):
            plt.plot(xs, ys, linestyle='-', marker='.', ms=2,
                     color=colors[i % len(colors)], label=labels[i])

        plt.plot(*START_POS, 'ro', ms=8, label='Start')
        plt.plot(*BUG_GOAL_POS, 'go', ms=8, label='Bug goal')
        plt.plot(*CABINET_POS, 'mo', ms=8, label='Cabinet')

        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(f"Path – {self.__class__.__name__}")
        plt.tight_layout()
        plt.savefig(f"{self.__class__.__name__.lower()}_path.png")
        plt.show()
