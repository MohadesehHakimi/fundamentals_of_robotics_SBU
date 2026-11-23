import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# sys.path.append('/usr/local/webots/lib/controller/python')
# os.environ['WEBOTS_HOME'] = '/usr/local/webots/'

from controller import Robot


# --- Utility Functions ---
def point_to_line_distance(P, A, B):
    """Compute perpendicular distance from point P to line AB"""
    if np.allclose(A, B):
        return np.linalg.norm(P - A)
    return np.abs(np.cross(B - A, A - P)) / np.linalg.norm(B - A)


def cross_2d(a, b):
    """2D cross product"""
    return a[0] * b[1] - a[1] * b[0]


def nearest_neighbor_sort(pts, max_dist=2.0):
    pts = list(pts)
    if not pts:
        return np.array([])
    sorted_pts = [pts.pop(0)]
    while pts:
        last = sorted_pts[-1]
        dists = [np.linalg.norm(p - last) for p in pts]
        idx = np.argmin(dists)
        if dists[idx] > max_dist:
            break
        sorted_pts.append(pts.pop(idx))
    return np.array(sorted_pts), pts


def normalize_angle(angle):
    """Normalize angle to range [-pi, pi]"""
    return math.atan2(math.sin(angle), math.cos(angle))


def get_heading(compass_vals):
    theta = math.atan2(compass_vals[0], compass_vals[1])
    return normalize_angle(theta)


def get_rho(pos, TARGET):
    return np.linalg.norm(np.array(TARGET) - np.array(pos[0:2]))


def get_alpha(pos, heading, TARGET):
    angle = math.atan2(TARGET[1] - pos[1], TARGET[0] - pos[0]) - heading
    return normalize_angle(angle)


def get_beta(alpha, heading, TARGET_HEADING):
    angle = TARGET_HEADING - heading - alpha
    return normalize_angle(angle)


def sensor_value_to_distance(value):
    """Convert sensor reading to distance using lookup table"""
    table = [
        (4095.00, 0.0),
        (2900.00, 0.1),
        (1800.00, 0.3),
        (1050.00, 0.5),
        (620.00, 0.7),
        (400.00, 1.0),
        (280.00, 1.3),
        (200.00, 1.5),
        (140.00, 1.7),
        (95.00, 2.0),
        (60.00, 3.0),
        (40.00, 4.0),
        (28.00, 5.0),
        (20.00, 6.0)
    ]
    if not isinstance(value, (int, float)) or value < 0:
        return -1
    table = sorted(table, key=lambda x: x[0], reverse=True)
    if value >= table[0][0] or value <= table[-1][0]:
        return -1
    for i in range(len(table) - 1):
        v1, d1 = table[i]
        v2, d2 = table[i + 1]
        if v2 <= value <= v1:
            t = (value - v1) / (v2 - v1)
            return d1 + t * (d2 - d1)
    return -1


# Recursive IEPF-based splitting
def split_segments(points, threshold=0.03):
    def recursive(pts):
        if len(pts) <= 2:
            return [(pts[0], pts[-1])]

        start, end = pts[0], pts[-1]
        side_dists = [
            (p, point_to_line_distance(p, start, end), np.sign(cross_2d(end - start, p - start)))
            for p in pts[1:-1]
        ]
        candidates = [x for x in side_dists if x[1] > threshold]

        if len(candidates) == 0:
            return [(start, end)]

        signs = [s for _, _, s in candidates]
        if abs(signs.count(1)) >= 2 or abs(signs.count(-1)) >= 2:
            max_p = max(candidates, key=lambda x: x[1])[0]
            idx = np.where((pts == max_p).all(axis=1))[0][0]
            left = recursive(pts[:idx + 1])
            right = recursive(pts[idx:])
            return left + right
        else:
            return [(start, end)]

    return recursive(points)


# Merge segments using MNE
def merge_segments(segments, all_points, mne_threshold=0.04):
    if not segments:
        return []
    merged = [segments[0]]
    segments = sorted(segments, key=lambda x: (x[0][0], x[0][1]))  # Sort segments by start point coordinates
    i = 1
    while i < len(segments):
        a = merged[-1][0]
        b = segments[i][1]
        idx1 = np.where(np.isclose(all_points, a).all(axis=1))[0]
        idx2 = np.where((all_points == b).all(axis=1))[0]
        if len(idx1) == 0 or len(idx2) == 0:
            merged.append(segments[i])
            i += 1
            continue
        idx1 = idx1[0]
        idx2 = idx2[0]
        if idx2 < idx1:
            idx1, idx2 = idx2, idx1
        segment_points = all_points[idx1:idx2 + 1]

        line_vec = b - a
        norm = np.linalg.norm(line_vec)
        if norm == 0:
            merged.append(segments[i])
            i += 1
            continue

        max_error = max([point_to_line_distance(p, a, b) for p in segment_points])
        mne = max_error / norm

        if mne < mne_threshold:
            merged[-1] = (merged[-1][0], b)  # merge
        else:
            merged.append(segments[i])
        i += 1
    return merged


# --- Robot Setup and Scanning ---

def rotate_and_scan(robot, sensors, sensor_angles, gps, compass, left_motor, right_motor, scanned_points):
    """Perform a 360-degree scan and collect points"""
    time_step = int(robot.getBasicTimeStep())
    rotation_speed = 2.0
    sample_interval = 0.1
    last_sample_time = 0.0
    last_heading = None
    total_rotation = 0.0

    while robot.step(time_step) != -1:
        current_time = robot.getTime()
        compass_vals = compass.getValues()
        heading = get_heading(compass_vals)

        if last_heading is not None:
            diff = heading - last_heading
            if diff > math.pi:
                diff -= 2 * math.pi
            elif diff < -math.pi:
                diff += 2 * math.pi
            total_rotation += abs(diff)

        last_heading = heading

        if total_rotation >= 2 * math.pi:
            left_motor.setVelocity(0)
            right_motor.setVelocity(0)
            break

        if current_time - last_sample_time >= sample_interval:
            pos = gps.getValues()
            distances = [s.getValue() for s in sensors]
            for i, raw_value in enumerate(distances):
                dist = sensor_value_to_distance(raw_value)
                if dist > 0:
                    angle = heading + sensor_angles[i]
                    x_sense = pos[0] + dist * math.sin(angle)
                    y_sense = pos[1] - dist * math.cos(angle)
                    scanned_points.append((x_sense, y_sense))
            last_sample_time = current_time

        left_motor.setVelocity(rotation_speed)
        right_motor.setVelocity(-rotation_speed)


# --- Robot Initialization ---
robot = Robot()
time_step = int(robot.getBasicTimeStep())

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

gps = robot.getDevice('gps')
gps.enable(time_step)

compass = robot.getDevice("compass")
compass.enable(time_step)

sensor_names = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
sensor_angles = [1.27, 0.77, 0.0, 5.21, 4.21, math.pi, 2.37, 1.87]
sensors = [robot.getDevice(name) for name in sensor_names]
for s in sensors:
    s.enable(time_step)

scanned_points = []

# Perform initial scan
rotate_and_scan(robot, sensors, sensor_angles, gps, compass, left_motor, right_motor, scanned_points)

TARGET = [1.0, 7.56]
TARGET_HEADING = 1.57
r = 0.0205
L = 0.053
MAX_SPEED = 6.28
sample_interval = 0.3
last_sample_time = 0

# Main loop
while robot.step(time_step) != -1:
    pos = gps.getValues()
    heading = get_heading(compass.getValues())

    rho = get_rho(pos, TARGET)
    alpha = get_alpha(pos, heading, TARGET)
    beta = get_beta(alpha, heading, TARGET_HEADING)

    if rho < 0.15:
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        rotate_and_scan(robot, sensors, sensor_angles, gps, compass, left_motor, right_motor, scanned_points)
        break

    if math.pi / 2 >= alpha >= -math.pi / 2:
        k_rho, k_beta, k_alpha = 1, -1, 2
    else:
        k_rho, k_beta, k_alpha = -1, 1, -2

    v = k_rho * rho
    omega = k_alpha * alpha + k_beta * beta

    phi_right = (2 * v + omega * L) / (2 * r)
    phi_left = (2 * v - omega * L) / (2 * r)

    phi_right = max(-MAX_SPEED, min(MAX_SPEED, phi_right))
    phi_left = max(-MAX_SPEED, min(MAX_SPEED, phi_left))

    left_motor.setVelocity(phi_left)
    right_motor.setVelocity(phi_right)

    current_time = robot.getTime()
    if current_time - last_sample_time >= sample_interval:
        distances = [s.getValue() for s in sensors]
        for i, raw_value in enumerate(distances):
            dist = sensor_value_to_distance(raw_value)
            if dist > 0:
                angle = heading + sensor_angles[i]
                x_sense = pos[0] + dist * math.sin(angle)
                y_sense = pos[1] - dist * math.cos(angle)
                scanned_points.append((x_sense, y_sense))
        last_sample_time = current_time

# store the points somewhere
np.savetxt("scanned_points.csv", np.array(scanned_points), delimiter=",")
points = np.loadtxt("scanned_points.csv", delimiter=",")


all_split_segments = []
remaining_points = list(points)
while remaining_points:
    c, remaining_points = nearest_neighbor_sort(remaining_points, max_dist=2.0)
    if len(c) < 5:
        continue
    # Apply Split
    split_threshold = 0.001
    split_result = split_segments(c, threshold=split_threshold)
    all_split_segments.extend(split_result)

# Apply Merge
merged_result = merge_segments(all_split_segments, points, mne_threshold=0.15)

# Plotting results
plt.figure()
plt.scatter(points[:, 0], points[:, 1], s=1, label='Scanned Points')
plt.title("Raw Scanned Environment")
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.savefig('raw_environment.png')
plt.close()

plt.figure()
for A, B in all_split_segments:
    plt.plot([A[0], B[0]], [A[1], B[1]], 'b')
plt.title("After Split")
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.axis('equal')
plt.grid(True)
plt.savefig('split_environment.png')
plt.close()

plt.figure()
for A, B in merged_result:
    plt.plot([A[0], B[0]], [A[1], B[1]], 'g')
plt.title("After Merge")
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.axis('equal')
plt.grid(True)
plt.savefig('merged_environment.png')
plt.close()
