import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import csv
import json
from datetime import datetime

from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
from PIL import Image
QR_SCANNING_AVAILABLE = True

os.environ['WEBOTS_HOME'] = '/Applications/Webots.app'
sys.path.append('/Applications/Webots.app/Contents/lib/controller/python')
from controller import Robot


# --- Constants ---
MAX_SPEED = 6.0
TARGET_POS = (-6.8, -6.5)
TARGET_THRESHOLD = 0.1  # meters

TARGET_WALL_DISTANCE_SIDE = 0.3
OBSTACLE_DISTANCE_THRESHOLD_FRONT = 0.35
WALL_FAR_DISTANCE_THRESHOLD = 0.45

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


def sensor_raw_to_distance(raw, table):
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


def dist(p1, p2):
    if p1 is None or p2 is None:
        return float('inf')
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def clamp(value, min_val, max_val):
    return max(min(value, max_val), min_val)


def lidar_to_global_xy(lidar, gps_pos, yaw):
    data = lidar.getRangeImage()
    res = lidar.getHorizontalResolution()
    fov = lidar.getFov()
    angles = np.linspace(-fov / 2, fov / 2, res)

    local_x, local_y = [], []
    for i, d in enumerate(data):
        if math.isinf(d) or d <= 0.01:
            continue
        angle = angles[i]
        lx = d * math.cos(angle)
        ly = -1 * d * math.sin(angle)
        local_x.append(lx)
        local_y.append(ly)

    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    world_x = gps_pos[0] + np.array(local_x) * cos_yaw - np.array(local_y) * sin_yaw
    world_y = gps_pos[1] + np.array(local_x) * sin_yaw + np.array(local_y) * cos_yaw
    return world_x.tolist(), world_y.tolist()


class WallFollowerWithLidar:
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
        self.gps = self.robot.getDevice("gps");
        self.gps.enable(self.time_step)
        self.imu = self.robot.getDevice("inertial unit");
        self.imu.enable(self.time_step)
        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.time_step)
        self.lidar.enablePointCloud()

        self.camera = self.robot.getDevice("camera");
        self.camera.enable(self.time_step)
        try:
            print(f"[CAMERA] Camera: {self.camera.getWidth()}x{self.camera.getHeight()}, FOV: {self.camera.getFov()}")
        except Exception:
            pass

        self.robot.step(self.time_step)
        g = self.gps.getValues()
        self.start_pos = (g[0], g[1])
        self.pos = self.start_pos
        self.last_logged = self.start_pos

        self.front_ids = (0, 7)
        self.left_ids = (5, 6)
        self.right_ids = (1, 2)

        self.lidar_points = []
        self.last_scan_time = self.robot.getTime()

        # QR Code scanning state
        self.detected_qr_codes = set()
        self.qr_data_list = []
        self.qr_scan_counter = 0
        self.qr_scan_interval = 10  # scan every ~10 frames

        self._stop()

    def _set_speed(self, l, r):
        self.left_motor.setVelocity(l * MAX_SPEED)
        self.right_motor.setVelocity(r * MAX_SPEED)

    def _stop(self):
        self._set_speed(0, 0)

    def _update_pose(self):
        g = self.gps.getValues()
        self.pos = (g[0], g[1])

    def _sensor_distances(self):
        raws = [s.getValue() for s in self.sensors]
        ds = [sensor_raw_to_distance(r, LASER_LOOKUP_TABLE) for r in raws]
        return [d if 0 <= d <= 0.6 else -1 for d in ds]

    def _front_blocked(self, ds):
        return any(0 <= ds[i] < OBSTACLE_DISTANCE_THRESHOLD_FRONT for i in self.front_ids)

    def _wall_follow(self, ds, side='left'):
        if self._front_blocked(ds):
            if side == 'left':
                self._set_speed(0.25, -0.25)
            else:
                self._set_speed(-0.25, 0.25)
            return

        ids = self.left_ids if side == 'left' else self.right_ids
        valid_distances = [ds[i] for i in ids if ds[i] != -1]
        wall_dist = min(valid_distances) if valid_distances else -1

        if wall_dist < 0 or wall_dist > WALL_FAR_DISTANCE_THRESHOLD:
            if side == 'left':
                self._set_speed(0.2, 0.4)
            else:
                self._set_speed(0.4, 0.2)
            return

        error = TARGET_WALL_DISTANCE_SIDE - wall_dist
        Kp = 1.5
        correction = Kp * error

        sign = 1 if side == 'left' else -1
        base_speed = 1.0
        l_speed = base_speed - sign * correction
        r_speed = base_speed + sign * correction

        self._set_speed(clamp(l_speed, -MAX_SPEED, MAX_SPEED), clamp(r_speed, -MAX_SPEED, MAX_SPEED))

    def _maybe_scan_lidar(self):
        current_time = self.robot.getTime()
        if current_time - self.last_scan_time >= 10.0:
            gps_vals = self.gps.getValues()
            yaw = self.imu.getRollPitchYaw()[2]
            wx, wy = lidar_to_global_xy(self.lidar, gps_vals[:2], yaw)
            for x, y in zip(wx, wy):
                self.lidar_points.append((x, y))
            self.last_scan_time = current_time
            print(f"[LIDAR] Sampled {len(wx)} points at t={current_time:.1f}s")

    def run(self):
        print(f"Starting at: {self.start_pos}")
        step_count = 0

        while self.robot.step(self.time_step) != -1:
            step_count += 1
            self._update_pose()
            ds = self._sensor_distances()
            self._wall_follow(ds, side='left')
            self._maybe_scan_lidar()

            # Periodic QR scanning
            if QR_SCANNING_AVAILABLE:
                self.qr_scan_counter += 1
                if self.qr_scan_counter >= self.qr_scan_interval:
                    self.qr_scan_counter = 0
                    new_qr_codes = scan_qr_codes(self.camera, self.detected_qr_codes)
                    if new_qr_codes:
                        g = self.gps.getValues()
                        yaw = self.imu.getRollPitchYaw()[2]
                        for qr in new_qr_codes:
                            qr['position'] = {'x': g[0], 'y': g[1], 'z': g[2]}
                            qr['yaw'] = yaw
                            self.qr_data_list.append(qr)
                        print(f"[QR] Detected {len(new_qr_codes)} new QR codes at ({g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f})")

            if step_count >= 5000 and dist(self.pos, TARGET_POS) < TARGET_THRESHOLD:
                print(f"Reached target position {TARGET_POS} after {step_count} steps.")
                break

        self._stop()
        print("Wall-following complete.")
        self._plot_lidar_points()
        self._save_lidar_to_csv()
        self._save_qr_outputs()

    def _plot_lidar_points(self):
        if not self.lidar_points:
            print("No LiDAR points recorded.")
            return
        xs, ys = zip(*self.lidar_points)
        plt.figure(figsize=(8, 6))
        plt.scatter(xs, ys, s=2, color='blue', label='LiDAR Points')

        # Overlay QR code positions as red X markers
        if self.qr_data_list:
            qxs, qys = [], []
            for qr in self.qr_data_list:
                if 'position' in qr:
                    qxs.append(qr['position']['x'])
                    qys.append(qr['position']['y'])
            if qxs:
                plt.scatter(qxs, qys, s=100, color='red', marker='x', label='QR Codes')
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("e-puck Wall Following – LiDAR Scan")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("epuck_wall_follow_scan.png")
        plt.show()

    def _save_lidar_to_csv(self, filename="epuck_lidar_points.csv"):
        if not self.lidar_points:
            print("No LiDAR data to save.")
            return
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y"])
            writer.writerows(self.lidar_points)
        print(f"Saved {len(self.lidar_points)} LiDAR points to '{filename}'.")

    def _save_qr_outputs(self):
        # Save QR positions CSV
        if self.qr_data_list:
            with open("epuck_qr_positions.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "z", "qr_data", "timestamp"]) 
                for qr in self.qr_data_list:
                    pos = qr.get('position', {})
                    writer.writerow([
                        pos.get('x', 0), pos.get('y', 0), pos.get('z', 0),
                        qr.get('data', ''), qr.get('timestamp', '')
                    ])
            print(f"[QR] Saved {len(self.qr_data_list)} QR code positions to epuck_qr_positions.csv")

        # Save QR JSON summary
        qr_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_qr_codes': len(self.qr_data_list),
            'scanning_enabled': QR_SCANNING_AVAILABLE,
            'qr_codes': self.qr_data_list if self.qr_data_list else []
        }
        with open("epuck_detected_qr_codes.json", "w") as jf:
            json.dump(qr_summary, jf, indent=2)
        print("[QR] Saved epuck_detected_qr_codes.json")


def scan_qr_codes(camera, existing_qr_codes, min_size=10000, min_coverage_ratio=0.8):
    if not QR_SCANNING_AVAILABLE:
        return []
    try:
        image_array = camera.getImageArray()
        if not image_array:
            return []
        width = camera.getWidth()
        height = camera.getHeight()
        if height <= 0 or width <= 0:
            height = len(image_array)
            width = len(image_array[0]) if height > 0 else 0
        if height == 0 or width == 0:
            return []
        image_data = []
        for row in image_array:
            for pixel in row:
                image_data.extend([pixel[0], pixel[1], pixel[2]])
        image = Image.frombytes('RGB', (width, height), bytes(image_data))
        if width > min_size or height > min_size:
            aspect_ratio = width / height
            if width > height:
                new_width = min_size
                new_height = int(min_size / aspect_ratio)
            else:
                new_height = min_size
                new_width = int(min_size * aspect_ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        candidates = []
        gray_image = image.convert('L')
        candidates.append(gray_image)
        candidates.append(image)
        min_side = min(image.size)
        if min_side < 240:
            scale = max(1, int(320 / max(1, min_side)))
            up_w, up_h = image.size[0] * scale, image.size[1] * scale
            candidates.append(gray_image.resize((up_w, up_h), Image.BICUBIC))
        variants = []
        for img in candidates:
            variants.append(img)
            variants.append(img.rotate(90, expand=True))
            variants.append(img.rotate(180, expand=True))
            variants.append(img.rotate(270, expand=True))
        qr_codes = []
        found_img_size = None
        for img in variants:
            qr_codes = pyzbar.decode(img, symbols=[ZBarSymbol.QRCODE])
            if qr_codes:
                found_img_size = img.size  # (width, height) of the image used for decoding
                break
        new_qr_codes = []
        for qr in qr_codes:
            qr_data = qr.data.decode('utf-8')
            qr_type = qr.type

            # Determine QR bounding box size
            qr_w, qr_h = 0, 0
            rect = getattr(qr, 'rect', None)
            if rect is not None:
                qr_w, qr_h = getattr(rect, 'width', 0), getattr(rect, 'height', 0)
            else:
                poly = getattr(qr, 'polygon', None) or []
                if poly:
                    xs = [p.x for p in poly]
                    ys = [p.y for p in poly]
                    qr_w = max(xs) - min(xs)
                    qr_h = max(ys) - min(ys)

            img_w, img_h = found_img_size if found_img_size else image.size
            coverage_w = (qr_w / max(1, img_w)) if img_w else 0
            coverage_h = (qr_h / max(1, img_h)) if img_h else 0

            # Only accept QR if width or height covers at least the threshold
            if (coverage_w >= min_coverage_ratio) or (coverage_h >= min_coverage_ratio):
                qr_id = f"{qr_type}:{qr_data}"
                if qr_id not in existing_qr_codes:
                    new_qr_codes.append({
                        'type': qr_type,
                        'data': qr_data,
                        'id': qr_id,
                        'timestamp': datetime.now().isoformat(),
                        'position': {'x': 0, 'y': 0, 'z': 0}
                    })
                    existing_qr_codes.add(qr_id)
        return new_qr_codes
    except Exception:
        return []


# --- Main ---
if __name__ == "__main__":
    controller = WallFollowerWithLidar()
    controller.run()
