import os
import sys

sys.path.append('/usr/local/webots/lib/controller/python')
os.environ['WEBOTS_HOME'] = '/usr/local/webots/'

import math
import json
import numpy as np
import datetime
import csv
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
from PIL import Image

from controller import Robot

from map_utils import load_map, compute_distance_map, get_free_space_indices_in_regions
from particle_filter import ParticleFilter
from visualizer import draw_particles_on_map
from map_utils import get_free_space_indices

# ----------------- Config -----------------
TIME_STEP = 8  # ms
BASE_SPEED = 1.2
WHEEL_RADIUS = 0.033  # m
WHEEL_BASE = 0.160  # m
SCAN_PERIOD = 2.0  # seconds
QR_SCANNING_AVAILABLE = True

# Load grid + meta
occupancy_grid = np.load("occupancy_grid.npy")
with open("occupancy_meta.json") as f:
    meta = json.load(f)
RESOLUTION = float(meta["resolution"])
X_MIN = float(meta["x_min"])
Y_MIN = float(meta["y_min"])

# ---- Prior rooms ----
PRIOR_ROOMS = [
    (-10, -6, -13, -8.5),
    (-10, -7.2, -8.5, -6.8),
    (-5.5, -3.5, -13, -8.5),
    (-3.5, 0, -13, -8.5),
    (-4, 0, -5, 0),
    (-3, 0, -6.5, -5.2)
]
ROOM_RECTS = PRIOR_ROOMS[:]

QR_TO_ROOM = {
    "1": PRIOR_ROOMS[0],
    "5": PRIOR_ROOMS[1],
    "7": PRIOR_ROOMS[2],
    "2": PRIOR_ROOMS[3],
    "6": PRIOR_ROOMS[4],
    "4": PRIOR_ROOMS[-1]
}

PRIOR_CONFIDENCE = 0.90

# Free cells inside rooms
free_in = get_free_space_indices_in_regions(
    occupancy_grid, X_MIN, Y_MIN, RESOLUTION, PRIOR_ROOMS
)

# Global free cells
free_global = get_free_space_indices(
    occupancy_grid, X_MIN, Y_MIN, RESOLUTION,
    x_range=(X_MIN, X_MIN + occupancy_grid.shape[1] * RESOLUTION),
    y_range=(Y_MIN, Y_MIN + occupancy_grid.shape[0] * RESOLUTION)
)


# Build initial particle set as a mixture
def make_init_free_indices_mixture(N):
    n_in = max(1, int(round(PRIOR_CONFIDENCE * N)))
    n_out = max(0, N - n_in)
    import random
    pick_in = [random.choice(free_in) for _ in range(n_in)] if free_in else []
    pick_out = [random.choice(free_global) for _ in range(n_out)] if free_global else []
    return pick_in + pick_out


def is_num(x): return (x is not None) and (not math.isnan(x)) and (not math.isinf(x))


def gaussian(x, mu, sigma):
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def lidar_local_xy(lidar):
    """Return LiDAR points in ROBOT frame as [(lx, ly), ...]."""
    data = lidar.getRangeImage()
    res = lidar.getHorizontalResolution()
    fov = lidar.getFov()
    angles = np.linspace(-fov / 2, fov / 2, res)

    out = []
    for i, d in enumerate(data):
        if not is_num(d) or d <= 0.01:
            continue
        a = angles[i]
        lx = d * math.cos(a)
        ly = -d * math.sin(a)  # +y left
        out.append((lx, ly))
    return out


# --- QR scanning helper ---
def scan_qr_codes(camera, existing_qr_codes, min_size=200, min_coverage_ratio=0.02):
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
                found_img_size = img.size
                break
        new_qr_codes = []
        for qr in qr_codes:
            qr_data = qr.data.decode('utf-8')
            qr_type = qr.type

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


def project_local_scan_to_world(scan_local, pose_xytheta):
    """Transform local (lx,ly) scan to world using given pose (x,y,theta)."""
    x0, y0, th = pose_xytheta
    c, s = math.cos(th), math.sin(th)
    out = []
    for lx, ly in scan_local:
        xw = x0 + lx * c - ly * s
        yw = y0 + lx * s + ly * c
        out.append((xw, yw))
    return out


def main():
    # --- Load map and build distance field ---
    occupancy_grid = load_map("occupancy_grid.npy")
    dist_cells = compute_distance_map(occupancy_grid)

    # Seed particles only in those rooms (over free cells)
    free_cells = get_free_space_indices_in_regions(
        occupancy_grid, X_MIN, Y_MIN, RESOLUTION, PRIOR_ROOMS
    )

    N = 150
    init_free_indices = make_init_free_indices_mixture(N)

    pf = ParticleFilter(
        num_particles=N,
        free_indices=init_free_indices,  # already sized N; PF will just read them
        x_min=X_MIN, y_min=Y_MIN, resolution=RESOLUTION,
        grid_height=occupancy_grid.shape[0],
        grid_width=occupancy_grid.shape[1],
        wheel_base=WHEEL_BASE,
        prior_regions=PRIOR_ROOMS,
        prior_outside_scale=(1.0 - PRIOR_CONFIDENCE) / PRIOR_CONFIDENCE
    )

    robot = Robot()

    # --- GPS (for plotting only) ---
    gps = robot.getDevice("gps")
    gps.enable(TIME_STEP)

    # camera
    camera = robot.getDevice("camera")
    camera.enable(TIME_STEP)
    try:
        print(f"[CAMERA] Camera: {camera.getWidth()}x{camera.getHeight()}, FOV: {camera.getFov()}")
    except Exception:
        pass

    # QR Code scanning state
    detected_qr_codes = set()
    qr_data_list = []
    qr_scan_counter = 0
    qr_scan_interval = 10  # every ~10 frames

    # LiDAR
    lidar = robot.getDevice("LDS-01")
    lidar.enable(TIME_STEP)
    lidar.enablePointCloud()
    try:
        lm1 = robot.getDevice("LDS-01_main_motor");
        lm2 = robot.getDevice("LDS-01_secondary_motor")
        lm1.setPosition(float('inf'));
        lm2.setPosition(float('inf'))
        lm1.setVelocity(30.0);
        lm2.setVelocity(60.0)
    except Exception:
        pass

    # Wheels
    right_motor = robot.getDevice("right wheel motor")
    left_motor = robot.getDevice("left wheel motor")
    right_motor.setPosition(float('inf'));
    left_motor.setPosition(float('inf'))
    right_motor.setVelocity(0.0);
    left_motor.setVelocity(0.0)

    # Encoders: prefer named; fallback to motor-attached
    try:
        right_enc = robot.getDevice("right wheel sensor")
        left_enc = robot.getDevice("left wheel sensor")
        right_enc.enable(TIME_STEP);
        left_enc.enable(TIME_STEP)
    except Exception:
        right_enc = right_motor.getPositionSensor()
        left_enc = left_motor.getPositionSensor()
        right_enc.enable(TIME_STEP);
        left_enc.enable(TIME_STEP)

    # Prime
    robot.step(TIME_STEP)
    prev_left = left_enc.getValue()
    prev_right = right_enc.getValue()

    # Odometry pose
    x = y = theta = 0.0

    # Braitenberg coefficients
    lw = lidar.getHorizontalResolution()
    lmax = lidar.getMaxRange()
    coeff = [6 * gaussian(i, lw / 4, lw / 12) for i in range(lw)]
    i0, i1 = int(0.25 * lw), int(0.5 * lw)

    # Timers & wheel travel accumulation between PF updates
    last_pf_t = robot.getTime()
    acc_dl = 0.0
    acc_dr = 0.0

    while robot.step(TIME_STEP) != -1:
        # ---- Obstacle avoidance ----
        qr_scan_counter += 1
        if qr_scan_counter >= qr_scan_interval:
            qr_scan_counter = 0
            new_qr_codes = scan_qr_codes(camera, detected_qr_codes)
            if new_qr_codes:
                # hard-lock PF to the matching room on first valid QR
                locked = False
                for qr in new_qr_codes:
                    qr_data = str(qr.get('data', '')).strip()
                    import re
                    token = ''.join(re.findall(r'\d+', qr_data)) or qr_data
                    if token in QR_TO_ROOM:
                        room_rect = QR_TO_ROOM[token]
                        pf.lock_to_room(room_rect, occupancy_grid,
                                        free_indices_func=get_free_space_indices_in_regions)
                        locked = True
                        print(f"[QR] '{qr_data}' -> LOCK to room {room_rect}")
                        break  # lock to the first recognized QR

        left_speed = BASE_SPEED
        right_speed = BASE_SPEED
        vals = lidar.getRangeImage()

        for i in range(i0, i1):
            j = lw - i - 1
            k = i - i0
            vi, vj = vals[i], vals[j]
            if is_num(vi) and is_num(vj):
                li = 1.0 - vi / lmax
                lj = 1.0 - vj / lmax
                left_speed += coeff[k] * (li - lj)
                right_speed += coeff[k] * (lj - li)

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        # ---- ODOM from encoders ----
        cl = left_enc.getValue();
        cr = right_enc.getValue()
        if is_num(cl) and is_num(cr):
            dl = (cl - prev_left) * WHEEL_RADIUS
            dr = (cr - prev_right) * WHEEL_RADIUS
            prev_left, prev_right = cl, cr

            # integrate pose
            dc = 0.5 * (dl + dr)
            dth = (dr - dl) / WHEEL_BASE
            x += dc * math.cos(theta + 0.5 * dth)
            y += dc * math.sin(theta + 0.5 * dth)
            theta = (theta + dth + math.pi) % (2 * math.pi) - math.pi

            # accumulate for PF motion update
            acc_dl += dl
            acc_dr += dr

        # ---- Every 5s: PF update ----
        now = robot.getTime()
        if now - last_pf_t >= SCAN_PERIOD:
            # 1) Take LiDAR snapshot in LOCAL frame
            scan_local = lidar_local_xy(lidar)

            # 2) Motion update with accumulated wheel travel
            pf.motion_update(acc_dl, acc_dr, occupancy_grid=occupancy_grid, distance_map=dist_cells, keep_free=True,
                             keep_in_bounds=True)
            acc_dl = 0.0;
            acc_dr = 0.0

            # 3) Sensor update + resample
            pf.sensor_update(scan_local, occupancy_grid, ROOM_RECTS)
            pf.resample()
            pf.roughen(0.01, 0.01, occupancy_grid=occupancy_grid, distance_map=dist_cells, keep_free=True,
                       keep_in_bounds=True)
            # 4) Estimate + draw
            est = pf.estimate_pose()

            # For visualization: project the LOCAL scan to world using the *estimate*
            scan_world_for_vis = project_local_scan_to_world(scan_local, est)

            gps_vals = gps.getValues()
            gps_xy = (gps_vals[0], gps_vals[1])

            draw_particles_on_map(
                occupancy_grid=occupancy_grid,
                x_min=X_MIN, y_min=Y_MIN, resolution=RESOLUTION,
                particles=pf.particles,
                estimate=est,
                scan_points_world=scan_world_for_vis,  # for viz only
                gps_xy=gps_xy,
                title=f"PF on Map (N={len(pf.particles)})",
            )

            print(f"[PF] t={now:.1f}s  est=({est[0]:.2f},{est[1]:.2f},{est[2]:.2f} rad)  pts={len(scan_local)}")
            last_pf_t = now

    if qr_data_list:
        with open("qr_positions.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z", "qr_data", "timestamp"])
            for qr in qr_data_list:
                pos = qr.get('position', {})
                writer.writerow([
                    pos.get('x', 0), pos.get('y', 0), pos.get('z', 0),
                    qr.get('data', ''), qr.get('timestamp', '')
                ])
        print(f"[QR] Saved {len(qr_data_list)} QR code positions to qr_positions.csv")

    qr_summary = {
        'total_qr_codes': len(qr_data_list),
        'scanning_enabled': QR_SCANNING_AVAILABLE,
        'qr_codes': qr_data_list if qr_data_list else []
    }
    with open("detected_qr_codes.json", "w") as jf:
        json.dump(qr_summary, jf, indent=2)
    print("[QR] Saved detected_qr_codes.json")


if __name__ == "__main__":
    main()
