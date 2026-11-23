import math

import numpy as np


def scan_rect_from_local(scan_local_points, particle_pose):
    if not scan_local_points:
        return None
    x, y, th = particle_pose
    c, s = math.cos(th), math.sin(th)
    xs, ys = [], []
    for lx, ly in scan_local_points:
        xw = x + lx * c - ly * s
        yw = y + lx * s + ly * c
        xs.append(xw);
        ys.append(yw)
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    return (xmin, xmax, ymin, ymax)


def rect_area(rect):
    xmin, xmax, ymin, ymax = rect
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def rect_iou(a, b):
    ax1, ax2, ay1, ay2 = a
    bx1, bx2, by1, by2 = b
    ix1 = max(ax1, bx1);
    ix2 = min(ax2, bx2)
    iy1 = max(ay1, by1);
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1);
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = rect_area(a) + rect_area(b) - inter
    return inter / ua if ua > 0 else 0.0


def area_similarity(a, b):
    aa = rect_area(a);
    ab = rect_area(b)
    denom = max(aa, ab, 1e-6)
    return 1.0 - abs(aa - ab) / denom


def perimeter_hit_ratio(rect, occupancy_grid, x_min, y_min, res, samples_per_edge=50):
    H, W = occupancy_grid.shape
    y_max = y_min + H * res
    xmin, xmax, ymin, ymax = rect
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    hits = 0;
    tot = 0
    xs = np.linspace(xmin, xmax, samples_per_edge)
    ys = np.linspace(ymin, ymax, samples_per_edge)
    # bottom/top
    for xw in xs:
        for yw in (ymin, ymax):
            col = int((xw - x_min) / res)
            row = int((y_max - yw) / res)
            if 0 <= row < H and 0 <= col < W:
                tot += 1
                if occupancy_grid[row, col] == 1:
                    hits += 1
    # left/right
    for yw in ys:
        for xw in (xmin, xmax):
            col = int((xw - x_min) / res)
            row = int((y_max - yw) / res)
            if 0 <= row < H and 0 <= col < W:
                tot += 1
                if occupancy_grid[row, col] == 1:
                    hits += 1
    return hits / tot if tot > 0 else 0.0


def prior_area_similarity(rect_scan, room_rects):
    if rect_scan is None or not room_rects:
        return 0.0
    a_scan = rect_area(rect_scan)
    best = 0.0
    for rr in room_rects:
        a_room = rect_area(rr)
        denom = max(a_scan, a_room, 1e-6)
        sim = 1.0 - abs(a_scan - a_room) / denom
        if sim > best:
            best = sim
    return best


def template_match_score(
        scan_local_points, particle_pose,
        occupancy_grid, room_rects,
        x_min, y_min, resolution,
        w_room=0.55, w_perim=0.25, w_prior_area=0.20, area_lambda=0.3
):
    r_scan = scan_rect_from_local(scan_local_points, particle_pose)
    if r_scan is None:
        return 1e-12

    # 1) match to rooms via IoU + area_sim
    best_room = 0.0
    for r_room in room_rects:
        iou = rect_iou(r_scan, r_room)
        asim = area_similarity(r_scan, r_room)
        best_room = max(best_room, iou + area_lambda * asim)

    # 2) perimeter alignment to map walls
    perim = perimeter_hit_ratio(r_scan, occupancy_grid, x_min, y_min, resolution)

    # 3) prior-area similarity (area only, independent of overlap)
    prior_area_sim = prior_area_similarity(r_scan, room_rects)

    # blend (weights sum to 1.0 by default)
    score = (w_room * best_room) + (w_perim * perim) + (w_prior_area * prior_area_sim)
    return max(score, 1e-12)
