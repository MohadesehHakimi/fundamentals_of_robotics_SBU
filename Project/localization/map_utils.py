import numpy as np
import scipy.ndimage


def get_free_space_indices_in_regions(occupancy_grid, x_min, y_min, resolution, regions):
    """
    regions: list of (xmin, xmax, ymin, ymax) in world coords.
    Returns list[(row, col)] for free cells inside the union of regions.
    """
    H, W = occupancy_grid.shape
    rows, cols = np.where(occupancy_grid == 0)

    xs = x_min + cols * resolution
    y_max = y_min + H * resolution
    ys = y_max - (rows + 1) * resolution

    mask = np.zeros_like(xs, dtype=bool)
    for (xmin, xmax, ymin, ymax) in regions:
        mask |= (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)

    return list(zip(rows[mask], cols[mask]))


def load_map(filename):
    grid = np.load(filename)
    return grid


def compute_distance_map(occupancy_grid):
    # 1 = obstacle, 0 = free
    obstacle_mask = occupancy_grid == 1
    distance_map = scipy.ndimage.distance_transform_edt(~obstacle_mask)
    return distance_map


def get_free_space_indices(occupancy_grid, x_min, y_min, resolution, x_range, y_range):
    free_indices = []
    for row in range(occupancy_grid.shape[0]):
        for col in range(occupancy_grid.shape[1]):
            if occupancy_grid[row, col] == 0:
                x = x_min + col * resolution
                y = y_min + (occupancy_grid.shape[0] - row - 1) * resolution
                if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
                    free_indices.append((row, col))
    return free_indices
