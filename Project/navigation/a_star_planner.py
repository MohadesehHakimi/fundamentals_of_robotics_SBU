import heapq
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

# Hard-coded start and goal positions in world coordinates (meters)
START_POS_WORLD = (-8.0, -8.0)
GOAL_QR_DATA = "002"  # The 'data' field of the target QR code
PATH_EXTENSION_METERS = 0.5  # How many meters to extend the path past the QR code
OBSTACLE_IGNORE_METERS = 0.2  # How many meters to ignore obstacles after passing a QR code
OBSTACLE_PROXIMITY_WEIGHT = 10.0  # Weight for wall avoidance penalty. Higher values mean more distance from walls.


def load_map_data():
    """Loads occupancy grid, metadata, and QR code data."""
    try:
        occupancy_grid = np.load("occupancy_grid.npy")
        with open("occupancy_meta.json", "r") as f:
            meta = json.load(f)
        with open("epuck_detected_qr_codes.json", "r") as f:
            qr_data = json.load(f)
        return occupancy_grid, meta, qr_data
    except FileNotFoundError as e:
        print(f"Error loading map data: {e}. Make sure the files are in the 'utils' directory.")
        return None, None, None


def world_to_grid(x, y, meta):
    """Converts world coordinates to grid coordinates."""
    # The y-axis is often flipped in image representations
    col = int((x - meta['x_min']) / meta['resolution'])
    row = int((meta['y_max'] - y) / meta['resolution'])
    return row, col


def grid_to_world(row, col, meta):
    """Converts grid coordinates to world coordinates."""
    x = col * meta['resolution'] + meta['x_min']
    y = meta['y_max'] - row * meta['resolution']
    return x, y


def a_star_search(grid, start, goal, qr_coords, resolution, dist_grid):
    """
    A* search algorithm to find the shortest path from start to goal.
    """
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {(r, c): float('inf') for r in range(rows) for c in range(cols)}
    g_score[start] = 0

    f_score = {(r, c): float('inf') for r in range(rows) for c in range(cols)}
    f_score[start] = heuristic(start, goal)

    # Dictionary to track distance from the last passed QR code
    dist_from_qr = {(r, c): float('inf') for r in range(rows) for c in range(cols)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (current[0] + dr, current[1] + dc)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # Obstacle check with QR pass-through logic
                ignore_dist_cells = OBSTACLE_IGNORE_METERS / resolution

                current_dist_from_last_qr = dist_from_qr.get(current, float('inf'))
                if current in qr_coords:
                    current_dist_from_last_qr = 0

                can_ignore_obstacle = current_dist_from_last_qr < ignore_dist_cells

                if grid[neighbor[0]][neighbor[1]] == 1 and not can_ignore_obstacle:  # Obstacle
                    continue

                move_cost = np.sqrt(dr ** 2 + dc ** 2)  # More accurate cost for diagonal moves

                # Add a penalty for being close to obstacles
                dist_to_obstacle = dist_grid[neighbor]
                penalty = 0
                if dist_to_obstacle > 0:
                    # The closer to an obstacle, the higher the penalty
                    penalty = OBSTACLE_PROXIMITY_WEIGHT / (dist_to_obstacle + 1e-5)

                tentative_g_score = g_score[current] + move_cost + penalty

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)

                    # Update distance from QR for the neighbor
                    if current in qr_coords:
                        dist_from_qr[neighbor] = 1
                    else:
                        dist_from_qr[neighbor] = dist_from_qr.get(current, float('inf')) + 1

                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found


def heuristic(a, b):
    """
    Euclidean distance heuristic for A*.
    """
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def reconstruct_path(came_from, current):
    """
    Reconstructs the path from the came_from map.
    """
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]


def plot_path(grid, path, start, goal, qr_coords, meta):
    """
    Plots the occupancy grid, path, start/goal points, and QR codes.
    """
    plt.figure(figsize=(12, 12))
    extent = [meta['x_min'], meta['x_max'], meta['y_min'], meta['y_max']]
    plt.imshow(grid, cmap='Greys', origin='upper', extent=extent)

    if path:
        path_world = [grid_to_world(r, c, meta) for r, c in path]
        path_x, path_y = zip(*path_world)
        plt.plot(path_x, path_y, '-b', linewidth=2, label='A* Path')

    start_world = grid_to_world(start[0], start[1], meta)
    goal_world = grid_to_world(goal[0], goal[1], meta)

    plt.plot(start_world[0], start_world[1], 'go', markersize=10, label='Start')
    plt.plot(goal_world[0], goal_world[1], 'ro', markersize=10, label='Goal')

    if qr_coords:
        qr_world = [grid_to_world(r, c, meta) for r, c in qr_coords]
        qr_x, qr_y = zip(*qr_world)
        plt.plot(qr_x, qr_y, 'ys', markersize=8, label='QR Codes (Doors)')

    plt.title('A* Path Planning on Occupancy Grid')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend()
    plt.grid(True)
    plt.savefig('a_star_path.png')
    plt.show()


def find_qr_goal_position(qr_data, goal_data):
    """Finds the world coordinates of the goal QR code."""
    for qr in qr_data.get('qr_codes', []):
        if qr.get('data') == goal_data:
            pos = qr.get('position')
            if pos and 'x' in pos and 'y' in pos:
                return (pos['x'], pos['y'])
    return None


def main():
    grid, meta, qr_data = load_map_data()
    if grid is None:
        return

    # Create a distance transform grid for obstacle penalty
    # This grid stores for each cell the distance to the nearest obstacle
    inverted_grid = 1 - grid
    dist_grid = distance_transform_edt(inverted_grid)

    # Find the goal position from QR data
    goal_pos_world = find_qr_goal_position(qr_data, GOAL_QR_DATA)
    if goal_pos_world is None:
        print(f"Error: QR code with data '{GOAL_QR_DATA}' not found in epuck_detected_qr_codes.json.")
        return

    print(f"Found QR code '{GOAL_QR_DATA}' at world position: {goal_pos_world}")

    # To convert world to grid, we need y_max, which isn't in meta.
    # We can derive it from grid shape.
    meta['y_max'] = meta['y_min'] + grid.shape[0] * meta['resolution']
    meta['x_max'] = meta['x_min'] + grid.shape[1] * meta['resolution']

    # Convert QR code positions to grid coordinates and mark as traversable
    qr_grid_coords = []
    if qr_data:
        for qr in qr_data['qr_codes']:
            pos = qr['position']
            qr_row, qr_col = world_to_grid(pos['x'], pos['y'], meta)
            if 0 <= qr_row < grid.shape[0] and 0 <= qr_col < grid.shape[1]:
                grid[qr_row, qr_col] = 0  # Mark as traversable
                qr_grid_coords.append((qr_row, qr_col))

    qr_grid_coords_set = set(qr_grid_coords)
    start_grid = world_to_grid(START_POS_WORLD[0], START_POS_WORLD[1], meta)
    goal_grid = world_to_grid(goal_pos_world[0], goal_pos_world[1], meta)

    # Check if start or goal are in an obstacle
    if grid[start_grid[0]][start_grid[1]] == 1:
        print("Start position is inside an obstacle.")
        return
    if grid[goal_grid[0]][goal_grid[1]] == 1:
        print("Goal position is inside an obstacle.")
        # Do not return, as we might be able to pass through it if it's right behind a door.

    path = a_star_search(grid, start_grid, goal_grid, qr_grid_coords_set, meta['resolution'], dist_grid)

    if path:
        print(f"Path found with {len(path)} points.")

        # Extend the path slightly beyond the goal QR code
        if len(path) >= 2:
            extension_cells = int(PATH_EXTENSION_METERS / meta['resolution'])
            last_point = path[-1]
            second_last_point = path[-2]

            # Normalize the direction vector (dx, dy)
            direction = (last_point[0] - second_last_point[0], last_point[1] - second_last_point[1])

            for i in range(1, extension_cells + 1):
                next_point = (last_point[0] + direction[0] * i, last_point[1] + direction[1] * i)

                # Check if the next point is within grid bounds
                if not (0 <= next_point[0] < grid.shape[0] and 0 <= next_point[1] < grid.shape[1]):
                    break  # Stop if out of bounds

                # Check if the next point is an obstacle
                if grid[next_point[0]][next_point[1]] == 1:
                    break  # Stop if it hits an obstacle

                path.append(next_point)

            print(f"Path extended to {len(path)} points.")

        # Save path to CSV
        path_world = [grid_to_world(r, c, meta) for r, c in path]
        df = pd.DataFrame(path_world, columns=['x', 'y'])
        df.to_csv('a_star_path.csv', index=False)
        print("Path saved to a_star_path.csv")
    else:
        print("No path found.")

    plot_path(grid, path, start_grid, goal_grid, qr_grid_coords, meta)


if __name__ == '__main__':
    main()
