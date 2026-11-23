import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# CONFIGURATION
CSV_FILE = 'epuck_lidar_points.csv'
GRID_RESOLUTION = 0.15  # meters per grid cell
MAP_PADDING = 1.0  # meters padding on each side
INFLATION_RADIUS = 0.01  # meters (inflate each obstacle by this amount)

# Load the point cloud
df = pd.read_csv(CSV_FILE)
points = df[['x', 'y']].to_numpy()

# Determine map bounds
x_min, y_min = points.min(axis=0) - MAP_PADDING
x_max, y_max = points.max(axis=0) + MAP_PADDING

# Compute grid size
width = int(np.ceil((x_max - x_min) / GRID_RESOLUTION))
height = int(np.ceil((y_max - y_min) / GRID_RESOLUTION))

# Initialize empty occupancy grid
occupancy_grid = np.zeros((height, width), dtype=np.uint8)

# Convert points to grid indices and mark occupied
for x, y in points:
    col = int((x - x_min) / GRID_RESOLUTION)
    row = int((y_max - y) / GRID_RESOLUTION)  # y flipped for image
    if 0 <= row < height and 0 <= col < width:
        occupancy_grid[row, col] = 1

# Obstacle Inflation
# inflate_radius_cells = int(np.ceil(INFLATION_RADIUS / GRID_RESOLUTION))
# structure = np.ones((2 * inflate_radius_cells + 1, 2 * inflate_radius_cells + 1), dtype=np.uint8)
# occupancy_grid = binary_dilation(occupancy_grid, structure=structure).astype(np.uint8)

np.save("occupancy_grid.npy", occupancy_grid)
meta = {
    "resolution": GRID_RESOLUTION,
    "x_min": float(x_min),
    "y_min": float(y_min)
}
with open("occupancy_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved occupancy_grid.npy and occupancy_meta.json")

fig, ax = plt.subplots(figsize=(10, 10))
extent = [x_min, x_max, y_min, y_max]
ax.imshow(occupancy_grid, cmap='Greys', origin='upper',
          extent=extent, interpolation='none')

x_ticks = np.arange(x_min, x_max, GRID_RESOLUTION)
y_ticks = np.arange(y_min, y_max, GRID_RESOLUTION)
ax.set_xticks(x_ticks, minor=True)
ax.set_yticks(y_ticks, minor=True)
ax.grid(which='minor', color='lightgray', linewidth=0.5)

ax.set_title("Occupancy Grid Map")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.set_aspect('equal')

plt.tight_layout()
plt.show()
