import matplotlib.pyplot as plt


def draw_particles_on_map(
        occupancy_grid, x_min, y_min, resolution,
        particles,
        estimate=None,  # (x, y, theta)
        scan_points_world=None,  # list[(x,y)] already in world frame
        gps_xy=None,  # (x,y) ground truth to mark
        title="PF Localization – Map Overlay"
):
    """
    Draw occupancy grid + particles + estimate, scan, and GPS marker.
    """
    H, W = occupancy_grid.shape
    x_max = x_min + W * resolution
    y_max = y_min + H * resolution

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [x_min, x_max, y_min, y_max]

    # Map background; origin='upper' matches our grid build
    ax.imshow(occupancy_grid, cmap='Greys', origin='upper', extent=extent, interpolation='none')

    # Particles
    xs = [p.x for p in particles]
    ys = [p.y for p in particles]
    ax.scatter(xs, ys, s=5, c='tab:gray', alpha=0.9, label='Particles')

    # Scan points (world)
    if scan_points_world:
        sx, sy = zip(*scan_points_world)
        ax.scatter(sx, sy, s=2, c='tab:blue', alpha=0.6, label='Scan (world)')

    # PF estimate
    if estimate is not None:
        ax.plot(estimate[0], estimate[1], marker='*', markersize=12, color='red', label='PF estimate')

    # GPS ground truth
    if gps_xy is not None:
        ax.plot(gps_xy[0], gps_xy[1], marker='o', markersize=8, color='green', label='GPS')

    ax.set_aspect('equal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right')
    fig.tight_layout()
    plt.show()
