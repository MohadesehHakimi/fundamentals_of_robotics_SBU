import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# sys.path.append('/usr/local/webots/lib/controller/python')
# os.environ['WEBOTS_HOME'] = '/usr/local/webots/'

from controller import Robot

# Robot parameters
L = 0.052  # distance between wheels [m]
r = 0.0205  # wheel radius [m]


def forward_kinematics(phi_r, phi_l, L, r, theta):
    x_dot_r = r * (phi_r + phi_l) / 2
    theta_dot_r = r * (phi_r - phi_l) / L
    y_dot_r = 0
    R_minus_theta = np.array([
        [np.cos(-theta), np.sin(-theta), 0],
        [-np.sin(-theta), np.cos(-theta), 0],
        [0, 0, 1]
    ])

    return np.dot(R_minus_theta, np.array([x_dot_r, y_dot_r, theta_dot_r]))


def update_position(position, phi_r, phi_l, L, r, dt):
    x, y, theta = position
    x_dot, y_dot, theta_dot = forward_kinematics(phi_r, phi_l, L, r, theta)
    x += x_dot * dt
    y += y_dot * dt
    theta += theta_dot * dt
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return x, y, theta


def velocity_case(case_id, t):
    if case_id == 1:
        return 2.0, 1.0
    elif case_id == 2:
        return -np.pi, np.pi
    elif case_id == 3:
        return -5 * np.cos(t), 5 * np.sin(t)
    else:
        return 0.0, 0.0


# --------- Choose motion case ---------
case_id = 3
# --------------------------------------

# Initialize robot
robot = Robot()
time_step = int(robot.getBasicTimeStep())
dt = time_step / 1000   # seconds

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

gps = robot.getDevice('gps')
gps.enable(time_step)

compass = robot.getDevice('compass')
compass.enable(time_step)

left_encoder = robot.getDevice('left wheel sensor')
right_encoder = robot.getDevice('right wheel sensor')
left_encoder.enable(time_step)
right_encoder.enable(time_step)

# initialize devices
for _ in range(10):
    robot.step(time_step)

last_left_pos = left_encoder.getValue()
last_right_pos = right_encoder.getValue()

# Initial pose for encoder-based tracking
enc_pose = (0.0, 0.0, 0.0)

# Logs
time_series = []
gps_positions = []
compass_angles = []
encoder_positions = []
encoder_angles = []
# run for 30 seconds
for step in range(int(30 / dt)):
    t = step * dt
    phi_l_cmd, phi_r_cmd = velocity_case(case_id, t)

    # Send constant motor commands
    left_motor.setVelocity(phi_l_cmd)
    right_motor.setVelocity(phi_r_cmd)

    # Read encoder at each time step
    left_pos = left_encoder.getValue()
    right_pos = right_encoder.getValue()
    phi_l_enc = (left_pos - last_left_pos) / dt
    phi_r_enc = (right_pos - last_right_pos) / dt
    last_left_pos = left_pos
    last_right_pos = right_pos

    # Update predicted pose using encoder data
    enc_pose = update_position(enc_pose, phi_r_enc, phi_l_enc, L, r, dt)
    encoder_positions.append((enc_pose[0], enc_pose[1]))
    encoder_angles.append(enc_pose[2])

    # Read GPS/Compass for comparison
    gps_val = gps.getValues()
    gps_positions.append((gps_val[0], gps_val[1]))

    compass_val = compass.getValues()
    heading = np.arctan2(compass_val[0], compass_val[1])
    heading = (heading + np.pi) % (2 * np.pi) - np.pi
    compass_angles.append(heading)
    # print(f"gps_x: {gps_val[0]:.3f}, gps_y: {gps_val[1]:.3f}, compass: {heading:.3f}, ")
    # print(f"enc_x: {enc_pose[0]:.3f}, enc_y: {enc_pose[1]:.3f}, enc_theta: {enc_pose[2]:.3f}")
    time_series.append(t)
    robot.step(time_step)

# convert to arrays
gps_positions = np.array(gps_positions)
encoder_positions = np.array(encoder_positions)

# --- Plot X-Y Path ---
plt.figure()
if case_id == 2:    # to avoid zoom in
    plt.plot(encoder_positions[:, 0], encoder_positions[:, 1], 'bo', label="Encoder", markersize=2)
    plt.plot(gps_positions[:, 0], gps_positions[:, 1], '--', label="GPS")
    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.3, 0.3)
else:
    plt.plot(encoder_positions[:, 0], encoder_positions[:, 1], label="Encoder")
    plt.plot(gps_positions[:, 0], gps_positions[:, 1], '--', label="GPS")

plt.title(f"Robot Path Comparison (Case {case_id})")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend()
plt.grid(True)
plt.axis('equal')

# --- Plot θ vs Time ---
plt.figure()
plt.plot(time_series, encoder_angles, label="Encoder θ(t)")
plt.plot(time_series, compass_angles, '--', label="Compass θ(t)")
plt.title(f"Orientation vs Time (Case {case_id})")
plt.xlabel("Time [s]")
plt.ylabel("θ [rad]")
plt.legend()
plt.grid(True)

plt.show()
