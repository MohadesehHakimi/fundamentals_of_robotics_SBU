import math
import os
import sys
import matplotlib.pyplot as plt

# SETUP Webots path on macOS
os.environ['WEBOTS_HOME'] = '/Applications/Webots.app'
sys.path.append('/Applications/Webots.app/Contents/lib/controller/python')

from controller import Robot

r = 0.0205  # wheel radius (meters)
L = 0.053  # distance between wheels (meters)
MAX_SPEED = 6.28  # e-puck motor speed limit
TARGET = [0.0, 0.0]  # target coordinates
k_rho = 1.5  # distance gain
k_alpha = 12.0  # heading gain
k_beta = -0.5  # orientation gain

robot = Robot()
time_step = int(robot.getBasicTimeStep())

# load devices
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
gps = robot.getDevice('gps')
gps.enable(time_step)
compass = robot.getDevice('compass')
compass.enable(time_step)

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0)
right_motor.setVelocity(0)


def normalize_angle(angle):
    """
    Normalize angle to [-π, π].
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def get_rho(pos):
    """
    distance from the robot to the goal.
    """
    return ((TARGET[0] - pos[0]) ** 2 + (TARGET[1] - pos[1]) ** 2) ** 0.5


def get_alpha(pos, heading):
    """
    heading error: angle between the robot’s orientation and the line to the goal.
    """
    angle = math.atan2(TARGET[1] - pos[1], TARGET[0] - pos[0]) - heading
    return normalize_angle(angle)


def get_beta(alpha, heading):
    """
    orientation error: how much the robot’s final heading differs from the goal line
    """
    angle = - heading - alpha
    return normalize_angle(angle)


def get_heading():
    compass_vals = compass.getValues()
    theta = math.atan2(compass_vals[0], compass_vals[1])
    return normalize_angle(theta)


robot.step(time_step)
path = []

while robot.step(time_step) != -1:
    # read state
    pos = gps.getValues()
    heading = get_heading()
    # record current position
    path.append((pos[0], pos[1]))
    print(f"Position: x={pos[0]:.2f}, y={pos[1]:.2f}, heading={math.degrees(heading):.2f}°")

    # compute control law
    rho = get_rho(pos)
    alpha = get_alpha(pos, heading)
    beta = get_beta(alpha, heading)
    print(f"rho={rho:.2f}, alpha={math.degrees(alpha):.2f}°, beta={math.degrees(beta):.2f}°")

    # stop condition
    if abs(rho) < 0.1:
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        print(f"Reached target at x={pos[0]:.4f}, y={pos[1]:.4f}")
        break

    # if goal is more than 90° behind, back up
    if abs(alpha) > math.radians(70):
        rho = -rho
        # rotate heading error into the “backwards” hemisphere
        alpha = normalize_angle(alpha + math.pi)
        beta = normalize_angle(beta + math.pi)

    v = k_rho * rho * math.cos(alpha)
    omega = k_alpha * alpha + k_beta * beta

    # convert to wheel speeds
    phi_right = (2 * v + omega * L) / (2 * r)
    phi_left = (2 * v - omega * L) / (2 * r)

    phi_right = max(-MAX_SPEED, min(MAX_SPEED, phi_right))
    phi_left = max(-MAX_SPEED, min(MAX_SPEED, phi_left))

    right_motor.setVelocity(phi_right)
    left_motor.setVelocity(phi_left)

#  plot the path
x_vals, y_vals = zip(*path)
plt.figure()
plt.plot(x_vals, y_vals, marker='o')
plt.plot(TARGET[0], TARGET[1], 'rx', label='Target', markersize=1)
plt.title('Robot Path')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
