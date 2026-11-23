import os
import sys
import math
import matplotlib.pyplot as plt

sys.path.append('/usr/local/webots/lib/controller/python')
os.environ['WEBOTS_HOME'] = '/usr/local/webots/'
from controller import Robot


theta_d = float(input("Enter the destination angle (in degrees): "))
theta_d = math.radians(theta_d)  # convert to radians
Kp = 5.0  # proportional gain
r = 0.0205  # wheel radius in meters
L = 0.053  # distance between wheels
angle_threshold = 0.01  # rad (about 0.5 degrees)

robot = Robot()
time_step = int(robot.getBasicTimeStep())

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


# Normalize angle to [-pi, pi]
def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


theta_d = normalize_angle(theta_d)
theta_vals = []
time_vals = []

start_time = robot.getTime()
while robot.step(time_step) != -1:
    t = robot.getTime()

    # Get current orientation from compass
    compass_vals = compass.getValues()
    theta = math.atan2(compass_vals[0], compass_vals[1])
    print(theta)

    # Calculate error
    e = normalize_angle(theta_d - theta)

    # Logging
    time_vals.append(t - start_time)
    theta_vals.append(theta)

    # Check if error is within the threshold
    if abs(e) < angle_threshold:
        break

    # Proportional controller
    omega = Kp * e

    # Convert to motor angular velocities
    phi_right = (L * omega) / (2 * r)
    phi_left = -phi_right

    right_motor.setVelocity(phi_right)
    left_motor.setVelocity(phi_left)

left_motor.setVelocity(0)
right_motor.setVelocity(0)

plt.figure()
plt.plot(time_vals, theta_vals, label="Current Heading")
plt.axhline(y=theta_d, color='r', linestyle='--', label="Target Heading")
plt.xlabel("Time (s)")
plt.ylabel("Theta (rad)")
plt.title("Robot Orientation Over Time")
plt.grid(True)
plt.legend()
plt.show()
