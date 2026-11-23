import os
import sys
import matplotlib.pyplot as plt

sys.path.append('/usr/local/webots/lib/controller/python')
os.environ['WEBOTS_HOME'] = '/usr/local/webots/'
from controller import Robot


r = 0.0205  # wheel radius (meters)
L = 0.053   # distance between wheels (meters)

duration = 30.0  # seconds to run each case

robot = Robot()
time_step = int(robot.getBasicTimeStep())

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
gps = robot.getDevice('gps')
gps.enable(time_step)

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0)
right_motor.setVelocity(0)

cases = [
    {"v": 0.04, "omega": 0.2, "label": "v=0.04 m/s, ω=0.2 rad/s"},
    {"v": 0.01, "omega": -0.3, "label": "v=0.01 m/s, ω=-0.3 rad/s"}
]

plt.figure()

for case in cases:
    v = case["v"]
    omega = case["omega"]
    label = case["label"]

    # Compute wheel angular velocities
    phi_right = (v + (L/2) * omega) / r
    phi_left = (v - (L/2) * omega) / r

    # Reset position logging
    x_vals = []
    y_vals = []

    # reset simulation time
    start_time = robot.getTime()
    while robot.step(time_step) != -1:
        t = robot.getTime()
        if t - start_time > duration:
            break

        # Apply velocities
        right_motor.setVelocity(phi_right)
        left_motor.setVelocity(phi_left)

        # read GPS values
        gps_vals = gps.getValues()
        x_vals.append(gps_vals[0])
        y_vals.append(gps_vals[1])

    # Stop motors
    right_motor.setVelocity(0)
    left_motor.setVelocity(0)

    # Plot trajectory
    plt.plot(x_vals, y_vals, label=label)

plt.title("Robot Trajectories for Different Commands")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
